"""
PC Edge Inference Engine for Mobile FaceNet Computation Partitioning.

Key guarantees:
1. Exact numerical parity with Android FaceNetModel.StandardizeOp and MaskDetectionModel.
2. Multi-slot warm-started TFLite Interpreter pool with ThreadPoolExecutor so multi-face
   requests (K >= 2) execute in parallel rather than serially.
3. Early-exit pruning in C2: run lightweight M3 (Mask) before heavy M4 (FaceNet) and skip M4
   whenever a mask is detected.
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import tensorflow as tf

    TFLiteInterpreter = tf.lite.Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter  # type: ignore


MASK_LABEL = "mask"
NO_MASK_LABEL = "no mask"


def preprocess_facenet(pil_img: Image.Image, img_size: int = 160) -> np.ndarray:
    """Replicate Android FaceNetModel.kt ResizeOp(BILINEAR) + StandardizeOp."""
    rgb = pil_img.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32)
    mean = float(np.mean(pixels))
    std = float(np.sqrt(np.mean((pixels - mean) ** 2)))
    std_adj = max(std, 1.0 / float(np.sqrt(pixels.size)))
    standardized = (pixels - mean) / std_adj
    return np.expand_dims(standardized, axis=0).astype(np.float32)


def preprocess_mask(pil_img: Image.Image, img_size: int = 224) -> np.ndarray:
    """Replicate Android MaskDetectionModel.kt ResizeOp(224, BILINEAR) + NormalizeOp(127.5, 127.5)."""
    rgb = pil_img.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32)
    normalized = (pixels - 127.5) / 127.5
    return np.expand_dims(normalized, axis=0).astype(np.float32)


@dataclass
class FaceInferenceResult:
    bbox: Optional[List[int]]  # [left, top, right, bottom] for C1; None for C2/C3
    mask_label: str
    embedding: List[float]
    mask_ms: float
    facenet_ms: float


class _InterpreterSlot:
    """One thread-isolated pair of (FaceNet, MaskDetector) TFLite interpreters."""

    def __init__(self, facenet_path: Path, mask_path: Path, num_threads: int) -> None:
        self.facenet = TFLiteInterpreter(
            model_path=str(facenet_path),
            num_threads=num_threads,
        )
        self.facenet.allocate_tensors()
        self.fn_in = self.facenet.get_input_details()[0]["index"]
        self.fn_out = self.facenet.get_output_details()[0]["index"]

        self.mask = TFLiteInterpreter(
            model_path=str(mask_path),
            num_threads=num_threads,
        )
        self.mask.allocate_tensors()
        self.mk_in = self.mask.get_input_details()[0]["index"]
        self.mk_out = self.mask.get_output_details()[0]["index"]

        # Warm up both interpreters so first live frame has no cold-start spike.
        dummy_fn = np.zeros((1, 160, 160, 3), dtype=np.float32)
        dummy_mk = np.zeros((1, 224, 224, 3), dtype=np.float32)
        for _ in range(2):
            self.facenet.set_tensor(self.fn_in, dummy_fn)
            self.facenet.invoke()
            self.mask.set_tensor(self.mk_in, dummy_mk)
            self.mask.invoke()

    def detect_mask(self, pil_img: Image.Image) -> Tuple[str, float]:
        t0 = time.perf_counter()
        tensor = preprocess_mask(pil_img)
        self.mask.set_tensor(self.mk_in, tensor)
        self.mask.invoke()
        logits = self.mask.get_tensor(self.mk_out)[0]
        label = MASK_LABEL if int(np.argmax(logits)) == 0 else NO_MASK_LABEL
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return label, elapsed_ms

    def extract_embedding(self, pil_img: Image.Image) -> Tuple[List[float], float]:
        t0 = time.perf_counter()
        tensor = preprocess_facenet(pil_img)
        self.facenet.set_tensor(self.fn_in, tensor)
        self.facenet.invoke()
        vec = self.facenet.get_tensor(self.fn_out)[0]
        # Round to 6 decimals to trim JSON size with zero impact on L2/cosine matching.
        emb = [round(float(x), 6) for x in vec]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return emb, elapsed_ms


class _FaceDetectorBackend:
    """Remote M2 face detector for Cut Point C1 (MediaPipe BlazeFace with OpenCV fallback)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.backend_name = "opencv-haar"
        self._mp_detector = None
        self._haar = None

        try:
            import mediapipe as mp

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5,
                )
                self.backend_name = "mediapipe-blazeface"
        except Exception:
            self._mp_detector = None

        if self._mp_detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            self.backend_name = "opencv-haar"

    def detect(self, pil_img: Image.Image) -> Tuple[List[List[int]], float]:
        t0 = time.perf_counter()
        rgb = np.asarray(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        boxes: List[List[int]] = []

        with self._lock:
            if self._mp_detector is not None:
                res = self._mp_detector.process(rgb)
                if res.detections:
                    for det in res.detections:
                        rb = det.location_data.relative_bounding_box
                        left = max(0, int(rb.xmin * w))
                        top = max(0, int(rb.ymin * h))
                        right = min(w, int((rb.xmin + rb.width) * w))
                        bottom = min(h, int((rb.ymin + rb.height) * h))
                        if right - left >= 16 and bottom - top >= 16:
                            boxes.append([left, top, right, bottom])
            elif self._haar is not None:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                rects = self._haar.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )
                for (x, y, bw, bh) in rects:
                    boxes.append([int(x), int(y), int(x + bw), int(y + bh)])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, elapsed_ms


class EdgeInferenceEngine:
    """Thread-safe parallel inference engine hosting C1, C2, and C3 pipelines."""

    def __init__(self, models_dir: Path, pool_size: int = 4) -> None:
        facenet_path = models_dir / "facenet.tflite"
        mask_path = models_dir / "mask_detector.tflite"
        if not facenet_path.exists():
            raise FileNotFoundError(f"Missing FaceNet model: {facenet_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing Mask model: {mask_path}")

        cpu_count = os.cpu_count() or 4
        threads_per_slot = max(2, cpu_count // max(1, min(pool_size, 4)))

        self.pool_size = pool_size
        self.threads_per_slot = threads_per_slot
        self._slots: queue.Queue[_InterpreterSlot] = queue.Queue()
        for _ in range(pool_size):
            self._slots.put(_InterpreterSlot(facenet_path, mask_path, threads_per_slot))

        self._executor = ThreadPoolExecutor(max_workers=pool_size)
        self._detector = _FaceDetectorBackend()

    @property
    def detector_backend(self) -> str:
        return self._detector.backend_name

    def _run_c2_single(
        self,
        pil_img: Image.Image,
        enable_mask: bool,
        bbox: Optional[List[int]] = None,
    ) -> FaceInferenceResult:
        slot = self._slots.get()
        try:
            mask_label = NO_MASK_LABEL
            mask_ms = 0.0
            if enable_mask:
                mask_label, mask_ms = slot.detect_mask(pil_img)

            # Early-exit pruning: skip heavy M4 FaceNet when wearing a mask.
            if enable_mask and mask_label == MASK_LABEL:
                return FaceInferenceResult(
                    bbox=bbox,
                    mask_label=mask_label,
                    embedding=[],
                    mask_ms=mask_ms,
                    facenet_ms=0.0,
                )

            embedding, facenet_ms = slot.extract_embedding(pil_img)
            return FaceInferenceResult(
                bbox=bbox,
                mask_label=mask_label,
                embedding=embedding,
                mask_ms=mask_ms,
                facenet_ms=facenet_ms,
            )
        finally:
            self._slots.put(slot)

    def _run_c3_single(self, pil_img: Image.Image) -> FaceInferenceResult:
        slot = self._slots.get()
        try:
            embedding, facenet_ms = slot.extract_embedding(pil_img)
            return FaceInferenceResult(
                bbox=None,
                mask_label=NO_MASK_LABEL,
                embedding=embedding,
                mask_ms=0.0,
                facenet_ms=facenet_ms,
            )
        finally:
            self._slots.put(slot)

    def infer_c2_crops(
        self,
        raw_images: List[bytes],
        enable_mask: bool = True,
    ) -> dict:
        t_start = time.perf_counter()
        pil_imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in raw_images]

        if len(pil_imgs) == 0:
            return {
                "results": [],
                "server_detect_ms": 0.0,
                "server_mask_ms": 0.0,
                "server_facenet_ms": 0.0,
                "server_infer_ms": 0.0,
            }

        if len(pil_imgs) == 1:
            results = [self._run_c2_single(pil_imgs[0], enable_mask)]
        else:
            futures = [
                self._executor.submit(self._run_c2_single, img, enable_mask)
                for img in pil_imgs
            ]
            results = [f.result() for f in futures]

        total_ms = (time.perf_counter() - t_start) * 1000.0
        # For parallel execution, sum shows work while wall_ms reflects actual latency.
        # Scale component breakdown proportionally so mask_ms + facenet_ms ~= total_ms.
        raw_mask = sum(r.mask_ms for r in results)
        raw_fn = sum(r.facenet_ms for r in results)
        raw_sum = raw_mask + raw_fn
        if raw_sum > 0:
            mask_ms = total_ms * (raw_mask / raw_sum)
            facenet_ms = total_ms * (raw_fn / raw_sum)
        else:
            mask_ms = 0.0
            facenet_ms = 0.0

        return {
            "results": [
                {"mask_label": r.mask_label, "embedding": r.embedding} for r in results
            ],
            "server_detect_ms": 0.0,
            "server_mask_ms": round(mask_ms, 2),
            "server_facenet_ms": round(facenet_ms, 2),
            "server_infer_ms": round(total_ms, 2),
        }

    def infer_c3_facenet(self, raw_images: List[bytes]) -> dict:
        t_start = time.perf_counter()
        pil_imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in raw_images]

        if len(pil_imgs) == 0:
            return {
                "results": [],
                "server_detect_ms": 0.0,
                "server_mask_ms": 0.0,
                "server_facenet_ms": 0.0,
                "server_infer_ms": 0.0,
            }

        if len(pil_imgs) == 1:
            results = [self._run_c3_single(pil_imgs[0])]
        else:
            futures = [self._executor.submit(self._run_c3_single, img) for img in pil_imgs]
            results = [f.result() for f in futures]

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "results": [
                {"mask_label": NO_MASK_LABEL, "embedding": r.embedding} for r in results
            ],
            "server_detect_ms": 0.0,
            "server_mask_ms": 0.0,
            "server_facenet_ms": round(total_ms, 2),
            "server_infer_ms": round(total_ms, 2),
        }

    def infer_c1_fullframe(self, raw_frame: bytes, enable_mask: bool = True) -> dict:
        t_start = time.perf_counter()
        full_img = Image.open(io.BytesIO(raw_frame)).convert("RGB")
        boxes, detect_ms = self._detector.detect(full_img)

        if not boxes:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "results": [],
                "server_detect_ms": round(detect_ms, 2),
                "server_mask_ms": 0.0,
                "server_facenet_ms": 0.0,
                "server_infer_ms": round(total_ms, 2),
            }

        crops = [full_img.crop((b[0], b[1], b[2], b[3])) for b in boxes]
        t_stage2 = time.perf_counter()
        if len(crops) == 1:
            results = [self._run_c2_single(crops[0], enable_mask, boxes[0])]
        else:
            futures = [
                self._executor.submit(self._run_c2_single, crop, enable_mask, box)
                for crop, box in zip(crops, boxes)
            ]
            results = [f.result() for f in futures]
        stage2_ms = (time.perf_counter() - t_stage2) * 1000.0

        raw_mask = sum(r.mask_ms for r in results)
        raw_fn = sum(r.facenet_ms for r in results)
        raw_sum = raw_mask + raw_fn
        if raw_sum > 0:
            mask_ms = stage2_ms * (raw_mask / raw_sum)
            facenet_ms = stage2_ms * (raw_fn / raw_sum)
        else:
            mask_ms = 0.0
            facenet_ms = 0.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "results": [
                {
                    "bbox": r.bbox,
                    "mask_label": r.mask_label,
                    "embedding": r.embedding,
                }
                for r in results
            ],
            "server_detect_ms": round(detect_ms, 2),
            "server_mask_ms": round(mask_ms, 2),
            "server_facenet_ms": round(facenet_ms, 2),
            "server_infer_ms": round(total_ms, 2),
        }
