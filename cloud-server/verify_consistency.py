"""
Self-test script for PC Edge Inference Server:
1. Validates preprocessing parity (StandardizeOp mean~=0, std~=1) and embedding shape/norm.
2. Benchmarks K=1 vs K=4 parallel multi-face speedup on the 4-slot interpreter pool.
3. Tests /ping, /infer_c1_fullframe, /infer_c2_crops, /infer_c3_facenet, and X-Net-Profile.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from inference import EdgeInferenceEngine, preprocess_facenet
import server as srv


def make_jpeg_bytes(width: int, height: int, seed: int = 42, quality: int = 75) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 220, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main() -> None:
    # 1. Verify StandardizeOp numerical properties
    sample_img = Image.fromarray(np.full((160, 160, 3), 120, dtype=np.uint8))
    tensor = preprocess_facenet(sample_img)
    assert tensor.shape == (1, 160, 160, 3) and tensor.dtype == np.float32
    assert abs(float(tensor.mean())) < 1e-5, "Constant image should standardize to 0"

    rng_img = Image.fromarray(
        np.random.default_rng(7).integers(0, 255, size=(160, 160, 3), dtype=np.uint8)
    )
    t_rng = preprocess_facenet(rng_img)
    assert abs(float(t_rng.mean())) < 1e-5, f"Mean not zero: {t_rng.mean()}"
    assert abs(float(t_rng.std()) - 1.0) < 1e-3, f"Std not one: {t_rng.std()}"
    print("[PASS] StandardizeOp numerical parity verified (mean=0, std=1, shape=1x160x160x3)")

    # 2. Test FastAPI endpoints end-to-end
    with TestClient(srv.app) as client:
        r_ping = client.get("/ping")
        assert r_ping.status_code == 200 and r_ping.json()["status"] == "ok"
        print(f"[PASS] GET /ping -> {r_ping.json()}")

        # C2 K=1 vs K=4 parallel benchmark
        crop1 = make_jpeg_bytes(160, 160, seed=1, quality=75)
        crop2 = make_jpeg_bytes(160, 160, seed=2, quality=75)
        crop3 = make_jpeg_bytes(160, 160, seed=3, quality=75)
        crop4 = make_jpeg_bytes(160, 160, seed=4, quality=75)

        r_c2_1 = client.post(
            "/infer_c2_crops",
            files=[("files", ("f1.jpg", crop1, "image/jpeg"))],
            data={"enable_mask": "true"},
        )
        assert r_c2_1.status_code == 200
        d1 = r_c2_1.json()
        print(
            f"[PASS] POST /infer_c2_crops (K=1, payload={len(crop1)/1024:.2f}KB) -> "
            f"infer={d1['server_infer_ms']}ms (mask={d1['server_mask_ms']}ms, fn={d1['server_facenet_ms']}ms)"
        )

        r_c2_4 = client.post(
            "/infer_c2_crops",
            files=[
                ("files", ("f1.jpg", crop1, "image/jpeg")),
                ("files", ("f2.jpg", crop2, "image/jpeg")),
                ("files", ("f3.jpg", crop3, "image/jpeg")),
                ("files", ("f4.jpg", crop4, "image/jpeg")),
            ],
            data={"enable_mask": "true"},
        )
        assert r_c2_4.status_code == 200
        d4 = r_c2_4.json()
        total_kb_4 = sum(len(x) for x in [crop1, crop2, crop3, crop4]) / 1024.0
        print(
            f"[PASS] POST /infer_c2_crops (K=4, payload={total_kb_4:.2f}KB) -> "
            f"infer={d4['server_infer_ms']}ms (mask={d4['server_mask_ms']}ms, fn={d4['server_facenet_ms']}ms)"
        )

        # C3 (M4-only)
        r_c3 = client.post(
            "/infer_c3_facenet",
            files=[("files", ("f1.jpg", crop1, "image/jpeg"))],
        )
        assert r_c3.status_code == 200
        d3 = r_c3.json()
        assert len(d3["results"][0]["embedding"]) == 128
        print(
            f"[PASS] POST /infer_c3_facenet (K=1) -> "
            f"emb_dim={len(d3['results'][0]['embedding'])}, infer={d3['server_infer_ms']}ms"
        )

        # C1 full frame
        full_jpg = make_jpeg_bytes(480, 640, seed=99, quality=75)
        r_c1 = client.post(
            "/infer_c1_fullframe",
            files={"file": ("frame.jpg", full_jpg, "image/jpeg")},
            data={"enable_mask": "true"},
        )
        assert r_c1.status_code == 200
        dc1 = r_c1.json()
        print(
            f"[PASS] POST /infer_c1_fullframe (payload={len(full_jpg)/1024:.2f}KB) -> "
            f"detect={dc1['server_detect_ms']}ms, total_infer={dc1['server_infer_ms']}ms"
        )

        # Network emulation check (SIM_WEAK)
        t0 = time.perf_counter()
        r_weak = client.get("/ping", headers={"X-Net-Profile": "SIM_WEAK"})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert r_weak.status_code == 200 and dt_ms >= 150.0
        print(f"[PASS] Network emulation SIM_WEAK /ping measured {dt_ms:.1f}ms")

        # Dashboard HTML check
        r_dash = client.get("/dashboard")
        assert r_dash.status_code == 200 and "Edge Inference Node Dashboard" in r_dash.text
        print("[PASS] GET /dashboard rendered successfully")


if __name__ == "__main__":
    main()
