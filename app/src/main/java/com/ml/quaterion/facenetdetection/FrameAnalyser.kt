/*
 * Copyright 2023 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ml.quaterion.facenetdetection

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.quaterion.facenetdetection.model.FaceNetModel
import com.ml.quaterion.facenetdetection.model.MaskDetectionModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.pow
import kotlin.math.sqrt

// Analyser class supporting All-Local, Cut-C1, Cut-C2, Cut-C3, and Part C Auto-Dynamic Partitioning.
class FrameAnalyser(
    context: Context,
    private var boundingBoxOverlay: BoundingBoxOverlay,
    private var model: FaceNetModel,
    val offloadClient: OffloadClient = OffloadClient(),
    val decisionEngine: PartitionDecisionEngine = PartitionDecisionEngine(offloadClient)
) : ImageAnalysis.Analyzer {

    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    private val detector = FaceDetection.getClient(realTimeOpts)

    private val nameScoreHashmap = HashMap<String, ArrayList<Float>>()

    // Atomic frame lock for backpressure dropping (M1 -> M2..M6).
    private val isProcessing = AtomicBoolean(false)

    // Synchronizes FaceNetModel inference vs runtime CPU/GPU hot-swap.
    private val modelLock = Any()

    // Store the face embeddings in a ( String , FloatArray ) ArrayList.
    @Volatile
    var faceList = ArrayList<Pair<String, FloatArray>>()

    private val maskDetectionModel = MaskDetectionModel(context)

    // Sliding window (10 frames) for smooth effective FPS estimation.
    private val recentFrameTimesMs = ArrayDeque<Double>(10)
    private var csvHeaderLogged = false

    // <-------------- User controls --------------------------->

    // Use any one of the two metrics, "cosine" or "l2"
    private val metricToBeUsed = "l2"

    // Use this variable to enable/disable mask detection.
    private val isMaskDetectionOn = true

    // Active partitioning mode selected in HUD Spinner.
    @Volatile
    var currentMode: PartitionMode = PartitionMode.ALL_LOCAL

    // Callback invoked on Main thread after every analyzed frame.
    var onMetricsUpdated: ((FrameMetrics) -> Unit)? = null

    // <-------------------------------------------------------->

    init {
        boundingBoxOverlay.drawMaskLabel = isMaskDetectionOn
    }

    /**
     * Thread-safely swaps the FaceNetModel instance (e.g., GPU <-> CPU-4T) and releases the old one.
     */
    fun updateFaceNetModel(newModel: FaceNetModel) {
        val oldModel: FaceNetModel
        synchronized(modelLock) {
            oldModel = model
            model = newModel
            recentFrameTimesMs.clear()
        }
        oldModel.close()
    }

    fun resetFpsWindow() {
        synchronized(recentFrameTimesMs) {
            recentFrameTimesMs.clear()
        }
    }

    private fun getThresholdsAndBackend(): Triple<Float, Float, String> {
        synchronized(modelLock) {
            return Triple(model.model.l2Threshold, model.model.cosineThreshold, model.activeBackend)
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(image: ImageProxy) {
        if (faceList.isEmpty() || !isProcessing.compareAndSet(false, true)) {
            image.close()
            return
        }

        val tFrameStartNs = System.nanoTime()
        val modeSnapshot = currentMode

        try {
            // M1: Frame acquisition & rotation correction (Anchored Local).
            val cameraXImage = image.image
            if (cameraXImage == null) {
                isProcessing.set(false)
                image.close()
                return
            }
            var frameBitmap = Bitmap.createBitmap(
                cameraXImage.width,
                cameraXImage.height,
                Bitmap.Config.ARGB_8888
            )
            frameBitmap.copyPixelsFromBuffer(image.planes[0].buffer)
            frameBitmap = BitmapUtils.rotateBitmap(
                frameBitmap,
                image.imageInfo.rotationDegrees.toFloat()
            )
            val tM1EndNs = System.nanoTime()

            if (!boundingBoxOverlay.areDimsInit) {
                boundingBoxOverlay.frameHeight = frameBitmap.height
                boundingBoxOverlay.frameWidth = frameBitmap.width
            }

            // Cut Point C1 skips local M2 MLKit detection and uploads the full rotated frame directly.
            if (modeSnapshot == PartitionMode.CUT_C1_FULL_REMOTE) {
                image.close()
                CoroutineScope(Dispatchers.IO).launch {
                    runC1FullRemote(frameBitmap, tFrameStartNs, tM1EndNs)
                }
                return
            }

            // Local M2 (part 1): MLKit face detection for ALL_LOCAL, CUT_C2, CUT_C3, AUTO_DYNAMIC.
            val inputImage = InputImage.fromBitmap(frameBitmap, 0)
            detector.process(inputImage)
                .addOnSuccessListener { faces ->
                    val tDetectEndNs = System.nanoTime()
                    val dispatcher = if (modeSnapshot == PartitionMode.ALL_LOCAL) {
                        Dispatchers.Default
                    } else {
                        Dispatchers.IO
                    }
                    CoroutineScope(dispatcher).launch {
                        runAfterLocalDetect(faces, frameBitmap, modeSnapshot, tFrameStartNs, tDetectEndNs)
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("Model", "MLKit face detection failed: ${e.message}")
                    isProcessing.set(false)
                }
                .addOnCompleteListener {
                    image.close()
                }
        } catch (e: Exception) {
            Log.e("Model", "Exception in analyze(): ${e.message}")
            isProcessing.set(false)
            image.close()
        }
    }

    /**
     * Mode 1: Cut Point C1 (Full-Frame Offloading: M1 Local -> M2+M3+M4 Remote -> M5+M6 Local).
     */
    private suspend fun runC1FullRemote(
        frameBitmap: Bitmap,
        tFrameStartNs: Long,
        tM1EndNs: Long
    ) {
        withContext(Dispatchers.IO) {
            val predictions = ArrayList<Prediction>()
            val (l2Threshold, cosineThreshold, _) = getThresholdsAndBackend()
            val localFaceList = faceList
            val netProfileTag = offloadClient.netProfile.headerValue
            val jpegQ = offloadClient.jpegQuality

            try {
                val res = offloadClient.inferC1FullFrame(frameBitmap, isMaskDetectionOn)
                var matchTotalNs = 0L

                for (item in res.payload.results) {
                    val b = item.bbox
                    if (b == null || b.size < 4) continue
                    val rect = Rect(b[0], b[1], b[2], b[3])
                    val maskLabel = if (isMaskDetectionOn) item.maskLabel else ""

                    if (!isMaskDetectionOn || maskLabel == maskDetectionModel.NO_MASK) {
                        if (item.embedding.isNotEmpty()) {
                            val tMatchStart = System.nanoTime()
                            val name = matchEmbedding(
                                item.embedding.toFloatArray(),
                                localFaceList,
                                l2Threshold,
                                cosineThreshold
                            )
                            matchTotalNs += (System.nanoTime() - tMatchStart)
                            predictions.add(Prediction(rect, name, maskLabel))
                        }
                    } else {
                        predictions.add(Prediction(rect, "Please remove the mask", maskLabel))
                    }
                }

                val tFrameEndNs = System.nanoTime()
                val prepMs = ((tM1EndNs - tFrameStartNs) / 1_000_000.0) + res.encodeMs
                val inferMs = res.payload.serverInferMs
                val netMs = (res.httpRttMs - inferMs).coerceAtLeast(0.0)
                val matchMs = matchTotalNs / 1_000_000.0
                val totalMs = (tFrameEndNs - tFrameStartNs) / 1_000_000.0

                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = PartitionMode.CUT_C1_FULL_REMOTE,
                        backend = "EDGE-C1",
                        faceCount = res.payload.results.size,
                        totalMs = totalMs,
                        prepMs = prepMs,
                        netMs = netMs,
                        inferMs = inferMs,
                        maskMs = res.payload.serverMaskMs,
                        faceNetMs = res.payload.serverFacenetMs,
                        matchMs = matchMs,
                        payloadKb = res.payloadBytes / 1024.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = String.format(
                            java.util.Locale.US,
                            "Edge M2(Detect): %.1fms",
                            res.payload.serverDetectMs
                        )
                    )
                )
            } catch (e: Exception) {
                val totalMs = (System.nanoTime() - tFrameStartNs) / 1_000_000.0
                Log.e("Offload", "C1 offload failed: ${e.message}")
                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = PartitionMode.CUT_C1_FULL_REMOTE,
                        backend = "EDGE-ERR",
                        faceCount = 0,
                        totalMs = totalMs,
                        prepMs = (tM1EndNs - tFrameStartNs) / 1_000_000.0,
                        netMs = totalMs,
                        inferMs = 0.0,
                        maskMs = 0.0,
                        faceNetMs = 0.0,
                        matchMs = 0.0,
                        payloadKb = 0.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = "[Offload Error] ${e.message?.take(48)}"
                    )
                )
            }
        }
    }

    /**
     * Dispatches ALL_LOCAL, CUT_C2_CROP_OFFLOAD, CUT_C3_FACENET_ONLY, and Part C AUTO_DYNAMIC.
     */
    private suspend fun runAfterLocalDetect(
        faces: List<Face>,
        cameraFrameBitmap: Bitmap,
        modeSnapshot: PartitionMode,
        tFrameStartNs: Long,
        tDetectEndNs: Long
    ) {
        if (modeSnapshot == PartitionMode.AUTO_DYNAMIC) {
            val decision = decisionEngine.decide(faces.size)
            if (decision.targetMode == PartitionMode.CUT_C2_CROP_OFFLOAD) {
                runC2CropOffload(
                    faces = faces,
                    cameraFrameBitmap = cameraFrameBitmap,
                    reportedMode = PartitionMode.AUTO_DYNAMIC,
                    tFrameStartNs = tFrameStartNs,
                    tDetectEndNs = tDetectEndNs,
                    extraNote = decision.reason,
                    allowLocalFallback = true,
                    backendTag = "AUTO-C2"
                )
            } else {
                runAllLocal(
                    faces = faces,
                    cameraFrameBitmap = cameraFrameBitmap,
                    tFrameStartNs = tFrameStartNs,
                    tDetectEndNs = tDetectEndNs,
                    reportedMode = PartitionMode.AUTO_DYNAMIC,
                    extraNote = decision.reason,
                    backendPrefix = "AUTO-",
                    netProfileOverride = offloadClient.netProfile.headerValue
                )
            }
            return
        }

        when (modeSnapshot) {
            PartitionMode.ALL_LOCAL -> runAllLocal(
                faces = faces,
                cameraFrameBitmap = cameraFrameBitmap,
                tFrameStartNs = tFrameStartNs,
                tDetectEndNs = tDetectEndNs
            )
            PartitionMode.CUT_C2_CROP_OFFLOAD -> runC2CropOffload(
                faces = faces,
                cameraFrameBitmap = cameraFrameBitmap,
                reportedMode = PartitionMode.CUT_C2_CROP_OFFLOAD,
                tFrameStartNs = tFrameStartNs,
                tDetectEndNs = tDetectEndNs,
                extraNote = ""
            )
            PartitionMode.CUT_C3_FACENET_ONLY -> runC3FaceNetOffload(
                faces = faces,
                cameraFrameBitmap = cameraFrameBitmap,
                tFrameStartNs = tFrameStartNs,
                tDetectEndNs = tDetectEndNs
            )
            else -> runAllLocal(
                faces = faces,
                cameraFrameBitmap = cameraFrameBitmap,
                tFrameStartNs = tFrameStartNs,
                tDetectEndNs = tDetectEndNs
            )
        }
    }

    /**
     * Mode 0: All-Local baseline (optimized with M3-before-M4 early-exit pruning).
     * Also serves as the local execution path and seamless failover target for AUTO_DYNAMIC.
     */
    private suspend fun runAllLocal(
        faces: List<Face>,
        cameraFrameBitmap: Bitmap,
        tFrameStartNs: Long,
        tDetectEndNs: Long,
        reportedMode: PartitionMode = PartitionMode.ALL_LOCAL,
        extraNote: String = "",
        backendPrefix: String = "",
        netProfileOverride: String = "LOCAL"
    ) {
        withContext(Dispatchers.Default) {
            val predictions = ArrayList<Prediction>()
            val localFaceList = faceList

            var cropTotalNs = 0L
            var maskTotalNs = 0L
            var faceNetTotalNs = 0L
            var matchTotalNs = 0L
            var activeBackend = getThresholdsAndBackend().third

            try {
                for (face in faces) {
                    try {
                        // M2 (part 2): Crop face ROI.
                        val tCropStart = System.nanoTime()
                        val croppedBitmap =
                            BitmapUtils.cropRectFromBitmap(cameraFrameBitmap, face.boundingBox)
                        cropTotalNs += (System.nanoTime() - tCropStart)

                        // M3 first (early-exit optimization): detect mask before running heavy M4 FaceNet!
                        var maskLabel = ""
                        if (isMaskDetectionOn) {
                            val tMaskStart = System.nanoTime()
                            maskLabel = maskDetectionModel.detectMask(croppedBitmap)
                            maskTotalNs += (System.nanoTime() - tMaskStart)
                        }

                        if (!isMaskDetectionOn || maskLabel == maskDetectionModel.NO_MASK) {
                            // M4: FaceNet 128-dim feature extraction (thread-safe hot-swap).
                            val tFaceNetStart = System.nanoTime()
                            val embedding: FloatArray
                            val l2Threshold: Float
                            val cosineThreshold: Float
                            synchronized(modelLock) {
                                activeBackend = model.activeBackend
                                embedding = model.getFaceEmbedding(croppedBitmap)
                                l2Threshold = model.model.l2Threshold
                                cosineThreshold = model.model.cosineThreshold
                            }
                            faceNetTotalNs += (System.nanoTime() - tFaceNetStart)

                            // M5: Feature clustering & distance matching.
                            val tMatchStart = System.nanoTime()
                            val bestName = matchEmbedding(
                                embedding,
                                localFaceList,
                                l2Threshold,
                                cosineThreshold
                            )
                            matchTotalNs += (System.nanoTime() - tMatchStart)

                            predictions.add(Prediction(face.boundingBox, bestName, maskLabel))
                        } else {
                            // Early-exit: skip M4 & M5 when masked.
                            predictions.add(
                                Prediction(face.boundingBox, "Please remove the mask", maskLabel)
                            )
                        }
                    } catch (e: Exception) {
                        Log.e("Model", "Exception in All-Local face loop: ${e.message}")
                        continue
                    }
                }

                val tFrameEndNs = System.nanoTime()
                val prepMs = ((tDetectEndNs - tFrameStartNs) + cropTotalNs) / 1_000_000.0
                val maskMs = maskTotalNs / 1_000_000.0
                val faceNetMs = faceNetTotalNs / 1_000_000.0
                val inferMs = maskMs + faceNetMs
                val matchMs = matchTotalNs / 1_000_000.0
                val totalMs = (tFrameEndNs - tFrameStartNs) / 1_000_000.0

                // Feed observed local compute cost back to the Part C online cost model.
                if (faces.isNotEmpty() && faceNetMs > 0.0) {
                    decisionEngine.recordLocalObservation(faces.size, inferMs)
                }

                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = reportedMode,
                        backend = "$backendPrefix$activeBackend",
                        faceCount = faces.size,
                        totalMs = totalMs,
                        prepMs = prepMs,
                        netMs = 0.0,
                        inferMs = inferMs,
                        maskMs = maskMs,
                        faceNetMs = faceNetMs,
                        matchMs = matchMs,
                        payloadKb = 0.0,
                        netProfile = netProfileOverride,
                        jpegQuality = offloadClient.jpegQuality,
                        note = extraNote
                    )
                )
            } catch (e: Exception) {
                Log.e("Model", "Fatal exception in runAllLocal(): ${e.message}")
                withContext(Dispatchers.Main) { isProcessing.set(false) }
            }
        }
    }

    /**
     * Mode 2: Cut Point C2 (Golden Crop Offload: M1+M2 Local -> M3+M4 Remote -> M5+M6 Local).
     * When allowLocalFallback is true (Part C AUTO_DYNAMIC), network/server errors fail over locally in-frame.
     */
    private suspend fun runC2CropOffload(
        faces: List<Face>,
        cameraFrameBitmap: Bitmap,
        reportedMode: PartitionMode,
        tFrameStartNs: Long,
        tDetectEndNs: Long,
        extraNote: String,
        allowLocalFallback: Boolean = false,
        backendTag: String = "EDGE-C2"
    ) {
        withContext(Dispatchers.IO) {
            val predictions = ArrayList<Prediction>()
            val (l2Threshold, cosineThreshold, _) = getThresholdsAndBackend()
            val localFaceList = faceList
            val netProfileTag = offloadClient.netProfile.headerValue
            val jpegQ = offloadClient.jpegQuality

            // Zero-face pruning: skip network call completely when K == 0!
            if (faces.isEmpty()) {
                val tEnd = System.nanoTime()
                val prepMs = (tDetectEndNs - tFrameStartNs) / 1_000_000.0
                val totalMs = (tEnd - tFrameStartNs) / 1_000_000.0
                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = reportedMode,
                        backend = backendTag,
                        faceCount = 0,
                        totalMs = totalMs,
                        prepMs = prepMs,
                        netMs = 0.0,
                        inferMs = 0.0,
                        maskMs = 0.0,
                        faceNetMs = 0.0,
                        matchMs = 0.0,
                        payloadKb = 0.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = extraNote.ifEmpty { "K=0 pruned (0 KB uploaded)" }
                    )
                )
                return@withContext
            }

            var cropTotalNs = 0L
            val crops = ArrayList<Bitmap>(faces.size)
            val validFaces = ArrayList<Face>(faces.size)
            for (face in faces) {
                try {
                    val tCropStart = System.nanoTime()
                    val cropped = BitmapUtils.cropRectFromBitmap(cameraFrameBitmap, face.boundingBox)
                    cropTotalNs += (System.nanoTime() - tCropStart)
                    crops.add(cropped)
                    validFaces.add(face)
                } catch (e: Exception) {
                    Log.e("Model", "Crop failed in C2: ${e.message}")
                }
            }

            if (crops.isEmpty()) {
                withContext(Dispatchers.Main) { isProcessing.set(false) }
                return@withContext
            }

            try {
                val res = offloadClient.inferC2Crops(crops, isMaskDetectionOn)
                var matchTotalNs = 0L

                val count = minOf(validFaces.size, res.payload.results.size)
                for (i in 0 until count) {
                    val face = validFaces[i]
                    val item = res.payload.results[i]
                    val maskLabel = if (isMaskDetectionOn) item.maskLabel else ""

                    if (!isMaskDetectionOn || maskLabel == maskDetectionModel.NO_MASK) {
                        if (item.embedding.isNotEmpty()) {
                            val tMatchStart = System.nanoTime()
                            val bestName = matchEmbedding(
                                item.embedding.toFloatArray(),
                                localFaceList,
                                l2Threshold,
                                cosineThreshold
                            )
                            matchTotalNs += (System.nanoTime() - tMatchStart)
                            predictions.add(Prediction(face.boundingBox, bestName, maskLabel))
                        }
                    } else {
                        predictions.add(
                            Prediction(face.boundingBox, "Please remove the mask", maskLabel)
                        )
                    }
                }

                val tFrameEndNs = System.nanoTime()
                val prepMs = ((tDetectEndNs - tFrameStartNs) + cropTotalNs) / 1_000_000.0 + res.encodeMs
                val inferMs = res.payload.serverInferMs
                val netMs = (res.httpRttMs - inferMs).coerceAtLeast(0.0)
                val matchMs = matchTotalNs / 1_000_000.0
                val totalMs = (tFrameEndNs - tFrameStartNs) / 1_000_000.0

                // Feed observed offload cost & throughput back to the Part C online cost model.
                decisionEngine.recordOffloadObservation(
                    faceCount = validFaces.size,
                    payloadBytes = res.payloadBytes,
                    netMs = netMs,
                    serverInferMs = inferMs
                )

                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = reportedMode,
                        backend = backendTag,
                        faceCount = validFaces.size,
                        totalMs = totalMs,
                        prepMs = prepMs,
                        netMs = netMs,
                        inferMs = inferMs,
                        maskMs = res.payload.serverMaskMs,
                        faceNetMs = res.payload.serverFacenetMs,
                        matchMs = matchMs,
                        payloadKb = res.payloadBytes / 1024.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = extraNote
                    )
                )
            } catch (e: Exception) {
                Log.e("Offload", "C2 offload failed: ${e.message}")
                if (allowLocalFallback) {
                    // Seamless fault-tolerant failover to local execution!
                    decisionEngine.reportOffloadFailure()
                    runAllLocal(
                        faces = validFaces,
                        cameraFrameBitmap = cameraFrameBitmap,
                        tFrameStartNs = tFrameStartNs,
                        tDetectEndNs = tDetectEndNs,
                        reportedMode = PartitionMode.AUTO_DYNAMIC,
                        extraNote = "⚡ Auto->Local (Failover: Edge dropped)",
                        backendPrefix = "AUTO-",
                        netProfileOverride = netProfileTag
                    )
                    return@withContext
                }
                val totalMs = (System.nanoTime() - tFrameStartNs) / 1_000_000.0
                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = reportedMode,
                        backend = "EDGE-ERR",
                        faceCount = validFaces.size,
                        totalMs = totalMs,
                        prepMs = ((tDetectEndNs - tFrameStartNs) + cropTotalNs) / 1_000_000.0,
                        netMs = totalMs,
                        inferMs = 0.0,
                        maskMs = 0.0,
                        faceNetMs = 0.0,
                        matchMs = 0.0,
                        payloadKb = 0.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = "[Offload Error] ${e.message?.take(48)}"
                    )
                )
            }
        }
    }

    /**
     * Mode 3: Cut Point C3 (Conditional Early-Exit Pruning: M1+M2+M3 Local -> M4-Only Remote -> M5+M6 Local).
     * Masked faces are pruned locally with zero upload and zero cloud compute.
     */
    private suspend fun runC3FaceNetOffload(
        faces: List<Face>,
        cameraFrameBitmap: Bitmap,
        tFrameStartNs: Long,
        tDetectEndNs: Long
    ) {
        withContext(Dispatchers.IO) {
            val predictions = ArrayList<Prediction>()
            val (l2Threshold, cosineThreshold, _) = getThresholdsAndBackend()
            val localFaceList = faceList
            val netProfileTag = offloadClient.netProfile.headerValue
            val jpegQ = offloadClient.jpegQuality

            var cropTotalNs = 0L
            var localMaskNs = 0L
            val unmaskedFaces = ArrayList<Face>()
            val unmaskedCrops = ArrayList<Bitmap>()
            var maskedCount = 0

            for (face in faces) {
                try {
                    val tCropStart = System.nanoTime()
                    val cropped = BitmapUtils.cropRectFromBitmap(cameraFrameBitmap, face.boundingBox)
                    cropTotalNs += (System.nanoTime() - tCropStart)

                    var maskLabel = maskDetectionModel.NO_MASK
                    if (isMaskDetectionOn) {
                        val tMaskStart = System.nanoTime()
                        maskLabel = maskDetectionModel.detectMask(cropped)
                        localMaskNs += (System.nanoTime() - tMaskStart)
                    }

                    if (isMaskDetectionOn && maskLabel != maskDetectionModel.NO_MASK) {
                        maskedCount += 1
                        predictions.add(
                            Prediction(face.boundingBox, "Please remove the mask", maskLabel)
                        )
                    } else {
                        unmaskedFaces.add(face)
                        unmaskedCrops.add(cropped)
                    }
                } catch (e: Exception) {
                    Log.e("Model", "C3 local M2/M3 failed: ${e.message}")
                }
            }

            val localMaskMs = localMaskNs / 1_000_000.0

            // If K==0 or every face is wearing a mask, skip cloud call completely!
            if (unmaskedCrops.isEmpty()) {
                val tEnd = System.nanoTime()
                val prepMs = ((tDetectEndNs - tFrameStartNs) + cropTotalNs) / 1_000_000.0
                val totalMs = (tEnd - tFrameStartNs) / 1_000_000.0
                val note = if (maskedCount > 0) {
                    "C3 pruned $maskedCount masked face(s) locally (0 KB)"
                } else {
                    "K=0 pruned (0 KB uploaded)"
                }
                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = PartitionMode.CUT_C3_FACENET_ONLY,
                        backend = "EDGE-C3",
                        faceCount = faces.size,
                        totalMs = totalMs,
                        prepMs = prepMs,
                        netMs = 0.0,
                        inferMs = localMaskMs,
                        maskMs = localMaskMs,
                        faceNetMs = 0.0,
                        matchMs = 0.0,
                        payloadKb = 0.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = note
                    )
                )
                return@withContext
            }

            try {
                val res = offloadClient.inferC3FaceNetOnly(unmaskedCrops)
                var matchTotalNs = 0L

                val count = minOf(unmaskedFaces.size, res.payload.results.size)
                for (i in 0 until count) {
                    val face = unmaskedFaces[i]
                    val item = res.payload.results[i]
                    if (item.embedding.isNotEmpty()) {
                        val tMatchStart = System.nanoTime()
                        val bestName = matchEmbedding(
                            item.embedding.toFloatArray(),
                            localFaceList,
                            l2Threshold,
                            cosineThreshold
                        )
                        matchTotalNs += (System.nanoTime() - tMatchStart)
                        predictions.add(
                            Prediction(face.boundingBox, bestName, maskDetectionModel.NO_MASK)
                        )
                    }
                }

                val tFrameEndNs = System.nanoTime()
                val prepMs = ((tDetectEndNs - tFrameStartNs) + cropTotalNs) / 1_000_000.0 + res.encodeMs
                val remoteFaceNetMs = res.payload.serverFacenetMs
                val netMs = (res.httpRttMs - res.payload.serverInferMs).coerceAtLeast(0.0)
                val matchMs = matchTotalNs / 1_000_000.0
                val totalMs = (tFrameEndNs - tFrameStartNs) / 1_000_000.0

                val note = if (maskedCount > 0) {
                    "C3 pruned $maskedCount masked, offloaded ${unmaskedCrops.size} M4"
                } else {
                    ""
                }

                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = PartitionMode.CUT_C3_FACENET_ONLY,
                        backend = "EDGE-C3",
                        faceCount = faces.size,
                        totalMs = totalMs,
                        prepMs = prepMs,
                        netMs = netMs,
                        inferMs = localMaskMs + remoteFaceNetMs,
                        maskMs = localMaskMs,
                        faceNetMs = remoteFaceNetMs,
                        matchMs = matchMs,
                        payloadKb = res.payloadBytes / 1024.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = note
                    )
                )
            } catch (e: Exception) {
                val totalMs = (System.nanoTime() - tFrameStartNs) / 1_000_000.0
                Log.e("Offload", "C3 offload failed: ${e.message}")
                publishFrameResult(
                    predictions = predictions,
                    metrics = buildMetrics(
                        mode = PartitionMode.CUT_C3_FACENET_ONLY,
                        backend = "EDGE-ERR",
                        faceCount = faces.size,
                        totalMs = totalMs,
                        prepMs = ((tDetectEndNs - tFrameStartNs) + cropTotalNs) / 1_000_000.0,
                        netMs = totalMs,
                        inferMs = localMaskMs,
                        maskMs = localMaskMs,
                        faceNetMs = 0.0,
                        matchMs = 0.0,
                        payloadKb = 0.0,
                        netProfile = netProfileTag,
                        jpegQuality = jpegQ,
                        note = "[Offload Error] ${e.message?.take(48)}"
                    )
                )
            }
        }
    }

    private fun matchEmbedding(
        embedding: FloatArray,
        localFaceList: ArrayList<Pair<String, FloatArray>>,
        l2Threshold: Float,
        cosineThreshold: Float
    ): String {
        nameScoreHashmap.clear()
        for (i in 0 until localFaceList.size) {
            val personName = localFaceList[i].first
            val score = if (metricToBeUsed == "cosine") {
                cosineSimilarity(embedding, localFaceList[i].second)
            } else {
                L2Norm(embedding, localFaceList[i].second)
            }
            val bucket = nameScoreHashmap[personName]
            if (bucket == null) {
                val list = ArrayList<Float>()
                list.add(score)
                nameScoreHashmap[personName] = list
            } else {
                bucket.add(score)
            }
        }

        val avgScores = nameScoreHashmap.values.map { scores -> scores.toFloatArray().average() }
        val names = nameScoreHashmap.keys.toTypedArray()
        nameScoreHashmap.clear()

        if (avgScores.isEmpty()) return "Unknown"
        return if (metricToBeUsed == "cosine") {
            val maxScore = avgScores.maxOrNull()!!
            if (maxScore > cosineThreshold) names[avgScores.indexOf(maxScore)] else "Unknown"
        } else {
            val minScore = avgScores.minOrNull()!!
            if (minScore > l2Threshold) "Unknown" else names[avgScores.indexOf(minScore)]
        }
    }

    private fun buildMetrics(
        mode: PartitionMode,
        backend: String,
        faceCount: Int,
        totalMs: Double,
        prepMs: Double,
        netMs: Double,
        inferMs: Double,
        maskMs: Double,
        faceNetMs: Double,
        matchMs: Double,
        payloadKb: Double,
        netProfile: String,
        jpegQuality: Int,
        note: String
    ): FrameMetrics {
        val effectiveFps: Double
        synchronized(recentFrameTimesMs) {
            if (recentFrameTimesMs.size >= 10) {
                recentFrameTimesMs.removeFirst()
            }
            recentFrameTimesMs.addLast(totalMs.coerceAtLeast(0.1))
            val avgTotalMs = recentFrameTimesMs.average()
            effectiveFps = if (avgTotalMs > 0.0) 1000.0 / avgTotalMs else 0.0
        }

        val metrics = FrameMetrics(
            timestampMs = System.currentTimeMillis(),
            mode = mode,
            backend = backend,
            faceCount = faceCount,
            fps = effectiveFps,
            totalMs = totalMs,
            prepMs = prepMs,
            netMs = netMs,
            inferMs = inferMs,
            maskMs = maskMs,
            faceNetMs = faceNetMs,
            matchMs = matchMs,
            payloadKb = payloadKb,
            netProfile = netProfile,
            jpegQuality = jpegQuality,
            note = note
        )

        if (!csvHeaderLogged) {
            Log.i(FrameMetrics.LOGCAT_CSV_TAG, FrameMetrics.CSV_HEADER)
            csvHeaderLogged = true
        }
        Log.i(FrameMetrics.LOGCAT_CSV_TAG, metrics.toCsvRow())
        return metrics
    }

    private suspend fun publishFrameResult(
        predictions: ArrayList<Prediction>,
        metrics: FrameMetrics
    ) {
        withContext(Dispatchers.Main) {
            boundingBoxOverlay.faceBoundingBoxes = predictions
            boundingBoxOverlay.invalidate()
            onMetricsUpdated?.invoke(metrics)
            isProcessing.set(false)
        }
    }

    private fun L2Norm(x1: FloatArray, x2: FloatArray): Float {
        return sqrt(x1.mapIndexed { i, xi -> (xi - x2[i]).pow(2) }.sum())
    }

    private fun cosineSimilarity(x1: FloatArray, x2: FloatArray): Float {
        val mag1 = sqrt(x1.map { it * it }.sum())
        val mag2 = sqrt(x2.map { it * it }.sum())
        val dot = x1.mapIndexed { i, xi -> xi * x2[i] }.sum()
        return dot / (mag1 * mag2)
    }
}
