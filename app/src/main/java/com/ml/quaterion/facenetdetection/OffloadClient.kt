package com.ml.quaterion.facenetdetection

import android.graphics.Bitmap
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import okhttp3.ConnectionPool
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

data class ServerFaceItem(
    @SerializedName("bbox") val bbox: List<Int>? = null,
    @SerializedName("mask_label") val maskLabel: String = "no mask",
    @SerializedName("embedding") val embedding: List<Float> = emptyList()
)

data class ServerInferPayload(
    @SerializedName("results") val results: List<ServerFaceItem> = emptyList(),
    @SerializedName("server_detect_ms") val serverDetectMs: Double = 0.0,
    @SerializedName("server_mask_ms") val serverMaskMs: Double = 0.0,
    @SerializedName("server_facenet_ms") val serverFacenetMs: Double = 0.0,
    @SerializedName("server_infer_ms") val serverInferMs: Double = 0.0
)

data class OffloadCallResult(
    val payload: ServerInferPayload,
    val httpRttMs: Double,
    val encodeMs: Double,
    val payloadBytes: Long
)

data class PingResult(
    val ok: Boolean,
    val rttMs: Double,
    val detail: String
)

/**
 * HTTP Keep-Alive client for offloading M2/M3/M4 to the PC Edge Inference Microservice.
 * Uses pure binary multipart/form-data (no Base64 bloat) and pre-resizes face crops to 160x160.
 */
class OffloadClient(
    @Volatile var baseUrl: String = DEFAULT_SERVER_URL,
    @Volatile var jpegQuality: Int = 75,
    @Volatile var netProfile: NetProfile = NetProfile.REAL
) {
    companion object {
        // Default points to localhost so `adb reverse tcp:8000 tcp:8000` works out of the box;
        // users can switch to PC LAN IP (e.g. http://192.168.x.x:8000) in the settings dialog.
        const val DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
        private const val CROP_TARGET_SIZE = 160
        private val JPEG_MEDIA_TYPE = "image/jpeg".toMediaType()
    }

    private val gson = Gson()

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectionPool(ConnectionPool(5, 5, TimeUnit.MINUTES))
        .connectTimeout(2000, TimeUnit.MILLISECONDS)
        .readTimeout(4000, TimeUnit.MILLISECONDS)
        .writeTimeout(3000, TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private fun normalizedUrl(path: String): String {
        val trimmed = baseUrl.trim().trimEnd('/')
        return "$trimmed$path"
    }

    private fun encodeJpeg(bitmap: Bitmap, quality: Int): ByteArray {
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality.coerceIn(20, 100), baos)
        return baos.toByteArray()
    }

    private fun resizeAndEncodeCrop(crop: Bitmap, quality: Int): ByteArray {
        val resized = if (crop.width != CROP_TARGET_SIZE || crop.height != CROP_TARGET_SIZE) {
            Bitmap.createScaledBitmap(crop, CROP_TARGET_SIZE, CROP_TARGET_SIZE, true)
        } else {
            crop
        }
        val bytes = encodeJpeg(resized, quality)
        if (resized !== crop) {
            resized.recycle()
        }
        return bytes
    }

    fun ping(): PingResult {
        val t0 = System.nanoTime()
        return try {
            val req = Request.Builder()
                .url(normalizedUrl("/ping"))
                .header("X-Net-Profile", netProfile.headerValue)
                .get()
                .build()
            client.newCall(req).execute().use { resp ->
                val rttMs = (System.nanoTime() - t0) / 1_000_000.0
                if (resp.isSuccessful) {
                    val body = resp.body?.string() ?: ""
                    PingResult(true, rttMs, body)
                } else {
                    PingResult(false, rttMs, "HTTP ${resp.code}")
                }
            }
        } catch (e: Exception) {
            val rttMs = (System.nanoTime() - t0) / 1_000_000.0
            PingResult(false, rttMs, e.message ?: "Connection failed")
        }
    }

    fun inferC1FullFrame(fullFrame: Bitmap, enableMask: Boolean): OffloadCallResult {
        val q = jpegQuality
        val tEncStart = System.nanoTime()
        val jpegBytes = encodeJpeg(fullFrame, q)
        val encodeMs = (System.nanoTime() - tEncStart) / 1_000_000.0

        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                "frame.jpg",
                jpegBytes.toRequestBody(JPEG_MEDIA_TYPE)
            )
            .addFormDataPart("enable_mask", enableMask.toString())
            .build()

        return executeMultipart("/infer_c1_fullframe", body, encodeMs, jpegBytes.size.toLong())
    }

    fun inferC2Crops(crops: List<Bitmap>, enableMask: Boolean): OffloadCallResult {
        val q = jpegQuality
        val tEncStart = System.nanoTime()
        var payloadBytes = 0L
        val builder = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("enable_mask", enableMask.toString())

        crops.forEachIndexed { idx, crop ->
            val bytes = resizeAndEncodeCrop(crop, q)
            payloadBytes += bytes.size.toLong()
            builder.addFormDataPart(
                "files",
                "face_$idx.jpg",
                bytes.toRequestBody(JPEG_MEDIA_TYPE)
            )
        }
        val encodeMs = (System.nanoTime() - tEncStart) / 1_000_000.0

        return executeMultipart("/infer_c2_crops", builder.build(), encodeMs, payloadBytes)
    }

    fun inferC3FaceNetOnly(unmaskedCrops: List<Bitmap>): OffloadCallResult {
        val q = jpegQuality
        val tEncStart = System.nanoTime()
        var payloadBytes = 0L
        val builder = MultipartBody.Builder()
            .setType(MultipartBody.FORM)

        unmaskedCrops.forEachIndexed { idx, crop ->
            val bytes = resizeAndEncodeCrop(crop, q)
            payloadBytes += bytes.size.toLong()
            builder.addFormDataPart(
                "files",
                "face_$idx.jpg",
                bytes.toRequestBody(JPEG_MEDIA_TYPE)
            )
        }
        val encodeMs = (System.nanoTime() - tEncStart) / 1_000_000.0

        return executeMultipart("/infer_c3_facenet", builder.build(), encodeMs, payloadBytes)
    }

    private fun executeMultipart(
        endpoint: String,
        body: MultipartBody,
        encodeMs: Double,
        payloadBytes: Long
    ): OffloadCallResult {
        val req = Request.Builder()
            .url(normalizedUrl(endpoint))
            .header("X-Net-Profile", netProfile.headerValue)
            .post(body)
            .build()

        val tNetStart = System.nanoTime()
        client.newCall(req).execute().use { resp ->
            val httpRttMs = (System.nanoTime() - tNetStart) / 1_000_000.0
            if (!resp.isSuccessful) {
                throw IllegalStateException("Server HTTP ${resp.code}: ${resp.message}")
            }
            val jsonStr = resp.body?.string() ?: throw IllegalStateException("Empty response body")
            val parsed = gson.fromJson(jsonStr, ServerInferPayload::class.java)
            return OffloadCallResult(
                payload = parsed,
                httpRttMs = httpRttMs,
                encodeMs = encodeMs,
                payloadBytes = payloadBytes
            )
        }
    }
}
