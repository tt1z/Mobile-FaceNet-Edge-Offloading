package com.ml.quaterion.facenetdetection

import java.util.Locale

/**
 * Fine-grained per-frame performance metrics (Part A Cost Profiling & Part B/C Evaluation).
 */
data class FrameMetrics(
    val timestampMs: Long,
    val mode: PartitionMode,
    val backend: String,
    val faceCount: Int,
    val fps: Double,
    val totalMs: Double,
    val prepMs: Double,
    val netMs: Double,
    val inferMs: Double,
    val maskMs: Double,
    val faceNetMs: Double,
    val matchMs: Double,
    val payloadKb: Double,
    val netProfile: String = "REAL",
    val jpegQuality: Int = 75,
    val note: String = ""
) {
    companion object {
        const val LOGCAT_CSV_TAG = "SA_PERF_CSV"
        const val CSV_HEADER =
            "timestamp_ms,mode,backend,net_profile,jpeg_q,faces,fps,total_ms,prep_ms,net_ms,infer_ms,mask_ms,facenet_ms,match_ms,payload_kb"
    }

    fun toHudText(): String {
        val line1 = String.format(
            Locale.US,
            "Mode: [%s] (%s|%s) | K=%d | FPS: %.1f",
            mode.tag,
            backend,
            netProfile,
            faceCount,
            fps
        )
        val line2 = String.format(
            Locale.US,
            "Total: %.1fms (Prep:%.1f | Net:%.1f | Infer:%.1f | M5:%.1f)",
            totalMs,
            prepMs,
            netMs,
            inferMs,
            matchMs
        )
        val line3 = String.format(
            Locale.US,
            "M3(Mask):%.1fms | M4(FN):%.1fms | Payload: %.1f KB (Q%d)",
            maskMs,
            faceNetMs,
            payloadKb,
            jpegQuality
        )
        return if (note.isNotEmpty()) {
            "$line1\n$line2\n$line3\n$note"
        } else {
            "$line1\n$line2\n$line3"
        }
    }

    fun toCsvRow(): String {
        return String.format(
            Locale.US,
            "%d,%s,%s,%s,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
            timestampMs,
            mode.tag,
            backend,
            netProfile,
            jpegQuality,
            faceCount,
            fps,
            totalMs,
            prepMs,
            netMs,
            inferMs,
            maskMs,
            faceNetMs,
            matchMs,
            payloadKb
        )
    }
}
