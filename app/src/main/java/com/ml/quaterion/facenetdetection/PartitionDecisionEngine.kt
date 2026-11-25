package com.ml.quaterion.facenetdetection

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.ceil

/**
 * Snapshot of one runtime dynamic partitioning decision (Part C).
 */
data class PartitionDecision(
    val targetMode: PartitionMode, // ALL_LOCAL or CUT_C2_CROP_OFFLOAD
    val reason: String,
    val localCostMs: Double,
    val offloadCostMs: Double,
    val rttMs: Double,
    val bandwidthKbps: Double,
    val isServerReachable: Boolean
)

/**
 * Component-and-Connector (C&C) Runtime Dynamic Partitioning Engine (Part C).
 *
 * Sub-components:
 * 1. Network Probe Connector: Periodic 2s background /ping coroutine tracking EMA RTT and reachability.
 * 2. Online Profiler: Tracks EMA local per-face latency, edge slot latency, crop size, and uplink throughput.
 * 3. Cost-Model Optimizer with Hysteresis: Solves min(Cost_local, Cost_offload) per frame at M2->M3 boundary
 *    with a hysteresis margin (delta = 10ms) and weak-network / offline fault-tolerant fallback.
 */
class PartitionDecisionEngine(
    private val offloadClient: OffloadClient,
    private val edgePoolSlots: Int = 4
) {
    companion object {
        private const val TAG = "PartitionEngine"
        private const val PROBE_INTERVAL_MS = 2000L
        private const val EMA_ALPHA = 0.35
        private const val HYSTERESIS_DELTA_MS = 10.0
        private const val MAX_RTT_FALLBACK_MS = 300.0
        private const val PRIOR_LOCAL_GPU_FACE_MS = 75.0
        private const val PRIOR_LOCAL_CPU_FACE_MS = 150.0
        private const val PRIOR_EDGE_SLOT_MS = 18.0
        private const val PRIOR_CROP_KB = 4.5
        private const val PRIOR_BANDWIDTH_KBPS = 5000.0 // 5 Mbps conservative initial prior
    }

    private val lock = Any()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var probeJob: Job? = null

    // Monitored runtime context (protected by lock)
    private var isServerReachable: Boolean = false
    private var consecutiveFailures: Int = 0
    private var emaRttMs: Double = 20.0
    private var emaBandwidthKbps: Double = PRIOR_BANDWIDTH_KBPS
    private var emaLocalFaceMs: Double = PRIOR_LOCAL_GPU_FACE_MS
    private var emaEdgeSlotMs: Double = PRIOR_EDGE_SLOT_MS
    private var emaCropKb: Double = PRIOR_CROP_KB

    // Last chosen execution path for hysteresis stability
    private var lastDynamicMode: PartitionMode = PartitionMode.ALL_LOCAL

    fun startNetworkProbe() {
        if (probeJob?.isActive == true) return
        probeJob = scope.launch {
            while (isActive) {
                runSingleProbe()
                delay(PROBE_INTERVAL_MS)
            }
        }
    }

    fun stopNetworkProbe() {
        probeJob?.cancel()
        probeJob = null
    }

    /**
     * Triggers an immediate out-of-cycle probe (e.g. right after user changes Server URL or NetProfile).
     */
    fun triggerImmediateProbe() {
        scope.launch {
            // Reset RTT/BW priors to match the newly chosen profile immediately before probe returns
            val profile = offloadClient.netProfile
            synchronized(lock) {
                when (profile) {
                    NetProfile.SIM_WEAK -> {
                        emaRttMs = 190.0
                        emaBandwidthKbps = 300.0
                    }
                    NetProfile.SIM_4G -> {
                        emaRttMs = 55.0
                        emaBandwidthKbps = 3000.0
                    }
                    NetProfile.SIM_5G -> {
                        emaRttMs = 18.0
                        emaBandwidthKbps = 20000.0
                    }
                    NetProfile.REAL -> {
                        emaBandwidthKbps = PRIOR_BANDWIDTH_KBPS
                    }
                }
            }
            runSingleProbe()
        }
    }

    /**
     * Adjusts local cost prior immediately when user toggles GPU <-> CPU-4T.
     */
    fun onLocalBackendChanged(isGpu: Boolean) {
        synchronized(lock) {
            emaLocalFaceMs = if (isGpu) PRIOR_LOCAL_GPU_FACE_MS else PRIOR_LOCAL_CPU_FACE_MS
        }
    }

    private fun runSingleProbe() {
        val res = offloadClient.ping()
        synchronized(lock) {
            if (res.ok) {
                if (!isServerReachable) {
                    // Fresh reconnect: adopt observed RTT directly instead of stale average.
                    emaRttMs = res.rttMs.coerceAtLeast(1.0)
                } else {
                    emaRttMs = ema(emaRttMs, res.rttMs.coerceAtLeast(1.0))
                }
                isServerReachable = true
                consecutiveFailures = 0
            } else {
                consecutiveFailures += 1
                if (consecutiveFailures >= 1) {
                    isServerReachable = false
                }
            }
        }
        Log.d(TAG, "Probe ok=${res.ok} rtt=${String.format(Locale.US, "%.1f", res.rttMs)}ms")
    }

    /**
     * Feeds actual observed local inference time (M3+M4) back into the online cost model.
     */
    fun recordLocalObservation(faceCount: Int, inferMs: Double) {
        if (faceCount <= 0 || inferMs <= 1.0) return
        val perFaceMs = inferMs / faceCount.toDouble()
        synchronized(lock) {
            emaLocalFaceMs = ema(emaLocalFaceMs, perFaceMs.coerceIn(5.0, 600.0))
        }
    }

    /**
     * Feeds actual observed C2/C3 offload timings and payload size back into the online cost model.
     */
    fun recordOffloadObservation(
        faceCount: Int,
        payloadBytes: Long,
        netMs: Double,
        serverInferMs: Double
    ) {
        if (faceCount <= 0) return
        val cropKb = (payloadBytes / 1024.0) / faceCount.toDouble()
        val parallelWaves = ceil(faceCount.toDouble() / edgePoolSlots.toDouble()).coerceAtLeast(1.0)
        val perSlotMs = serverInferMs / parallelWaves

        synchronized(lock) {
            isServerReachable = true
            consecutiveFailures = 0
            if (cropKb in 0.5..80.0) {
                emaCropKb = ema(emaCropKb, cropKb)
            }
            if (perSlotMs in 2.0..500.0) {
                emaEdgeSlotMs = ema(emaEdgeSlotMs, perSlotMs)
            }
            // Estimate effective uplink throughput from net overhead above baseline RTT.
            val txMs = (netMs - emaRttMs).coerceAtLeast(1.0)
            if (payloadBytes > 512) {
                // (bytes * 8 bits) / txMs (ms) == kbps
                val measuredKbps = (payloadBytes * 8.0) / txMs
                emaBandwidthKbps = ema(emaBandwidthKbps, measuredKbps.coerceIn(80.0, 100000.0))
            }
        }
    }

    /**
     * Immediately marks edge node unreachable on runtime request failure so the next frame falls back locally.
     */
    fun reportOffloadFailure() {
        synchronized(lock) {
            consecutiveFailures += 1
            isServerReachable = false
            lastDynamicMode = PartitionMode.ALL_LOCAL
        }
    }

    /**
     * Evaluates the online cost model at the M2 -> M3 boundary for K detected faces.
     */
    fun decide(faceCount: Int): PartitionDecision {
        synchronized(lock) {
            val rtt = emaRttMs
            val bwKbps = emaBandwidthKbps
            val reachable = isServerReachable

            if (faceCount <= 0) {
                return PartitionDecision(
                    targetMode = PartitionMode.ALL_LOCAL,
                    reason = String.format(Locale.US, "⚡ Auto: K=0 (RTT=%.0fms, skip offload)", rtt),
                    localCostMs = 0.0,
                    offloadCostMs = 0.0,
                    rttMs = rtt,
                    bandwidthKbps = bwKbps,
                    isServerReachable = reachable
                )
            }

            // Online Cost Model:
            // Cost_local(K) = K * t_local_face
            val costLocal = faceCount * emaLocalFaceMs

            // Cost_offload(K, RTT, B) = RTT + (K * Size_crop_KB * 1024 * 8 / (B_kbps * 1000)) * 1000 + ceil(K/P) * t_edge_slot
            // Simplifies to: RTT + (K * Size_crop_KB * 8.192 / B_kbps * 1000) + waves * t_edge_slot
            val payloadBits = faceCount * emaCropKb * 1024.0 * 8.0
            val txTimeMs = payloadBits / bwKbps // bits / kbps = ms
            val waves = ceil(faceCount.toDouble() / edgePoolSlots.toDouble()).coerceAtLeast(1.0)
            val costOffload = rtt + txTimeMs + waves * emaEdgeSlotMs

            // Hard safety fallback 1: Edge server disconnected / unreachable.
            if (!reachable) {
                lastDynamicMode = PartitionMode.ALL_LOCAL
                return PartitionDecision(
                    targetMode = PartitionMode.ALL_LOCAL,
                    reason = "⚡ Auto->Local: Edge offline/unreachable",
                    localCostMs = costLocal,
                    offloadCostMs = costOffload,
                    rttMs = rtt,
                    bandwidthKbps = bwKbps,
                    isServerReachable = false
                )
            }

            // Hard safety fallback 2: Severe weak network (RTT > 300ms).
            if (rtt > MAX_RTT_FALLBACK_MS) {
                lastDynamicMode = PartitionMode.ALL_LOCAL
                return PartitionDecision(
                    targetMode = PartitionMode.ALL_LOCAL,
                    reason = String.format(
                        Locale.US,
                        "⚡ Auto->Local: High RTT=%.0fms (>300ms)",
                        rtt
                    ),
                    localCostMs = costLocal,
                    offloadCostMs = costOffload,
                    rttMs = rtt,
                    bandwidthKbps = bwKbps,
                    isServerReachable = true
                )
            }

            // Hysteresis switching to prevent oscillation around the break-even threshold.
            val chosen = when (lastDynamicMode) {
                PartitionMode.ALL_LOCAL -> {
                    if (costOffload + HYSTERESIS_DELTA_MS < costLocal) {
                        PartitionMode.CUT_C2_CROP_OFFLOAD
                    } else {
                        PartitionMode.ALL_LOCAL
                    }
                }
                else -> {
                    if (costLocal + HYSTERESIS_DELTA_MS < costOffload) {
                        PartitionMode.ALL_LOCAL
                    } else {
                        PartitionMode.CUT_C2_CROP_OFFLOAD
                    }
                }
            }
            lastDynamicMode = chosen

            val reason = if (chosen == PartitionMode.CUT_C2_CROP_OFFLOAD) {
                String.format(
                    Locale.US,
                    "⚡ Auto->C2: K=%d RTT=%.0fms | L:%.0fms > C2:%.0fms",
                    faceCount,
                    rtt,
                    costLocal,
                    costOffload
                )
            } else {
                String.format(
                    Locale.US,
                    "⚡ Auto->Local: K=%d RTT=%.0fms | L:%.0fms <= C2:%.0fms",
                    faceCount,
                    rtt,
                    costLocal,
                    costOffload
                )
            }

            return PartitionDecision(
                targetMode = chosen,
                reason = reason,
                localCostMs = costLocal,
                offloadCostMs = costOffload,
                rttMs = rtt,
                bandwidthKbps = bwKbps,
                isServerReachable = true
            )
        }
    }

    private fun ema(prev: Double, sample: Double): Double {
        return EMA_ALPHA * sample + (1.0 - EMA_ALPHA) * prev
    }
}
