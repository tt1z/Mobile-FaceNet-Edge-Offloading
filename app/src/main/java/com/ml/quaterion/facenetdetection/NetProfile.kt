package com.ml.quaterion.facenetdetection

/**
 * Controllable network emulation profiles sent via X-Net-Profile header.
 * REAL applies zero artificial latency/throttling (for genuine WiFi / 5G / USB measurements).
 */
enum class NetProfile(val headerValue: String, val displayName: String) {
    REAL("REAL", "REAL (真实网络)"),
    SIM_5G("SIM_5G", "SIM_5G (+12ms, 20Mbps)"),
    SIM_4G("SIM_4G", "SIM_4G (+45ms, 3Mbps)"),
    SIM_WEAK("SIM_WEAK", "SIM_WEAK (+180ms, 300Kbps)");

    override fun toString(): String = displayName

    companion object {
        fun fromHeader(value: String?): NetProfile {
            return values().firstOrNull { it.headerValue == value } ?: REAL
        }
    }
}
