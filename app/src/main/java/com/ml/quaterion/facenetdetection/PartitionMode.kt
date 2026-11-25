package com.ml.quaterion.facenetdetection

/**
 * Computation partitioning modes aligned with the M1~M6 Dataflow DAG:
 * - ALL_LOCAL: Full pipeline (M1..M6) runs locally (with M3-before-M4 early exit).
 * - CUT_C1_FULL_REMOTE: M1 local, full frame uploaded for remote M2+M3+M4, M5+M6 local.
 * - CUT_C2_CROP_OFFLOAD: M1+M2 local, K cropped 160x160 faces uploaded for remote M3+M4, M5+M6 local.
 * - CUT_C3_FACENET_ONLY: M1+M2+M3 local (masked faces pruned locally with 0 offload), only unmasked faces uploaded for remote M4.
 * - AUTO_DYNAMIC: Runtime adaptive switching via PartitionDecisionEngine (Part C).
 */
enum class PartitionMode(val displayName: String, val tag: String) {
    ALL_LOCAL("All-Local (本地)", "ALL_LOCAL"),
    CUT_C1_FULL_REMOTE("Cut-C1 (全图卸载)", "CUT_C1"),
    CUT_C2_CROP_OFFLOAD("Cut-C2 (裁剪卸载)", "CUT_C2"),
    CUT_C3_FACENET_ONLY("Cut-C3 (Mask本地+M4卸载)", "CUT_C3"),
    AUTO_DYNAMIC("Auto-Dynamic (动态)", "AUTO");

    override fun toString(): String = displayName
}
