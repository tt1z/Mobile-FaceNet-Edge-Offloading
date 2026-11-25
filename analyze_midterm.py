"""
Mid-term Report Statistical Analysis & High-DPI Publication Chart Generator.

Auto-discovers standardized 15-column Part A/B CSV logs in `--input-dir` (default: perf_logs/),
classifies each CSV automatically by (mode, backend, net_profile, jpeg_q, faceCount, pruned_ratio),
selects the latest recording per condition, and generates summary tables + 300 DPI figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

ROOT = Path(__file__).resolve().parent


def load_stats(path: Path) -> Optional[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "net_profile" not in reader.fieldnames or "jpeg_q" not in reader.fieldnames:
            # Skip legacy 13-column CSVs
            return None
        rows = [r for r in reader if r.get("backend") and r.get("backend") != "EDGE-ERR"]

    if not rows:
        return None

    def arr(col: str) -> np.ndarray:
        return np.array([float(r[col]) for r in rows], dtype=np.float64)

    total = arr("total_ms")
    prep = arr("prep_ms")
    net = arr("net_ms")
    infer = arr("infer_ms")
    mask = arr("mask_ms")
    fn = arr("facenet_ms")
    match = arr("match_ms")
    payload = arr("payload_kb")
    fps_col = arr("fps")
    faces = arr("faces")

    mean_total = float(np.mean(total))
    eff_fps = 1000.0 / mean_total if mean_total > 0 else 0.0
    remote_detect = np.maximum(0.0, infer - mask - fn)

    # Dominant mode/backend/profile/quality/K in this recording
    def mode_val(col: str) -> str:
        vals = [r[col] for r in rows]
        return max(set(vals), key=vals.count)

    return {
        "file": path.name,
        "n": len(rows),
        "mode": mode_val("mode"),
        "backend": mode_val("backend"),
        "net_profile": mode_val("net_profile"),
        "jpeg_q": int(float(mode_val("jpeg_q"))),
        "k_mean": float(np.mean(faces)),
        "k_round": int(round(float(np.mean(faces)))),
        "total_raw": total,
        "total_mean": mean_total,
        "total_std": float(np.std(total)),
        "total_p50": float(np.percentile(total, 50)),
        "total_p95": float(np.percentile(total, 95)),
        "eff_fps": eff_fps,
        "hud_fps_mean": float(np.mean(fps_col)),
        "prep_mean": float(np.mean(prep)),
        "net_mean": float(np.mean(net)),
        "infer_mean": float(np.mean(infer)),
        "remote_detect_mean": float(np.mean(remote_detect)),
        "mask_mean": float(np.mean(mask)),
        "fn_mean": float(np.mean(fn)),
        "match_mean": float(np.mean(match)),
        "payload_mean": float(np.mean(payload)),
        "pruned_ratio": float(np.mean(payload == 0.0)) * 100.0,
    }


def discover_experiments(input_dir: Path) -> Dict[str, dict]:
    """
    Scan all CSVs sorted by filename (timestamp) ascending so the latest recording
    for each experimental condition automatically takes precedence.
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    stats: Dict[str, dict] = {}

    for path in csv_files:
        s = load_stats(path)
        if s is None:
            continue

        mode = s["mode"]
        backend = s["backend"]
        net = s["net_profile"]
        q = s["jpeg_q"]
        k = s["k_round"]

        # Only classify K>=1 recordings into single-face K=1 benchmark slots when k == 1
        if k != 1:
            continue

        if mode == "ALL_LOCAL":
            if backend == "GPU":
                stats["exp1_local_gpu"] = s
            elif backend == "CPU-4T":
                stats["exp1_local_cpu"] = s
        elif mode == "CUT_C1" and q == 75:
            if net == "REAL":
                stats["exp1_c1_real"] = s
                stats["exp2_c1_real"] = s
            elif net == "SIM_5G":
                stats["exp2_c1_5g"] = s
            elif net == "SIM_4G":
                stats["exp2_c1_4g"] = s
            elif net == "SIM_WEAK":
                stats["exp2_c1_weak"] = s
        elif mode == "CUT_C2":
            if net == "REAL":
                if q == 50:
                    stats["exp4_c2_q50"] = s
                elif q == 75:
                    # First C2@REAL,Q75 populates exp1/exp2 if not set; latest updates exp4_c2_q75
                    if "exp1_c2_real" not in stats:
                        stats["exp1_c2_real"] = s
                        stats["exp2_c2_real"] = s
                    stats["exp4_c2_q75"] = s
                elif q == 95:
                    stats["exp4_c2_q95"] = s
            elif q == 75:
                if net == "SIM_5G":
                    stats["exp2_c2_5g"] = s
                elif net == "SIM_4G":
                    stats["exp2_c2_4g"] = s
                elif net == "SIM_WEAK":
                    stats["exp2_c2_weak"] = s
        elif mode == "CUT_C3" and net == "REAL" and q == 75:
            if s["pruned_ratio"] >= 50.0 or s["payload_mean"] < 1.0:
                stats["exp4_c3_masked"] = s
            else:
                stats["exp1_c3_real"] = s
                stats["exp4_c3_unmasked"] = s

    return stats


def plot_fig1_partitioning(stats: Dict[str, dict], out_dir: Path) -> bool:
    candidates = [
        ("exp1_local_gpu", "All-Local\n(GPU)"),
        ("exp1_local_cpu", "All-Local\n(CPU-4T)"),
        ("exp1_c1_real", "Cut-C1\n(全图卸载)"),
        ("exp1_c2_real", "Cut-C2\n(裁剪卸载)"),
        ("exp1_c3_real", "Cut-C3\n(M3本地+M4卸载)"),
    ]
    present = [(k, lbl) for k, lbl in candidates if k in stats]
    if len(present) < 2:
        return False

    keys = [k for k, _ in present]
    labels = [lbl for _, lbl in present]
    x = np.arange(len(keys))

    prep = np.array([stats[k]["prep_mean"] for k in keys])
    net = np.array([stats[k]["net_mean"] for k in keys])
    rem_det = np.array([stats[k]["remote_detect_mean"] for k in keys])
    mask = np.array([stats[k]["mask_mean"] for k in keys])
    fn = np.array([stats[k]["fn_mean"] for k in keys])
    match = np.array([stats[k]["match_mean"] for k in keys])
    totals = np.array([stats[k]["total_mean"] for k in keys])
    other = np.maximum(0.0, totals - (prep + net + rem_det + mask + fn + match))
    fps = np.array([stats[k]["eff_fps"] for k in keys])
    payloads = np.array([stats[k]["payload_mean"] for k in keys])

    fig, ax1 = plt.subplots(figsize=(10.8, 6.0))
    w = 0.50

    ax1.bar(x, prep, w, label="Prep (本地采集/检测/裁剪/编码)", color="#4E79A7")
    b1 = prep
    ax1.bar(x, net, w, bottom=b1, label="Net (网络传输与往返)", color="#F28E2B")
    b2 = b1 + net
    ax1.bar(x, rem_det, w, bottom=b2, label="Edge M2 (云端人脸检测)", color="#76B7B2")
    b3 = b2 + rem_det
    ax1.bar(x, mask, w, bottom=b3, label="M3 (口罩检测)", color="#59A14F")
    b4 = b3 + mask
    ax1.bar(x, fn, w, bottom=b4, label="M4 (FaceNet 特征提取)", color="#E15759")
    b5 = b4 + fn
    ax1.bar(x, match, w, bottom=b5, label="M5 (特征比对)", color="#B07AA1")
    b6 = b5 + match
    ax1.bar(x, other, w, bottom=b6, label="调度/序列化杂项开销", color="#D1D5DB")

    for i in range(len(keys)):
        kb_txt = f"{payloads[i]:.1f} KB" if payloads[i] > 0 else "0 KB"
        ax1.text(
            x[i],
            totals[i] + 4,
            f"{totals[i]:.1f} ms\n({kb_txt})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#1F2937",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("单帧平均端到端耗时 (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(totals) * 1.32)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        fps,
        color="#111827",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="有效帧率 (Effective FPS)",
        zorder=10,
    )
    for i in range(len(keys)):
        offset = -1.5 if keys[i] == "exp1_c3_real" else 1.0
        ax2.text(
            x[i],
            fps[i] + offset,
            f"{fps[i]:.1f} FPS",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#9CA3AF", alpha=0.92),
            zorder=11,
        )
    ax2.set_ylabel("有效帧率 (FPS = 1000 / Total)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(fps) * 1.18)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, ncol=2, framealpha=0.92)
    ax1.set_title(
        "实验 1：不同计算切分方案的单帧延迟分解与有效帧率对比（K=1, REAL/LOCAL, Q75）",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_partitioning_breakdown.png")
    plt.close(fig)
    return True


def plot_fig2_network(stats: Dict[str, dict], out_dir: Path) -> bool:
    regimes = [
        ("REAL\n(直连)", "exp2_c1_real", "exp2_c2_real"),
        ("SIM_5G\n(+12ms,20Mbps)", "exp2_c1_5g", "exp2_c2_5g"),
        ("SIM_4G\n(+45ms,3Mbps)", "exp2_c1_4g", "exp2_c2_4g"),
        ("SIM_WEAK\n(+180ms,300Kbps)", "exp2_c1_weak", "exp2_c2_weak"),
    ]
    present = [r for r in regimes if r[1] in stats and r[2] in stats]
    if len(present) < 2:
        return False

    profiles = [r[0] for r in present]
    c1_keys = [r[1] for r in present]
    c2_keys = [r[2] for r in present]

    c1_tot = [stats[k]["total_mean"] for k in c1_keys]
    c2_tot = [stats[k]["total_mean"] for k in c2_keys]
    c1_net = [stats[k]["net_mean"] for k in c1_keys]
    c2_net = [stats[k]["net_mean"] for k in c2_keys]
    c1_kb = float(np.mean([stats[k]["payload_mean"] for k in c1_keys]))
    c2_kb = float(np.mean([stats[k]["payload_mean"] for k in c2_keys]))

    x = np.arange(len(profiles))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    ax1.plot(x, c1_tot, marker="s", linewidth=2.5, markersize=8, color="#E15759", label=f"Cut-C1 全图卸载 (~{c1_kb:.1f} KB/帧)")
    ax1.plot(x, c2_tot, marker="o", linewidth=2.5, markersize=8, color="#2A9D8F", label=f"Cut-C2 裁剪卸载 (~{c2_kb:.1f} KB/帧)")
    if "exp1_local_cpu" in stats:
        local_cpu = stats["exp1_local_cpu"]["total_mean"]
        ax1.axhline(local_cpu, linestyle="--", linewidth=1.8, color="#264653", label=f"All-Local CPU-4T 基线 ({local_cpu:.1f} ms)")
    if "exp1_local_gpu" in stats:
        local_gpu = stats["exp1_local_gpu"]["total_mean"]
        ax1.axhline(local_gpu, linestyle=":", linewidth=1.8, color="#6C757D", label=f"All-Local GPU 基线 ({local_gpu:.1f} ms)")

    for i in range(len(x)):
        c1_above = c1_tot[i] >= c2_tot[i]
        ax1.annotate(
            f"{c1_tot[i]:.1f}",
            (x[i], c1_tot[i]),
            textcoords="offset points",
            xytext=(0, 10 if c1_above else -16),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#C0392B",
        )
        ax1.annotate(
            f"{c2_tot[i]:.1f}",
            (x[i], c2_tot[i]),
            textcoords="offset points",
            xytext=(0, -16 if c1_above else 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1D7A6E",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(profiles, fontsize=9)
    ax1.set_ylabel("单帧平均端到端总延迟 (ms)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) 不同网络环境下的端到端总延迟与最优切分点跃迁", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(c1_tot + c2_tot) * 1.18)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    w = 0.36
    b1 = ax2.bar(x - w / 2, c1_net, w, color="#E15759", alpha=0.9, label=f"Cut-C1 网络耗时 (~{c1_kb:.1f} KB)")
    b2 = ax2.bar(x + w / 2, c2_net, w, color="#2A9D8F", alpha=0.9, label=f"Cut-C2 网络耗时 (~{c2_kb:.1f} KB)")
    for bars, col in [(b1, "#C0392B"), (b2, "#1D7A6E")]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(c1_net + c2_net) * 0.015,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=col,
            )

    if c1_kb > 0:
        reduction = (1.0 - c2_kb / c1_kb) * 100.0
        net_drop_txt = ""
        if c1_net[-1] > 0:
            net_drop_txt = f"\n弱网/末档 C2 净网络开销比 C1 降低 {(1.0 - c2_net[-1]/c1_net[-1])*100:.1f}%"
        ax2.text(
            0.44,
            0.66,
            f"漏斗效应流量缩减: {c1_kb:.1f} KB → {c2_kb:.2f} KB (-{reduction:.1f}%){net_drop_txt}",
            transform=ax2.transAxes,
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFBEB", ec="#D97706"),
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(profiles, fontsize=9)
    ax2.set_ylabel("纯网络往返与传输开销 Net (ms)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) 剥离服务端推理后的纯网络传输耗时对比", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(c1_net + c2_net) * 1.24)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle("实验 2：网络带宽与 RTT 对 Cut-C1 / Cut-C2 计算划分性能的影响（K=1, Q75）", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "fig2_network_sensitivity.png")
    plt.close(fig)
    return True


def plot_fig3_exp4(stats: Dict[str, dict], out_dir: Path) -> bool:
    has_q = all(k in stats for k in ["exp4_c2_q50", "exp4_c2_q75", "exp4_c2_q95"])
    has_c3 = all(k in stats for k in ["exp4_c3_unmasked", "exp4_c3_masked"])
    if not has_q and not has_c3:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    if has_q:
        q_keys = ["exp4_c2_q50", "exp4_c2_q75", "exp4_c2_q95"]
        q_labels = ["Q50\n(高压缩)", "Q75\n(推荐平衡点)", "Q95\n(极高画质)"]
        q_kb = [stats[k]["payload_mean"] for k in q_keys]
        q_net = [stats[k]["net_mean"] for k in q_keys]
        x = np.arange(len(q_keys))

        bars = ax1.bar(x, q_kb, width=0.45, color=["#60A5FA", "#2563EB", "#1E3A8A"], alpha=0.88, label="单帧上传流量 (KB)")
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h * 0.48, f"{h:.2f} KB", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
        ax1.set_xticks(x)
        ax1.set_xticklabels(q_labels, fontsize=9.5)
        ax1.set_ylabel("单帧上传数据量 Payload (KB)", fontsize=10, fontweight="bold", color="#1E3A8A")
        ax1.set_ylim(0, max(q_kb) * 1.28)
        ax1.grid(axis="y", linestyle="--", alpha=0.35)

        ax1_r = ax1.twinx()
        ax1_r.plot(x, q_net, color="#DC2626", marker="D", linewidth=2.2, markersize=7, label="纯网络传输耗时 Net (ms)")
        for i in range(len(x)):
            ax1_r.text(
                x[i],
                q_net[i] + max(q_net) * 0.07,
                f"{q_net[i]:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
                color="#DC2626",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FCA5A5", alpha=0.95),
            )
        ax1_r.set_ylabel("纯网络往返与传输耗时 Net (ms)", fontsize=10, fontweight="bold", color="#DC2626")
        ax1_r.set_ylim(0, max(q_net) * 1.28)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
    else:
        ax1.text(0.5, 0.5, "未检测到 Q50/Q75/Q95 完整对比数据", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("(a) Cut-C2 下 JPEG 压缩质量对传输体积与网络耗时的权衡", fontsize=11, fontweight="bold")

    if has_c3:
        u = stats["exp4_c3_unmasked"]
        m = stats["exp4_c3_masked"]
        metrics_labels = ["总延迟\nTotal (ms)", "网络耗时\nNet (ms)", "云端 M4\nFaceNet (ms)", "上传流量\nPayload (KB×10)"]
        u_vals = [u["total_mean"], u["net_mean"], u["fn_mean"], u["payload_mean"] * 10.0]
        m_vals = [m["total_mean"], m["net_mean"], m["fn_mean"], m["payload_mean"] * 10.0]
        u_txt = [f"{u['total_mean']:.1f}ms", f"{u['net_mean']:.1f}ms", f"{u['fn_mean']:.1f}ms", f"{u['payload_mean']:.2f}KB"]
        m_txt = [f"{m['total_mean']:.1f}ms", f"{m['net_mean']:.1f}ms", f"{m['fn_mean']:.1f}ms", f"{m['payload_mean']:.2f}KB"]

        x2 = np.arange(len(metrics_labels))
        w2 = 0.36
        ax2.bar(x2 - w2 / 2, u_vals, w2, color="#F59E0B", label="未戴口罩 (正常卸载 M4)")
        ax2.bar(x2 + w2 / 2, m_vals, w2, color="#10B981", label="戴口罩 (本地 M3 早退剪枝)")
        for i in range(len(x2)):
            ax2.text(x2[i] - w2 / 2, u_vals[i] + 2.5, u_txt[i], ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B45309")
            ax2.text(x2[i] + w2 / 2, m_vals[i] + 2.5, m_txt[i], ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")

        kb_drop = (1.0 - m["payload_mean"] / max(0.01, u["payload_mean"])) * 100.0
        fn_drop = (1.0 - m["fn_mean"] / max(0.01, u["fn_mean"])) * 100.0
        ax2.text(
            0.55,
            0.74,
            f"早退剪枝收益:\n• 流量暴降 -{kb_drop:.1f}% ({m['pruned_ratio']:.1f}% 帧零上传)\n• 云端 M4 算力节省 -{fn_drop:.1f}%",
            transform=ax2.transAxes,
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color="#065F46",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#10B981"),
        )
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metrics_labels, fontsize=9)
        ax2.set_ylabel("数值（延迟 ms / 流量按 KB×10 缩放展示）", fontsize=9.5, fontweight="bold")
        ax2.set_ylim(0, max(u_vals) * 1.25)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        ax2.legend(loc="upper right", fontsize=8.5)
    else:
        ax2.text(0.5, 0.5, "未检测到 Cut-C3 戴口罩/未戴口罩对比数据", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("(b) Cut-C3 条件早退剪枝实测收益对比（REAL, Q75）", fontsize=11, fontweight="bold")

    fig.suptitle("实验 4：压缩率参数权衡与 Cut-C3 条件早退剪枝架构洞察", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "fig3_compression_and_pruning.png")
    plt.close(fig)
    return True


def plot_fig4_distributions(stats: Dict[str, dict], out_dir: Path) -> bool:
    candidates = [
        ("exp1_local_gpu", "Local\nGPU"),
        ("exp1_local_cpu", "Local\nCPU-4T"),
        ("exp1_c1_real", "Cut-C1\nREAL"),
        ("exp1_c2_real", "Cut-C2\nREAL"),
        ("exp1_c3_real", "Cut-C3\nREAL"),
        ("exp2_c1_4g", "Cut-C1\nSIM_4G"),
        ("exp2_c2_4g", "Cut-C2\nSIM_4G"),
    ]
    present = [(k, lbl) for k, lbl in candidates if k in stats]
    if len(present) < 2:
        return False

    keys = [k for k, _ in present]
    labels = [lbl for _, lbl in present]
    data = [stats[k]["total_raw"] for k in keys]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#DC2626", markeredgecolor="#DC2626", markersize=5),
        medianprops=dict(color="#111827", linewidth=2),
    )
    palette = ["#93C5FD", "#60A5FA", "#FCA5A5", "#6EE7B7", "#FCD34D", "#F87171", "#34D399"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    max_y = max(float(np.max(stats[k]["total_raw"])) for k in keys)
    for i, k in enumerate(keys):
        p95 = stats[k]["total_p95"]
        ax.text(
            i + 1,
            float(np.max(stats[k]["total_raw"])) + max_y * 0.025,
            f"P95:\n{p95:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#374151",
        )

    ax.set_ylabel("单帧端到端总延迟分布 (ms)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max_y * 1.18)
    ax.set_title("主要切分方案的端到端延迟箱线图与 P95 尾部稳定性对比（红菱形为均值）", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_latency_boxplot.png")
    plt.close(fig)
    return True


def write_markdown_summary(stats: Dict[str, dict], out_dir: Path, input_dir: Path) -> Path:
    def row(label: str, k: str) -> str:
        if k not in stats:
            return f"| {label} | - | 未采集 | - | - | - | - | - | - | - | - | - |"
        s = stats[k]
        return (
            f"| {label} ({s['file']}) | {s['n']} | {s['total_mean']:.1f} ± {s['total_std']:.1f} | "
            f"{s['total_p50']:.1f} | {s['total_p95']:.1f} | **{s['eff_fps']:.2f}** | "
            f"{s['prep_mean']:.1f} | {s['net_mean']:.1f} | {s['mask_mean']:.1f} | "
            f"{s['fn_mean']:.1f} | {s['match_mean']:.2f} | **{s['payload_mean']:.2f}** |"
        )

    unique_files = {s["file"]: s["n"] for s in stats.values()}
    total_frames = sum(unique_files.values())

    md = f"""# 中期报告实验数据自动统计汇总（Part A & Part B）

> **数据源目录**：`{input_dir}`（自动识别最新有效实验文件 **{len(unique_files)} 组**，有效样本总计 **{total_frames} 帧**）

## 实验 1：不同计算切分方案横向对比（$K=1$, `REAL/LOCAL`, `Q75`）

| 切分方案 | 帧数 N | 总延迟均值±标准差 (ms) | P50 (ms) | P95 (ms) | 有效 FPS | Prep (ms) | Net (ms) | M3 Mask (ms) | M4 FaceNet (ms) | M5 Match (ms) | 上传负载 (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row("All-Local GPU", "exp1_local_gpu")}
{row("All-Local CPU-4T", "exp1_local_cpu")}
{row("Cut-C1 全图卸载", "exp1_c1_real")}
{row("Cut-C2 裁剪卸载", "exp1_c2_real")}
{row("Cut-C3 M3本地+M4卸载", "exp1_c3_real")}

## 实验 2：网络带宽与 RTT 敏感度对比（`Cut-C1` vs `Cut-C2`, $K=1$, `Q75`）

| 实验组 | 帧数 N | 总延迟均值±标准差 (ms) | P50 (ms) | P95 (ms) | 有效 FPS | Prep (ms) | Net (ms) | M3 Mask (ms) | M4 FaceNet (ms) | M5 Match (ms) | 上传负载 (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row("Cut-C1 @ REAL", "exp2_c1_real")}
{row("Cut-C2 @ REAL", "exp2_c2_real")}
{row("Cut-C1 @ SIM_5G", "exp2_c1_5g")}
{row("Cut-C2 @ SIM_5G", "exp2_c2_5g")}
{row("Cut-C1 @ SIM_4G", "exp2_c1_4g")}
{row("Cut-C2 @ SIM_4G", "exp2_c2_4g")}
{row("Cut-C1 @ SIM_WEAK", "exp2_c1_weak")}
{row("Cut-C2 @ SIM_WEAK", "exp2_c2_weak")}

## 实验 4：压缩率权衡（JPEG Quality）与 `Cut-C3` 条件早退剪枝

| 实验组 | 帧数 N | 总延迟均值±标准差 (ms) | P50 (ms) | P95 (ms) | 有效 FPS | Prep (ms) | Net (ms) | M3 Mask (ms) | M4 FaceNet (ms) | M5 Match (ms) | 上传负载 (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row("Cut-C2 @ Q50", "exp4_c2_q50")}
{row("Cut-C2 @ Q75", "exp4_c2_q75")}
{row("Cut-C2 @ Q95", "exp4_c2_q95")}
{row("Cut-C3 未戴口罩", "exp4_c3_unmasked")}
{row("Cut-C3 戴口罩早退", "exp4_c3_masked")}
"""
    out_md = out_dir / "midterm_summary.md"
    out_md.write_text(md, encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-discover CSV logs and generate Part B charts.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "perf_logs", help="Directory containing sa_perf_*.csv files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "midterm_report_results", help="Directory to save charts and summary")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = discover_experiments(input_dir)
    if not stats:
        raise SystemExit(f"[ERROR] No valid 15-column sa_perf_*.csv files found in {input_dir}")

    generated: List[str] = []
    if plot_fig1_partitioning(stats, out_dir):
        generated.append("fig1_partitioning_breakdown.png")
    if plot_fig2_network(stats, out_dir):
        generated.append("fig2_network_sensitivity.png")
    if plot_fig3_exp4(stats, out_dir):
        generated.append("fig3_compression_and_pruning.png")
    if plot_fig4_distributions(stats, out_dir):
        generated.append("fig4_latency_boxplot.png")

    out_md = write_markdown_summary(stats, out_dir, input_dir)
    print(f"[OK] Matched {len(stats)} condition slots from {input_dir}")
    print(f"[OK] Generated charts ({len(generated)}): {', '.join(generated)}")
    print(f"[OK] Summary markdown: {out_md}")


if __name__ == "__main__":
    main()
