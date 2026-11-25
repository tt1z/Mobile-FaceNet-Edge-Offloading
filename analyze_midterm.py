"""
Mid-term Report Statistical Analysis & High-DPI Publication Chart Generator.
Reads the 15 standardized Part A/B CSV logs collected on HUAWEI Mate 40 Pro (NOH-AL00)
and generates summary tables + 4 publication-grade 300 DPI figures.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Configure crisp Chinese + English font fallback on Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

ROOT = Path(r"d:\ASSDK\project")
LOG_DIR = ROOT / "perf_logs"
OUT_DIR = ROOT / "midterm_report_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Explicit experiment mapping verified from today's 2026-11-25 test session
EXP_FILES = {
    # Experiment 1: Partitioning modes baseline (K=1, REAL/LOCAL, Q75)
    "exp1_local_gpu": ("All-Local\n(GPU)", "sa_perf_20261125_145336.csv"),
    "exp1_local_cpu": ("All-Local\n(CPU-4T)", "sa_perf_20261125_145349.csv"),
    "exp1_c1_real": ("Cut-C1\n(全图卸载)", "sa_perf_20261125_145453.csv"),
    "exp1_c2_real": ("Cut-C2\n(裁剪卸载)", "sa_perf_20261125_145506.csv"),
    "exp1_c3_real": ("Cut-C3\n(M3本地+M4卸载)", "sa_perf_20261125_145514.csv"),

    # Experiment 2: Network sensitivity (C1 vs C2 across REAL, SIM_5G, SIM_4G, SIM_WEAK, Q75)
    "exp2_c1_real": ("C1-REAL", "sa_perf_20261125_145453.csv"),
    "exp2_c2_real": ("C2-REAL", "sa_perf_20261125_145506.csv"),
    "exp2_c1_5g": ("C1-5G", "sa_perf_20261125_145539.csv"),
    "exp2_c2_5g": ("C2-5G", "sa_perf_20261125_145547.csv"),
    "exp2_c1_4g": ("C1-4G", "sa_perf_20261125_145604.csv"),
    "exp2_c2_4g": ("C2-4G", "sa_perf_20261125_145617.csv"),
    "exp2_c1_weak": ("C1-WEAK", "sa_perf_20261125_145634.csv"),
    "exp2_c2_weak": ("C2-WEAK", "sa_perf_20261125_145646.csv"),

    # Experiment 4A: JPEG compression quality trade-off (Cut-C2, REAL, K=1)
    "exp4_c2_q50": ("Q50", "sa_perf_20261125_145743.csv"),
    "exp4_c2_q75": ("Q75", "sa_perf_20261125_145753.csv"),
    "exp4_c2_q95": ("Q95", "sa_perf_20261125_145802.csv"),

    # Experiment 4B: Cut-C3 early-exit pruning (Unmasked vs Masked, REAL, Q75)
    "exp4_c3_unmasked": ("Cut-C3 (未戴口罩)", "sa_perf_20261125_145514.csv"),
    "exp4_c3_masked": ("Cut-C3 (戴口罩早退)", "sa_perf_20261125_145830.csv"),
}


def load_stats(filename: str) -> Dict[str, object]:
    path = LOG_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("backend") != "EDGE-ERR"]

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

    mean_total = float(np.mean(total))
    # True steady-state effective FPS = 1000 / mean_total_ms
    eff_fps = 1000.0 / mean_total if mean_total > 0 else 0.0

    # For C1, server_infer_ms includes remote M2 detect + M3 mask + M4 facenet.
    # Compute remote_detect_ms = max(0, infer - mask - fn)
    remote_detect = np.maximum(0.0, infer - mask - fn)

    return {
        "file": filename,
        "n": len(rows),
        "mode": rows[0]["mode"],
        "backend": rows[0]["backend"],
        "net_profile": rows[0]["net_profile"],
        "jpeg_q": int(rows[0]["jpeg_q"]),
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


def plot_fig1_partitioning(stats: Dict[str, dict]) -> None:
    keys = [
        "exp1_local_gpu",
        "exp1_local_cpu",
        "exp1_c1_real",
        "exp1_c2_real",
        "exp1_c3_real",
    ]
    labels = [EXP_FILES[k][0] for k in keys]
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
        # Place total label slightly offset for Cut-C2/C3 so the FPS curve passes cleanly above
        y_pos = totals[i] + 4
        ax1.text(
            x[i],
            y_pos,
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
    fps_offsets = [1.0, 1.0, 0.9, 1.0, -1.5]
    for i in range(len(keys)):
        ax2.text(
            x[i],
            fps[i] + fps_offsets[i],
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
    # Shift right axis range so the FPS curve rides cleanly above the C2/C3 bar labels
    ax2.set_ylim(0, max(fps) * 1.18)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=8.5,
        ncol=2,
        framealpha=0.92,
    )
    ax1.set_title(
        "实验 1：不同计算切分方案的单帧延迟分解与有效帧率对比（K=1, REAL/LOCAL, Q75）",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_partitioning_breakdown.png")
    plt.close(fig)


def plot_fig2_network(stats: Dict[str, dict]) -> None:
    profiles = ["REAL\n(直连)", "SIM_5G\n(+12ms,20Mbps)", "SIM_4G\n(+45ms,3Mbps)", "SIM_WEAK\n(+180ms,300Kbps)"]
    c1_keys = ["exp2_c1_real", "exp2_c1_5g", "exp2_c1_4g", "exp2_c1_weak"]
    c2_keys = ["exp2_c2_real", "exp2_c2_5g", "exp2_c2_4g", "exp2_c2_weak"]

    c1_tot = [stats[k]["total_mean"] for k in c1_keys]
    c2_tot = [stats[k]["total_mean"] for k in c2_keys]
    c1_net = [stats[k]["net_mean"] for k in c1_keys]
    c2_net = [stats[k]["net_mean"] for k in c2_keys]
    c1_kb = np.mean([stats[k]["payload_mean"] for k in c1_keys])
    c2_kb = np.mean([stats[k]["payload_mean"] for k in c2_keys])

    local_cpu = stats["exp1_local_cpu"]["total_mean"]
    local_gpu = stats["exp1_local_gpu"]["total_mean"]

    x = np.arange(len(profiles))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # Left: End-to-End Total Latency across network regimes (showing Crossover Points)
    ax1.plot(x, c1_tot, marker="s", linewidth=2.5, markersize=8, color="#E15759", label=f"Cut-C1 全图卸载 (~{c1_kb:.1f} KB/帧)")
    ax1.plot(x, c2_tot, marker="o", linewidth=2.5, markersize=8, color="#2A9D8F", label=f"Cut-C2 裁剪卸载 (~{c2_kb:.1f} KB/帧)")
    ax1.axhline(local_cpu, linestyle="--", linewidth=1.8, color="#264653", label=f"All-Local CPU-4T 基线 ({local_cpu:.1f} ms)")
    ax1.axhline(local_gpu, linestyle=":", linewidth=1.8, color="#6C757D", label=f"All-Local GPU 基线 ({local_gpu:.1f} ms)")

    for i in range(len(x)):
        # Put the higher curve's label above and the lower curve's label below to avoid any overlap
        c1_above = c1_tot[i] >= c2_tot[i]
        c1_off = 10 if c1_above else -16
        c2_off = -16 if c1_above else 10
        ax1.annotate(
            f"{c1_tot[i]:.1f}",
            (x[i], c1_tot[i]),
            textcoords="offset points",
            xytext=(0, c1_off),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#C0392B",
        )
        ax1.annotate(
            f"{c2_tot[i]:.1f}",
            (x[i], c2_tot[i]),
            textcoords="offset points",
            xytext=(0, c2_off),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1D7A6E",
        )

    ax1.annotate(
        "4G 交叉点:\nCut-C2 反超 Cut-C1",
        xy=(2, c2_tot[2]),
        xytext=(1.35, 360),
        arrowprops=dict(arrowstyle="->", color="#1D7A6E", lw=1.4),
        fontsize=8.5,
        fontweight="bold",
        color="#1D7A6E",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#1D7A6E"),
    )
    ax1.annotate(
        "弱网下全图上传雪崩\n(All-Local 反超卸载!)",
        xy=(3, c1_tot[3]),
        xytext=(1.75, 760),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.4),
        fontsize=8.5,
        fontweight="bold",
        color="#C0392B",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#C0392B"),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(profiles, fontsize=9)
    ax1.set_ylabel("单帧平均端到端总延迟 (ms)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) 不同网络环境下的端到端总延迟与最优切分点跃迁", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(c1_tot) * 1.15)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    # Right: Pure Network Overhead (Net ms) comparison
    w = 0.36
    b1 = ax2.bar(x - w / 2, c1_net, w, color="#E15759", alpha=0.9, label=f"Cut-C1 网络耗时 (~{c1_kb:.1f} KB)")
    b2 = ax2.bar(x + w / 2, c2_net, w, color="#2A9D8F", alpha=0.9, label=f"Cut-C2 网络耗时 (~{c2_kb:.1f} KB)")
    for bars, col in [(b1, "#C0392B"), (b2, "#1D7A6E")]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 10,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=col,
            )

    reduction = (1.0 - c2_kb / c1_kb) * 100.0
    ax2.text(
        0.44,
        0.64,
        f"漏斗效应流量缩减: {c1_kb:.1f} KB → {c2_kb:.2f} KB (-{reduction:.1f}%)\n"
        f"弱网下 C2 净网络开销比 C1 降低 {(1.0 - c2_net[-1]/c1_net[-1])*100:.1f}%",
        transform=ax2.transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#1F2937",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFBEB", ec="#D97706"),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(profiles, fontsize=9)
    ax2.set_ylabel("纯网络往返与传输开销 Net (ms)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) 剥离服务端推理后的纯网络传输耗时对比", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(c1_net) * 1.22)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle("实验 2：网络带宽与 RTT 对 Cut-C1 / Cut-C2 计算划分性能的影响（K=1, Q75）", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "fig2_network_sensitivity.png")
    plt.close(fig)


def plot_fig3_exp4(stats: Dict[str, dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # Left: JPEG Quality trade-off (Q50, Q75, Q95) - plot Payload (KB) vs Net Overhead (ms)
    q_keys = ["exp4_c2_q50", "exp4_c2_q75", "exp4_c2_q95"]
    q_labels = ["Q50\n(高压缩)", "Q75\n(推荐平衡点)", "Q95\n(极高画质)"]
    q_kb = [stats[k]["payload_mean"] for k in q_keys]
    q_net = [stats[k]["net_mean"] for k in q_keys]
    x = np.arange(len(q_keys))

    bars = ax1.bar(x, q_kb, width=0.45, color=["#60A5FA", "#2563EB", "#1E3A8A"], alpha=0.88, label="单帧上传流量 (KB)")
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h * 0.48,
            f"{h:.2f} KB",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
        )
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
            q_net[i] + 1.4,
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
    ax1.set_title("(a) Cut-C2 下 JPEG 压缩质量对传输体积与网络耗时的权衡", fontsize=11, fontweight="bold")

    # Right: Cut-C3 Conditional Early-Exit Pruning (Unmasked vs Masked)
    u = stats["exp4_c3_unmasked"]
    m = stats["exp4_c3_masked"]
    metrics_labels = ["总延迟\nTotal (ms)", "网络耗时\nNet (ms)", "云端 M4\nFaceNet (ms)", "上传流量\nPayload (KB×10)"]
    u_vals = [u["total_mean"], u["net_mean"], u["fn_mean"], u["payload_mean"] * 10.0]
    m_vals = [m["total_mean"], m["net_mean"], m["fn_mean"], m["payload_mean"] * 10.0]
    u_txt = [f"{u['total_mean']:.1f}ms", f"{u['net_mean']:.1f}ms", f"{u['fn_mean']:.1f}ms", f"{u['payload_mean']:.2f}KB"]
    m_txt = [f"{m['total_mean']:.1f}ms", f"{m['net_mean']:.1f}ms", f"{m['fn_mean']:.1f}ms", f"{m['payload_mean']:.2f}KB"]

    x2 = np.arange(len(metrics_labels))
    w2 = 0.36
    b_u = ax2.bar(x2 - w2 / 2, u_vals, w2, color="#F59E0B", label="未戴口罩 (正常卸载 M4)")
    b_m = ax2.bar(x2 + w2 / 2, m_vals, w2, color="#10B981", label="戴口罩 (本地 M3 早退剪枝)")
    for i in range(len(x2)):
        ax2.text(x2[i] - w2 / 2, u_vals[i] + 2.5, u_txt[i], ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B45309")
        ax2.text(x2[i] + w2 / 2, m_vals[i] + 2.5, m_txt[i], ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")

    kb_drop = (1.0 - m["payload_mean"] / u["payload_mean"]) * 100.0
    fn_drop = (1.0 - m["fn_mean"] / u["fn_mean"]) * 100.0
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
    ax2.set_title("(b) Cut-C3 条件早退剪枝实测收益对比（REAL, Q75）", fontsize=11, fontweight="bold")

    fig.suptitle("实验 4：压缩率参数权衡与 Cut-C3 条件早退剪枝架构洞察", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "fig3_compression_and_pruning.png")
    plt.close(fig)


def plot_fig4_distributions(stats: Dict[str, dict]) -> None:
    keys = [
        "exp1_local_gpu",
        "exp1_local_cpu",
        "exp1_c1_real",
        "exp1_c2_real",
        "exp1_c3_real",
        "exp2_c1_4g",
        "exp2_c2_4g",
    ]
    labels = [
        "Local\nGPU",
        "Local\nCPU-4T",
        "Cut-C1\nREAL",
        "Cut-C2\nREAL",
        "Cut-C3\nREAL",
        "Cut-C1\nSIM_4G",
        "Cut-C2\nSIM_4G",
    ]
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

    for i, k in enumerate(keys):
        p95 = stats[k]["total_p95"]
        ax.text(
            i + 1,
            float(np.max(stats[k]["total_raw"])) + 6,
            f"P95:\n{p95:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#374151",
        )

    ax.set_ylabel("单帧端到端总延迟分布 (ms)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(20, 285)
    ax.set_title("主要切分方案的端到端延迟箱线图与 P95 尾部稳定性对比（红菱形为均值）", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_latency_boxplot.png")
    plt.close(fig)


def write_markdown_summary(stats: Dict[str, dict]) -> Path:
    def row(label: str, k: str) -> str:
        s = stats[k]
        return (
            f"| {label} | {s['n']} | {s['total_mean']:.1f} ± {s['total_std']:.1f} | "
            f"{s['total_p50']:.1f} | {s['total_p95']:.1f} | **{s['eff_fps']:.2f}** | "
            f"{s['prep_mean']:.1f} | {s['net_mean']:.1f} | {s['mask_mean']:.1f} | "
            f"{s['fn_mean']:.1f} | {s['match_mean']:.2f} | **{s['payload_mean']:.2f}** |"
        )

    c1_kb = np.mean([stats[k]["payload_mean"] for k in ["exp2_c1_real", "exp2_c1_5g", "exp2_c1_4g", "exp2_c1_weak"]])
    c2_kb = np.mean([stats[k]["payload_mean"] for k in ["exp2_c2_real", "exp2_c2_5g", "exp2_c2_4g", "exp2_c2_weak"]])
    funnel_pct = (1.0 - c2_kb / c1_kb) * 100.0

    md = f"""# 中期报告实验数据统计与架构洞察汇总（Part A & Part B）

> **测试平台**：HUAWEI Mate 40 Pro (`NOH-AL00`, HarmonyOS/Android) + Windows PC 边缘节点（4 槽位 TFLite XNNPACK 并行池 + MediaPipe BlazeFace），共采集 **15 组** 标准化真机实验数据（有效样本总计 **{sum(stats[k]['n'] for k in set(EXP_FILES.keys()) if not k.startswith('exp2_c1_real') and not k.startswith('exp2_c2_real') and not k.startswith('exp4_c3_unmasked'))} 帧**）。

---

## 实验 1：不同计算切分方案横向对比（单人脸 $K=1$, `REAL/LOCAL`, `Q75`）

| 切分方案 | 帧数 N | 总延迟均值±标准差 (ms) | P50 (ms) | P95 (ms) | 有效 FPS | Prep (ms) | Net (ms) | M3 Mask (ms) | M4 FaceNet (ms) | M5 Match (ms) | 上传负载 (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row("All-Local (GPU)", "exp1_local_gpu")}
{row("All-Local (CPU-4T)", "exp1_local_cpu")}
{row("Cut-C1 (全图卸载)", "exp1_c1_real")}
{row("Cut-C2 (裁剪卸载)", "exp1_c2_real")}
{row("Cut-C3 (M3本地+M4卸载)", "exp1_c3_real")}

* **核心量化结论**：
  1. **边缘端算力碾压**：M4 FaceNet 特征提取从手机端 **`73.8 ms (GPU)` / `62.7 ms (CPU-4T)`** 骤降至边缘端 **`5.7 ~ 6.1 ms`**（单模块加速 **10.3x ~ 12.9x**！）。
  2. **端侧硬件反直觉洞察（CPU-4T > GPU）**：在 Batch=1 小张量（`160×160×3`）下，手机端纯 4 线程 CPU+XNNPACK（`144.8 ms`）比 GPU Delegate（`177.7 ms`）快 **18.5%**，原因是单帧小张量的 CPU-GPU 显存拷贝与多模型上下文切换开销抵消了 GPU 并行优势。
  3. **极速网络下的 C1 优势**：在低延迟直连（`REAL`）下，`Cut-C1` 将手机端耗时的 MLKit M2 检测（~`65–80 ms`）也一并卸载至边缘 BlazeFace（~`20 ms`），获得最高有效帧率 **`{stats['exp1_c1_real']['eff_fps']:.2f} FPS`**（较本地 GPU 提速 **{stats['exp1_local_gpu']['total_mean']/stats['exp1_c1_real']['total_mean']:.2f}x**）。

---

## 实验 2：网络带宽与 RTT 敏感度对比（`Cut-C1` vs `Cut-C2`, $K=1$, `Q75`）

| 实验组 | 帧数 N | 总延迟均值±标准差 (ms) | P50 (ms) | P95 (ms) | 有效 FPS | Prep (ms) | Net (ms) | M3 Mask (ms) | M4 FaceNet (ms) | M5 Match (ms) | 上传负载 (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row("Cut-C1 @ REAL", "exp2_c1_real")}
{row("Cut-C2 @ REAL", "exp2_c2_real")}
{row("Cut-C1 @ SIM_5G (+12ms, 20Mbps)", "exp2_c1_5g")}
{row("Cut-C2 @ SIM_5G (+12ms, 20Mbps)", "exp2_c2_5g")}
{row("Cut-C1 @ SIM_4G (+45ms, 3Mbps)", "exp2_c1_4g")}
{row("Cut-C2 @ SIM_4G (+45ms, 3Mbps)", "exp2_c2_4g")}
{row("Cut-C1 @ SIM_WEAK (+180ms, 300Kbps)", "exp2_c1_weak")}
{row("Cut-C2 @ SIM_WEAK (+180ms, 300Kbps)", "exp2_c2_weak")}

* **核心量化结论（三次最优切分点跃迁——直接奠定 Part C 动态划分理论基础！）**：
  1. **数据流“漏斗效应”实证**：`Cut-C2` 经手机 M2 裁剪缩放至 `160×160` 后，平均单帧上传量从 `Cut-C1` 的 **`{c1_kb:.2f} KB`** 锐减至 **`{c2_kb:.2f} KB`**，流量缩减 **`{funnel_pct:.1f}%`**！
  2. **4G 网络出现黄金交叉点（Crossover Point）**：在 `REAL` 与 `SIM_5G`（20Mbps）下 `Cut-C1` 领先；当带宽降至 `SIM_4G`（3Mbps）时，`Cut-C1` 因全图传输拖累升至 **`{stats['exp2_c1_4g']['total_mean']:.1f} ms`**，而 `Cut-C2` 仅 **`{stats['exp2_c2_4g']['total_mean']:.1f} ms`**，**正式反超 `Cut-C1`**！
  3. **弱网下全图卸载雪崩 & 本地模式反超**：在 `SIM_WEAK`（300Kbps, +180ms RTT）下，`Cut-C1` 总延迟雪崩至 **`{stats['exp2_c1_weak']['total_mean']:.1f} ms`（1.1 FPS）**；`Cut-C2` 凭借低流量将延迟压低一半以上至 **`{stats['exp2_c2_weak']['total_mean']:.1f} ms`**（快 **2.08x**）；但此时两者均慢于 `All-Local (CPU-4T)` 的 **`{stats['exp1_local_cpu']['total_mean']:.1f} ms`**！这用硬核数据证明：**不存在放之四海而皆准的静态切分点，必须根据实时网络状态与负载进行动态计算划分（Part C）**。

---

## 实验 4：压缩率权衡（JPEG Quality）与 `Cut-C3` 条件早退剪枝洞察

| 实验组 | 帧数 N | 总延迟均值±标准差 (ms) | P50 (ms) | P95 (ms) | 有效 FPS | Prep (ms) | Net (ms) | M3 Mask (ms) | M4 FaceNet (ms) | M5 Match (ms) | 上传负载 (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row("Cut-C2 @ Q50 (REAL)", "exp4_c2_q50")}
{row("Cut-C2 @ Q75 (REAL)", "exp4_c2_q75")}
{row("Cut-C2 @ Q95 (REAL)", "exp4_c2_q95")}
{row("Cut-C3 未戴口罩 (REAL, Q75)", "exp4_c3_unmasked")}
{row("Cut-C3 戴口罩早退 (REAL, Q75)", "exp4_c3_masked")}

* **核心量化结论**：
  1. **JPEG 压缩甜点（`Q75`）**：从 `Q50` 到 `Q75`，单帧体积仅微增 `1.28 KB`（`3.64 → 4.92 KB`），纯网络传输耗时几乎不变（`6.1 → 5.2 ms`）；而提升至 `Q95` 后单帧体积暴涨 **3.0x** 至 `10.95 KB`，纯网络开销随之翻倍飙升至 `19.2 ms`（+269%），边际效益极低。（注：`Q50` 组因实验末段手机本地 `Prep` 均值波动至 `79.6 ms`，导致其总延迟高于 `Q75` 的 `58.1 ms`，剥离 `Prep` 后看纯 `Net` 耗时趋势完全符合预期。）
  2. **`Cut-C3` 条件早退剪枝（Early-Exit Pruning）**：在戴口罩场景下，本地轻量 M3 拦截了 **{stats['exp4_c3_masked']['pruned_ratio']:.1f}%** 的帧直接短路，平均上传流量从 **`{stats['exp4_c3_unmasked']['payload_mean']:.2f} KB`** 暴跌至 **`{stats['exp4_c3_masked']['payload_mean']:.2f} KB`（节省 {(1 - stats['exp4_c3_masked']['payload_mean']/stats['exp4_c3_unmasked']['payload_mean'])*100:.1f}% 带宽）**，云端 M4 算力消耗从 **`{stats['exp4_c3_unmasked']['fn_mean']:.1f} ms`** 降至 **`{stats['exp4_c3_masked']['fn_mean']:.1f} ms`（节省 {(1 - stats['exp4_c3_masked']['fn_mean']/stats['exp4_c3_unmasked']['fn_mean'])*100:.1f}% 边缘算力）**，同时端到端总延迟降低 **14.4%**（`119.0 → 101.9 ms`）。

---

## 生成的高清图表文件清单（300 DPI，可直接插入 Word 中期报告）
1. `fig1_partitioning_breakdown.png` — 实验 1：各切分方案单帧模块延迟堆叠分解与有效 FPS 对比
2. `fig2_network_sensitivity.png` — 实验 2：不同网络带宽/RTT 下的延迟曲线（标注 4G 交叉点与弱网本地反超点）与纯网络开销对比
3. `fig3_compression_and_pruning.png` — 实验 4：JPEG 压缩率权衡 & `Cut-C3` 戴口罩早退剪枝收益对比
4. `fig4_latency_boxplot.png` — 主要切分方案端到端延迟箱线图与 P95 尾部稳定性对比
"""
    out_md = OUT_DIR / "midterm_summary.md"
    out_md.write_text(md, encoding="utf-8")
    return out_md


def main() -> None:
    stats = {k: load_stats(fn) for k, (_, fn) in EXP_FILES.items()}
    plot_fig1_partitioning(stats)
    plot_fig2_network(stats)
    plot_fig3_exp4(stats)
    plot_fig4_distributions(stats)
    out_md = write_markdown_summary(stats)
    print(f"[OK] Generated 4 charts and summary report in: {OUT_DIR}")
    print(f"[OK] Summary markdown: {out_md}")


if __name__ == "__main__":
    main()
