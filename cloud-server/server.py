"""
FastAPI Edge Inference Microservice for Mobile FaceNet Computation Partitioning.

Endpoints:
- GET  /ping                 : Lightweight RTT probe (respects X-Net-Profile emulation).
- POST /infer_c1_fullframe   : Cut Point C1 (M2 + M3 + M4 remote on full camera frame).
- POST /infer_c2_crops       : Cut Point C2 (M3 + M4 remote on K cropped 160x160 faces).
- POST /infer_c3_facenet     : Cut Point C3 (M4-only remote after mobile local M3 pruning).
- GET  /api/stats            : Live telemetry JSON for monitoring & demo.
- GET  /dashboard            : Real-time single-page Edge Node Dashboard for live defense.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from inference import EdgeInferenceEngine

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Controllable Network Emulation Profiles (extra_rtt_ms, uplink_kbps)
# REAL introduces zero artificial delay for genuine WiFi / 5G / USB testing.
NET_PROFILES: Dict[str, Dict[str, Optional[float]]] = {
    "REAL": {"extra_rtt_ms": 0.0, "uplink_kbps": None},
    "SIM_5G": {"extra_rtt_ms": 12.0, "uplink_kbps": 20000.0},
    "SIM_4G": {"extra_rtt_ms": 45.0, "uplink_kbps": 3000.0},
    "SIM_WEAK": {"extra_rtt_ms": 180.0, "uplink_kbps": 300.0},
}

engine: Optional[EdgeInferenceEngine] = None
recent_events: Deque[dict] = deque(maxlen=60)
mode_counts: Dict[str, int] = {"CUT_C1": 0, "CUT_C2": 0, "CUT_C3": 0}
total_requests: int = 0
last_net_profile: str = "REAL"
server_start_time: float = time.time()


async def apply_network_emulation(profile_name: Optional[str], payload_bytes: int) -> float:
    """Sleep asynchronously to emulate RTT + uplink bandwidth throttling without tainting server_infer_ms."""
    global last_net_profile
    key = (profile_name or "REAL").strip().upper()
    if key not in NET_PROFILES:
        key = "REAL"
    last_net_profile = key

    cfg = NET_PROFILES[key]
    extra_rtt_ms = float(cfg["extra_rtt_ms"] or 0.0)
    uplink_kbps = cfg["uplink_kbps"]

    tx_ms = 0.0
    if uplink_kbps and payload_bytes > 0:
        # (bytes * 8 bits) / (kbps * 1000 bits/sec) * 1000 ms/sec = bytes * 8 / kbps
        tx_ms = (payload_bytes * 8.0) / float(uplink_kbps)

    total_sim_ms = extra_rtt_ms + tx_ms
    if total_sim_ms > 0.0:
        await asyncio.sleep(total_sim_ms / 1000.0)
    return round(total_sim_ms, 2)


def record_event(
    cut_mode: str,
    net_profile: str,
    faces: int,
    payload_bytes: int,
    resp: dict,
    sim_delay_ms: float,
) -> None:
    global total_requests
    total_requests += 1
    if cut_mode in mode_counts:
        mode_counts[cut_mode] += 1
    recent_events.appendleft(
        {
            "ts": time.strftime("%H:%M:%S"),
            "mode": cut_mode,
            "net_profile": net_profile,
            "faces": faces,
            "payload_kb": round(payload_bytes / 1024.0, 2),
            "detect_ms": resp["server_detect_ms"],
            "mask_ms": resp["server_mask_ms"],
            "facenet_ms": resp["server_facenet_ms"],
            "infer_ms": resp["server_infer_ms"],
            "sim_net_ms": sim_delay_ms,
        }
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    pool_size = int(os.environ.get("POOL_SIZE", "4"))
    print(f"[EdgeServer] Loading TFLite models from {MODELS_DIR} (pool_size={pool_size})...")
    t0 = time.perf_counter()
    engine = EdgeInferenceEngine(MODELS_DIR, pool_size=pool_size)
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(
        f"[EdgeServer] Ready in {elapsed:.1f}ms | pool={engine.pool_size} "
        f"| threads/slot={engine.threads_per_slot} | C1 detector={engine.detector_backend}"
    )
    yield


app = FastAPI(title="FaceNet Edge Inference Server", version="0.2.0", lifespan=lifespan)


@app.get("/ping")
async def ping(x_net_profile: Optional[str] = Header(default="REAL")):
    sim_ms = await apply_network_emulation(x_net_profile, payload_bytes=0)
    return {
        "status": "ok",
        "server_time_ms": int(time.time() * 1000),
        "net_profile": last_net_profile,
        "sim_rtt_ms": sim_ms,
        "pool_size": engine.pool_size if engine else 0,
        "detector": engine.detector_backend if engine else "none",
    }


@app.post("/infer_c1_fullframe")
async def infer_c1_fullframe(
    file: UploadFile = File(...),
    enable_mask: bool = Form(default=True),
    x_net_profile: Optional[str] = Header(default="REAL"),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    raw = await file.read()
    sim_ms = await apply_network_emulation(x_net_profile, payload_bytes=len(raw))

    resp = await asyncio.to_thread(engine.infer_c1_fullframe, raw, enable_mask)
    record_event("CUT_C1", last_net_profile, len(resp["results"]), len(raw), resp, sim_ms)
    return resp


@app.post("/infer_c2_crops")
async def infer_c2_crops(
    files: List[UploadFile] = File(...),
    enable_mask: bool = Form(default=True),
    x_net_profile: Optional[str] = Header(default="REAL"),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    raws = [await f.read() for f in files]
    payload_bytes = sum(len(b) for b in raws)
    sim_ms = await apply_network_emulation(x_net_profile, payload_bytes=payload_bytes)

    resp = await asyncio.to_thread(engine.infer_c2_crops, raws, enable_mask)
    record_event("CUT_C2", last_net_profile, len(raws), payload_bytes, resp, sim_ms)
    return resp


@app.post("/infer_c3_facenet")
async def infer_c3_facenet(
    files: List[UploadFile] = File(...),
    x_net_profile: Optional[str] = Header(default="REAL"),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    raws = [await f.read() for f in files]
    payload_bytes = sum(len(b) for b in raws)
    sim_ms = await apply_network_emulation(x_net_profile, payload_bytes=payload_bytes)

    resp = await asyncio.to_thread(engine.infer_c3_facenet, raws)
    record_event("CUT_C3", last_net_profile, len(raws), payload_bytes, resp, sim_ms)
    return resp


@app.get("/api/stats")
async def api_stats():
    events_list = list(recent_events)
    avg_infer = (
        round(sum(e["infer_ms"] for e in events_list) / len(events_list), 2)
        if events_list
        else 0.0
    )
    avg_payload = (
        round(sum(e["payload_kb"] for e in events_list) / len(events_list), 2)
        if events_list
        else 0.0
    )
    return {
        "uptime_sec": int(time.time() - server_start_time),
        "pool_size": engine.pool_size if engine else 0,
        "threads_per_slot": engine.threads_per_slot if engine else 0,
        "detector": engine.detector_backend if engine else "none",
        "total_requests": total_requests,
        "mode_counts": mode_counts,
        "last_net_profile": last_net_profile,
        "avg_infer_ms": avg_infer,
        "avg_payload_kb": avg_payload,
        "recent": events_list[:20],
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Edge Inference Node Dashboard — FaceNet Offloading</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8; --green: #34d399; --amber: #fbbf24; --rose: #fb7185; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { margin: 0 0 6px; font-size: 20px; color: var(--accent); }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 18px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
  .label { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .val { font-size: 24px; font-weight: 700; }
  .c1 { color: var(--rose); } .c2 { color: var(--green); } .c3 { color: var(--amber); } .acc { color: var(--accent); }
  table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; border: 1px solid var(--border); font-size: 13px; }
  th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: #0b1220; color: var(--muted); font-weight: 600; }
  .pill { padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; display: inline-block; }
  .pill-CUT_C1 { background: rgba(251,113,133,0.2); color: var(--rose); }
  .pill-CUT_C2 { background: rgba(52,211,153,0.2); color: var(--green); }
  .pill-CUT_C3 { background: rgba(251,191,36,0.2); color: var(--amber); }
</style>
</head>
<body>
  <h1>⚡ Mobile-Edge FaceNet Inference Node Dashboard</h1>
  <div class="sub" id="meta">Connecting to edge inference node...</div>
  <div class="grid">
    <div class="card"><div class="label">Total Offload Requests</div><div class="val acc" id="totalReq">0</div></div>
    <div class="card"><div class="label">Avg Edge Inference (ms)</div><div class="val c2" id="avgInfer">0.0</div></div>
    <div class="card"><div class="label">Avg Upload Payload (KB)</div><div class="val c3" id="avgPayload">0.0</div></div>
    <div class="card"><div class="label">Active Net Profile</div><div class="val acc" id="netProfile">REAL</div></div>
    <div class="card"><div class="label">C1 / C2 / C3 Hits</div><div class="val" id="modeHits">0 / 0 / 0</div></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>Cut Point</th><th>Net Profile</th><th>Faces (K)</th>
        <th>Payload (KB)</th><th>M2 Detect</th><th>M3 Mask</th><th>M4 FaceNet</th>
        <th>Total Edge Infer</th><th>Sim Net Delay</th>
      </tr>
    </thead>
    <tbody id="tbody"><tr><td colspan="10" style="color:#94a3b8;">Waiting for mobile client requests...</td></tr></tbody>
  </table>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('meta').textContent =
      `Pool: ${d.pool_size} parallel slots (${d.threads_per_slot} threads/slot) | C1 Detector: ${d.detector} | Uptime: ${d.uptime_sec}s`;
    document.getElementById('totalReq').textContent = d.total_requests;
    document.getElementById('avgInfer').textContent = d.avg_infer_ms.toFixed(1);
    document.getElementById('avgPayload').textContent = d.avg_payload_kb.toFixed(2);
    document.getElementById('netProfile').textContent = d.last_net_profile;
    document.getElementById('modeHits').textContent =
      `${d.mode_counts.CUT_C1} / ${d.mode_counts.CUT_C2} / ${d.mode_counts.CUT_C3}`;
    const tb = document.getElementById('tbody');
    if (d.recent.length === 0) return;
    tb.innerHTML = d.recent.map(e => `
      <tr>
        <td>${e.ts}</td>
        <td><span class="pill pill-${e.mode}">${e.mode}</span></td>
        <td>${e.net_profile}</td>
        <td>${e.faces}</td>
        <td><b>${e.payload_kb.toFixed(2)} KB</b></td>
        <td>${e.detect_ms.toFixed(1)} ms</td>
        <td>${e.mask_ms.toFixed(1)} ms</td>
        <td>${e.facenet_ms.toFixed(1)} ms</td>
        <td><b>${e.infer_ms.toFixed(1)} ms</b></td>
        <td>${e.sim_net_ms.toFixed(1)} ms</td>
      </tr>
    `).join('');
  } catch (err) {
    document.getElementById('meta').textContent = 'Disconnected: ' + err;
  }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
