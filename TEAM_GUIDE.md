# 团队协作部署、新 UI 说明与真机测试手册

> **项目仓库**：`https://github.com/tt1z/Mobile-FaceNet-Edge-Offloading.git`  
> **适用对象**：项目组成员（颜嘉进、胡雪斌、苏世杰）  
> **目标**：从 GitHub 拉取代码后，10 分钟内完成 **PC 边缘推理服务端 + Android 手机端** 部署，熟悉新 UI 控件功能并独立采集真机实验数据。

---

## 一、 代码拉取与端云部署步骤

### 1. 克隆仓库
```powershell
git clone https://github.com/tt1z/Mobile-FaceNet-Edge-Offloading.git
cd Mobile-FaceNet-Edge-Offloading
```
目录结构说明：
- `app/`：Android 客户端源码（Kotlin + CameraX + MLKit + TFLite + OkHttp3）
- `cloud-server/`：PC 端 Python 边缘推理微服务（FastAPI + 4 槽位 TFLite 并行池 + MediaPipe）
- `perf_logs/`：从手机导出的性能测试 `.csv` 原始数据
- `analyze_midterm.py`：一键统计分析与 300 DPI 高清图表生成脚本

---

### 2. 启动 PC 端边缘推理服务端（`cloud-server/`）
PC 端使用 **`uv`** 管理 Python 3.11 虚拟环境（无需手动配 Anaconda）：

```powershell
cd cloud-server

# 1. 首次拉取：自动创建 .venv 并安装依赖（已配置清华源）
uv sync --python 3.11

# 2. 启动边缘推理服务器（默认监听 0.0.0.0:8000）
.\.venv\Scripts\python.exe server.py
```
看到控制台输出 `Uvicorn running on http://0.0.0.0:8000` 即启动成功。  
👉 **电脑浏览器打开** `http://localhost:8000/dashboard` 可实时查看边缘节点收到的请求、上传包体大小（KB）与推理耗时。

---

### 3. 编译运行 Android 客户端（真机调试）
1. 用 **Android Studio** 打开仓库根目录（注意：不要修改 Gradle 7.4 版本和 `settings.gradle` 中的阿里云镜像）。
2. 手机开启 **USB 调试** 并通过数据线连接电脑（或使用 WiFi 同局域网）。
3. **打通端云通信链路（二选一，推荐方式 A）**：
   * **方式 A：USB 数据线直连（最稳定，零防火墙干扰）**
     在终端执行一次端口转发（每次重插数据线需重跑一次）：
     ```powershell
     adb reverse tcp:8000 tcp:8000
     ```
     手机端设置保持默认地址 `http://127.0.0.1:8000` 即可。
   * **方式 B：真实 WiFi / 手机热点无线直连（测真实无线网络时使用）**
     确保手机与电脑连在**同一 WiFi 或热点**下，用 `ipconfig` 查电脑局域网 IP（如 `192.168.x.x`），在手机 App 设置里填入 `http://192.168.x.x:8000`（如连不上请检查 Windows 防火墙是否放行 8000 端口）。
4. 在 Android Studio 点击 **Run ▶** 将 App 安装到真机。首次启动若提示选择人脸照片目录，按 `主目录 -> 人物姓名子目录 -> 人脸照片` 结构选择（已有缓存则直接点弹窗的 **`LOAD`** 秒加载）。

---

## 二、 手机软件新 UI 界面与控件全解

App 打开并加载人脸底库后，屏幕顶部会实时显示 **性能监控 HUD** 与 **4 个交互控制按钮**：

```text
┌────────────────────────────────────────────────────────────────┐
│ Mode: [CUT_C2] (EDGE-C2|REAL) | K=1 | FPS: 9.9                 │  <-- 实时 HUD
│ Total: 100.6ms (Prep:80.6 | Net:6.4 | Infer:9.1 | M5:0.5)      │
│ M3(Mask):3.4ms | M4(FN):5.7ms | Payload: 4.8 KB (Q75)          │
└────────────────────────────────────────────────────────────────┘
  [ Cut-C2 (裁剪卸载) ▼ ]   [ GPU ○ ]   [ REC ]   [ ⚙ ]          <-- 控制栏
```

### 1. 顶部 HUD 各指标含义
| 字段 | 含义 | 说明 |
| :--- | :--- | :--- |
| **`Mode`** | 当前计算切分模式 | `ALL_LOCAL` / `CUT_C1` / `CUT_C2` / `CUT_C3` / `AUTO` |
| **`(backend \| netProfile)`** | 实际执行后端与网络档位 | 如 `GPU`、`CPU-4T`、`EDGE-C1/C2/C3`；网络档位 `LOCAL` / `REAL` / `SIM_4G` 等 |
| **`K`** | 当前画面检测到的人脸数 | $K=0$ 时 `Cut-C2 / C3` 会自动剪枝，上传 `0 KB` |
| **`FPS`** | 实时滑动平均有效帧率 | 由单帧端到端总延迟决定（`1000 / Total`） |
| **`Total`** | 端到端单帧总耗时 (ms) | 包含预处理、网络、推理、比对与调度开销 |
| **`Prep`** | 本地预处理耗时 (ms) | 相机旋转 + 本地人脸检测/裁剪（如启用）+ JPEG 编码 |
| **`Net`** | 纯网络往返与传输开销 (ms) | 已剥离服务端计算耗时（`HTTP_RTT - Server_Infer`） |
| **`Infer`** | 模型推理总耗时 (ms) | `M3(Mask) + M4(FaceNet)`（`Cut-C1` 还含云端 M2 检测） |
| **`M5`** | 本地特征聚类比对耗时 (ms) | 128 维向量 L2 距离计算，通常 `< 1 ms` |
| **`Payload`** | 单帧上传网络字节数 (KB) | 直观对比 `Cut-C1` 全图（~25 KB）与 `Cut-C2` 裁剪（~4.8 KB） |

### 2. 控制栏 4 个核心按钮功能
1. **切分模式下拉框（最左侧）**：
   * **`All-Local (本地)`**：M1~M6 全部在手机运行（已优化为先测口罩，戴口罩直接跳过 FaceNet）。
   * **`Cut-C1 (全图卸载)`**：手机跳过 MLKit 人脸检测，直接把 `480×640` 完整帧发给 PC 端执行 M2+M3+M4。
   * **`Cut-C2 (裁剪卸载)`**：手机本地 M2 检测并裁剪缩放到 `160×160`（约 4~5 KB/脸）发给 PC 端执行 M3+M4。
   * **`Cut-C3 (Mask本地+M4卸载)`**：手机本地跑完 M1+M2+M3，**戴口罩直接在手机端拦截（0 KB 上传）**，未戴口罩才把人脸发给 PC 端只跑 M4。
   * **`Auto-Dynamic (动态)`**：运行时动态划分模式（Part C 接入）。
2. **`GPU` 开关（中左）**：
   * 在 **`GPU Delegate`** 与 **`CPU-4T`（纯 4 线程 CPU + XNNPACK）** 之间实时热切换。
   * ⚠️ **注意**：该开关**仅影响 `All-Local` 模式**；在 `Cut-C1 / C2 / C3` 模式下 FaceNet 跑在 PC 端，无需切换此开关。
3. **`REC` / `STOP(N)` 按钮（中右）**：
   * **录制每帧的性能耗时数据到 `.csv` 表格**（⚠️ **不是录视频**），按钮变红并实时显示已录帧数 `STOP(25)`。
   * 再次点击停止，文件自动保存至手机私有目录：  
     `/sdcard/Android/data/com.ml.quaterion.facenetdetection/files/Documents/sa_perf_YYYYMMDD_HHMMSS.csv`
4. **`⚙` 设置按钮（最右侧）**：
   * **Server URL**：修改 PC 服务端地址（USB 转发填 `http://127.0.0.1:8000`，WiFi 填 `http://<PC_IP>:8000`）。
   * **`Test Ping`**：测试与 PC 端连通性并显示实测 RTT（绿色 `OK!` 表示通路正常）。
   * **Network Profile**：切换 **`REAL`（真实网络直连）** 或可控仿真档位（`SIM_5G` / `SIM_4G` / `SIM_WEAK`）。
   * **JPEG Quality**：切换上传图片压缩质量 **`Q50` / `Q75`（默认）/ `Q95`**。

---

## 三、 标准化真机数据测试清单

> **采集规范**：每组实验保持人脸稳定在画面中，点击 **`REC`** 连续录制 **`10~15 秒`（约 30~60 帧）** 后点击 **`STOP`**。多组实验可连续录制，系统会按时间戳自动分开保存。

| 实验组 | 控制变量 | 手机端具体操作步骤 | 核心验证目标 |
| :--- | :--- | :--- | :--- |
| **实验 1：切分点横向对比**<br>*(必做)* | 4 种切分模式 + 端侧硬件<br>(`K=1`, `REAL`, `Q75`) | 1. 选 `All-Local`，开 `GPU` 录 1 组<br>2. 关 `GPU`（变为 `CPU-4T`）录 1 组<br>3. 切 `Cut-C1` 录 1 组<br>4. 切 `Cut-C2` 录 1 组<br>5. 切 `Cut-C3` 录 1 组 | 对比 M1~M6 各模块耗时占比、验证 `Cut-C2` 流量缩减与边缘 10x 算力加速 |
| **实验 2：网络环境敏感度**<br>*(必做)* | 不同带宽与 RTT<br>(`Cut-C1` vs `Cut-C2`, `K=1`, `Q75`) | 点 `⚙` 依次切换以下网络档位，每档分别录 `Cut-C1` 和 `Cut-C2` 各 1 组：<br>1. `REAL`（或真实 WiFi）<br>2. `SIM_5G` (+12ms, 20Mbps)<br>3. `SIM_4G` (+45ms, 3Mbps)<br>4. `SIM_WEAK` (+180ms, 300Kbps) | 寻找 4G 网络下的 **C1/C2 交叉点**，实证弱网下全图卸载雪崩及本地反超现象 |
| **实验 3：多人同框负载**<br>*(强烈推荐)* | 人脸数量 $K=1$ vs $K=2 \sim 3$<br>(`REAL`, `Q75`) | 用另一台手机/电脑屏幕打开多人正脸合照对准摄像头，分别录制：<br>1. `All-Local (GPU)`<br>2. `All-Local (CPU-4T)`<br>3. `Cut-C2` | 实证手机端 $O(K)$ 串行阻塞导致 FPS 暴跌，对比 PC 端 4 槽位并行池的加速比 |
| **实验 4：压缩率与早退剪枝**<br>*(加分项)* | JPEG 质量 & 口罩遮挡<br>(`REAL`) | **4A 压缩率**：选 `Cut-C2`，在 `⚙` 中分别切 `Q50`、`Q75`、`Q95` 各录 1 组<br>**4B 早退剪枝**：选 `Cut-C3`（`Q75`），戴口罩或用手遮住口鼻录 1 组 | 寻找压缩率甜点（`Q75`），实证 `Cut-C3` 本地拦截戴口罩人脸实现 0 KB 零流量 |

---

## 四、 数据导出与一键绘图

所有测试组录制完成后，把手机连上电脑执行：

```powershell
# 1. 一键拉取手机里所有录制好的 CSV 到电脑 perf_logs 目录
adb pull /sdcard/Android/data/com.ml.quaterion.facenetdetection/files/Documents/. d:\ASSDK\project\perf_logs\

# 2. 若实验文件与 analyze_midterm.py 中预设文件名一致（或更新映射后），运行分析脚本生成图表
cloud-server\.venv\Scripts\python.exe analyze_midterm.py
```
运行后即可在 `midterm_report_results/` 目录查看生成的 **300 DPI 高清对比图** 与 **Markdown 统计汇总表**。
