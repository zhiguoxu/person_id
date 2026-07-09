# 🤖 机器人视觉人物识别系统

> Robot Vision Person Identification System

## 概述

为机器人提供毫秒级、高精度的人物识别系统。接收高帧率视频流，识别画面中的人物并返回稳定的 Person ID。

### 核心特性

- **双层处理架构**：Tier 1 快速追踪 (3ms/帧) + Tier 2 深度识别 (异步)，兼顾实时性与识别深度。
- **四模态识别**：人脸 (ArcFace) + 全身 (OSNet) + 体型比例 + VLM 仲裁。
- **智能追踪**：BoT-SORT 多目标追踪 + 时空约束恢复。
- **动态自适应 UI**：前端自动检测任意摄像头的物理分辨率与比例（支持 16:9, 4:3 等），杜绝传输过程中的图像变形和 BBox 偏移。
- **丰富可视化**：提供 17 点骨骼 (Skeleton)、追踪轨迹 (Trail)、姿态角标、Track ID 及实时事件瀑布流与时序调试面板 (Pipeline Debug)。
- **自适应换装**：衣橱记忆库 + 交叉覆写机制。

## 快速开始

> 架构: 前端(本地浏览器/摄像头) ←WebSocket→ 后端(本地或远程 CUDA 服务器)

### 1. 部署后端环境

推荐使用集成的脚本自动管理 Conda 环境和 CUDA 依赖：

```bash
# 自动创建 conda 环境 (person_id) 并安装 PyTorch、onnxruntime-gpu 及各种依赖
bash install.sh

# 启动服务器 (自动激活环境)
bash deploy.sh
```

> **注意**: 如果使用纯 CPU 或 Mac 环境，安装脚本会自动切换对应的依赖项。默认监听端口为 `10003`。

### 2. 打开前端

直接用浏览器打开 `frontend/index.html` 即可。摄像头在本地采集，帧通过 WebSocket 发送到服务器处理。

> 后端地址配置在 `frontend/js/config.js`。如果在本地运行，会自动连接 `localhost:10003`；如果连接远程服务器，可修改对应的 IP 地址。

### 3. 视频来源的三种模式

| 模式 | 视频数据路径 | 说明 |
| --- | --- | --- |
| 本地摄像头 | 浏览器 getUserMedia → 抓帧上传 | URL 输入框留空, 点 Start Camera |
| 前端拉流 (旧) | 浏览器 flv.js 播放 → 抓帧上传 | 填入流地址, 点 Start Camera |
| **服务端拉流 (推荐)** | **服务器直接拉流识别, 前端仅观看** | 点「📡 设备推流」获取地址 → 点「▶ 服务端拉流」 |

服务端拉流模式下, 识别在后台持续运行, **不依赖浏览器页面保持打开**; 页面打开时服务端把
处理帧 (JPEG) 和识别结果实时推送给前端渲染, 多个页面可同时观看同一路流。

识别路径按视频流**原生分辨率无损处理**, 分辨率完全跟随设备推流动态适配, 换设备无需改配置
(实际分辨率可在 `stream/status` 的 `stream_width/stream_height` 中确认)。解码帧直接进
pipeline, 无缩放、无 JPEG 重压缩 —— 对比旧的浏览器抓帧路径是 640 宽 + JPEG 0.7 有损上传。

推给网页的**预览帧**默认限宽 1280 (`stream_preview_max_width`)、JPEG 质量 80
(`stream_preview_jpeg_quality`) 以省带宽, 这两项只影响观看清晰度, 不影响识别质量和框的精度。

相关接口 (camera_id 即设备 device-sn):

```
POST /api/{camera_id}/refresh_stream   # 开启设备推流 (ISS start_stream), 返回 flv_url
POST /api/{camera_id}/stream/start     # 开启服务端拉流消费, body: {"url": "..."}
POST /api/{camera_id}/stream/stop      # 停止服务端拉流消费
GET  /api/{camera_id}/stream/status    # 查询消费状态 (running/connected/fps/viewers)
```

## 项目结构

```
person_id/
├── docs/                   # 文档
│   ├── background.md       # 需求背景
│   ├── scenario_analysis.md # 场景分析
│   ├── research_report.md  # 技术调研
│   ├── frontend_design.md  # 前端设计
│   └── implementation_plan.md # 实施方案
├── src/                    # 后端源码
│   ├── config.py           # 全局配置
│   ├── detection/          # YOLO11-Pose 检测
│   ├── features/           # 特征提取 (人脸/全身/体型)
│   ├── gallery/            # 特征底库 + SQLite
│   ├── identity/           # 身份消歧 + VLM 仲裁
│   ├── tracking/           # BoT-SORT 追踪
│   ├── attention/          # 注意力引擎
│   ├── perception/         # 主动感知建议
│   ├── pipeline/           # Tier 1/2 流水线
│   └── api/                # FastAPI + WebSocket
├── frontend/               # 前端可视化
│   ├── index.html
│   ├── css/style.css
│   └── js/                 # 模块化 JS (视频采集/Canvas叠加层/通信)
├── deploy.sh               # 运行启动脚本
├── install.sh              # 依赖安装脚本
└── tests/                  # 测试
```

## 技术栈

| 组件 | 技术 |
|---|---|
| 检测 + 姿态 | YOLO11-Pose (n + x) |
| 追踪 | BoT-SORT (boxmot) |
| 人脸 | InsightFace (ArcFace-R100) |
| 全身 ReID | OSNet-AIN |
| VLM 仲裁 | Qwen-VL-Max |
| 持久化 | SQLite (aiosqlite) |
| API | FastAPI + WebSocket |
| 前端 | HTML5 + Canvas + Vanilla JS |
| 推理加速 | CUDA (onnxruntime-gpu) / CPU |

## 文档

- [需求背景](docs/background.md)
- [场景深度分析](docs/scenario_analysis.md)
- [技术调研报告](docs/research_report.md)
- [前端可视化设计](docs/frontend_design.md)
- [最终实施方案](docs/implementation_plan.md)

## 许可证

Internal use only.
