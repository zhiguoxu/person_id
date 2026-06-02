# 🤖 机器人视觉人物识别系统 — 最终实施方案

> **目标**：为机器人提供毫秒级、高精度的人物识别系统。接收高帧率视频流，识别画面中的人物并返回稳定的 Person ID。  
> **原则**：效果优先，算力充足（CUDA GPU），追踪优先于识别。  
> **算力环境**：CUDA GPU（onnxruntime-gpu, torch.cuda, ultralytics CUDA）  
> **状态**：✅ v2.3 适配前后端分离部署
> **部署**：后端 `8.145.38.125:10003` (CUDA) ← WebSocket → 前端 (本地浏览器)

| 版本 | 日期 | 变更内容 |
|---|---|---|
| v2.1 | 2026-05-29 | 初版确认：双层架构、四模态、前端可视化 |
| v2.1.1 | 2026-05-29 | 确认 CUDA 算力环境，开始编码 |
| v2.2 | 2026-05-29 | 首次完整编码：47 文件 / 10,249 行 |
| v2.3 | 2026-05-29 | 前后端分离：本地摄像头 + 远程 CUDA 推理 (端口 10003) |

---

## 已确认的决策

| 决策项 | 最终选择 |
|---|---|
| 帧率 | **10+ FPS**，双层处理架构 |
| 声纹模块 | **第二版**再集成 |
| 体型比例 + 重排序 | **全部加入** |
| 持久化 | **SQLite** |
| 前端 | **第二版**，先专注后端核心逻辑 |
| 注意力引擎 | **简化版**（无语音/手势模块） |
| 回调接口 | **空函数占位** |

---

## 一、系统总览

### 1.1 双层处理架构 (Two-Tier Architecture)

```
摄像头 (10-30 FPS 视频流)
  │
  ├──→ [Tier 1: 每帧处理, ~3ms/帧]
  │    YOLO11n-pose (2.9M 参数, 轻量)
  │      → 人体检测 + 17 关键点
  │    BoT-SORT 追踪器 (+ SOLIDER ReID 特征)
  │      → 稳定 Track ID, 平滑轨迹
  │    身份缓存查询
  │      → Track_ID → Person_ID 映射
  │    输出: 每帧的追踪结果 + 身份状态
  │
  └──→ [Tier 2: 关键帧处理, ~1 FPS / 按需触发]
       触发条件:
         a) 新 Track 出现 (未知身份)
         b) Track 重现 (遮挡后恢复)
         c) 定期刷新 (每 30 秒)
         d) 人脸质量提升 (更好的角度/距离)
       处理内容:
         YOLO11x-pose (58.8M 参数, 精确)
           → 精确关键点 + 姿态分桶
         InsightFace (SCRFD + ArcFace-R100)
           → 人脸检测 + 512 维人脸特征
         SOLIDER ReID
           → 2048 维全身特征
         体型比例提取
           → 骨骼几何特征 (零额外成本)
         底库匹配 + K-Reciprocal 重排序
         四重阈值歧义消除
         VLM 仲裁 (如需)
       结果注入 → Tier 1 身份缓存
```

### 1.2 核心设计原则

| 原则 | 说明 |
|---|---|
| **追踪 > 识别** | 一旦 Track 建立且身份确认，绝对信赖追踪结果，不重复调用重模型 |
| **效果 > 速度** | 使用最重量级的模型（YOLO11x-pose, SOLIDER, ArcFace-R100） |
| **异步解耦** | Tier 2 全部异步化，Tier 1 永远 <5ms 返回 |
| **人类在环** | 系统不确定时主动求助，人工反馈是最高优先级铁律 |
| **特征动态演化** | 底库特征随时间衰减更新，自适应换装 |

---

## 二、技术选型

| 组件 | 选择 | 版本/参数 | Python 包 |
|---|---|---|---|
| 人体检测 (Tier 1) | YOLO11n-pose | 2.9M, ~1.8ms | `ultralytics` |
| 人体检测 (Tier 2) | YOLO11x-pose | 58.8M, ~12ms, mAP 69.5% | `ultralytics` |
| 多目标追踪 | BoT-SORT + ReID | 相机运动补偿 | `boxmot` |
| 人脸检测 | InsightFace SCRFD | WiderFace Hard 96.1% | `insightface` |
| 人脸识别 | ArcFace-R100 | LFW 99.83%, 512 维 | `insightface` |
| 全身 ReID | SOLIDER | Rank-1 96.9%, mAP 93.9% | `fast-reid` |
| 重排序 | K-Reciprocal | mAP +2-4%, <1ms | 自实现 |
| VLM 仲裁 | Qwen-VL-Max | API | `openai` |
| 持久化 | SQLite | — | `aiosqlite` |

### 依赖清单

```
# 核心检测与追踪
ultralytics>=8.3.0        # YOLO11-pose
boxmot>=11.0.0             # BoT-SORT 追踪器

# 人脸识别
insightface>=0.7.3         # SCRFD + ArcFace
onnxruntime-gpu>=1.18.0    # InsightFace GPU 推理

# ReID (SOLIDER 从源码安装)
# git clone https://github.com/tinyvision/SOLIDER-REID

# 深度学习基础
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
numpy>=1.26.0
scipy>=1.12.0

# 服务层
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.6.0

# VLM 仲裁
openai>=1.14.0             # Qwen-VL-Max API (兼容 OpenAI 协议)

# 持久化
aiosqlite>=0.20.0          # SQLite 异步驱动

# 工具
pillow>=10.2.0
scikit-learn>=1.4.0
loguru>=0.7.0
```

---

## 三、核心模块详细设计

### 3.1 模块 1：人体检测与姿态分析 (`detection/`)

#### 处理流程

```
输入帧 (BGR, 1920×1080)
  │
  ├─► YOLO 推理 (Tier 1: yolo11n / Tier 2: yolo11x)
  │     输出: List[Detection]
  │       ├─ bbox: (x1, y1, x2, y2)
  │       ├─ conf: float
  │       └─ keypoints: (17, 3)  # 17 个关键点 (x, y, conf)
  │
  ├─► 快速拦截: 无检测 → 短路返回
  │
  ├─► 正脸判断 (零额外成本)
  │     条件: nose.conf > 0.5 AND l_eye.conf > 0.3 AND r_eye.conf > 0.3
  │
  └─► 姿态分桶 (Pose Bucket)
        ├─ FRONTAL:  鼻子+双眼可见
        ├─ LEFT:     鼻子+仅左耳可见
        ├─ RIGHT:    鼻子+仅右耳可见
        ├─ BACK:     鼻子不可见, 耳朵可见
        └─ UNKNOWN:  关键点不足
```

#### 姿态分桶算法

```python
def classify_pose(keypoints: np.ndarray) -> PoseBucket:
    """
    基于 COCO 17 关键点判断人体朝向
    keypoints shape: (17, 3) — x, y, confidence
    """
    CONF_THRESH = 0.3
    
    has_nose = keypoints[0, 2] > CONF_THRESH
    has_l_eye = keypoints[1, 2] > CONF_THRESH
    has_r_eye = keypoints[2, 2] > CONF_THRESH
    has_l_ear = keypoints[3, 2] > CONF_THRESH
    has_r_ear = keypoints[4, 2] > CONF_THRESH
    has_eyes = has_l_eye and has_r_eye

    if has_nose and has_eyes:
        return PoseBucket.FRONTAL
    elif has_nose and has_l_ear and not has_r_ear:
        return PoseBucket.LEFT
    elif has_nose and has_r_ear and not has_l_ear:
        return PoseBucket.RIGHT
    elif not has_nose and (has_l_ear or has_r_ear):
        return PoseBucket.BACK
    else:
        return PoseBucket.UNKNOWN
```

---

### 3.2 模块 2：双模态特征提取 (`features/`)

#### 人脸特征提取

```
YOLO 人体检测框
  │
  ├─► 裁剪人体区域
  ├─► InsightFace SCRFD 人脸检测
  │     输出: face_bbox, landmarks(5点), det_score
  │
  ├─► 人脸质量评估 (复合评分)
  │     ├─ blur_score:     cv2.Laplacian(face).var()  → [0,1]
  │     ├─ size_score:     min(w,h) / MIN_FACE_SIZE   → [0,1]
  │     ├─ landmark_conf:  mean(landmark_confidences)  → [0,1]
  │     ├─ pose_score:     1.0 - abs(yaw)/90           → [0,1]
  │     ├─ lighting:       gray_histogram_std           → [0,1]
  │     └─ quality = 0.25*blur + 0.15*size + 0.15*landmark + 0.30*pose + 0.15*light
  │         阈值: quality < 0.4 → 丢弃不入库
  │
  └─► ArcFace-R100 特征提取
        输出: face_embedding (512维, L2 归一化)
```

#### 全身 ReID 特征提取

```
YOLO 人体检测框
  │
  ├─► 裁剪人体区域 (带 10% padding)
  ├─► 预处理: Resize 256×128, ImageNet normalize
  ├─► 水平翻转测试增强 (TTA)
  └─► SOLIDER 推理
        输出: body_embedding (2048维, L2 归一化)
        最终特征 = mean(原图特征, 翻转图特征)
```

#### 体型比例特征 (零额外成本)

```python
@dataclass
class BodyProportions:
    """基于 COCO 17 关键点的骨骼几何特征"""
    torso_leg_ratio: float        # 躯干/腿比例
    shoulder_hip_ratio: float     # 肩宽/髋宽比例
    arm_torso_ratio: float        # 手臂/躯干比例
    head_body_ratio: float        # 头/身体比例
    relative_height_px: float     # 帧内相对高度 (像素)
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.torso_leg_ratio,
            self.shoulder_hip_ratio,
            self.arm_torso_ratio,
            self.head_body_ratio,
        ])
```

---

### 3.3 模块 3：双模态特征底库 (`gallery/`)

#### 核心概念

```
PersonProfile (每个已知人物)
  │
  ├─ 人脸池 (Face Bank) — 永恒锚点
  │   ├─ FRONTAL 桶: [特征1(质量0.92), 特征2(质量0.85), ...]  最多5个
  │   ├─ LEFT 桶:    [特征1(质量0.71), ...]
  │   ├─ RIGHT 桶:   [特征1(质量0.68), ...]
  │   └─ BACK 桶:    [特征1(质量0.55), ...]
  │   衰减半衰期: 365 天 (几乎不衰减)
  │
  ├─ 衣橱记忆库 (Wardrobe) — 周期锚点
  │   ├─ Outfit #1: 红色外套, 最后穿 2026-05-28, 全身特征(2048维)
  │   ├─ Outfit #2: 黑色T恤, 最后穿 2026-05-25, 全身特征(2048维)
  │   └─ ...  最多20套
  │   衰减半衰期: 30 天
  │
  ├─ 体型比例 (Body Proportions)
  │   └─ 累积平均的骨骼比例向量
  │
  └─ 元数据
      ├─ person_id, display_name
      ├─ created_at, last_seen, total_appearances
      └─ vlm_description (VLM 生成的文字描述)
```

#### 数据结构

```python
@dataclass
class FeatureEntry:
    embedding: np.ndarray        # L2 归一化特征向量
    pose_bucket: PoseBucket
    quality_score: float
    timestamp: float
    source_image: Optional[bytes] = None  # JPEG, 供 VLM 使用
    
    def time_decay_weight(self, now: float, half_life_days: float) -> float:
        age_days = (now - self.timestamp) / 86400
        return 0.5 ** (age_days / half_life_days)

@dataclass
class OutfitRecord:
    body_embedding: np.ndarray   # 2048维
    quality_score: float
    first_seen: float
    last_seen: float
    seen_count: int = 1
    
    def recency_weight(self, now: float) -> float:
        days_since = (now - self.last_seen) / 86400
        if days_since < 1:   return 1.0
        elif days_since < 7: return 0.85
        elif days_since < 30: return 0.6
        elif days_since < 90: return 0.3
        else: return 0.1

@dataclass
class PersonProfile:
    person_id: str
    display_name: str
    
    # 人脸池 (按姿态分桶)
    face_features: dict[PoseBucket, list[FeatureEntry]]
    # 衣橱记忆库
    wardrobe: list[OutfitRecord]
    # 体型比例
    body_proportions: Optional[BodyProportions] = None
    body_proportions_samples: int = 0
    # VLM 文字描述
    vlm_description: Optional[str] = None
    # 元数据
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_appearances: int = 0
    
    # 配置
    MAX_FACES_PER_BUCKET: int = 5
    MAX_OUTFITS: int = 20
    FACE_HALF_LIFE_DAYS: float = 365.0
    OUTFIT_HALF_LIFE_DAYS: float = 30.0
```

#### 匹配算法

**人脸匹配 — 姿态感知 Max-Pooling**：
1. 优先在相同姿态桶中搜索（正脸 vs 正脸）
2. 如果同桶无结果，扩展到相邻桶
3. 返回所有人的最高余弦相似度（加权时间衰减 + 质量加权）

**全身匹配 — 衣橱遍历**：
1. 遍历目标人物的所有衣橱记录
2. 对每条记录计算余弦相似度
3. 应用近因权重（昨天穿的 > 三个月前穿的）

**双模态融合**：
- 人脸权重: 0.50（最强身份信号）
- 全身权重: 0.35（中等信号）
- 体型比例: 0.15（辅助信号）
- **捷径**：人脸质量 > 0.7 且匹配度 > 0.75 → 直接确认，跳过融合
- 权重按模态可用性和质量动态归一化

**K-Reciprocal 重排序**：
- 在融合匹配后对全身 ReID 结果应用重排序
- 底库 < 50 人时 < 1ms
- 提升 mAP +2-4%

#### 入库策略

**人脸入库**：
- 质量 < 0.4 → 拒绝入库
- 每个姿态桶最多 5 个特征
- 桶满时，替换质量最低的

**衣橱更新**：
- 与已有衣橱匹配 (sim > 0.85) → 更新 last_seen + EMA 融合特征
- 新衣服 → 添加新记录
- 超容量 → 淘汰最久未见的

**交叉覆写**（化学反应）：
- 人脸确认身份后，将当前全身特征追加到该人的衣橱库
- 实现自动换装适应

---

### 3.4 模块 4：身份歧义消除引擎 (`identity/`)

#### 四重阈值

```
ReID 阶段:  X_reid = 0.72 (确信),  Y_reid = 0.55 (疑似)
VLM 阶段:   X_vlm  = 0.80 (确信),  Y_vlm  = 0.60 (疑似)
```

#### 状态判定

| 状态 | 条件 | 后续动作 |
|---|---|---|
| **确信 (Confident)** | 仅一人 ≥ X, 且远超第二名 | 直接绑定身份 |
| **疑似 (Suspected)** | Y ≤ 最高 < X | 交给 VLM 仲裁 |
| **冲突 (Conflict)** | 多人 ≥ X (如双胞胎) | 交给 VLM 仲裁 |
| **陌生 (Stranger)** | 所有人 < Y | 创建新身份 |

#### 阶段递进

```
新目标 → 提取特征 → 融合匹配 + 重排序
  │
  ├─ ReID 确信 → 绑定身份 ✅
  ├─ ReID 陌生 → 创建新身份 ✅
  └─ ReID 疑似/冲突 → VLM 仲裁
       ├─ VLM 确信 → 绑定身份 ✅
       └─ VLM 仍不确定 → 主动交互回调 (占位)
            └─ 人工确认 → 强制入库 (最高优先级铁律) ✅
```

#### VLM 仲裁 Prompt

```python
SYSTEM_PROMPT = """你是一个专业的人物识别助手。对比两张图片，
判断是否是同一人。分析维度：
1. 面部特征（脸型、五官）
2. 体型特征（身高、体型）
3. 发型特征
4. 衣着特征（注意：同一人可能换衣服）
5. 其他显著特征（配饰、纹身等）

输出 JSON:
{
    "is_same_person": true/false/null,
    "confidence": 0.0~1.0,
    "reasoning": "详细分析",
    "distinguishing_features": ["特征1", "特征2"]
}"""
```

---

### 3.5 模块 5：追踪引擎 (`tracking/`)

#### Tier 1 追踪器

```python
class TrackingEngine:
    def __init__(self):
        # BoT-SORT 追踪器 (via boxmot)
        self.tracker = BoTSORT(
            reid_weights="solider_r50.pt",
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,           # 30 帧 @ 10 FPS = 3 秒
            match_thresh=0.8,
            cmc_method="sparseOptFlow"  # 相机运动补偿
        )
        
        # 身份缓存: Track_ID → PersonIdentity
        self.identity_cache: dict[int, PersonIdentity] = {}
        
        # 时空记忆: 丢失 Track 的位置记忆
        self.spatial_memory: dict[int, SpatialMemory] = {}
```

#### 时空约束

```python
SPATIAL_TIMEOUT_SEC = 10.0       # 记忆有效期
SPATIAL_DISTANCE_PX = 200.0      # 最大中心距离

def check_spatial_memory(self, track) -> Optional[SpatialMemory]:
    """
    如果丢失的 Track 在短时间内、附近位置重现，
    大概率还是同一人 → 避免唤醒重模型
    """
```

---

### 3.6 模块 6：注意力引擎 (简化版) (`attention/`)

```python
class AttentionScore:
    # 基础分 (简化版, 无语音/手势模块)
    area_score: float = 0.0          # 人体面积占比 [0,1]
    center_score: float = 0.0        # 与画面中心的距离 [0,1]
    face_visibility: float = 0.0     # 正脸可见度 [0,1]
    approaching_bonus: float = 0.0   # 主动靠近 [0,0.15]
    momentum_bonus: float = 0.0      # 注意力惯性 [0,0.2]
    
    @property
    def base_score(self) -> float:
        return (self.area_score * 0.3 + 
                self.center_score * 0.3 + 
                self.face_visibility * 0.4)
    
    @property
    def total_score(self) -> float:
        return self.base_score * (1.0 + self.approaching_bonus + 
                                  self.momentum_bonus)
```

---

### 3.7 模块 7：主动感知建议 (`perception/`)

```python
class ActivePerceptionAdvisor:
    """告诉机器人"怎么做能改善识别效果" — 回调/建议系统"""
    
    def suggest(self, target, identity_status) -> Optional[PerceptionAdvice]:
        # 背影 → 建议移动位置获取正脸
        # 人脸质量低 → 建议靠近
        # 身份不确定 → 建议口头询问
        # 完全陌生 → 建议打招呼
        pass  # 占位实现
```

---

## 四、API 设计

### 核心接口

#### `POST /api/vision/process_frame` — 主处理接口

```python
class ProcessFrameRequest(BaseModel):
    frame_base64: str
    audio_pcm_base64: Optional[str] = None  # 预留, 第二版声纹
    asr_text: Optional[str] = None
    reid_threshold: Optional[float] = None
    vlm_threshold: Optional[float] = None

class TrackedPersonResponse(BaseModel):
    track_id: int
    person_id: Optional[str]
    display_name: Optional[str]
    bbox: list[float]              # [x1, y1, x2, y2] 归一化
    confidence: float
    status: str                    # "confirmed" | "identifying" | "spatial_inferred"
    pose_bucket: str
    attention_score: float
    is_current_target: bool

class ProcessFrameResponse(BaseModel):
    timestamp: float
    frame_id: int
    persons: list[TrackedPersonResponse]
    current_target: Optional[TrackedPersonResponse]
    pending_identifications: int
    processing_time_ms: float
```

#### `GET /api/vision/current_target` — 高频旁路接口

```python
class CurrentTargetResponse(BaseModel):
    person_id: Optional[str]
    display_name: Optional[str]
    bbox: Optional[list[float]]
    confidence: float
    last_update_ms: float
```

#### `POST /api/vision/confirm_identity` — 人工确认接口

```python
class ConfirmIdentityRequest(BaseModel):
    track_id: int
    confirmed_person_id: str
    confirmed_name: Optional[str] = None
    source: str = "human"
```

#### `POST /api/vision/callback` — 主动交互回调 (占位)

```python
async def trigger_callback(payload: IdentityCallbackPayload):
    """占位: 第二版对接机器人主脑"""
    logger.info(f"Identity callback triggered: {payload.event_type}")
    pass
```

#### `ws://host:port/ws/vision` — WebSocket 视频流接口

```python
# 双向通信:
#   客户端 → 服务端: Binary JPEG 帧 (640×480, quality=0.7, ~30KB)
#   服务端 → 客户端: JSON 处理结果 (含 pipeline_debug 调试信息)
#   客户端 → 服务端: JSON 配置更新 / 操作命令
#   服务端 → 客户端: JSON 异步事件推送

# 背压控制: 客户端在上一帧结果返回前不发送新帧
# 自适应帧率: 根据后端延迟动态调整 (5-30 FPS)
```

#### 前端辅助 REST 接口

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 返回前端页面 (StaticFiles) |
| `/api/config` | GET | 获取当前配置 (初始化控制面板) |
| `/api/config` | PUT | 批量更新配置 |
| `/api/gallery/persons` | GET | 获取所有已知人物列表 |
| `/api/gallery/person/{id}` | GET | 获取单个人物详情 |

---

### 4.2 前端可视化测试平台

> 完整设计详见 [frontend_design.md](./frontend_design.md)

**技术栈**：HTML5 + Vanilla JS + CSS (暗色 Glassmorphism 主题)  
**通信**：WebSocket (双向，二进制帧 + JSON 结果)

**核心面板**：

| 面板 | 功能 |
|---|---|
| **Video Panel** | 摄像头画面 + Canvas 叠加层 (检测框/关键点/轨迹/标签) |
| **Pipeline Debug** | 算法流水线各阶段：状态/耗时/缩略图/匹配分数条形图 |
| **Controls** | 8 个实时阈值滑块 (Detection/ReID X,Y/VLM X,Y/Quality/Face Shortcut/Outfit Match) |
| **Event Timeline** | 水平滚动事件时间线 (新人物/身份确认/追踪丢失/VLM 仲裁) |
| **Person Gallery** | 已识别人物卡片列表，点击展开底库详情 |

**关键特性**：
- Canvas 双层结构：video 底层 + 透明 canvas 叠加层（高性能渲染）
- 匹配分数条形图 + 阈值线：直观看到阈值调整对决策的影响
- 背压控制：上一帧未处理完不发送新帧，避免服务端过载
- 管理员可直接在界面手动确认身份（点击"确认为 Alice"按钮）

---

## 五、代码目录结构

```
person_id/
├── docs/                           # 文档
│   ├── background.md               # 需求背景
│   ├── scenario_analysis.md        # 场景深度分析
│   ├── research_report.md          # 行业调研报告
│   ├── frontend_design.md          # 前端可视化设计
│   └── implementation_plan.md      # 本文档 (实施方案)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # 全局配置 (阈值、模型路径等)
│   │
│   ├── detection/                  # 模块 1: 人体检测与姿态分析
│   │   ├── __init__.py
│   │   ├── yolo_pose.py            # YOLO11 Pose 封装 (n/x 双模式)
│   │   └── pose_classifier.py      # 姿态分桶算法
│   │
│   ├── features/                   # 模块 2: 特征提取
│   │   ├── __init__.py
│   │   ├── face_extractor.py       # InsightFace ArcFace 封装
│   │   ├── body_extractor.py       # SOLIDER ReID 封装
│   │   ├── quality_assessor.py     # 复合质量评估
│   │   └── body_proportions.py     # 体型比例特征
│   │
│   ├── gallery/                    # 模块 3: 特征底库
│   │   ├── __init__.py
│   │   ├── data_models.py          # PersonProfile, FeatureEntry, OutfitRecord
│   │   ├── matcher.py              # GalleryMatcher (姿态分桶 Max-Pooling)
│   │   ├── updater.py              # GalleryUpdater (入库、衣橱、交叉覆写)
│   │   ├── reranker.py             # K-Reciprocal 重排序
│   │   └── persistence.py          # SQLite 持久化
│   │
│   ├── identity/                   # 模块 4: 身份歧义消除
│   │   ├── __init__.py
│   │   ├── resolver.py             # AmbiguityResolver (四重阈值)
│   │   ├── vlm_arbitrator.py       # VLM 仲裁器
│   │   └── multi_modal_fusion.py   # 自适应多模态融合
│   │
│   ├── tracking/                   # 模块 5: 追踪引擎
│   │   ├── __init__.py
│   │   ├── engine.py               # TrackingEngine (BoT-SORT)
│   │   └── spatial_memory.py       # 时空约束机制
│   │
│   ├── attention/                  # 模块 6: 注意力引擎 (简化版)
│   │   ├── __init__.py
│   │   └── engine.py               # AttentionEngine
│   │
│   ├── perception/                 # 模块 7: 主动感知建议
│   │   ├── __init__.py
│   │   └── advisor.py              # ActivePerceptionAdvisor (占位)
│   │
│   ├── pipeline/                   # 主流水线编排
│   │   ├── __init__.py
│   │   ├── tier1.py                # Tier 1: 每帧快速追踪
│   │   ├── tier2.py                # Tier 2: 关键帧深度识别
│   │   ├── temporal_aggregator.py  # 质量加权时序特征聚合
│   │   └── orchestrator.py         # 总调度器
│   │
│   └── api/                        # API 服务层
│       ├── __init__.py
│       ├── server.py               # FastAPI 应用
│       ├── routes.py               # 路由定义
│       ├── schemas.py              # Pydantic 模型
│       └── websocket.py            # WebSocket 视频流接口
│
├── frontend/                       # 前端可视化测试平台
│   ├── index.html                  # 主页面
│   ├── css/
│   │   └── style.css               # 暗色主题样式 (Glassmorphism)
│   └── js/
│       ├── app.js                  # 应用入口 + 初始化
│       ├── websocket.js            # WebSocket 管理 + 背压控制
│       ├── video-capture.js        # 摄像头采集 (getUserMedia)
│       ├── overlay-renderer.js     # Canvas 叠加渲染
│       ├── pipeline-panel.js       # 算法流程可视化面板
│       ├── controls-panel.js       # 阈值控制面板
│       ├── events-timeline.js      # 事件时间线
│       └── person-gallery.js       # 人物画廊
│
├── models/                         # 预训练模型存放
│   └── .gitkeep
│
├── data/                           # 运行时数据
│   ├── gallery.db                  # SQLite 底库
│   └── logs/
│
├── tests/                          # 测试
│   ├── test_detection.py
│   ├── test_features.py
│   ├── test_gallery.py
│   ├── test_identity.py
│   ├── test_tracking.py
│   └── test_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## 六、主循环处理流程

```python
class VisionOrchestrator:
    """视觉系统总调度器"""
    
    async def process_frame(self, frame: np.ndarray) -> ProcessFrameResponse:
        """
        每帧调用 (10-30 FPS)
        Tier 1: <5ms 返回
        """
        t0 = time.perf_counter()
        
        # ===== Tier 1: 快速追踪 =====
        
        # Step 1: 轻量检测 (~2ms)
        detections = self.detector_fast.detect(frame)  # yolo11n
        if not detections:
            return empty_response()
        
        # Step 2: 追踪器更新 (~1ms)
        tracked = self.tracker.update(frame, detections)
        
        # Step 3: 身份缓存查询 (<0.1ms)
        for person in tracked:
            if person.track_id in self.identity_cache:
                person.person_id = self.identity_cache[person.track_id]
                person.status = "confirmed"
            else:
                person.status = "identifying"
        
        # Step 4: 触发 Tier 2 (按需, 异步)
        for person in tracked:
            if self._should_trigger_tier2(person):
                asyncio.create_task(
                    self._tier2_identify(person, frame)
                )
        
        # Step 5: 注意力评分 (<0.1ms)
        scores = self.attention.compute_scores(tracked, frame.shape)
        target = self.attention.select_target(scores)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return self._build_response(tracked, scores, target, elapsed_ms)
    
    def _should_trigger_tier2(self, person: TrackedPerson) -> bool:
        """判断是否触发 Tier 2 深度识别"""
        if person.status == "identifying":
            return True  # 新目标
        if person.track_id in self._tier2_pending:
            return False  # 已在队列中
        if time.time() - person.last_tier2_time > 30:
            return True  # 定期刷新
        return False
    
    async def _tier2_identify(self, person, frame):
        """
        Tier 2: 异步深度识别 (~30ms + VLM 可选)
        """
        # 精确检测 + 关键点
        det = self.detector_heavy.detect_single(frame, person.bbox)  # yolo11x
        
        # 姿态分桶
        pose = classify_pose(det.keypoints)
        
        # 人脸特征
        face_emb, face_quality = None, 0.0
        if pose != PoseBucket.BACK:
            face_result = self.face_extractor.extract(frame, det.bbox)
            if face_result:
                face_emb = face_result.embedding
                face_quality = face_result.quality
        
        # 全身 ReID 特征
        body_emb = self.body_extractor.extract(frame, det.bbox)
        
        # 体型比例
        proportions = BodyProportions.from_keypoints(det.keypoints)
        
        # 时序聚合
        agg_body = self.temporal_agg.add_and_get(
            person.track_id, body_emb, face_quality or 0.5
        )
        
        # 底库匹配
        face_results = self.gallery.match_face(face_emb, pose) if face_emb else []
        body_results = self.gallery.match_body(agg_body or body_emb)
        
        # 重排序
        if body_results:
            body_results = self.reranker.rerank(body_results)
        
        # 体型匹配
        prop_results = self.gallery.match_proportions(proportions) if proportions else []
        
        # 自适应融合
        fused = self.fusion.fuse(
            face_result=face_results[0] if face_results else None,
            body_result=body_results[0] if body_results else None,
            proportion_result=prop_results[0] if prop_results else None,
            face_quality=face_quality,
        )
        
        # 歧义消除
        decision = self.resolver.resolve_reid(fused)
        
        if decision.status == IdentityStatus.CONFIDENT:
            self._inject_identity(person.track_id, decision)
            self._update_gallery(decision.best_match, face_emb, body_emb, 
                                proportions, pose, face_quality, frame, det.bbox)
        elif decision.status == IdentityStatus.STRANGER:
            new_id = self._create_new_person(face_emb, body_emb, proportions, 
                                              pose, face_quality, frame, det.bbox)
            self._inject_identity(person.track_id, new_id)
        else:
            # 疑似/冲突 → VLM 仲裁
            vlm_result = await self.vlm.arbitrate(
                self._crop_person(frame, det.bbox),
                self._get_candidate_images(decision.candidates)
            )
            vlm_decision = self.resolver.resolve_vlm(vlm_result)
            
            if vlm_decision.status == IdentityStatus.CONFIDENT:
                self._inject_identity(person.track_id, vlm_decision)
            else:
                # 占位: 主动交互回调
                await self._trigger_callback(person, vlm_decision)
```

---

## 七、关键参数配置

```python
class Config:
    # === 模型配置 ===
    YOLO_FAST_MODEL = "yolo11n-pose.pt"     # Tier 1
    YOLO_HEAVY_MODEL = "yolo11x-pose.pt"    # Tier 2
    INSIGHTFACE_MODEL = "buffalo_l"
    VLM_MODEL = "qwen-vl-max"
    VLM_API_KEY = "sk-xxx"
    VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # === 检测阈值 ===
    YOLO_CONFIDENCE = 0.5
    KEYPOINT_CONFIDENCE = 0.3
    MIN_FACE_SIZE = 40  # 像素
    
    # === ReID 四重阈值 ===
    REID_CONFIDENT_THRESHOLD = 0.72   # X_reid
    REID_SUSPECTED_THRESHOLD = 0.55   # Y_reid
    VLM_CONFIDENT_THRESHOLD = 0.80    # X_vlm
    VLM_SUSPECTED_THRESHOLD = 0.60    # Y_vlm
    FACE_SHORTCUT_THRESHOLD = 0.75    # 人脸捷径
    
    # === 底库配置 ===
    MAX_FACES_PER_BUCKET = 5
    MAX_OUTFITS = 20
    FACE_HALF_LIFE_DAYS = 365.0
    OUTFIT_HALF_LIFE_DAYS = 30.0
    QUALITY_ENROLL_THRESHOLD = 0.4
    QUALITY_UPDATE_THRESHOLD = 0.7
    OUTFIT_MATCH_THRESHOLD = 0.85
    
    # === 追踪配置 ===
    TRACK_BUFFER_FRAMES = 30
    SPATIAL_TIMEOUT_SEC = 10.0
    SPATIAL_DISTANCE_PX = 200.0
    TIER2_REFRESH_INTERVAL_SEC = 30.0
    
    # === 融合权重 ===
    FACE_BASE_WEIGHT = 0.50
    BODY_BASE_WEIGHT = 0.35
    PROPORTION_BASE_WEIGHT = 0.15
    
    # === 时序聚合 ===
    TEMPORAL_WINDOW_SIZE = 5
    
    # === 服务配置 ===
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    CALLBACK_URL = ""  # 占位, 第二版对接
    GALLERY_DB_PATH = "data/gallery.db"
    LOG_LEVEL = "INFO"
```

---

## 八、验证方案

### 单元测试

| 模块 | 测试内容 | 验证指标 |
|---|---|---|
| `detection/` | YOLO 推理、姿态分桶 | 检测准确率、分桶正确率 |
| `features/` | 人脸/全身特征、质量评估、体型比例 | 特征维度、L2 范数 |
| `gallery/` | 匹配、入库、衣橱更新、重排序 | Rank-1、库容量控制 |
| `identity/` | 四重阈值判定 | 状态分类正确率 |
| `tracking/` | 追踪连续性、时空约束 | ID 稳定性、切换率 |

### 集成测试

```bash
pytest tests/ -v
python -m src.benchmark --frames 100
```

### 手动验证

启动后端: `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`

测试场景:
1. 单人正脸 → 确认身份
2. 转头 → 姿态分桶切换 + 多角度特征入库
3. 离开再回来 → 时空约束 + 重识别
4. 换衣服 → 衣橱交叉覆写
5. 多人场景 → 注意力引擎选择
6. 遮挡后恢复 → Track ID 稳定性

---

## 九、第二版规划 (预告)

| 功能 | 优先级 | 依赖 |
|---|---|---|
| 声纹识别 (ECAPA-TDNN) | 高 | `speechbrain` |
| 前端 Webcam 可视化 | 高 | HTML/JS |
| 语音模块注意力加权 | 中 | 声纹模块 |
| 手势检测注意力加权 | 中 | — |
| 步态识别 | 低 | 需 >10 FPS |
| 主动交互回调对接 | 高 | 机器人主脑接口 |
