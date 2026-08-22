"""
机器人视觉人物识别系统 - 全局配置

加载风格与 voice_server 对齐 (pydantic-settings):
- 基底 config_files/config.yaml + 环境覆盖 config_files/config_{APP_ENV}.yaml
  (APP_ENV 默认 dev, 两层 yaml 深合并)
- APP_ 前缀环境变量可覆盖顶层字段, 优先级高于 yaml
- 模块级单例 ``config``: 启动期状态(lifespan 里套过一次 DB 覆盖后冻结),
  启动期消费(模型加载/建连)读它; 运行期 hot 字段一律经
  ``src.configs.override.current_config()`` 读生效快照

所有可调参数集中管理。阈值参数经 web 控制台在线编辑(含 Controls 滑块,
统一落 DB 覆盖层, 多实例同步), 改完下一次读取生效。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from voice_agent_common.config_models.cos_config import COSConfig
from voice_agent_common.config_models.log_stream_config import LogStreamConfig
from voice_agent_common.config_models.redis_config import RedisConfig

# ==============================================================================
# 项目路径
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


class VideoRecordConfig(BaseModel):
    """拉流录像配置(pipeline/video_recorder.py)。

    consume/start→stop 期间自动落盘并上传 COS; 断线重连仍属同一拉流会话。
    单段上限避免超大文件; 超长自动切新段(新 DB 行 + 新 COS 对象)。
    消费点逐帧现读 current_config().video_record, 全部字段热生效(encoder
    对已在写的段不生效, 下个段起效)。
    """
    enabled: bool = False
    max_seconds: float = 1800.0  # 单段最长 30 分钟
    fps: float = 10.0            # 录制帧率(读流线程节流)
    max_width: int = 1280        # 录制限宽(像素), 0=原分辨率
    # 编码器选择: auto = NVENC 可用则用(GPU 独立编码单元, 不占 CUDA 核心,
    # 大规模路数必选), 否则退 libx264(CPU)。nvenc/x264 强制指定。
    # 均出 H.264 mp4(网页可播); ffmpeg 缺失时最终兜底 cv2 mp4v(仅可下载)
    encoder: str = "auto"  # auto | nvenc | x264
    # 仅在看到人时录制: 开段要求识别结果里当前有人; 连续 no_person_seconds
    # 秒无人则收段上传(段尾会带上这段无人画面), 下次看到人另起新段(新视频)
    require_person: bool = True
    no_person_seconds: float = 5.0
    # 开段的可见性新鲜度窗口(秒): 最近这么多秒内看到过人才允许开段。
    # 识别结果按处理帧率(stream_max_fps)更新, 比读流帧率慢; 断流重连后
    # 旧的"有人"标志可能过期, 该窗口兜住这两种滞后。不宜小于识别帧间隔
    person_fresh_seconds: float = 3.0


class HardwareConfig(BaseModel):
    """硬件与计算设备配置"""
    device: str = "cuda:0"  # 统一计算设备，替代原有的 yolo_device, reid_device 等

    @property
    def insightface_ctx_id(self) -> int:
        if self.device == "cpu":
            return -1
        return int(self.device.split(":")[-1]) if ":" in self.device else 0


class DetectionConfig(BaseModel):
    """检测模块配置"""
    # YOLO 模型
    yolo_fast_model: str = "yolo11n-pose.pt"  # Tier 1 轻量模型
    yolo_confidence: float = 0.5  # 检测置信度阈值
    yolo_iou_threshold: float = 0.7  # NMS IoU 阈值
    yolo_max_det: int = 10  # 最大检测数

    # 关键点
    # 最小人体像素高度 — Tier1 硬门槛, 过小的人直接不追踪/不入库。
    # 调高可收紧注册: 只对靠得够近、分辨率够高的人建档 (人脸/人体特征更可靠)。
    min_person_height_px: int = 120


class FaceConfig(BaseModel):
    """人脸识别配置"""
    insightface_model: str = "buffalo_l"  # InsightFace 模型包 (仅用于 SCRFD 检测)
    det_size: tuple[int, int] = (640, 640)  # 人脸检测输入尺寸

    # 入库人脸最小像素边长 (人脸 bbox 短边) — 小于此值不入库。
    # eDifFIQA 分数无法完全反映分辨率, 太小的脸即使"清晰"也只是低分辨率特征, 故额外加尺寸门槛。
    min_face_size: int = 60

    # 人脸识别模型 — ArcFace / AdaFace 可切换
    recognition_backend: str = "adaface"  # "arcface" 或 "adaface"
    arcface_model: str = "w600k_r50.onnx"  # ArcFace ONNX 模型文件名 (在 MODELS_DIR 下)
    adaface_model: str = "adaface_ir101.onnx"  # AdaFace ONNX 模型文件名 (在 MODELS_DIR 下)

    # 人脸质量评估模型 — eDifFIQA 变体可切换
    # tiny: MobileFaceNet (~1.7M params, ~0.3ms)  — 最快, 精度一般
    # small: IResNet-18   (~11M params,  ~1ms)    — 平衡
    # medium: IResNet-50  (~44M params,  ~2ms)    — 较高精度
    # large: IResNet-100  (~65M params,  ~3ms)    — 最高精度, 跨模型泛化最好
    ediffiqa_variant: str = "medium"  # "tiny", "small", "medium", "large"


class ReIDConfig(BaseModel):
    """人员重识别配置"""
    # SOLIDER (Swin-Small, CVPR 2023) — 全身 ReID 特征提取
    reid_model_name: str = "solider_swin_small"  # ReID 模型名
    reid_model_weights: str = ""  # 模型权重路径 (空=自动查找 models/solider_swin_small_reid.pth)
    reid_input_size: tuple[int, int] = (384, 128)  # 输入尺寸 (H, W) — SOLIDER 标准尺寸
    reid_pixel_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)  # 像素均值 (SOLIDER-REID swin_small 标准)
    reid_pixel_std: tuple[float, float, float] = (0.5, 0.5, 0.5)  # 像素标准差 (SOLIDER-REID swin_small 标准)
    use_flip_test: bool = False  # 水平翻转测试增强


class GalleryConfig(BaseModel):
    """特征底库配置"""
    # 人脸库
    max_faces_per_bucket: int = 5  # 每个姿态桶最多特征数
    face_match_half_life_days: float = 365.0  # 匹配端质心权重半衰期

    # 衣橱库
    max_outfits: int = 20  # 最大衣橱记录数
    outfit_match_threshold: float = 0.85  # 衣橱匹配阈值 (同一套衣服)

    # 入库质量门槛 — 人脸与人体分开管理
    # 注册人脸要求更高质量: 人脸是强标识, 一张糊脸入库会污染质心、引发误识;
    # 人体特征容错更高 (多桶 + 时间衰减), 门槛可略低以保证覆盖率。
    face_quality_enroll_threshold: float = 0.55   # 人脸入库最低质量分 (eDifFIQA large 评估)
    body_quality_enroll_threshold: float = 0.40   # 人体/衣橱入库最低质量分 (清晰度+完整度)
    ediffiqa_enroll_variant: str = "large"  # 入库质量评估模型变体 (独立于 Tier1, 默认最大)

    # 入库衰减 — 统一量纲: 半衰期 (天)
    face_enroll_half_life_days: float = 100.0  # 人脸入库半衰期 (发型/妆容变化慢)
    body_enroll_half_life_days: float = 50.0  # 人体入库半衰期 (换装导致变化快)

    # 人体库
    max_body_per_bucket: int = 3  # 每个姿态桶最多人体特征数


class MatchingConfig(BaseModel):
    """匹配与融合配置"""
    # 四级置信度阈值 (A > B > C, 无 D)
    A_threshold: float = 0.85  # 笃定 (唯一终态)
    A_margin: float = 0.20  # 笃定所需最小 margin
    B_threshold: float = 0.72  # 确定
    B_margin: float = 0.10  # 确定所需最小 margin
    C_threshold: float = 0.55  # 怀疑/陌生 分界线

    # Body Top-K Blend 参数
    blend_alpha: float = 0.7  # peak 权重 (1-α 为 depth 权重)
    cross_pose_discount: float = 0.7  # 跨姿态投票权折扣 (同姿态=1.0)
    wardrobe_boost_gamma: float = 0.2  # wardrobe 提升因子 (贝叶斯 lift ≈ 1.5×)

    # 多模态融合
    face_base_weight: float = 0.7  # 人脸基础权重 (multi_modal_fusion.fuse 使用)
    body_base_weight: float = 0.2  # 全身基础权重
    proportion_base_weight: float = 0.1  # 体型比例基础权重

    # Sigmoid 门控参数 (各模态独立校准)
    face_gate_q0: float = 0.3  # 人脸质量翻转点
    face_gate_k: float = 10.0  # 人脸质量门控斜率
    body_gate_q0: float = 0.5  # 人体质量翻转点
    body_gate_k: float = 10.0  # 人体质量门控斜率


class TrackingConfig(BaseModel):
    """追踪引擎配置"""
    # BoT-SORT 参数
    track_high_thresh: float = 0.5  # 高置信度检测阈值
    track_low_thresh: float = 0.1  # 低置信度检测阈值
    new_track_thresh: float = 0.6  # 新轨迹创建阈值
    track_buffer: int = 30  # 轨迹缓冲帧数
    match_thresh: float = 0.8  # 匹配阈值
    cmc_method: str = "sof"  # 相机运动补偿方法 (ecc/orb/sof/sift)


class MultiFrameConfig(BaseModel):
    """多帧处理配置"""
    # Tier1 帧收集
    recent_min_interval: float = 0.25  # RecentBuffer 帧间最小间隔 (时间多样性)

    # Tier2 质量缓存
    face_pool_size: int = 10  # 人脸质量缓存容量
    body_pool_size: int = 10  # 人体质量缓存容量

    # 聚合质量阈值 — 低于此质量的帧不参与身份聚合 (影响识别, 间接影响入库特征来源)
    agg_min_face_quality: float = 0.20  # 人脸聚合最低质量 (收紧: 0.1 → 0.2)
    agg_min_body_quality: float = 0.30  # 人体聚合最低质量

    # Tier2 (ReID) 调度 (注意力目标基准间隔)
    tier2_fast_interval: float = 1.0  # IDENTIFYING/SUSPECTED/CONFLICT 间隔
    tier2_slow_interval: float = 5.0  # CONFIDENT/STRANGER 间隔

    # Tier3 (VLM) 调度
    vlm_cooldown: float = 5.0  # VLM 冷却周期 (注意力目标)

    # DEFINITE 后台富化
    definite_enrich_interval: float = 10.0  # 富化周期 todo: 临时短一点

    # 注意力差异化
    non_attention_factor: float = 2.0  # 非注意力目标: 所有间隔 × 2


class VLMConfig(BaseModel):
    """VLM 仲裁配置"""
    enabled: bool = False  # 是否启用 Tier3 VLM 仲裁
    model: str = "qwen-vl-max"
    # 必填、无代码默认 (与 voice_server 口径一致): 密钥只放环境 yaml
    # (config_{APP_ENV}.yaml), 不进代码、不进基底 config.yaml。
    # 不启用 VLM 的环境也须显式写 api_key: "" (fail-fast, 缺配置起不来)
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_sec: float = 30.0
    max_retries: int = 2
    max_candidates: int = 3


class VoiceEmbedExtractorConfig(BaseModel):
    """声音 embedding 提取配置(VoiceEmbedExtractor 专用, 纯"信号→向量", 无身份决策)。

    比对阈值/注册门控等决策参数在 agent_server 侧(声纹库与花名册同生命周期)。
    配方依据 voice_agent/test/speaker_id/ 真实数据评测(recipe_findings.md)。
    """
    enabled: bool = False
    # 说话人 embedding 模型(ONNX), 缺失时启动自动下载
    model_path: str = str(
        MODELS_DIR / "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx")
    provider: str = "cuda"      # 本机 H20 实测 6.8ms/次(CPU 37ms)
    num_threads: int = 4        # 仅 provider=cpu 时生效

    # 子段平均配方参数(评测锁定, 一般无需改动)
    seg_window_sec: float = 3.0
    seg_hop_sec: float = 1.5


class Config(BaseSettings):
    """
    系统总配置

    所有模块的配置参数集中管理, 加载机制与 voice_server 一致:
    yaml(基底 + 环境覆盖) + APP_ 前缀环境变量。
    阈值参数经在线编辑/Controls 滑块调整(落 DB 覆盖层, 见 configs/override.py)。
    服务自身参数 (host/port/拉流等, 原 ServerConfig) 扁平在顶层,
    与 voice/agent server 的配置结构保持一致。
    """
    # 服务版本号 (web 控制台「系统配置」页展示; 在线编辑锁定, 见 editable_fields)
    version: str = "0.2.0"

    # ── 服务自身 (原 ServerConfig, 顶层扁平) ──
    host: str = "0.0.0.0"
    port: int = 10003  # 远程 CUDA 服务器端口
    log_level: str = "INFO"
    gallery_db_path: str = str(DATA_DIR / "gallery.db")

    # 图像矫正
    image_correction_enabled: bool = False  # 是否启用镜头畸变矫正

    # ISS 直播流 API (device-sn 由请求的 camera_id 动态传入, 不在此写死)
    # test/prod 两套环境, 由前端界面选择、每次请求通过 env 参数指定
    iss_api_url_test: str = "https://iss-test.joyin-ai.com"  # ISS 测试环境
    iss_api_url_prod: str = "https://iss-prod.joyin-ai.com"  # ISS 生产环境

    # 服务端拉流消费 (StreamConsumer)
    stream_max_fps: float = 15.0  # 处理帧率上限 (拉到的多余帧直接丢弃)
    # 识别路径分辨率: 0 = 不缩放, 按视频流原生分辨率处理 (无损)。
    # 仅当算力不足时才设为正数 (如 1280) 用等比缩小换速度。
    stream_proc_max_width: int = 0
    # ---- 前端预览帧 (下面两项只影响网页观看的带宽/清晰度) ----
    # 识别用的是解码原帧, 与预览参数无关; 识别结果坐标随 frame_w/frame_h 下发,
    # 预览图缩放不会导致框偏移。
    # 1280/q80 约 90KB/帧, 12fps 约 9Mbps; 链路窄时可在 Controls 面板调小。
    stream_preview_max_width: int = 1280  # 预览帧最大宽度, 0 = 原生分辨率
    stream_preview_jpeg_quality: int = 80  # 预览帧 JPEG 质量 (1-100)
    stream_reconnect_delay: float = 2.0  # 拉流断开后的重连间隔 (秒)
    # GPU 解码(NVDEC, pipeline/gpu_decoder.py): ffmpeg -hwaccel cuda 拉流,
    # 解码走显卡独立 ASIC 不占 CPU(大规模路数必选)。本机 ffmpeg 不支持
    # cuda hwaccel 时自动退回 cv2 CPU 软解; 单次打开失败也按次退回
    stream_gpu_decode: bool = True

    # ---- 拉流自动恢复 (pipeline/restream.py) ----
    # 连续拉流失败 (打开失败/连上但读不到几帧就断) 达到该次数后, 自动重新
    # 开启设备推流并切换到新地址。控制台 Controls 面板可调。
    stream_restream_fail_threshold: int = 3
    # 重推流前查设备是否在线 (设备关机后无法推流, 不在线则跳过): 直读
    # voice_server 写入 Redis 的 ws:online 在线标记 (key 格式的事实源:
    # packages/common/voice_agent_common/utils/live_events.py)。
    # 连接复用 config.redis 的 host/密码; 标记在 voice_server 的主库,
    # db 与 namespace 须与其配置一致 (redis.db / live_namespace)
    voice_online_redis_db: int = 2
    voice_live_namespace: str = "default"

    # 拉流录像(consume 期间自动录制上传 COS)。默认值即生产值, yaml 只写偏离项
    video_record: VideoRecordConfig = VideoRecordConfig()

    # WebSocket
    ws_max_frame_size: int = 1024 * 1024  # 1MB 最大帧大小
    # 配置在线编辑(DB 覆盖层)的存储, 与 voice/agent 同款 session_store 表结构。
    # 必填、无代码默认 (与 voice_server 口径一致), 只放环境 yaml: dev 先落本地
    # SQLite, 切共享 MySQL 时改成 mysql+aiomysql://... 即可, 代码零改动
    db_url: str
    # 录像上传 COS (与 voice_server 同桶同密钥约定); 密钥/bucket 只放环境 yaml
    cos: COSConfig
    hardware: HardwareConfig = HardwareConfig()
    detection: DetectionConfig = DetectionConfig()
    face: FaceConfig = FaceConfig()
    reid: ReIDConfig = ReIDConfig()
    gallery: GalleryConfig = GalleryConfig()
    matching: MatchingConfig = MatchingConfig()
    tracking: TrackingConfig = TrackingConfig()
    multiframe: MultiFrameConfig = MultiFrameConfig()
    # vlm/redis 为必填块 (无代码默认): 含密钥/连接地址, 只在环境 yaml 里配置,
    # 缺失则启动直接失败 (fail-fast); 不启用的环境显式写空值 (见各类 docstring)
    vlm: VLMConfig
    voice_embed: VoiceEmbedExtractorConfig = VoiceEmbedExtractorConfig()
    # Redis 连接 (拉流期望状态持久化 + 日志聚合 Stream, 共用连接参数)。
    # 复用 common 的 RedisConfig(全字段必填、无代码默认, 与 voice/agent 同款):
    # 连接参数只放环境 yaml, 缺配置起不来; 日志聚合转发无条件建连, 连不上
    # 启动即失败——与 voice/agent 口径一致
    redis: RedisConfig
    # 日志聚合 Stream(独立连接/独立 db): 本进程日志 XADD 到 Redis Stream,
    # 由 console_server 统一消费入库, 配置须与 voice/agent/console 一致
    log_stream: LogStreamConfig = LogStreamConfig()

    model_config = SettingsConfigDict(
        yaml_file_encoding='utf-8',
        env_prefix="APP_"
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        run_env = os.getenv("APP_ENV", "dev")
        current_path = os.path.dirname(__file__)
        path_prefix = f"{current_path}/config_files/"
        base_yaml = path_prefix + "config.yaml"
        env_yaml = path_prefix + f"config_{run_env}.yaml"

        # 优先级: 靠前的源更高。env 必须排在 yaml 前, 否则 APP_* 环境变量
        # 盖不过 yaml 里已写的字段(与 voice_server 口径一致)
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=[base_yaml, env_yaml], deep_merge=True),
        )

    def to_dict(self) -> dict:
        """序列化为字典 (用于 API 返回)"""
        return self.model_dump()

    def iss_api_url(self, env: str) -> str:
        """按环境名取 ISS 服务地址 (env: "test" | "prod")。"""
        if env == "prod":
            return self.iss_api_url_prod
        return self.iss_api_url_test

    # 可调参数定义 (单一来源)
    # key → (config_section_name, attr_name, min, max, step, group, label)
    # section 为空串 = 顶层字段 (原 ServerConfig, 已扁平到 Config 顶层)
    _TUNABLE_DEFS: dict = {
        "A_THRESHOLD": ("matching", "A_threshold", 0, 1, 0.01, "reid", "A Threshold (笃定)"),
        "B_THRESHOLD": ("matching", "B_threshold", 0, 1, 0.01, "reid", "B Threshold (确定)"),
        "C_THRESHOLD": ("matching", "C_threshold", 0, 1, 0.01, "reid", "C Threshold (怀疑)"),
        "FACE_QUALITY_ENROLL_THRESHOLD": ("gallery", "face_quality_enroll_threshold", 0, 1, 0.05, "quality", "人脸入库质量门槛"),
        "BODY_QUALITY_ENROLL_THRESHOLD": ("gallery", "body_quality_enroll_threshold", 0, 1, 0.05, "quality", "人体入库质量门槛"),
        "MIN_FACE_SIZE": ("face", "min_face_size", 0, 200, 5, "quality", "入库人脸最小像素"),
        "MIN_PERSON_HEIGHT_PX": ("detection", "min_person_height_px", 0, 400, 10, "quality", "最小人体像素高度"),
        "AGG_MIN_FACE_QUALITY": ("multiframe", "agg_min_face_quality", 0, 1, 0.05, "quality", "人脸聚合最低质量"),
        "AGG_MIN_BODY_QUALITY": ("multiframe", "agg_min_body_quality", 0, 1, 0.05, "quality", "人体聚合最低质量"),
        "OUTFIT_MATCH_THRESHOLD": ("gallery", "outfit_match_threshold", 0, 1, 0.01, "matching", "衣橱匹配阈值"),
        "STREAM_RESTREAM_FAIL_THRESHOLD": ("", "stream_restream_fail_threshold", 1, 20, 1, "stream", "自动重推流失败次数阈值"),
        # 预览带宽 (只影响网页观看, 不影响识别): 观看卡顿调小, 网速好调大
        "STREAM_PREVIEW_MAX_WIDTH": ("", "stream_preview_max_width", 320, 1920, 160, "stream", "预览帧最大宽度(px)"),
        "STREAM_PREVIEW_JPEG_QUALITY": ("", "stream_preview_jpeg_quality", 30, 95, 5, "stream", "预览帧 JPEG 质量"),
    }

    def get_tunable_params(self) -> dict:
        """获取可调参数当前值及元数据 (供前端滑块渲染)。"""
        result = {}
        for key, (section, attr, mn, mx, step, group, label) in self._TUNABLE_DEFS.items():
            cfg_section = getattr(self, section) if section else self
            result[key] = {
                "value": getattr(cfg_section, attr),
                "min": mn, "max": mx, "step": step,
                "group": group, "label": label,
            }
        return result

    def tunable_key_paths(self) -> dict[str, str]:
        """滑块键名 → 配置点路径 (供 /api/params 的 PUT 转写成 DB 覆盖项)。"""
        mapping = {
            key: f"{section}.{attr}" if section else attr
            for key, (section, attr, *_) in self._TUNABLE_DEFS.items()
        }
        # 仅通过顶部按钮控制, 不在 Controls 面板显示
        mapping["IMAGE_CORRECTION_ENABLED"] = "image_correction_enabled"
        return mapping


config = Config()
