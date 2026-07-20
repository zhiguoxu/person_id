"""
机器人视觉人物识别系统 - 全局配置

所有可调参数集中管理。支持通过 WebSocket 实时更新阈值参数。
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

# ==============================================================================
# 项目路径
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class HardwareConfig(BaseModel):
    """硬件与计算设备配置"""
    device: str = "cuda:2"  # 统一计算设备，替代原有的 yolo_device, reid_device 等

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
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_sec: float = 30.0
    max_retries: int = 2
    max_candidates: int = 3


class VoiceEmbedExtractorConfig(BaseModel):
    """声音 embedding 提取配置(VoiceEmbedExtractor 专用, 纯"信号→向量", 无身份决策)。

    比对阈值/注册门控等决策参数在 agent_server 侧(声纹库与花名册同生命周期)。
    配方依据 voice_agent/test/speaker_id/ 真实数据评测(recipe_findings.md)。
    """
    enabled: bool = True
    # 说话人 embedding 模型(ONNX), 缺失时启动自动下载
    model_path: str = str(
        MODELS_DIR / "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx")
    provider: str = "cuda"      # 本机 H20 实测 6.8ms/次(CPU 37ms)
    num_threads: int = 4        # 仅 provider=cpu 时生效

    # 子段平均配方参数(评测锁定, 一般无需改动)
    seg_window_sec: float = 3.0
    seg_hop_sec: float = 1.5


class ServerConfig(BaseModel):
    """服务配置"""
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

    def iss_api_url(self, env: str) -> str:
        """按环境名取 ISS 服务地址 (env: "test" | "prod")。"""
        if env == "prod":
            return self.iss_api_url_prod
        return self.iss_api_url_test

    # 服务端拉流消费 (StreamConsumer)
    stream_max_fps: float = 15.0  # 处理帧率上限 (拉到的多余帧直接丢弃)
    # 识别路径分辨率: 0 = 不缩放, 按视频流原生分辨率处理 (无损)。
    # 仅当算力不足时才设为正数 (如 1280) 用等比缩小换速度。
    stream_proc_max_width: int = 0
    # ---- 前端预览帧 (下面两项只影响网页观看的带宽/清晰度) ----
    # 识别用的是解码原帧, 与预览参数无关; 识别结果坐标随 frame_w/frame_h 下发,
    # 预览图缩放不会导致框偏移。
    stream_preview_max_width: int = 1280  # 预览帧最大宽度, 0 = 原生分辨率
    stream_preview_jpeg_quality: int = 80  # 预览帧 JPEG 质量 (1-100)
    stream_reconnect_delay: float = 2.0  # 拉流断开后的重连间隔 (秒)

    # WebSocket
    ws_max_frame_size: int = 1024 * 1024  # 1MB 最大帧大小


class Config(BaseModel):
    """
    系统总配置
    
    所有模块的配置参数集中管理。
    阈值参数支持通过 WebSocket 实时更新。
    """
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    multiframe: MultiFrameConfig = Field(default_factory=MultiFrameConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    voice_embed: VoiceEmbedExtractorConfig = Field(default_factory=VoiceEmbedExtractorConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def to_dict(self) -> dict:
        """序列化为字典 (用于 API 返回)"""
        return self.model_dump()

    def update_from_dict(self, updates: dict[str, float | bool]) -> list[str]:
        """
        从扁平化的 key-value 字典更新配置。
        
        支持格式: {"REID_CONFIDENT_THRESHOLD": 0.72, "YOLO_CONFIDENCE": 0.5}
        返回成功更新的 key 列表。
        """
        updated_keys: list[str] = []
        # 映射: 扁平化 KEY → (子配置对象, 属性名)
        flat_map = self._build_flat_map()

        for key, value in updates.items():
            key_upper = key.upper()
            if key_upper in flat_map:
                sub_config, attr_name = flat_map[key_upper]
                # 自动类型转换: 如果目标字段是 bool 类型, 确保设置为 bool
                current = getattr(sub_config, attr_name, None)
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    # 像素类整型字段 (min_face_size 等): 滑块会传 float, 落库前取整
                    value = int(round(float(value)))
                setattr(sub_config, attr_name, value)
                updated_keys.append(key_upper)

        return updated_keys

    # 可调参数定义 (单一来源)
    # key → (config_section_name, attr_name, min, max, step, group, label)
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
    }

    def get_tunable_params(self) -> dict:
        """获取可调参数当前值及元数据 (供前端滑块渲染)。"""
        result = {}
        for key, (section, attr, mn, mx, step, group, label) in self._TUNABLE_DEFS.items():
            cfg_section = getattr(self, section)
            result[key] = {
                "value": getattr(cfg_section, attr),
                "min": mn, "max": mx, "step": step,
                "group": group, "label": label,
            }
        return result

    def _build_flat_map(self) -> dict[str, tuple[BaseModel, str]]:
        """构建扁平化键名 → (子配置, 属性名) 映射 (供 update_from_dict 写入)。"""
        mapping: dict[str, tuple[BaseModel, str]] = {}
        for key, (section, attr, *_) in self._TUNABLE_DEFS.items():
            mapping[key] = (getattr(self, section), attr)
        # 仅通过顶部按钮控制, 不在 Controls 面板显示
        mapping["IMAGE_CORRECTION_ENABLED"] = (self.server, "image_correction_enabled")
        return mapping


def load_config() -> Config:
    """
    加载配置并设为全局单例。
    优先级: 环境变量 > .env 文件 > 默认值
    """
    # 从项目根目录加载 .env 文件
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    config = Config()

    # 从环境变量覆盖关键配置
    if api_key := os.environ.get("VLM_API_KEY"):
        config.vlm.api_key = api_key
    if base_url := os.environ.get("VLM_BASE_URL"):
        config.vlm.base_url = base_url
    if device := os.environ.get("CUDA_DEVICE"):
        config.hardware.device = device
    if db_path := os.environ.get("GALLERY_DB_PATH"):
        config.server.gallery_db_path = db_path

    # 服务器配置
    if host := os.environ.get("SERVER_HOST"):
        config.server.host = host
    if port := os.environ.get("SERVER_PORT"):
        config.server.port = int(port)

    _set_instance(config)
    return config


# ==============================================================================
# 全局单例
# ==============================================================================

_instance: Config | None = None


def _set_instance(config: Config) -> None:
    """设置全局 Config 单例（仅供 load_config 调用）。"""
    global _instance
    _instance = config


def get_config() -> Config:
    """获取全局 Config 单例。

    必须先调用 ``load_config()`` 初始化。

    Returns:
        全局 Config 实例。

    Raises:
        RuntimeError: 如果 ``load_config()`` 未被调用。
    """
    if _instance is None:
        raise RuntimeError("Config not initialized. Call load_config() first.")
    return _instance
