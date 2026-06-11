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


class DetectionConfig(BaseModel):
    """检测模块配置"""
    # YOLO 模型
    yolo_fast_model: str = "yolo11n-pose.pt"  # Tier 1 轻量模型
    yolo_heavy_model: str = "yolo11x-pose.pt"  # Tier 2 精确模型
    yolo_device: str = "cuda:0"  # CUDA 设备
    yolo_confidence: float = 0.5  # 检测置信度阈值
    yolo_iou_threshold: float = 0.7  # NMS IoU 阈值
    yolo_max_det: int = 20  # 最大检测数

    # 关键点
    keypoint_confidence: float = 0.3  # 关键点最低置信度
    min_person_height_px: int = 80  # 最小人体像素高度


class FaceConfig(BaseModel):
    """人脸识别配置"""
    insightface_model: str = "buffalo_l"  # InsightFace 模型包
    insightface_ctx_id: int = 0  # CUDA 设备 ID
    det_size: tuple[int, int] = (640, 640)  # 人脸检测输入尺寸

    min_face_size: int = 40  # 最小人脸像素尺寸


class ReIDConfig(BaseModel):
    """人员重识别配置"""
    # SOLIDER 模型 (暂用 OSNet 占位, SOLIDER 需从源码集成)
    reid_model_name: str = "osnet_ain_x1_0"  # ReID 模型名
    reid_model_weights: str = ""  # 模型权重路径 (空=自动从缓存查找)
    reid_device: str = "cuda:0"  # CUDA 设备
    reid_input_size: tuple[int, int] = (256, 128)  # 输入尺寸 (H, W)
    use_flip_test: bool = False  # 水平翻转测试增强


class GalleryConfig(BaseModel):
    """特征底库配置"""
    # 人脸库
    max_faces_per_bucket: int = 5  # 每个姿态桶最多特征数
    face_match_half_life_days: float = 365.0  # 匹配端质心权重半衰期

    # 衣橱库
    max_outfits: int = 20  # 最大衣橱记录数
    outfit_half_life_days: float = 30.0  # 衣橱衰减半衰期
    outfit_match_threshold: float = 0.85  # 衣橱匹配阈值 (同一套衣服)

    # 入库质量门槛
    quality_enroll_threshold: float = 0.4  # 入库最低质量分

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
    face_base_weight: float = 0.60  # 人脸基础权重 (multi_modal_fusion.fuse 使用)
    body_base_weight: float = 0.3  # 全身基础权重
    proportion_base_weight: float = 0.1  # 体型比例基础权重


class TrackingConfig(BaseModel):
    """追踪引擎配置"""
    # BoT-SORT 参数
    track_high_thresh: float = 0.5  # 高置信度检测阈值
    track_low_thresh: float = 0.1  # 低置信度检测阈值
    new_track_thresh: float = 0.6  # 新轨迹创建阈值
    track_buffer: int = 30  # 轨迹缓冲帧数
    match_thresh: float = 0.8  # 匹配阈值
    cmc_method: str = "sof"  # 相机运动补偿方法 (ecc/orb/sof/sift)

    # 时空约束
    spatial_timeout_sec: float = 10.0  # 空间记忆有效期
    spatial_distance_px: float = 200.0  # 最大关联距离

    # Tier 2 触发
    tier2_refresh_interval_sec: float = 30.0  # 定期刷新间隔


class TemporalConfig(BaseModel):
    """时序聚合配置"""
    window_size: int = 5  # 滑动窗口大小 (帧数)


class MultiFrameConfig(BaseModel):
    """多帧处理配置"""
    # Tier1 帧收集
    recent_min_interval: float = 0.25  # RecentBuffer 帧间最小间隔 (时间多样性)

    # Tier2 质量缓存
    face_pool_size: int = 10  # 人脸质量缓存容量
    body_pool_size: int = 10  # 人体质量缓存容量

    # 聚合质量阈值
    agg_min_face_quality: float = 0.3  # 人脸聚合最低质量
    agg_min_body_quality: float = 0.2  # 人体聚合最低质量

    # Tier2 (ReID) 调度 (注意力目标基准间隔)
    tier2_fast_interval: float = 1.0  # IDENTIFYING/SUSPECTED/CONFLICT 间隔
    tier2_slow_interval: float = 5.0  # CONFIDENT/STRANGER 间隔

    # Tier3 (VLM) 调度
    vlm_cooldown: float = 5.0  # VLM 冷却周期 (注意力目标)

    # DEFINITE 后台富化
    definite_enrich_interval: float = 20.0  # 富化周期

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


class ServerConfig(BaseModel):
    """服务配置"""
    host: str = "0.0.0.0"
    port: int = 10003  # 远程 CUDA 服务器端口
    log_level: str = "INFO"
    gallery_db_path: str = str(DATA_DIR / "gallery.db")

    # WebSocket
    ws_max_frame_size: int = 1024 * 1024  # 1MB 最大帧大小
    ws_send_timeout: float = 5.0  # 发送超时


class Config(BaseModel):
    """
    系统总配置
    
    所有模块的配置参数集中管理。
    阈值参数支持通过 WebSocket 实时更新。
    """
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    multiframe: MultiFrameConfig = Field(default_factory=MultiFrameConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def to_dict(self) -> dict:
        """序列化为字典 (用于 API 返回)"""
        return self.model_dump()

    def update_from_dict(self, updates: dict[str, float]) -> list[str]:
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
                setattr(sub_config, attr_name, value)
                updated_keys.append(key_upper)

        return updated_keys

    def get_tunable_params(self) -> dict:
        """
        获取可通过前端滑块调整的参数。
        返回: {参数名: {value, min, max, step, group, label}}
        """
        return {
            "YOLO_CONFIDENCE": {
                "value": self.detection.yolo_confidence,
                "min": 0, "max": 1, "step": 0.05,
                "group": "detection", "label": "Detection Confidence",
            },
            "A_THRESHOLD": {
                "value": self.matching.A_threshold,
                "min": 0, "max": 1, "step": 0.01,
                "group": "reid", "label": "A Threshold (笃定)",
            },
            "B_THRESHOLD": {
                "value": self.matching.B_threshold,
                "min": 0, "max": 1, "step": 0.01,
                "group": "reid", "label": "B Threshold (确定)",
            },
            "C_THRESHOLD": {
                "value": self.matching.C_threshold,
                "min": 0, "max": 1, "step": 0.01,
                "group": "reid", "label": "C Threshold (怀疑)",
            },
            "QUALITY_ENROLL_THRESHOLD": {
                "value": self.gallery.quality_enroll_threshold,
                "min": 0, "max": 1, "step": 0.05,
                "group": "quality", "label": "入库质量门槛",
            },
            "OUTFIT_MATCH_THRESHOLD": {
                "value": self.gallery.outfit_match_threshold,
                "min": 0, "max": 1, "step": 0.01,
                "group": "matching", "label": "衣橱匹配阈值",
            },
        }

    def _build_flat_map(self) -> dict[str, tuple[BaseModel, str]]:
        """构建扁平化键名 → (子配置, 属性名) 映射"""
        mapping: dict[str, tuple[BaseModel, str]] = {}
        key_to_config = {
            "YOLO_CONFIDENCE": (self.detection, "yolo_confidence"),
            "YOLO_IOU_THRESHOLD": (self.detection, "yolo_iou_threshold"),
            "KEYPOINT_CONFIDENCE": (self.detection, "keypoint_confidence"),
            "A_THRESHOLD": (self.matching, "A_threshold"),
            "B_THRESHOLD": (self.matching, "B_threshold"),
            "C_THRESHOLD": (self.matching, "C_threshold"),
            "FACE_BASE_WEIGHT": (self.matching, "face_base_weight"),
            "BODY_BASE_WEIGHT": (self.matching, "body_base_weight"),
            "PROPORTION_BASE_WEIGHT": (self.matching, "proportion_base_weight"),
            "QUALITY_ENROLL_THRESHOLD": (self.gallery, "quality_enroll_threshold"),
            "OUTFIT_MATCH_THRESHOLD": (self.gallery, "outfit_match_threshold"),

            "SPATIAL_TIMEOUT_SEC": (self.tracking, "spatial_timeout_sec"),
            "SPATIAL_DISTANCE_PX": (self.tracking, "spatial_distance_px"),
            "TIER2_REFRESH_INTERVAL_SEC": (self.tracking, "tier2_refresh_interval_sec"),
        }
        mapping.update(key_to_config)
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
        config.detection.yolo_device = device
        config.face.insightface_ctx_id = int(device.split(":")[-1]) if ":" in device else 0
        config.reid.reid_device = device
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
