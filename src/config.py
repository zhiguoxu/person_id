"""
机器人视觉人物识别系统 - 全局配置

所有可调参数集中管理。支持通过 WebSocket 实时更新阈值参数。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ==============================================================================
# 项目路径
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@dataclass
class DetectionConfig:
    """检测模块配置"""
    # YOLO 模型
    yolo_fast_model: str = "yolo11n-pose.pt"       # Tier 1 轻量模型
    yolo_heavy_model: str = "yolo11x-pose.pt"      # Tier 2 精确模型
    yolo_device: str = "cuda:0"                     # CUDA 设备
    yolo_confidence: float = 0.5                    # 检测置信度阈值
    yolo_iou_threshold: float = 0.7                 # NMS IoU 阈值
    yolo_max_det: int = 20                          # 最大检测数

    # 关键点
    keypoint_confidence: float = 0.3                # 关键点最低置信度
    min_person_height_px: int = 80                  # 最小人体像素高度


@dataclass
class FaceConfig:
    """人脸识别配置"""
    insightface_model: str = "buffalo_l"            # InsightFace 模型包
    insightface_ctx_id: int = 0                     # CUDA 设备 ID
    det_size: tuple[int, int] = (640, 640)          # 人脸检测输入尺寸

    # 人脸质量评估权重
    quality_blur_weight: float = 0.25               # 模糊度权重
    quality_size_weight: float = 0.15               # 尺寸权重
    quality_landmark_weight: float = 0.15           # 关键点置信度权重
    quality_pose_weight: float = 0.30               # 姿态角权重
    quality_lighting_weight: float = 0.15           # 光照权重
    min_face_size: int = 40                         # 最小人脸像素尺寸


@dataclass
class ReIDConfig:
    """人员重识别配置"""
    # SOLIDER 模型 (暂用 OSNet 占位, SOLIDER 需从源码集成)
    reid_model_name: str = "osnet_ain_x1_0"            # ReID 模型名
    reid_model_weights: str = ""                       # 模型权重路径 (空=自动从缓存查找)
    reid_device: str = "cuda:0"                     # CUDA 设备
    reid_input_size: tuple[int, int] = (256, 128)   # 输入尺寸 (H, W)
    use_flip_test: bool = True                      # 水平翻转测试增强


@dataclass
class GalleryConfig:
    """特征底库配置"""
    # 人脸库
    max_faces_per_bucket: int = 5                   # 每个姿态桶最多特征数
    face_half_life_days: float = 365.0              # 人脸特征衰减半衰期

    # 衣橱库
    max_outfits: int = 20                           # 最大衣橱记录数
    outfit_half_life_days: float = 30.0             # 衣橱衰减半衰期
    outfit_match_threshold: float = 0.85            # 衣橱匹配阈值 (同一套衣服)

    # 入库质量门槛
    quality_enroll_threshold: float = 0.4           # 入库最低质量分
    quality_update_threshold: float = 0.7           # 更新特征最低质量分


@dataclass
class MatchingConfig:
    """匹配与融合配置"""
    # ReID 四重阈值
    reid_confident_threshold: float = 0.72          # X_reid: 确信阈值
    reid_suspected_threshold: float = 0.55          # Y_reid: 疑似阈值

    # VLM 四重阈值
    vlm_confident_threshold: float = 0.80           # X_vlm: 确信阈值
    vlm_suspected_threshold: float = 0.60           # Y_vlm: 疑似阈值

    # 人脸捷径
    face_shortcut_threshold: float = 0.75           # 人脸质量+匹配度均高时直接确认
    face_shortcut_quality: float = 0.7              # 触发捷径的最低人脸质量

    # 多模态融合基础权重
    face_base_weight: float = 0.50                  # 人脸权重
    body_base_weight: float = 0.35                  # 全身权重
    proportion_base_weight: float = 0.15            # 体型比例权重

    # 相似度差距阈值 (第一名和第二名的差距)
    confident_margin: float = 0.10                  # 确信需要的最小差距


@dataclass
class TrackingConfig:
    """追踪引擎配置"""
    # BoT-SORT 参数
    track_high_thresh: float = 0.5                  # 高置信度检测阈值
    track_low_thresh: float = 0.1                   # 低置信度检测阈值
    new_track_thresh: float = 0.6                   # 新轨迹创建阈值
    track_buffer: int = 30                          # 轨迹缓冲帧数
    match_thresh: float = 0.8                       # 匹配阈值
    cmc_method: str = "sof"                         # 相机运动补偿方法 (ecc/orb/sof/sift)

    # 时空约束
    spatial_timeout_sec: float = 10.0               # 空间记忆有效期
    spatial_distance_px: float = 200.0              # 最大关联距离

    # Tier 2 触发
    tier2_refresh_interval_sec: float = 30.0        # 定期刷新间隔


@dataclass
class TemporalConfig:
    """时序聚合配置"""
    window_size: int = 5                            # 滑动窗口大小 (帧数)


@dataclass
class VLMConfig:
    """VLM 仲裁配置"""
    model: str = "qwen-vl-max"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_sec: float = 30.0
    max_retries: int = 2


@dataclass
class ServerConfig:
    """服务配置"""
    host: str = "0.0.0.0"
    port: int = 10003                               # 远程 CUDA 服务器端口
    log_level: str = "INFO"
    gallery_db_path: str = str(DATA_DIR / "gallery.db")

    # WebSocket
    ws_max_frame_size: int = 1024 * 1024            # 1MB 最大帧大小
    ws_send_timeout: float = 5.0                    # 发送超时


@dataclass
class Config:
    """
    系统总配置
    
    所有模块的配置参数集中管理。
    阈值参数支持通过 WebSocket 实时更新。
    """
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def to_dict(self) -> dict:
        """序列化为字典 (用于 API 返回)"""
        return asdict(self)

    def update_from_dict(self, updates: dict) -> list[str]:
        """
        从扁平化的 key-value 字典更新配置。
        
        支持格式: {"REID_CONFIDENT_THRESHOLD": 0.72, "YOLO_CONFIDENCE": 0.5}
        返回成功更新的 key 列表。
        """
        updated_keys = []
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
                "min": 0.1, "max": 0.9, "step": 0.05,
                "group": "detection", "label": "Detection Confidence",
            },
            "REID_CONFIDENT_THRESHOLD": {
                "value": self.matching.reid_confident_threshold,
                "min": 0.50, "max": 0.95, "step": 0.01,
                "group": "reid", "label": "ReID Confident (X)",
            },
            "REID_SUSPECTED_THRESHOLD": {
                "value": self.matching.reid_suspected_threshold,
                "min": 0.30, "max": 0.80, "step": 0.01,
                "group": "reid", "label": "ReID Suspected (Y)",
            },
            "VLM_CONFIDENT_THRESHOLD": {
                "value": self.matching.vlm_confident_threshold,
                "min": 0.50, "max": 0.95, "step": 0.01,
                "group": "vlm", "label": "VLM Confident (X)",
            },
            "VLM_SUSPECTED_THRESHOLD": {
                "value": self.matching.vlm_suspected_threshold,
                "min": 0.30, "max": 0.80, "step": 0.01,
                "group": "vlm", "label": "VLM Suspected (Y)",
            },
            "QUALITY_ENROLL_THRESHOLD": {
                "value": self.gallery.quality_enroll_threshold,
                "min": 0.10, "max": 0.90, "step": 0.05,
                "group": "quality", "label": "Face Quality Min",
            },
            "FACE_SHORTCUT_THRESHOLD": {
                "value": self.matching.face_shortcut_threshold,
                "min": 0.50, "max": 0.95, "step": 0.01,
                "group": "matching", "label": "Face Shortcut",
            },
            "OUTFIT_MATCH_THRESHOLD": {
                "value": self.gallery.outfit_match_threshold,
                "min": 0.50, "max": 0.95, "step": 0.01,
                "group": "matching", "label": "Outfit Match",
            },
        }

    def _build_flat_map(self) -> dict:
        """构建扁平化键名 → (子配置, 属性名) 映射"""
        mapping = {}
        key_to_config = {
            "YOLO_CONFIDENCE": (self.detection, "yolo_confidence"),
            "YOLO_IOU_THRESHOLD": (self.detection, "yolo_iou_threshold"),
            "KEYPOINT_CONFIDENCE": (self.detection, "keypoint_confidence"),
            "REID_CONFIDENT_THRESHOLD": (self.matching, "reid_confident_threshold"),
            "REID_SUSPECTED_THRESHOLD": (self.matching, "reid_suspected_threshold"),
            "VLM_CONFIDENT_THRESHOLD": (self.matching, "vlm_confident_threshold"),
            "VLM_SUSPECTED_THRESHOLD": (self.matching, "vlm_suspected_threshold"),
            "FACE_SHORTCUT_THRESHOLD": (self.matching, "face_shortcut_threshold"),
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
    加载配置。
    优先级: 环境变量 > 默认值
    """
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

    return config
