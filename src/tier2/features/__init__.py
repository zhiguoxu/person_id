"""
特征提取模块

提供人脸、全身和体型比例的特征提取与评估。
工厂函数使用 cache 保证 GPU 模型只加载一次。
"""

from __future__ import annotations

from functools import cache

from src.tier2.features.body_extractor import BodyExtractor
from src.tier2.features.face_extractor import FaceExtractor, FaceResult
from src.tier2.features.face_quality_assessor import FaceQualityAssessor


@cache
def get_face_extractor() -> FaceExtractor:
    """创建人脸特征提取器（单例缓存）。"""
    return FaceExtractor()


@cache
def get_body_extractor() -> BodyExtractor:
    """创建人体 ReID 特征提取器（单例缓存）。"""
    return BodyExtractor()


@cache
def get_quality_assessor() -> FaceQualityAssessor:
    """创建质量评估器（单例缓存）。"""
    return FaceQualityAssessor()

__all__ = [
    "FaceExtractor",
    "FaceResult",
    "BodyExtractor",
    "FaceQualityAssessor",
    "get_face_extractor",
    "get_body_extractor",
    "get_quality_assessor",
]
