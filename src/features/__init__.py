"""
特征提取模块

提供人脸、全身和体型比例的特征提取与评估。
"""
from src.features.body_extractor import BodyExtractor
from src.features.body_proportions import extract_proportions, proportions_similarity
from src.features.face_extractor import FaceExtractor, FaceResult
from src.features.quality_assessor import QualityAssessor


def create_face_extractor(config):
    """创建人脸特征提取器。"""
    return FaceExtractor(config.face)


def create_body_extractor(config):
    """创建全身 ReID 特征提取器。"""
    return BodyExtractor(config.reid)


__all__ = [
    "FaceExtractor",
    "FaceResult",
    "BodyExtractor",
    "QualityAssessor",
    "extract_proportions",
    "proportions_similarity",
    "create_face_extractor",
    "create_body_extractor",
]
