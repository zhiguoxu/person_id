"""High-quality lens distortion correction with parameter scaling."""

from pydantic import BaseModel, ConfigDict

import cv2
import numpy as np


image_size = (2560, 1440)
camera_matrix = [[1215.26939, 0.0, 1295.73231],
                 [0.0, 1215.73825, 720.93368],
                 [0.0, 0.0, 1.0]]
dist_coeffs = [-0.311263478, 0.102889558, -0.000213350609, 0.0000179681805, -0.0151980338]


class CameraParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]


_camera_params = CameraParams(
    camera_matrix=np.array(camera_matrix, dtype=np.float64),
    dist_coeffs=np.array(dist_coeffs, dtype=np.float64),
    image_size=image_size
)


def _scale_camera_matrix(
        camera_matrix: np.ndarray,
        calib_size: tuple[int, int],
        actual_size: tuple[int, int],
) -> np.ndarray:
    sx = actual_size[0] / calib_size[0]
    sy = actual_size[1] / calib_size[1]
    s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    return s @ camera_matrix


def _undistort_image(
        img: np.ndarray,
        camera_params: CameraParams,
        alpha: float = 0,
) -> np.ndarray:
    h, w = img.shape[:2]
    actual_size = (w, h)
    camera_matrix = _scale_camera_matrix(
        camera_params.camera_matrix, camera_params.image_size, actual_size,
    )
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, camera_params.dist_coeffs, actual_size, alpha=alpha,
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, camera_params.dist_coeffs,
        None, new_camera_matrix, actual_size, cv2.CV_16SC2,
    )
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    return undistorted[y: y + rh, x: x + rw]


def correct_frame(
        img: np.ndarray,
        camera_params: CameraParams = _camera_params,
        alpha: float = 0,
) -> np.ndarray:
    """对已解码的 BGR 帧做镜头畸变矫正 (供服务端拉流路径使用, 免去 JPEG 编解码往返)。"""
    return _undistort_image(img, camera_params, alpha=alpha)


def correct_image_bytes(
        image_bytes: bytes,
        camera_params: CameraParams = _camera_params,
        alpha: float = 0,
) -> bytes:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    undistorted = _undistort_image(img, camera_params, alpha=alpha)
    success, encoded = cv2.imencode(
        ".jpg", undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise RuntimeError("Failed to encode corrected image")
    return encoded.tobytes()
