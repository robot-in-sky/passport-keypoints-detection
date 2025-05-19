import math
from collections.abc import Sequence

import cv2
import numpy as np


def validate_pages(  # noqa: ANN201, PLR0913
        points: list[list[float]],
        image_size: tuple[int, int], *,
        target_aspect_ratio: float = 4.0 / 3.0,
        aspect_tolerance: float = 0.15,
        min_area_ratio: float = 0.2,
        max_page_tilt_degrees: float = 45.0):
    """Проверяет валидность координат углов разворота паспорта с учётом перспективного наклона.

    Параметры:
    - points: список из 4 точек в порядке [top-left, top-right, bottom-right, bottom-left]
    - image_size: (width, height) — размер изображения
    - target_aspect_ratio: ожидаемое соотношение сторон (по умолчанию 4:3)
    - aspect_tolerance: допустимое отклонение aspect ratio (0.15 = ±15%)
    - min_area_ratio: минимальная доля площади документа от всего изображения
    - max_tilt_degrees: максимально допустимый угол отклонения от плоскости (по осям X и Y)
    - focal_length: фокусное расстояние камеры в пикселях (по умолчанию для смартфона)

    Возвращает:
    - True, если страница валидна
    - False, если нет
    """
    if len(points) != 6:  # noqa: PLR2004
        return False

    points_top = (points[0], points[1], points[3], points[2])
    points_bottom = (points[2], points[3], points[5], points[4])

    page_min_area_ratio = min_area_ratio / 2

    val_top = validate_page(points_top, image_size,
                            target_aspect_ratio=target_aspect_ratio,
                            aspect_tolerance=aspect_tolerance,
                            min_area_ratio=page_min_area_ratio,
                            max_tilt_degrees=max_page_tilt_degrees)

    val_bottom = validate_page(points_bottom, image_size,
                            target_aspect_ratio=target_aspect_ratio,
                            aspect_tolerance=aspect_tolerance,
                            min_area_ratio=page_min_area_ratio,
                            max_tilt_degrees=max_page_tilt_degrees)

    return val_top and val_bottom


def validate_page(  # noqa: PLR0913
    points: Sequence[list[float]],
    image_size: tuple[int, int], *,
    target_aspect_ratio: float = 4.0 / 3.0,
    aspect_tolerance: float = 0.15,
    min_area_ratio: float = 0.2,
    max_tilt_degrees: float = 45.0,
    focal_length: float = 1500.0
) -> bool:
    """Проверяет валидность координат углов прямоугольной страницы с учётом перспективного наклона.

    Параметры:
    - points: список из 4 точек в порядке [top-left, top-right, bottom-right, bottom-left]
    - image_size: (width, height) — размер изображения
    - target_aspect_ratio: ожидаемое соотношение сторон (по умолчанию 4:3)
    - aspect_tolerance: допустимое отклонение aspect ratio (0.15 = ±15%)
    - min_area_ratio: минимальная доля площади документа от всего изображения
    - max_tilt_degrees: максимально допустимый угол отклонения от плоскости (по осям X и Y)
    - focal_length: фокусное расстояние камеры в пикселях (по умолчанию для смартфона)

    Возвращает:
    - True, если страница валидна
    - False, если нет
    """
    if len(points) != 4:
        return False

    pts = np.array(points, dtype=np.float32)

    hull = cv2.convexHull(pts)
    if len(hull) != 4:
        return False

    doc_area = cv2.contourArea(pts)
    img_area = image_size[0] * image_size[1]
    if img_area == 0 or doc_area / img_area < min_area_ratio:
        return False

    scale = 100.0  # Масштаб для численной стабильности
    target_width = target_aspect_ratio * scale
    target_height = scale

    dst_pts = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)

    try:
        H, _ = cv2.findHomography(dst_pts, pts)  # mapping from real-world plane to image
    except cv2.error:
        return False

    # Матрица камеры
    w, h = image_size
    cx, cy = w / 2, h / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    tilt_x, tilt_y = _estimate_tilt_from_homography(H, K)
    if abs(tilt_x) > max_tilt_degrees or abs(tilt_y) > max_tilt_degrees:
        return False

    # Проверка соотношения сторон
    try:
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        transformed = cv2.perspectiveTransform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
    except cv2.error:
        return False


    def distance(p1, p2):  # noqa: ANN202
        return np.linalg.norm(p1 - p2)

    top = distance(transformed[0], transformed[1])
    bottom = distance(transformed[3], transformed[2])
    left = distance(transformed[0], transformed[3])
    right = distance(transformed[1], transformed[2])

    aspect_ratio = (top + bottom) / (left + right)
    lower_bound = target_aspect_ratio * (1 - aspect_tolerance)
    upper_bound = target_aspect_ratio * (1 + aspect_tolerance)
    return lower_bound <= aspect_ratio <= upper_bound


def _estimate_tilt_from_homography(H: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """Оценивает наклон плоскости документа по гомографии и матрице камеры."""
    K_inv = np.linalg.inv(K)
    Rt = K_inv @ H

    r1 = Rt[:, 0]
    r2 = Rt[:, 1]

    # Усреднённая норма
    norm = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    r1 /= norm
    r2 /= norm
    r3 = np.cross(r1, r2)

    R = np.stack([r1, r2, r3], axis=1)

    # Извлечение наклонов
    tilt_x = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    tilt_y = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)))

    return tilt_x, tilt_y
