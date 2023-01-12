import cv2
import numpy as np


PARAMS = {
    "LIGHT": {
        "n_lines": 0,
        "n_points": 1,
        "n_curves": 2,
    },
    "MEDIUM": {
        "n_lines": 2,
        "n_points": 4,
        "n_curves": 8,
    },
    "HEAVY": {
        "n_lines": 3,
        "n_points": 5,
        "n_curves": 14,
    },
}


WHITE = 255


def add_random_line(mask: np.ndarray, thickness: int) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
    x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
    mask = cv2.line(mask, (x1, y1), (x2, y2), color=WHITE, thickness=thickness)
    return mask


def add_random_point(mask: np.ndarray, thickness: int) -> np.ndarray:
    h, w = mask.shape[:2]
    x, y = np.random.randint(0, w), np.random.randint(0, h)
    mask = cv2.circle(mask, (x, y), radius=thickness, color=WHITE, thickness=-1)
    return mask


def add_random_curve(mask: np.ndarray, thickness: int) -> np.ndarray:
    h, w = mask.shape[:2]
    center_x, center_y = np.random.randint(w // 3, 2 * w // 3), np.random.randint(
        h // 3, 2 * h // 3
    )
    axis_x, axis_y = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
    start_angle = np.random.randint(0, 270)
    end_angle = start_angle + np.random.randint(0, 90)
    mask = cv2.ellipse(
        mask,
        (center_x, center_y),
        (axis_x, axis_y),
        np.random.randint(0, 180),
        start_angle,
        end_angle,
        WHITE,
        thickness,
    )
    return mask


def generate_random_mask(
    h: int,
    w: int,
    n_lines: int,
    n_points: int,
    n_curves: int,
    min_width: int,
    max_width: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n_lines):
        mask = add_random_line(mask, np.random.randint(min_width, max_width))
    for _ in range(n_points):
        mask = add_random_point(mask, np.random.randint(min_width, max_width))
    for _ in range(n_curves):
        mask = add_random_curve(mask, np.random.randint(min_width, max_width))
    return mask


def generate_random_mask_by_degree(
    h: int, w: int, degree: str, min_width: int, max_width: int
) -> np.ndarray:
    params = PARAMS[degree]
    return generate_random_mask(
        h, w, **params, min_width=min_width, max_width=max_width
    )
