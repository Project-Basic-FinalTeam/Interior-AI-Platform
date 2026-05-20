import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import pillow_avif  # AVIF 이미지 로딩용

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# =========================================================
# 0. 사용자 설정값
# =========================================================

INPUT_IMAGE_PATH = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Depth\iphone16pro_test.jpg")
OUTPUT_JSON_PATH = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Depth\iphone16pro_test.json")

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# iPhone 16 Pro 1x main camera 기준
DEVICE_NAME = "iPhone 16 Pro"
EQUIVALENT_FOCAL_LENGTH_MM = 24.0  # 35mm 환산 초점거리
FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_HEIGHT_MM = 24.0

# 실제 촬영 조건
ORIGINAL_IMAGE_WIDTH = 5712
ORIGINAL_IMAGE_HEIGHT = 4284
CAMERA_HEIGHT_M = 1.30
KNOWN_FRONT_WALL_WIDTH_M = 3.24

# 기하 계산은 원본 전체 해상도에서 하면 메모리를 많이 쓰므로 축소해서 처리
# JSON에는 원본 좌표로 다시 환산해서 저장함
GEOMETRY_MAX_WIDTH = 1600

# 디버그 이미지 저장
SAVE_DEBUG_IMAGES = True

# 재현성
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# 1. 이미지 로딩
# =========================================================

def load_image_rgb(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"입력 이미지가 존재하지 않습니다: {image_path}")

    return Image.open(image_path).convert("RGB")


# =========================================================
# 2. iPhone 16 Pro 24mm 기준 intrinsic 계산
# =========================================================

def calculate_intrinsics_from_35mm_equiv(width: int, height: int):
    """
    24mm 35mm 환산 초점거리를 이용해 pixel 단위 fx, fy, cx, cy를 근사 계산한다.
    """
    full_frame_diagonal_mm = math.sqrt(FULL_FRAME_WIDTH_MM ** 2 + FULL_FRAME_HEIGHT_MM ** 2)
    image_diagonal_px = math.sqrt(width ** 2 + height ** 2)

    focal_px = image_diagonal_px / full_frame_diagonal_mm * EQUIVALENT_FOCAL_LENGTH_MM

    fx = focal_px
    fy = focal_px
    cx = width / 2.0
    cy = height / 2.0

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "method": "35mm_equivalent_focal_length_based",
        "equivalent_focal_length_mm": EQUIVALENT_FOCAL_LENGTH_MM,
    }


def scale_intrinsics(intrinsics: dict, scale_x: float, scale_y: float):
    return {
        "fx": float(intrinsics["fx"] * scale_x),
        "fy": float(intrinsics["fy"] * scale_y),
        "cx": float(intrinsics["cx"] * scale_x),
        "cy": float(intrinsics["cy"] * scale_y),
        "method": intrinsics["method"],
        "equivalent_focal_length_mm": intrinsics["equivalent_focal_length_mm"],
    }


# =========================================================
# 3. Depth Anything V2 Metric Indoor 추론
# =========================================================

def predict_metric_depth(image: Image.Image, model_id: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] torch device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    width, height = image.size

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np[depth_np < 0] = 0

    return depth_np


# =========================================================
# 4. 기하 계산용 리사이즈
# =========================================================

def resize_for_geometry(image: Image.Image, depth_m: np.ndarray, max_width: int):
    original_width, original_height = image.size

    if original_width <= max_width:
        scale = 1.0
        geom_width = original_width
        geom_height = original_height
    else:
        scale = max_width / original_width
        geom_width = int(round(original_width * scale))
        geom_height = int(round(original_height * scale))

    image_np = np.array(image)
    image_geom = cv2.resize(image_np, (geom_width, geom_height), interpolation=cv2.INTER_AREA)
    depth_geom = cv2.resize(depth_m, (geom_width, geom_height), interpolation=cv2.INTER_CUBIC)

    scale_x = geom_width / original_width
    scale_y = geom_height / original_height

    return image_geom, depth_geom.astype(np.float32), {
        "original_width": original_width,
        "original_height": original_height,
        "geometry_width": geom_width,
        "geometry_height": geom_height,
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
    }


# =========================================================
# 5. depth map -> 3D point cloud
# =========================================================

def depth_to_point_cloud(depth_m: np.ndarray, intrinsics: dict) -> np.ndarray:
    h, w = depth_m.shape

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_3d = np.stack([x, y, z], axis=-1).astype(np.float32)
    return points_3d


# =========================================================
# 6. RANSAC 평면 추정 공통 함수
# =========================================================

def fit_plane_from_3_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return None

    normal = normal / norm
    d = -np.dot(normal, p1)

    return normal.astype(np.float32), float(d)


def point_plane_distance(points: np.ndarray, normal: np.ndarray, d: float):
    return np.abs(points @ normal + d)


def plane_distance_from_camera_origin(normal: np.ndarray, d: float):
    return abs(d) / (np.linalg.norm(normal) + 1e-8)


# =========================================================
# 7. 바닥 plane 추정
# =========================================================

def estimate_floor_plane(points_3d: np.ndarray, depth_m: np.ndarray):
    h, w = depth_m.shape

    valid_mask = depth_m > 0.05

    # 바닥은 이미지 하단에 있을 가능성이 높음
    bottom_mask = np.zeros_like(valid_mask, dtype=bool)
    bottom_mask[int(h * 0.42):, :] = True

    candidate_mask = valid_mask & bottom_mask
    candidate_points = points_3d[candidate_mask]

    if len(candidate_points) < 1000:
        raise RuntimeError("바닥 후보 점이 너무 적습니다. 이미지 하단에 바닥이 충분히 보이는지 확인하세요.")

    max_samples = 15000
    if len(candidate_points) > max_samples:
        idx = np.random.choice(len(candidate_points), max_samples, replace=False)
        sampled = candidate_points[idx]
    else:
        sampled = candidate_points

    valid_depth_values = depth_m[valid_mask]
    median_depth = float(np.median(valid_depth_values))
    threshold_m = max(0.04, median_depth * 0.015)

    best = None
    best_score = -1

    iterations = 800

    for _ in range(iterations):
        ids = np.random.choice(len(sampled), 3, replace=False)
        plane = fit_plane_from_3_points(sampled[ids[0]], sampled[ids[1]], sampled[ids[2]])

        if plane is None:
            continue

        normal, d = plane

        # 카메라 좌표계에서 이미지 y 방향이 아래쪽이므로,
        # 수평 촬영 시 바닥 normal은 대체로 y축 성분이 큼
        y_alignment = abs(float(normal[1]))

        if y_alignment < 0.45:
            continue

        distances = point_plane_distance(sampled, normal, d)
        inlier_count = int(np.sum(distances < threshold_m))

        # y축 정렬이 좋은 plane에 가산점
        score = inlier_count * (0.7 + 0.3 * y_alignment)

        if score > best_score:
            best_score = score
            best = {
                "normal": normal,
                "d": float(d),
                "threshold_m": float(threshold_m),
                "inlier_count_sampled": inlier_count,
                "y_alignment": y_alignment,
            }

    if best is None:
        raise RuntimeError("바닥 plane 추정에 실패했습니다.")

    all_points = points_3d.reshape(-1, 3)
    dist = point_plane_distance(all_points, best["normal"], best["d"]).reshape(h, w)

    floor_mask = (dist < best["threshold_m"]) & valid_mask

    # 이미지 너무 위쪽의 오검출 제거
    floor_mask[:int(h * 0.25), :] = False

    floor_mask_uint8 = (floor_mask.astype(np.uint8) * 255)
    kernel = np.ones((7, 7), np.uint8)
    floor_mask_uint8 = cv2.morphologyEx(floor_mask_uint8, cv2.MORPH_OPEN, kernel)
    floor_mask_uint8 = cv2.morphologyEx(floor_mask_uint8, cv2.MORPH_CLOSE, kernel)
    floor_mask_uint8 = cv2.medianBlur(floor_mask_uint8, 5)

    floor_mask = floor_mask_uint8 > 0

    estimated_camera_height_raw = plane_distance_from_camera_origin(best["normal"], best["d"])

    best["estimated_camera_height_m_raw"] = float(estimated_camera_height_raw)
    best["known_camera_height_m"] = float(CAMERA_HEIGHT_M)

    if estimated_camera_height_raw > 1e-6:
        best["scale_factor_from_camera_height"] = float(CAMERA_HEIGHT_M / estimated_camera_height_raw)
    else:
        best["scale_factor_from_camera_height"] = None

    best["floor_mask"] = floor_mask

    return best


# =========================================================
# 8. 바닥-벽 경계 후보 추출
# =========================================================

def extract_floor_wall_boundary(floor_mask: np.ndarray):
    h, w = floor_mask.shape

    boundary_y_by_x = np.full(w, -1, dtype=np.float32)
    xs = []
    ys = []

    for x in range(w):
        y_candidates = np.where(floor_mask[:, x])[0]

        if len(y_candidates) == 0:
            continue

        y_top = int(np.min(y_candidates))

        if y_top < int(h * 0.20):
            continue

        boundary_y_by_x[x] = y_top
        xs.append(x)
        ys.append(y_top)

    if len(xs) < w * 0.10:
        return None, [], None

    xs = np.array(xs, dtype=np.int32)
    ys = np.array(ys, dtype=np.float32)

    full_xs = np.arange(xs.min(), xs.max() + 1)
    interp_ys = np.interp(full_xs, xs, ys)

    # smoothing
    window = max(11, w // 70)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / window
    smooth_ys = np.convolve(interp_ys, kernel, mode="same")

    boundary_y_by_x[:] = -1
    boundary_y_by_x[full_xs] = smooth_ys

    boundary_mask = np.zeros((h, w), dtype=np.uint8)

    for x, y in zip(full_xs, smooth_ys):
        y_int = int(np.clip(round(y), 0, h - 1))
        boundary_mask[y_int, x] = 255

    boundary_mask = cv2.dilate(boundary_mask, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        boundary_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=max(60, w // 7),
        maxLineGap=max(20, w // 25),
    )

    line_segments = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0].tolist()
            pixel_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            line_segments.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "pixel_length_geometry": float(pixel_length),
            })

    if len(line_segments) == 0:
        x1 = int(full_xs[0])
        y1 = int(smooth_ys[0])
        x2 = int(full_xs[-1])
        y2 = int(smooth_ys[-1])
        pixel_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        line_segments.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "pixel_length_geometry": float(pixel_length),
        })

    return boundary_y_by_x, line_segments, boundary_mask


# =========================================================
# 9. 정면 벽 plane 탐지
# =========================================================

def estimate_front_wall_plane(points_3d: np.ndarray, depth_m: np.ndarray, boundary_y_by_x):
    h, w = depth_m.shape

    valid_mask = depth_m > 0.05

    if boundary_y_by_x is None:
        # boundary가 없으면 상단~중단부를 벽 후보로 사용
        wall_candidate_mask = valid_mask.copy()
        wall_candidate_mask[int(h * 0.75):, :] = False
        wall_candidate_mask[:int(h * 0.05), :] = False
    else:
        wall_candidate_mask = np.zeros((h, w), dtype=bool)

        for x in range(w):
            by = boundary_y_by_x[x]
            if by < 0:
                continue

            y_end = max(0, int(by) - 8)
            wall_candidate_mask[int(h * 0.05):y_end, x] = True

        wall_candidate_mask &= valid_mask

    candidate_points = points_3d[wall_candidate_mask]

    if len(candidate_points) < 1000:
        return {
            "detected": False,
            "reason": "정면 벽 후보 점이 너무 적습니다.",
            "wall_mask": np.zeros((h, w), dtype=bool),
        }

    max_samples = 18000
    if len(candidate_points) > max_samples:
        idx = np.random.choice(len(candidate_points), max_samples, replace=False)
        sampled = candidate_points[idx]
    else:
        sampled = candidate_points

    valid_depth_values = depth_m[valid_mask]
    median_depth = float(np.median(valid_depth_values))
    threshold_m = max(0.05, median_depth * 0.018)

    best = None
    best_score = -1

    iterations = 1000

    for _ in range(iterations):
        ids = np.random.choice(len(sampled), 3, replace=False)
        plane = fit_plane_from_3_points(sampled[ids[0]], sampled[ids[1]], sampled[ids[2]])

        if plane is None:
            continue

        normal, d = plane

        # 정면 벽은 보통 카메라 z축 방향과 normal이 유사함.
        z_alignment = abs(float(normal[2]))
        y_alignment = abs(float(normal[1]))

        # 바닥/천장 plane 제거
        if z_alignment < 0.40:
            continue

        if y_alignment > 0.75:
            continue

        distances = point_plane_distance(sampled, normal, d)
        inlier_count = int(np.sum(distances < threshold_m))

        score = inlier_count * (0.7 + 0.3 * z_alignment)

        if score > best_score:
            best_score = score
            best = {
                "normal": normal,
                "d": float(d),
                "threshold_m": float(threshold_m),
                "inlier_count_sampled": inlier_count,
                "z_alignment": z_alignment,
                "y_alignment": y_alignment,
            }

    if best is None:
        return {
            "detected": False,
            "reason": "정면 벽 plane을 안정적으로 찾지 못했습니다.",
            "wall_mask": np.zeros((h, w), dtype=bool),
        }

    all_points = points_3d.reshape(-1, 3)
    dist = point_plane_distance(all_points, best["normal"], best["d"]).reshape(h, w)

    wall_mask = (dist < best["threshold_m"]) & wall_candidate_mask

    wall_mask_uint8 = wall_mask.astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    wall_mask_uint8 = cv2.morphologyEx(wall_mask_uint8, cv2.MORPH_OPEN, kernel)
    wall_mask_uint8 = cv2.morphologyEx(wall_mask_uint8, cv2.MORPH_CLOSE, kernel)

    # 가장 큰 연결 성분만 사용
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask_uint8, connectivity=8)

    if num_labels <= 1:
        return {
            "detected": False,
            "reason": "정면 벽 연결 성분을 찾지 못했습니다.",
            "wall_mask": np.zeros((h, w), dtype=bool),
        }

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])

    wall_mask_clean = labels == largest_label

    if largest_area < 1000:
        return {
            "detected": False,
            "reason": "정면 벽 후보 영역이 너무 작습니다.",
            "wall_mask": wall_mask_clean,
        }

    best["detected"] = True
    best["wall_mask"] = wall_mask_clean
    best["area_pixels_geometry"] = largest_area

    return best


# =========================================================
# 10. 정면 벽 가로 길이 추정
# =========================================================

def get_median_point_3d(points_3d: np.ndarray, x: int, y: int, patch_size: int = 5):
    h, w, _ = points_3d.shape

    half = patch_size // 2
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)

    patch = points_3d[y1:y2, x1:x2].reshape(-1, 3)
    valid = patch[:, 2] > 0.05
    patch = patch[valid]

    if len(patch) == 0:
        return points_3d[y, x].astype(np.float32)

    return np.median(patch, axis=0).astype(np.float32)


def estimate_front_wall_width_from_mask(wall_mask: np.ndarray, boundary_y_by_x, points_3d: np.ndarray):
    h, w = wall_mask.shape

    ys, xs = np.where(wall_mask)

    if len(xs) < 500:
        return {
            "detected": False,
            "reason": "정면 벽 mask 픽셀이 너무 적습니다.",
        }

    # 너무 극단적인 가장자리 오검출을 피하기 위해 분위수 사용
    x_left = int(np.percentile(xs, 3))
    x_right = int(np.percentile(xs, 97))

    if x_right <= x_left:
        return {
            "detected": False,
            "reason": "정면 벽 좌우 경계를 계산하지 못했습니다.",
        }

    def find_bottom_y(x):
        if boundary_y_by_x is not None and 0 <= x < len(boundary_y_by_x) and boundary_y_by_x[x] >= 0:
            return int(np.clip(round(boundary_y_by_x[x]), 0, h - 1))

        col_ys = np.where(wall_mask[:, x])[0]
        if len(col_ys) == 0:
            return int(np.percentile(ys, 95))

        return int(np.max(col_ys))

    y_left = find_bottom_y(x_left)
    y_right = find_bottom_y(x_right)

    p_left = get_median_point_3d(points_3d, x_left, y_left, patch_size=7)
    p_right = get_median_point_3d(points_3d, x_right, y_right, patch_size=7)

    width_raw_m = float(np.linalg.norm(p_right - p_left))
    pixel_width_geometry = float(math.sqrt((x_right - x_left) ** 2 + (y_right - y_left) ** 2))

    return {
        "detected": True,
        "left_bottom_pixel_geometry": {"x": int(x_left), "y": int(y_left)},
        "right_bottom_pixel_geometry": {"x": int(x_right), "y": int(y_right)},
        "left_bottom_point_3d_raw_m": {
            "x": float(p_left[0]),
            "y": float(p_left[1]),
            "z": float(p_left[2]),
        },
        "right_bottom_point_3d_raw_m": {
            "x": float(p_right[0]),
            "y": float(p_right[1]),
            "z": float(p_right[2]),
        },
        "estimated_width_raw_m": width_raw_m,
        "known_width_m": float(KNOWN_FRONT_WALL_WIDTH_M),
        "pixel_width_geometry": pixel_width_geometry,
    }


def mask_to_polygon(mask: np.ndarray, max_points: int = 12):
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)

    epsilon = 0.015 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    points = approx.reshape(-1, 2)

    if len(points) > max_points:
        # 너무 많으면 bounding rectangle로 단순화
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ])

    return [{"x": int(x), "y": int(y)} for x, y in points]


# =========================================================
# 11. scale 보정
# =========================================================

def decide_final_scale(floor_plane: dict, front_wall_width: dict):
    height_scale = floor_plane.get("scale_factor_from_camera_height")

    wall_scale = None
    if front_wall_width.get("detected") and front_wall_width.get("estimated_width_raw_m", 0) > 1e-6:
        wall_scale = KNOWN_FRONT_WALL_WIDTH_M / front_wall_width["estimated_width_raw_m"]

    if wall_scale is not None:
        selected = "known_front_wall_width_3.24m"
        final_scale = wall_scale
    elif height_scale is not None:
        selected = "known_camera_height_1.30m"
        final_scale = height_scale
    else:
        selected = "none"
        final_scale = 1.0

    return {
        "selected_scale_method": selected,
        "final_scale_factor": float(final_scale),
        "scale_factor_from_known_front_wall_width": None if wall_scale is None else float(wall_scale),
        "scale_factor_from_camera_height": None if height_scale is None else float(height_scale),
        "note": "정면 벽 3.24m 탐지가 성공하면 해당 값을 최종 scale로 사용하고, 실패하면 카메라 높이 1.30m 기반 scale을 사용합니다.",
    }


# =========================================================
# 12. 경계선 길이 계산 및 원본 좌표 환산
# =========================================================

def geometry_pixel_to_original_pixel(x: int, y: int, resize_info: dict):
    scale_x = resize_info["scale_x"]
    scale_y = resize_info["scale_y"]

    ox = int(round(x / scale_x))
    oy = int(round(y / scale_y))

    ox = int(np.clip(ox, 0, resize_info["original_width"] - 1))
    oy = int(np.clip(oy, 0, resize_info["original_height"] - 1))

    return ox, oy


def polygon_geometry_to_original(polygon, resize_info: dict):
    result = []

    for p in polygon:
        ox, oy = geometry_pixel_to_original_pixel(p["x"], p["y"], resize_info)
        result.append({"x": ox, "y": oy})

    return result


def enrich_boundary_edges(line_segments, points_3d, resize_info, final_scale_factor):
    edges = []

    for i, line in enumerate(line_segments, start=1):
        x1, y1 = line["x1"], line["y1"]
        x2, y2 = line["x2"], line["y2"]

        p1_raw = get_median_point_3d(points_3d, x1, y1, patch_size=7)
        p2_raw = get_median_point_3d(points_3d, x2, y2, patch_size=7)

        raw_length_m = float(np.linalg.norm(p2_raw - p1_raw))
        calibrated_length_m = raw_length_m * final_scale_factor

        p1_cal = p1_raw * final_scale_factor
        p2_cal = p2_raw * final_scale_factor

        ox1, oy1 = geometry_pixel_to_original_pixel(x1, y1, resize_info)
        ox2, oy2 = geometry_pixel_to_original_pixel(x2, y2, resize_info)

        edges.append({
            "edge_id": f"floor_wall_edge_{i}",
            "type": "floor_wall_boundary_candidate",
            "pixel_start_geometry": {"x": int(x1), "y": int(y1)},
            "pixel_end_geometry": {"x": int(x2), "y": int(y2)},
            "pixel_start_original": {"x": int(ox1), "y": int(oy1)},
            "pixel_end_original": {"x": int(ox2), "y": int(oy2)},
            "point_3d_start_raw_m": {
                "x": float(p1_raw[0]),
                "y": float(p1_raw[1]),
                "z": float(p1_raw[2]),
            },
            "point_3d_end_raw_m": {
                "x": float(p2_raw[0]),
                "y": float(p2_raw[1]),
                "z": float(p2_raw[2]),
            },
            "point_3d_start_calibrated_m": {
                "x": float(p1_cal[0]),
                "y": float(p1_cal[1]),
                "z": float(p1_cal[2]),
            },
            "point_3d_end_calibrated_m": {
                "x": float(p2_cal[0]),
                "y": float(p2_cal[1]),
                "z": float(p2_cal[2]),
            },
            "length_raw_m": raw_length_m,
            "length_calibrated_m": calibrated_length_m,
            "pixel_length_geometry": float(line["pixel_length_geometry"]),
        })

    return edges


# =========================================================
# 13. 디버그 이미지 저장
# =========================================================

def save_debug_images(
    image_geom_rgb,
    depth_geom,
    floor_mask,
    boundary_mask,
    wall_mask,
    boundary_edges,
    front_wall_width,
    output_json_path: Path,
):
    base = output_json_path.with_suffix("")

    image_bgr = cv2.cvtColor(image_geom_rgb, cv2.COLOR_RGB2BGR)

    # depth 시각화
    valid = depth_geom > 0.05
    depth_vis = np.zeros_like(depth_geom, dtype=np.uint8)

    if np.any(valid):
        d_min = np.percentile(depth_geom[valid], 2)
        d_max = np.percentile(depth_geom[valid], 98)
        depth_norm = np.clip((depth_geom - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)

    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(base) + "_depth.png", depth_color)

    # floor overlay
    floor_overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[floor_mask] = (0, 255, 0)
    floor_overlay = cv2.addWeighted(floor_overlay, 0.75, green, 0.25, 0)
    cv2.imwrite(str(base) + "_floor_mask.png", floor_overlay)

    # boundary overlay
    boundary_overlay = image_bgr.copy()

    if boundary_mask is not None:
        boundary_overlay[boundary_mask > 0] = (0, 0, 255)

    for edge in boundary_edges:
        x1 = edge["pixel_start_geometry"]["x"]
        y1 = edge["pixel_start_geometry"]["y"]
        x2 = edge["pixel_end_geometry"]["x"]
        y2 = edge["pixel_end_geometry"]["y"]

        cv2.line(boundary_overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = f'{edge["length_calibrated_m"]:.2f}m'
        cv2.putText(
            boundary_overlay,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(base) + "_boundary_overlay.png", boundary_overlay)

    # front wall overlay
    wall_overlay = image_bgr.copy()

    if wall_mask is not None and np.any(wall_mask):
        blue = np.zeros_like(image_bgr)
        blue[wall_mask] = (255, 0, 0)
        wall_overlay = cv2.addWeighted(wall_overlay, 0.70, blue, 0.30, 0)

    if front_wall_width.get("detected"):
        p1 = front_wall_width["left_bottom_pixel_geometry"]
        p2 = front_wall_width["right_bottom_pixel_geometry"]

        cv2.line(
            wall_overlay,
            (p1["x"], p1["y"]),
            (p2["x"], p2["y"]),
            (0, 255, 255),
            4,
        )
        cv2.putText(
            wall_overlay,
            f'front wall = {KNOWN_FRONT_WALL_WIDTH_M:.2f}m',
            (p1["x"], max(30, p1["y"] - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(base) + "_front_wall_overlay.png", wall_overlay)


# =========================================================
# 14. JSON 구성
# =========================================================

def build_result_json(
    image_path,
    output_path,
    model_id,
    image,
    depth_original,
    depth_geom,
    original_intrinsics,
    geometry_intrinsics,
    resize_info,
    floor_plane,
    boundary_edges,
    front_wall_plane,
    front_wall_width,
    scale_info,
):
    original_width, original_height = image.size

    valid_depth = depth_original[depth_original > 0.05]

    if len(valid_depth) > 0:
        depth_stats = {
            "min_m": float(np.min(valid_depth)),
            "max_m": float(np.max(valid_depth)),
            "mean_m": float(np.mean(valid_depth)),
            "median_m": float(np.median(valid_depth)),
            "p10_m": float(np.percentile(valid_depth, 10)),
            "p90_m": float(np.percentile(valid_depth, 90)),
        }
    else:
        depth_stats = {
            "min_m": None,
            "max_m": None,
            "mean_m": None,
            "median_m": None,
            "p10_m": None,
            "p90_m": None,
        }

    wall_polygon_geometry = []
    wall_polygon_original = []

    if front_wall_plane.get("detected"):
        wall_polygon_geometry = mask_to_polygon(front_wall_plane["wall_mask"])
        wall_polygon_original = polygon_geometry_to_original(wall_polygon_geometry, resize_info)

    front_wall_json = {
        "detected": bool(front_wall_plane.get("detected", False)),
        "known_width_m": float(KNOWN_FRONT_WALL_WIDTH_M),
        "plane": None,
        "mask_area_pixels_geometry": int(front_wall_plane.get("area_pixels_geometry", 0)),
        "polygon_geometry": wall_polygon_geometry,
        "polygon_original": wall_polygon_original,
        "width_estimation": front_wall_width,
    }

    if front_wall_plane.get("detected"):
        front_wall_json["plane"] = {
            "normal_raw": [
                float(front_wall_plane["normal"][0]),
                float(front_wall_plane["normal"][1]),
                float(front_wall_plane["normal"][2]),
            ],
            "d_raw": float(front_wall_plane["d"]),
            "threshold_m": float(front_wall_plane["threshold_m"]),
            "z_alignment": float(front_wall_plane["z_alignment"]),
            "y_alignment": float(front_wall_plane["y_alignment"]),
        }

    if front_wall_width.get("detected"):
        raw_width = front_wall_width["estimated_width_raw_m"]
        calibrated_width = raw_width * scale_info["final_scale_factor"]

        front_wall_json["width_estimation"]["estimated_width_calibrated_m"] = float(calibrated_width)
        front_wall_json["width_estimation"]["scale_error_after_calibration_m"] = float(
            abs(KNOWN_FRONT_WALL_WIDTH_M - calibrated_width)
        )

        # 원본 좌표도 추가
        left_g = front_wall_width["left_bottom_pixel_geometry"]
        right_g = front_wall_width["right_bottom_pixel_geometry"]

        left_o = geometry_pixel_to_original_pixel(left_g["x"], left_g["y"], resize_info)
        right_o = geometry_pixel_to_original_pixel(right_g["x"], right_g["y"], resize_info)

        front_wall_json["width_estimation"]["left_bottom_pixel_original"] = {
            "x": int(left_o[0]),
            "y": int(left_o[1]),
        }
        front_wall_json["width_estimation"]["right_bottom_pixel_original"] = {
            "x": int(right_o[0]),
            "y": int(right_o[1]),
        }

    result = {
        "input_image": str(image_path),
        "output_json": str(output_path),
        "model": {
            "name": model_id,
            "task": "single_image_metric_depth_estimation",
            "scene_type": "indoor",
        },
        "image": {
            "width": int(original_width),
            "height": int(original_height),
            "megapixel_info": "24MP, 5712x4284 기준",
        },
        "camera": {
            "device": DEVICE_NAME,
            "lens": "1x main camera",
            "equivalent_focal_length_mm": float(EQUIVALENT_FOCAL_LENGTH_MM),
            "f_number": 1.78,
            "iso": 160,
            "exposure_compensation_ev": 0,
            "shutter_speed": "1/120s",
            "camera_height_m": float(CAMERA_HEIGHT_M),
            "shooting_assumption": "수평 촬영",
            "intrinsics_original": original_intrinsics,
            "intrinsics_geometry": geometry_intrinsics,
            "note": "24mm 35mm 환산 초점거리와 이미지 해상도를 이용해 intrinsic을 근사했습니다.",
        },
        "resize_for_geometry": resize_info,
        "depth_statistics_original_m": depth_stats,
        "floor_plane": {
            "normal_raw": [
                float(floor_plane["normal"][0]),
                float(floor_plane["normal"][1]),
                float(floor_plane["normal"][2]),
            ],
            "d_raw": float(floor_plane["d"]),
            "threshold_m": float(floor_plane["threshold_m"]),
            "inlier_count_sampled": int(floor_plane["inlier_count_sampled"]),
            "estimated_camera_height_m_raw": float(floor_plane["estimated_camera_height_m_raw"]),
            "known_camera_height_m": float(CAMERA_HEIGHT_M),
            "scale_factor_from_camera_height": floor_plane.get("scale_factor_from_camera_height"),
        },
        "front_wall": front_wall_json,
        "scale_calibration": scale_info,
        "floor_wall_boundary_edges": boundary_edges,
        "llm_placement_context": {
            "summary": "바닥-벽 경계 후보선, 정면 벽 후보 영역, 보정된 m 단위 길이를 제공합니다.",
            "recommended_usage": "가구 배치 가능 벽면과 바닥 경계 기준 좌표를 LLM 또는 메인 서버에 전달할 때 사용합니다.",
            "important_warning": "정면 벽이 가구나 커튼 등으로 많이 가려져 있으면 벽 탐지와 길이 보정이 불안정할 수 있습니다.",
        },
    }

    return result


# =========================================================
# 15. main
# =========================================================

def main():
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/9] 이미지 로딩 중...")
    image = load_image_rgb(INPUT_IMAGE_PATH)

    original_width, original_height = image.size
    print(f"[INFO] image size: {original_width} x {original_height}")

    if (original_width, original_height) != (ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT):
        print("[WARN] 설정된 기준 해상도 5712x4284와 실제 이미지 해상도가 다릅니다.")
        print("[WARN] 실제 이미지 해상도 기준으로 intrinsic을 다시 계산합니다.")

    print("[2/9] iPhone 16 Pro 24mm intrinsic 계산 중...")
    original_intrinsics = calculate_intrinsics_from_35mm_equiv(original_width, original_height)

    print("[3/9] Depth Anything V2 Metric Indoor 추론 중...")
    depth_original = predict_metric_depth(image, MODEL_ID)

    print("[4/9] 기하 계산용 이미지/depth 축소 중...")
    image_geom, depth_geom, resize_info = resize_for_geometry(
        image=image,
        depth_m=depth_original,
        max_width=GEOMETRY_MAX_WIDTH,
    )

    geometry_intrinsics = scale_intrinsics(
        original_intrinsics,
        resize_info["scale_x"],
        resize_info["scale_y"],
    )

    print(f"[INFO] geometry size: {resize_info['geometry_width']} x {resize_info['geometry_height']}")

    print("[5/9] depth map을 3D point cloud로 변환 중...")
    points_3d = depth_to_point_cloud(depth_geom, geometry_intrinsics)

    print("[6/9] 바닥 plane 추정 중...")
    floor_plane = estimate_floor_plane(points_3d, depth_geom)

    print("[INFO] raw estimated camera height:",
          f"{floor_plane['estimated_camera_height_m_raw']:.3f} m")
    print("[INFO] height scale candidate:",
          floor_plane.get("scale_factor_from_camera_height"))

    print("[7/9] 바닥-벽 경계 후보선 추출 중...")
    boundary_y_by_x, line_segments, boundary_mask = extract_floor_wall_boundary(floor_plane["floor_mask"])

    print(f"[INFO] floor-wall boundary candidate lines: {len(line_segments)}")

    print("[8/9] 정면 벽 plane 탐지 및 3.24m 기준 보정 중...")
    front_wall_plane = estimate_front_wall_plane(points_3d, depth_geom, boundary_y_by_x)

    if front_wall_plane.get("detected"):
        front_wall_width = estimate_front_wall_width_from_mask(
            wall_mask=front_wall_plane["wall_mask"],
            boundary_y_by_x=boundary_y_by_x,
            points_3d=points_3d,
        )
    else:
        front_wall_width = {
            "detected": False,
            "reason": front_wall_plane.get("reason", "정면 벽 탐지 실패"),
        }

    scale_info = decide_final_scale(floor_plane, front_wall_width)
    final_scale_factor = scale_info["final_scale_factor"]

    print("[INFO] selected scale method:", scale_info["selected_scale_method"])
    print("[INFO] final scale factor:", final_scale_factor)

    boundary_edges = enrich_boundary_edges(
        line_segments=line_segments,
        points_3d=points_3d,
        resize_info=resize_info,
        final_scale_factor=final_scale_factor,
    )

    print("[9/9] JSON 저장 중...")
    result_json = build_result_json(
        image_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_JSON_PATH,
        model_id=MODEL_ID,
        image=image,
        depth_original=depth_original,
        depth_geom=depth_geom,
        original_intrinsics=original_intrinsics,
        geometry_intrinsics=geometry_intrinsics,
        resize_info=resize_info,
        floor_plane=floor_plane,
        boundary_edges=boundary_edges,
        front_wall_plane=front_wall_plane,
        front_wall_width=front_wall_width,
        scale_info=scale_info,
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    if SAVE_DEBUG_IMAGES:
        save_debug_images(
            image_geom_rgb=image_geom,
            depth_geom=depth_geom,
            floor_mask=floor_plane["floor_mask"],
            boundary_mask=boundary_mask,
            wall_mask=front_wall_plane.get("wall_mask", None),
            boundary_edges=boundary_edges,
            front_wall_width=front_wall_width,
            output_json_path=OUTPUT_JSON_PATH,
        )

    print("\n[DONE] 저장 완료")
    print(f"JSON: {OUTPUT_JSON_PATH}")
    print(f"Depth image: {OUTPUT_JSON_PATH.with_suffix('')}_depth.png")
    print(f"Floor mask: {OUTPUT_JSON_PATH.with_suffix('')}_floor_mask.png")
    print(f"Boundary overlay: {OUTPUT_JSON_PATH.with_suffix('')}_boundary_overlay.png")
    print(f"Front wall overlay: {OUTPUT_JSON_PATH.with_suffix('')}_front_wall_overlay.png")

    print("\n[SUMMARY]")
    print(f"- 정면 벽 탐지 여부: {front_wall_plane.get('detected', False)}")

    if front_wall_width.get("detected"):
        print(f"- 정면 벽 raw 추정 가로: {front_wall_width['estimated_width_raw_m']:.3f} m")
        print(f"- 정면 벽 보정 기준 가로: {KNOWN_FRONT_WALL_WIDTH_M:.3f} m")
        print(f"- 최종 scale factor: {final_scale_factor:.4f}")
    else:
        print(f"- 정면 벽 가로 추정 실패: {front_wall_width.get('reason')}")

    for edge in boundary_edges:
        print(
            f'- {edge["edge_id"]}: '
            f'raw={edge["length_raw_m"]:.3f}m, '
            f'calibrated={edge["length_calibrated_m"]:.3f}m'
        )


if __name__ == "__main__":
    main()