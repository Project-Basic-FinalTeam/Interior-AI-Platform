"""
실내 이미지 기반 Depth 추정 및 바닥-벽 경계 후보선 추출 코드

이 코드는 단일 실내 이미지를 입력으로 받아 Depth Anything V2 Metric Indoor 모델을 이용해
픽셀 단위의 metric depth map(미터 단위 깊이 지도)을 생성한다. 이후 depth map을 카메라 좌표계
기준의 3D point cloud로 변환하고, 이미지 하단부의 점들을 기반으로 RANSAC 알고리즘을 사용해
바닥 plane(평면)을 추정한다.

추정된 바닥 영역으로부터 바닥-벽 경계 후보선을 추출하고, 각 경계선의 픽셀 좌표, 3D 좌표,
추정 길이, 평균 깊이값 등을 계산하여 JSON 파일로 저장한다. 또한 결과 검증을 위해 depth map,
floor mask, boundary overlay 등의 디버그 이미지를 함께 저장한다.

시연용 보정:
- 특정 이미지 1장 시연 기준으로 depth scale calibration(스케일 보정)을 적용한다.
- result_10 기준 왼쪽 경계선 4.07m → 약 3.0m 수준으로 맞추기 위해 DEPTH_SCALE_FACTOR = 0.74 사용.
- 오른쪽 대표선은 책상 오른쪽 다리 바닥점 → 오른쪽 벽 하단점으로 수동 지정한다.
- 오른쪽 대표선의 표시 길이는 0.93m로 고정한다.
- RANSAC 결과가 실행마다 달라지지 않도록 random seed를 고정한다.
- HoughLinesP로 중복 검출된 바닥-벽 후보선 중 대표 경계선만 선택한다.
- boundary overlay에서는 빨간 raw boundary 후보점을 숨기고 파란 대표선만 표시한다.

주의:
- 카메라의 실제 intrinsic parameter(내부 파라미터)를 알 수 없기 때문에,
  horizontal FOV 값을 가정하여 3D 좌표와 길이를 계산한다.
- 따라서 출력되는 길이값은 실측값이 아니라 추정값이다.
- DEPTH_SCALE_FACTOR와 MANUAL_RIGHT_EDGE_LENGTH_M은 일반 모델 보정값이 아니라
  특정 시연 이미지 기준 수동 보정값이다.
"""

import json
import math
import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import pillow_avif  # AVIF 이미지 로딩용 등록 플러그인

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# =========================
# 1. 경로 및 설정값
# =========================

INPUT_IMAGE_PATH = Path(
    r"C:\Users\khrha\Desktop\SWPJ_4\Interior-AI-Platform\depth\base_test\test_10.png"
)

RESULT_NUMBER = "10-3"

BASE_OUTPUT_DIR = Path(
    r"C:\Users\khrha\Desktop\SWPJ_4\Interior-AI-Platform\depth\base_test"
)

RESULT_NAME = f"result_{RESULT_NUMBER}"

OUTPUT_DIR = BASE_OUTPUT_DIR / RESULT_NAME
OUTPUT_JSON_PATH = OUTPUT_DIR / f"{RESULT_NAME}.json"

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# 카메라 시야각 가정값
HORIZONTAL_FOV_DEG = 70.0

# 디버그 이미지 저장 여부
SAVE_DEBUG_IMAGES = True

# 실행 결과 고정
RANDOM_SEED = 42

# 최종 JSON에 남길 대표 경계선 개수
DEMO_MAX_EDGES = 2

# =========================
# 시연 이미지 전용 scale 보정값
# =========================
# 현재 result_10 기준:
# 왼쪽 경계선 4.07m → 약 3.0m
# 오른쪽 경계선 1.25m → 약 0.93m
DEPTH_SCALE_FACTOR = 0.74


# =========================
# 오른쪽 경계선 수동 보정
# =========================
# 이미지 1장 시연용:
# 오른쪽 대표 경계선을 "책상 오른쪽 다리 바닥점 → 오른쪽 벽 하단점"으로 고정
MANUAL_RIGHT_EDGE_OVERRIDE = True

# 원본 이미지 1448x1086 기준 좌표입니다.
# 책상 오른쪽 다리 바닥 근처 좌표
# 실행 결과를 보고 시작점이 다리보다 오른쪽/왼쪽이면 x 값을 조금 조정하면 됩니다.
MANUAL_RIGHT_EDGE_START = {
    "x": 1005,
    "y": 813,
}

# 오른쪽 벽 하단 끝 좌표
MANUAL_RIGHT_EDGE_END = {
    "x": 1368,
    "y": 896,
}

# 시연에서 보여줄 오른쪽 구간 길이
MANUAL_RIGHT_EDGE_LENGTH_M = 0.93


# =========================
# 2. 이미지 로딩
# =========================

def load_image_rgb(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"입력 이미지가 존재하지 않습니다: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return image


# =========================
# 3. Depth Anything V2 Metric Indoor 추론
# =========================

def predict_metric_depth(image: Image.Image, model_id: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 사용 device: {device}")

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


def apply_depth_scale_calibration(depth_m: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    특정 시연 이미지 기준 depth scale calibration.

    Depth Anything V2 Metric Indoor 결과가 실제 방 크기보다 크게 추정될 경우,
    depth map 전체에 동일한 scale factor를 곱해 3D 좌표와 길이값을 함께 보정한다.

    주의:
    - 이 보정은 특정 시연 이미지 기준 수동 보정이다.
    - 일반화된 실측 보정값이 아니다.
    - 정확한 보정을 위해서는 실제 기준 길이, 카메라 FOV, 카메라 intrinsic parameter가 필요하다.
    """

    calibrated_depth = depth_m.copy()

    valid_mask = calibrated_depth > 0.05
    calibrated_depth[valid_mask] = calibrated_depth[valid_mask] * scale_factor

    return calibrated_depth


# =========================
# 4. depth map을 3D point cloud로 변환
# =========================

def make_camera_intrinsics(width: int, height: int, horizontal_fov_deg: float):
    hfov_rad = math.radians(horizontal_fov_deg)

    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return fx, fy, cx, cy


def depth_to_point_cloud(depth_m: np.ndarray, horizontal_fov_deg: float):
    height, width = depth_m.shape
    fx, fy, cx, cy = make_camera_intrinsics(width, height, horizontal_fov_deg)

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_3d = np.stack([x, y, z], axis=-1).astype(np.float32)

    return points_3d, {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "horizontal_fov_deg": horizontal_fov_deg,
    }


# =========================
# 5. 바닥 plane RANSAC 추정
# =========================

def fit_plane_from_3_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return None

    normal = normal / norm
    d = -np.dot(normal, p1)

    return normal, d


def plane_distance(points, normal, d):
    return np.abs(points @ normal + d)


def estimate_floor_plane_ransac(points_3d: np.ndarray, depth_m: np.ndarray):
    height, width = depth_m.shape

    y_start = int(height * 0.45)

    valid_mask = depth_m > 0.05

    bottom_mask = np.zeros_like(valid_mask, dtype=bool)
    bottom_mask[y_start:, :] = True

    candidate_mask = valid_mask & bottom_mask
    candidate_points = points_3d[candidate_mask]

    if len(candidate_points) < 500:
        raise RuntimeError(
            "바닥 후보 점이 너무 적습니다. 이미지에 바닥이 충분히 보이는지 확인하세요."
        )

    max_sample_points = 12000

    if len(candidate_points) > max_sample_points:
        idx = np.random.choice(len(candidate_points), max_sample_points, replace=False)
        sampled_points = candidate_points[idx]
    else:
        sampled_points = candidate_points

    best_normal = None
    best_d = None
    best_inlier_count = 0

    median_depth = float(np.median(depth_m[valid_mask]))
    distance_threshold = max(0.04, median_depth * 0.015)

    iterations = 600

    for _ in range(iterations):
        ids = np.random.choice(len(sampled_points), 3, replace=False)

        plane = fit_plane_from_3_points(
            sampled_points[ids[0]],
            sampled_points[ids[1]],
            sampled_points[ids[2]],
        )

        if plane is None:
            continue

        normal, d = plane

        distances = plane_distance(sampled_points, normal, d)
        inlier_count = int(np.sum(distances < distance_threshold))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None:
        raise RuntimeError("바닥 평면 추정에 실패했습니다.")

    all_dist = plane_distance(
        points_3d.reshape(-1, 3),
        best_normal,
        best_d,
    ).reshape(height, width)

    floor_mask = (all_dist < distance_threshold) & valid_mask

    upper_cut = int(height * 0.30)
    floor_mask[:upper_cut, :] = False

    floor_mask_uint8 = floor_mask.astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    floor_mask_uint8 = cv2.morphologyEx(floor_mask_uint8, cv2.MORPH_OPEN, kernel)
    floor_mask_uint8 = cv2.morphologyEx(floor_mask_uint8, cv2.MORPH_CLOSE, kernel)

    floor_mask = floor_mask_uint8 > 0

    return {
        "normal": best_normal,
        "d": float(best_d),
        "distance_threshold_m": float(distance_threshold),
        "inlier_count_sampled": int(best_inlier_count),
        "floor_mask": floor_mask,
    }


# =========================
# 6. 바닥-벽 경계선 후보 추출
# =========================

def extract_floor_wall_boundary_lines(floor_mask: np.ndarray):
    height, width = floor_mask.shape

    boundary_points = []

    for x in range(width):
        ys = np.where(floor_mask[:, x])[0]

        if len(ys) == 0:
            continue

        y_top = int(np.min(ys))

        if y_top < int(height * 0.25):
            continue

        boundary_points.append((x, y_top))

    if len(boundary_points) < width * 0.15:
        return [], None

    boundary_points = np.array(boundary_points, dtype=np.int32)

    xs = boundary_points[:, 0]
    ys = boundary_points[:, 1]

    full_xs = np.arange(xs.min(), xs.max() + 1)
    full_ys = np.interp(full_xs, xs, ys)

    window = max(9, width // 80)

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    smooth_ys = np.convolve(full_ys, kernel, mode="same")

    boundary_mask = np.zeros((height, width), dtype=np.uint8)

    for x, y in zip(full_xs, smooth_ys):
        y_int = int(np.clip(round(y), 0, height - 1))
        boundary_mask[y_int, x] = 255

    boundary_mask = cv2.dilate(
        boundary_mask,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    lines = cv2.HoughLinesP(
        boundary_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=max(50, width // 6),
        maxLineGap=max(20, width // 25),
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
                "pixel_length": float(pixel_length),
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
            "pixel_length": float(pixel_length),
        })

    return line_segments, boundary_mask


# =========================
# 7. 선분의 3D 길이 계산
# =========================

def get_point_3d_at_pixel(points_3d: np.ndarray, x: int, y: int, patch_size: int = 5):
    height, width, _ = points_3d.shape

    half = patch_size // 2

    x1 = max(0, x - half)
    x2 = min(width, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(height, y + half + 1)

    patch = points_3d[y1:y2, x1:x2].reshape(-1, 3)

    valid = patch[:, 2] > 0.05
    patch = patch[valid]

    if len(patch) == 0:
        return points_3d[y, x].astype(float)

    return np.median(patch, axis=0).astype(float)


def enrich_lines_with_metric_length(line_segments, points_3d: np.ndarray):
    enriched = []

    for i, line in enumerate(line_segments, start=1):
        x1, y1 = line["x1"], line["y1"]
        x2, y2 = line["x2"], line["y2"]

        p1 = get_point_3d_at_pixel(points_3d, x1, y1)
        p2 = get_point_3d_at_pixel(points_3d, x2, y2)

        length_m = float(np.linalg.norm(p2 - p1))

        sample_count = 30

        xs = np.linspace(x1, x2, sample_count).astype(int)
        ys = np.linspace(y1, y2, sample_count).astype(int)

        depth_values = []

        for x, y in zip(xs, ys):
            z = points_3d[y, x, 2]

            if z > 0.05:
                depth_values.append(float(z))

        mean_depth_m = float(np.mean(depth_values)) if depth_values else None

        enriched.append({
            "edge_id": f"floor_wall_edge_{i}",
            "type": "floor_wall_boundary_candidate",
            "pixel_start": {
                "x": int(x1),
                "y": int(y1),
            },
            "pixel_end": {
                "x": int(x2),
                "y": int(y2),
            },
            "point_3d_start_m": {
                "x": float(p1[0]),
                "y": float(p1[1]),
                "z": float(p1[2]),
            },
            "point_3d_end_m": {
                "x": float(p2[0]),
                "y": float(p2[1]),
                "z": float(p2[2]),
            },
            "length_m_estimated": length_m,
            "pixel_length": float(line["pixel_length"]),
            "mean_depth_m": mean_depth_m,
            "confidence_note": (
                "카메라 FOV, 내부파라미터, 수동 scale factor 기반 추정값입니다. "
                "실측값이 아닙니다."
            ),
        })

    return enriched


# =========================
# 7-1. 시연용 대표 경계선 선택
# =========================

def get_edge_angle_deg(edge):
    x1 = edge["pixel_start"]["x"]
    y1 = edge["pixel_start"]["y"]
    x2 = edge["pixel_end"]["x"]
    y2 = edge["pixel_end"]["y"]

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:
        return 90.0

    return math.degrees(math.atan2(dy, dx))


def get_edge_mid_x(edge):
    return (edge["pixel_start"]["x"] + edge["pixel_end"]["x"]) / 2.0


def get_edge_mid_y(edge):
    return (edge["pixel_start"]["y"] + edge["pixel_end"]["y"]) / 2.0


def clone_selected_edge(edge, new_id, reason):
    selected = copy.deepcopy(edge)

    selected["edge_id"] = new_id
    selected["is_demo_selected"] = True
    selected["selection_reason"] = reason
    selected["confidence_note"] = (
        "시연용으로 중복 경계 후보를 제거한 대표 바닥-벽 경계선입니다. "
        "카메라 FOV, 내부파라미터, 수동 scale factor 기반 추정값이며 실측값이 아닙니다."
    )

    return selected


def select_demo_boundary_edges(edges, image_width, image_height, max_edges=2):
    if len(edges) == 0:
        return []

    filtered = []

    for edge in edges:
        angle = get_edge_angle_deg(edge)
        mid_y = get_edge_mid_y(edge)
        pixel_length = edge["pixel_length"]

        if pixel_length < image_width * 0.18:
            continue

        if not (image_height * 0.55 <= mid_y <= image_height * 0.85):
            continue

        if abs(angle) > 35:
            continue

        filtered.append(edge)

    if len(filtered) == 0:
        fallback_edges = sorted(edges, key=lambda e: e["pixel_length"], reverse=True)
        selected = []

        for i, edge in enumerate(fallback_edges[:max_edges], start=1):
            selected.append(
                clone_selected_edge(
                    edge,
                    f"demo_floor_wall_edge_fallback_{i}",
                    "필터링 결과가 없어 긴 선분 기준으로 대표 경계선을 선택",
                )
            )

        return selected

    left_candidates = [
        edge for edge in filtered
        if get_edge_angle_deg(edge) < -3
    ]

    right_candidates = [
        edge for edge in filtered
        if get_edge_angle_deg(edge) > 3
    ]

    selected = []

    if left_candidates:
        left_edge = max(left_candidates, key=lambda e: e["pixel_length"])

        selected.append(
            clone_selected_edge(
                left_edge,
                "demo_floor_wall_edge_left",
                "왼쪽 바닥-벽 경계 후보 중 가장 긴 선분을 대표선으로 선택",
            )
        )

    if right_candidates:
        right_edge = max(
            right_candidates,
            key=lambda e: (get_edge_mid_x(e), e["pixel_length"]),
        )

        selected.append(
            clone_selected_edge(
                right_edge,
                "demo_floor_wall_edge_right",
                "오른쪽 바닥-벽 경계 후보 중 가장 오른쪽에 위치한 선분을 대표선으로 선택",
            )
        )

    if len(selected) < max_edges:
        selected_keys = {
            (
                edge["pixel_start"]["x"],
                edge["pixel_start"]["y"],
                edge["pixel_end"]["x"],
                edge["pixel_end"]["y"],
            )
            for edge in selected
        }

        remaining_edges = []

        for edge in filtered:
            key = (
                edge["pixel_start"]["x"],
                edge["pixel_start"]["y"],
                edge["pixel_end"]["x"],
                edge["pixel_end"]["y"],
            )

            if key not in selected_keys:
                remaining_edges.append(edge)

        remaining_edges = sorted(
            remaining_edges,
            key=lambda e: e["pixel_length"],
            reverse=True,
        )

        for edge in remaining_edges:
            if len(selected) >= max_edges:
                break

            selected.append(
                clone_selected_edge(
                    edge,
                    f"demo_floor_wall_edge_extra_{len(selected) + 1}",
                    "대표선 개수가 부족하여 긴 선분 기준으로 추가 선택",
                )
            )

    return selected[:max_edges]


# =========================
# 7-2. 오른쪽 대표 경계선 수동 보정
# =========================

def make_manual_right_edge(points_3d: np.ndarray):
    x1 = int(MANUAL_RIGHT_EDGE_START["x"])
    y1 = int(MANUAL_RIGHT_EDGE_START["y"])
    x2 = int(MANUAL_RIGHT_EDGE_END["x"])
    y2 = int(MANUAL_RIGHT_EDGE_END["y"])

    p1 = get_point_3d_at_pixel(points_3d, x1, y1)
    p2 = get_point_3d_at_pixel(points_3d, x2, y2)

    raw_length_m = float(np.linalg.norm(p2 - p1))
    pixel_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    sample_count = 30
    xs = np.linspace(x1, x2, sample_count).astype(int)
    ys = np.linspace(y1, y2, sample_count).astype(int)

    depth_values = []

    for x, y in zip(xs, ys):
        z = points_3d[y, x, 2]

        if z > 0.05:
            depth_values.append(float(z))

    mean_depth_m = float(np.mean(depth_values)) if depth_values else None

    return {
        "edge_id": "demo_floor_wall_edge_right",
        "type": "manual_demo_right_boundary",
        "pixel_start": {
            "x": x1,
            "y": y1,
        },
        "pixel_end": {
            "x": x2,
            "y": y2,
        },
        "point_3d_start_m": {
            "x": float(p1[0]),
            "y": float(p1[1]),
            "z": float(p1[2]),
        },
        "point_3d_end_m": {
            "x": float(p2[0]),
            "y": float(p2[1]),
            "z": float(p2[2]),
        },

        # 실제 depth 기반 계산값은 따로 보관
        "length_m_raw_after_scale": raw_length_m,

        # 시연 표시용 길이는 0.93m로 고정
        "length_m_estimated": float(MANUAL_RIGHT_EDGE_LENGTH_M),

        "pixel_length": float(pixel_length),
        "mean_depth_m": mean_depth_m,
        "is_demo_selected": True,
        "is_manual_override": True,
        "selection_reason": (
            "이미지 1장 시연용으로 오른쪽 경계선을 책상 오른쪽 다리 바닥점에서 "
            "오른쪽 벽 하단점까지 수동 지정했습니다."
        ),
        "confidence_note": (
            "오른쪽 경계선은 시연 이미지 기준 수동 보정값입니다. "
            "표시 길이 0.93m는 사용자가 지정한 기준 길이입니다."
        ),
    }


def apply_manual_right_edge_override(edges, points_3d: np.ndarray, max_edges=2):
    if not MANUAL_RIGHT_EDGE_OVERRIDE:
        return edges[:max_edges]

    manual_right_edge = make_manual_right_edge(points_3d)

    # 기존 오른쪽 대표선은 제거하고, 왼쪽/기타 대표선만 유지
    kept_edges = []

    for edge in edges:
        if edge["edge_id"] == "demo_floor_wall_edge_right":
            continue

        kept_edges.append(edge)

    # 최종 개수가 max_edges를 넘지 않도록 왼쪽 대표선 1개 + 수동 오른쪽 1개 구성
    kept_edges = kept_edges[:max_edges - 1]
    kept_edges.append(manual_right_edge)

    return kept_edges


# =========================
# 8. 디버그 이미지 저장
# =========================

def draw_safe_label(image, text, x, y):
    height, width = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    safe_x = int(np.clip(x + 10, 10, width - text_w - 10))
    safe_y = int(np.clip(y - 12, text_h + 10, height - 10))

    cv2.putText(
        image,
        text,
        (safe_x, safe_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def save_debug_images(
    image: Image.Image,
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    boundary_mask,
    edges,
    output_json_path: Path,
):
    base = output_json_path.with_suffix("")

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    valid = depth_m > 0.05
    depth_vis = np.zeros_like(depth_m, dtype=np.uint8)

    if np.any(valid):
        d_min = np.percentile(depth_m[valid], 2)
        d_max = np.percentile(depth_m[valid], 98)
        depth_norm = np.clip((depth_m - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(base) + "_depth.png", depth_colormap)

    floor_vis = image_bgr.copy()
    floor_overlay = np.zeros_like(image_bgr)

    floor_overlay[floor_mask] = (0, 255, 0)

    floor_vis = cv2.addWeighted(floor_vis, 0.75, floor_overlay, 0.25, 0)
    cv2.imwrite(str(base) + "_floor_mask.png", floor_vis)

    boundary_vis = image_bgr.copy()

    if boundary_mask is not None:
        boundary_vis[boundary_mask > 0] = (0, 0, 255)

    for edge in edges:
        x1 = edge["pixel_start"]["x"]
        y1 = edge["pixel_start"]["y"]
        x2 = edge["pixel_end"]["x"]
        y2 = edge["pixel_end"]["y"]

        cv2.line(boundary_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(boundary_vis, (x1, y1), 6, (0, 255, 255), -1)
        cv2.circle(boundary_vis, (x2, y2), 6, (0, 255, 255), -1)

        label = f'~{edge["length_m_estimated"]:.2f} m'
        draw_safe_label(boundary_vis, label, x1, y1)

    cv2.imwrite(str(base) + "_boundary_overlay.png", boundary_vis)


# =========================
# 9. JSON 저장
# =========================

def build_output_json(
    image_path,
    output_path,
    model_id,
    image,
    depth_m_raw,
    depth_m_calibrated,
    intrinsics,
    floor_plane,
    edges,
    raw_edge_count=None,
):
    width, height = image.size

    valid_raw_depth = depth_m_raw[depth_m_raw > 0.05]
    valid_calibrated_depth = depth_m_calibrated[depth_m_calibrated > 0.05]

    def make_depth_stats(valid_depth):
        if len(valid_depth) > 0:
            return {
                "min_m": float(np.min(valid_depth)),
                "max_m": float(np.max(valid_depth)),
                "mean_m": float(np.mean(valid_depth)),
                "median_m": float(np.median(valid_depth)),
                "p10_m": float(np.percentile(valid_depth, 10)),
                "p90_m": float(np.percentile(valid_depth, 90)),
            }

        return {
            "min_m": None,
            "max_m": None,
            "mean_m": None,
            "median_m": None,
            "p10_m": None,
            "p90_m": None,
        }

    raw_depth_stats = make_depth_stats(valid_raw_depth)
    calibrated_depth_stats = make_depth_stats(valid_calibrated_depth)

    result = {
        "input_image": str(image_path),
        "output_dir": str(output_path.parent),
        "output_json": str(output_path),
        "model": {
            "name": model_id,
            "task": "single_image_metric_depth_estimation",
            "scene_type": "indoor",
        },
        "image": {
            "width": int(width),
            "height": int(height),
        },
        "camera_assumption": {
            "note": (
                "카메라 내부파라미터가 없어서 horizontal_fov_deg 기반으로 "
                "3D 좌표와 길이를 추정했습니다."
            ),
            "intrinsics": intrinsics,
        },
        "depth_scale_calibration": {
            "enabled": True,
            "scale_factor": float(DEPTH_SCALE_FACTOR),
            "mode": "single_demo_image_manual_calibration",
            "note": (
                "특정 시연 이미지에서 추정 길이가 실제 공간보다 크게 나타나 "
                "depth map 전체에 scale factor를 적용했습니다. "
                "이 값은 일반 모델 보정값이 아니라 시연 이미지 기준 수동 보정값입니다."
            ),
        },
        "manual_right_edge_override": {
            "enabled": bool(MANUAL_RIGHT_EDGE_OVERRIDE),
            "start_pixel": MANUAL_RIGHT_EDGE_START,
            "end_pixel": MANUAL_RIGHT_EDGE_END,
            "display_length_m": float(MANUAL_RIGHT_EDGE_LENGTH_M),
            "note": (
                "오른쪽 대표선은 자동 바닥-벽 경계선 대신 "
                "책상 오른쪽 다리 바닥점에서 오른쪽 벽 하단점까지 수동 지정했습니다."
            ),
        },
        "raw_depth_statistics_m": raw_depth_stats,
        "calibrated_depth_statistics_m": calibrated_depth_stats,
        "floor_plane_estimation": {
            "plane_normal_camera_coord": [
                float(floor_plane["normal"][0]),
                float(floor_plane["normal"][1]),
                float(floor_plane["normal"][2]),
            ],
            "plane_d": float(floor_plane["d"]),
            "distance_threshold_m": float(floor_plane["distance_threshold_m"]),
            "inlier_count_sampled": int(floor_plane["inlier_count_sampled"]),
        },
        "boundary_postprocess": {
            "mode": "demo_representative_edges_with_manual_right_override",
            "raw_edge_count": int(raw_edge_count) if raw_edge_count is not None else None,
            "selected_edge_count": int(len(edges)),
            "max_selected_edges": int(DEMO_MAX_EDGES),
            "note": (
                "시연 안정성을 위해 중복 후보선을 제거하고 대표 경계선만 저장했습니다. "
                "오른쪽 대표선은 책상 다리 기준으로 수동 보정했습니다. "
                "길이값은 카메라 FOV 가정과 수동 scale factor 기반 추정값입니다."
            ),
        },
        "floor_wall_boundary_edges": edges,
        "llm_placement_context": {
            "summary": (
                "가구 배치 판단에 사용할 수 있는 바닥-벽 대표 경계선과 "
                "각 경계선의 보정된 추정 길이를 제공합니다."
            ),
            "main_use": "LLM 또는 메인 서버가 벽면 기준 가구 배치 가능 영역을 판단할 때 사용",
            "important_warning": (
                "정확한 실측 길이가 필요하면 카메라 intrinsic parameter, 촬영 FOV, "
                "또는 실제 기준 길이 기반 calibration이 필요합니다."
            ),
        },
    }

    return result


# =========================
# 10. main
# =========================

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] 입력 이미지 경로:", INPUT_IMAGE_PATH)
    print("[INFO] 출력 폴더 경로:", OUTPUT_DIR)
    print("[INFO] 출력 JSON 경로:", OUTPUT_JSON_PATH)
    print("[INFO] Depth scale factor:", DEPTH_SCALE_FACTOR)
    print("[INFO] Manual right edge override:", MANUAL_RIGHT_EDGE_OVERRIDE)

    print("[1/6] 이미지 로딩 중...")
    image = load_image_rgb(INPUT_IMAGE_PATH)

    print("[2/6] Depth Anything V2 Metric Indoor 추론 중...")
    depth_m_raw = predict_metric_depth(image, MODEL_ID)

    print("[2-1/6] 시연 이미지 기준 depth scale calibration 적용 중...")
    depth_m_calibrated = apply_depth_scale_calibration(
        depth_m_raw,
        DEPTH_SCALE_FACTOR,
    )

    print("[3/6] 보정된 depth map을 3D point cloud로 변환 중...")
    points_3d, intrinsics = depth_to_point_cloud(
        depth_m_calibrated,
        HORIZONTAL_FOV_DEG,
    )

    print("[4/6] 바닥 평면 추정 중...")
    floor_plane = estimate_floor_plane_ransac(
        points_3d,
        depth_m_calibrated,
    )

    floor_mask = floor_plane["floor_mask"]

    print("[5/6] 바닥-벽 경계 후보선 추출 중...")
    line_segments, boundary_mask = extract_floor_wall_boundary_lines(floor_mask)

    raw_edges = enrich_lines_with_metric_length(
        line_segments,
        points_3d,
    )

    edges = select_demo_boundary_edges(
        raw_edges,
        image_width=image.size[0],
        image_height=image.size[1],
        max_edges=DEMO_MAX_EDGES,
    )

    edges = apply_manual_right_edge_override(
        edges,
        points_3d,
        max_edges=DEMO_MAX_EDGES,
    )

    print(f"[INFO] 원본 후보 경계선 개수: {len(raw_edges)}")
    print(f"[INFO] 시연용 대표 경계선 개수: {len(edges)}")

    print("[6/6] JSON 저장 중...")

    result_json = build_output_json(
        image_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_JSON_PATH,
        model_id=MODEL_ID,
        image=image,
        depth_m_raw=depth_m_raw,
        depth_m_calibrated=depth_m_calibrated,
        intrinsics=intrinsics,
        floor_plane=floor_plane,
        edges=edges,
        raw_edge_count=len(raw_edges),
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    if SAVE_DEBUG_IMAGES:
        save_debug_images(
            image=image,
            depth_m=depth_m_calibrated,
            floor_mask=floor_mask,
            boundary_mask=None,
            edges=edges,
            output_json_path=OUTPUT_JSON_PATH,
        )

    print("[완료] 저장 완료")
    print(f"[완료] 출력 폴더: {OUTPUT_DIR}")
    print(f"[완료] JSON 저장 위치: {OUTPUT_JSON_PATH}")
    print(f"[완료] 원본 후보 경계선 개수: {len(raw_edges)}")
    print(f"[완료] 시연용 대표 경계선 개수: {len(edges)}")

    print("[완료] 생성 파일:")
    print(f"- {OUTPUT_JSON_PATH}")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_depth.png")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_floor_mask.png")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_boundary_overlay.png")

    for edge in edges:
        print(
            f'- {edge["edge_id"]}: '
            f'{edge["length_m_estimated"]:.3f} m, '
            f'pixel {edge["pixel_start"]} -> {edge["pixel_end"]}'
        )


if __name__ == "__main__":
    main()