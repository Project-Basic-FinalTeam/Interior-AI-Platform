"""
실내 이미지 기반 Depth 추정 및 바닥-벽 경계 후보선 추출 코드
- 26.05.29 공유 이미지 기준 후보정(post-processing) 포함 버전

이 코드는 단일 실내 이미지를 입력으로 받아 Depth Anything V2 Metric Indoor 모델을 이용해
픽셀 단위의 metric depth map(미터 단위 깊이 지도)을 생성한다. 이후 depth map을 카메라 좌표계
기준의 3D point cloud(포인트 클라우드)로 변환하고, 이미지 하단부의 점들을 기반으로 RANSAC
알고리즘을 사용해 바닥 plane(평면)을 추정한다.

후보정 포함 사항:
1. floor mask(바닥 마스크) 추가 정리
2. 선택적 object exclusion mask(객체 제외 마스크) 반영
3. 바닥-벽 경계 후보선 중 수평선만 필터링
4. 유사한 y 좌표와 x 구간을 가진 중복 edge(경계선) 병합
5. 필요 시 길이 scale factor(스케일 계수) 보정

주의:
- 카메라의 실제 intrinsic parameter(내부 파라미터)를 알 수 없기 때문에,
  horizontal FOV 값을 가정하여 3D 좌표와 길이를 계산한다.
- 따라서 출력되는 길이값은 실측값이 아니라 추정값이다.
- 정확한 실측 길이가 필요한 경우, 카메라 calibration(보정) 또는 기준 객체 기반 scale 보정이 필요하다.
"""

import json
import math
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

INPUT_IMAGE_PATH = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Interior-AI-Platform\depth\base_test\test_11.jpg")

# 여기 숫자만 그때그때 바꾸면 됩니다.
# 예: 1 -> result_1, 2 -> result_2, 11 -> result_11
RESULT_NUMBER = '11_1'

BASE_OUTPUT_DIR = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Interior-AI-Platform\depth\base_test")
RESULT_NAME = f"result_{RESULT_NUMBER}"

# 최종 output(출력) 폴더
OUTPUT_DIR = BASE_OUTPUT_DIR / RESULT_NAME

# 최종 JSON(제이슨) 저장 경로
OUTPUT_JSON_PATH = OUTPUT_DIR / f"{RESULT_NAME}.json"

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# 카메라 시야각을 모르면 길이 계산이 추정값이 됩니다.
# 스마트폰 일반 광각 사진 기준으로 65~75도 정도를 1차 가정값으로 둘 수 있습니다.
HORIZONTAL_FOV_DEG = 70.0

# 디버그 이미지 저장 여부
SAVE_DEBUG_IMAGES = True


# =========================
# 1-1. 후보정 설정값
# =========================

USE_DEMO_POSTPROCESSING = True

# 실제 기준 길이가 없으면 1.0 유지하세요.
# 예: 수납장 실제 너비가 1.60m인데 코드가 1.45m로 추정했다면 1.60 / 1.45 입력
LENGTH_SCALE_FACTOR = 1.0

# 수평선으로 인정할 기울기 허용치
# dy / dx 값이 이 값보다 작아야 수평선으로 인정합니다.
HORIZONTAL_SLOPE_TOL = 0.03

# 바닥-벽 경계선이 나올 수 있는 y 위치 범위
# test_11 기준 경계는 이미지 높이의 약 70% 부근입니다.
BOUNDARY_Y_MIN_RATIO = 0.55
BOUNDARY_Y_MAX_RATIO = 0.82

# 너무 짧은 후보선 제거 기준
MIN_BOUNDARY_PIXEL_LENGTH = 200

# 비슷한 y 좌표의 중복 선분을 병합할 때 허용할 y 차이
MERGE_Y_TOLERANCE_PX = 8

# 같은 선으로 볼 수 있는 x 방향 간격
MERGE_X_GAP_TOLERANCE_PX = 80

# 최종적으로 남길 최대 바닥-벽 경계 후보선 개수
# test_11처럼 가구가 경계를 가리는 경우 2~3개 정도가 적절합니다.
MAX_BOUNDARY_SEGMENTS = 3

# floor mask 후보정 시 상단 제거 비율
POSTPROCESS_FLOOR_UPPER_CUT_RATIO = 0.45

# 객체 제외 마스크가 있다면 경로를 넣으세요.
# YOLO/SAM 결과로 만든 객체 mask가 있을 때 사용합니다.
# 흰색/양수 픽셀 영역을 바닥 후보에서 제거합니다.
# 예: Path(r"C:\...\object_mask.png")
OBJECT_EXCLUSION_MASK_PATH = None

# 객체 제외 마스크를 조금 넓혀서 가구 가장자리 오탐을 줄이는 값
OBJECT_EXCLUSION_DILATE_KERNEL = 21


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

    # 원본 이미지 크기로 resize
    width, height = image.size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.detach().cpu().numpy().astype(np.float32)

    # 비정상값 제거
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np[depth_np < 0] = 0

    return depth_np


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
# 5. 바닥 plane(평면) RANSAC 추정
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

    # 바닥은 보통 이미지 하단부에 많이 보이므로 하단 55% 영역을 후보로 사용
    y_start = int(height * 0.45)

    valid_mask = depth_m > 0.05
    bottom_mask = np.zeros_like(valid_mask, dtype=bool)
    bottom_mask[y_start:, :] = True

    candidate_mask = valid_mask & bottom_mask
    candidate_points = points_3d[candidate_mask]

    if len(candidate_points) < 500:
        raise RuntimeError("바닥 후보 점이 너무 적습니다. 이미지에 바닥이 충분히 보이는지 확인하세요.")

    # 연산량 제한을 위한 샘플링
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

    all_dist = plane_distance(points_3d.reshape(-1, 3), best_normal, best_d).reshape(height, width)
    floor_mask = (all_dist < distance_threshold) & valid_mask

    # 이미지 상단의 잘못된 바닥 후보 제거
    upper_cut = int(height * 0.30)
    floor_mask[:upper_cut, :] = False

    # morphology(형태학 연산)로 마스크 정리
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
# 5-1. 후보정: 객체 제외 마스크 로딩
# =========================

def load_optional_object_exclusion_mask(mask_path, image_size):
    """
    YOLO/SAM 등에서 얻은 객체 mask가 있을 때 바닥 후보에서 제외하기 위한 함수.
    mask_path가 None이면 사용하지 않는다.

    image_size: PIL image.size 형식의 (width, height)
    """
    if mask_path is None:
        return None

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"객체 제외 마스크가 존재하지 않습니다: {mask_path}")

    width, height = image_size
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"객체 제외 마스크를 읽을 수 없습니다: {mask_path}")

    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    object_mask = mask > 0

    # 객체 가장자리까지 넉넉하게 제외
    if OBJECT_EXCLUSION_DILATE_KERNEL > 1:
        k = OBJECT_EXCLUSION_DILATE_KERNEL
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        object_mask_uint8 = object_mask.astype(np.uint8) * 255
        object_mask_uint8 = cv2.dilate(object_mask_uint8, kernel, iterations=1)
        object_mask = object_mask_uint8 > 0

    return object_mask


# =========================
# 5-2. 후보정: floor mask 정리
# =========================

def postprocess_floor_mask_for_demo(floor_mask: np.ndarray, object_exclusion_mask=None) -> np.ndarray:
    """
    시연용 floor mask(바닥 마스크) 후보정 함수.
    모델 추론 결과 자체는 유지하고, 바닥 영역 마스크만 후처리한다.
    """
    height, width = floor_mask.shape

    mask = floor_mask.astype(np.uint8) * 255

    # 너무 위쪽에서 잡힌 바닥 후보 제거
    # test_11에서는 바닥-벽 경계가 이미지 높이의 약 70% 부근이므로
    # 상단 45% 정도는 바닥 후보에서 제외해도 안전함
    mask[:int(height * POSTPROCESS_FLOOR_UPPER_CUT_RATIO), :] = 0

    # 작은 노이즈 제거
    open_kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # 끊긴 바닥 영역 연결
    close_kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # 객체 영역 제외는 마지막에 적용한다.
    # 앞에서 적용하면 close 연산이 객체 영역을 다시 메울 수 있다.
    if object_exclusion_mask is not None:
        if object_exclusion_mask.shape != mask.shape:
            raise ValueError("object_exclusion_mask와 floor_mask의 크기가 다릅니다.")
        mask[object_exclusion_mask] = 0

    return mask > 0


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

        # 해당 column(열)에서 가장 위에 있는 바닥 픽셀을 바닥-벽 경계 후보로 사용
        y_top = int(np.min(ys))

        # 너무 위쪽 값은 제외
        if y_top < int(height * 0.25):
            continue

        boundary_points.append((x, y_top))

    if len(boundary_points) < width * 0.15:
        return [], None

    boundary_points = np.array(boundary_points, dtype=np.int32)

    # 1차 smoothing(평활화)
    xs = boundary_points[:, 0]
    ys = boundary_points[:, 1]

    # 빠진 x는 interpolation(보간)
    full_xs = np.arange(xs.min(), xs.max() + 1)
    full_ys = np.interp(full_xs, xs, ys)

    # 이동 평균 smoothing(평활화)
    window = max(9, width // 80)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    smooth_ys = np.convolve(full_ys, kernel, mode="same")

    boundary_mask = np.zeros((height, width), dtype=np.uint8)

    for x, y in zip(full_xs, smooth_ys):
        y_int = int(np.clip(round(y), 0, height - 1))
        boundary_mask[y_int, x] = 255

    # 선을 조금 두껍게 만들어 HoughLinesP가 잡기 쉽게 처리
    boundary_mask = cv2.dilate(boundary_mask, np.ones((3, 3), np.uint8), iterations=1)

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

    # Hough가 실패하면 전체 boundary(경계)를 하나의 선분으로 fallback(대체)
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
# 6-1. 후보정: 경계선 필터링 및 병합
# =========================

def normalize_line_segment(line):
    """
    x1이 x2보다 크면 좌우를 정렬한다.
    """
    x1, y1 = line["x1"], line["y1"]
    x2, y2 = line["x2"], line["y2"]

    if x1 <= x2:
        return dict(line)

    return {
        "x1": x2,
        "y1": y2,
        "x2": x1,
        "y2": y1,
        "pixel_length": float(line["pixel_length"]),
    }


def is_valid_horizontal_boundary(line, image_height: int) -> bool:
    """
    시연용으로 사용할 수 있는 수평 바닥-벽 경계 후보선만 남긴다.
    """
    x1, y1 = line["x1"], line["y1"]
    x2, y2 = line["x2"], line["y2"]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx < MIN_BOUNDARY_PIXEL_LENGTH:
        return False

    slope = dy / (dx + 1e-6)
    if slope > HORIZONTAL_SLOPE_TOL:
        return False

    y_mean = (y1 + y2) / 2.0
    y_min = image_height * BOUNDARY_Y_MIN_RATIO
    y_max = image_height * BOUNDARY_Y_MAX_RATIO

    if not (y_min <= y_mean <= y_max):
        return False

    return True


def postprocess_boundary_lines(line_segments, image_width: int, image_height: int):
    """
    경계 후보선 후보정 함수.

    처리 내용:
    1. 좌우 방향 정렬
    2. 너무 짧은 선 제거
    3. 기울어진 선 제거
    4. y 위치가 비슷하고 x 구간이 겹치거나 가까운 선 병합
    5. 긴 선 우선으로 최종 후보 개수 제한
    """
    if not line_segments:
        return []

    valid_lines = []

    for line in line_segments:
        line = normalize_line_segment(line)

        if is_valid_horizontal_boundary(line, image_height):
            valid_lines.append(line)

    if not valid_lines:
        return []

    # y 위치와 x 시작점 기준 정렬
    valid_lines.sort(
        key=lambda l: (
            int(((l["y1"] + l["y2"]) / 2) // MERGE_Y_TOLERANCE_PX),
            l["x1"],
        )
    )

    clusters = []

    for line in valid_lines:
        x1, y1 = line["x1"], line["y1"]
        x2, y2 = line["x2"], line["y2"]
        y_mean = (y1 + y2) / 2.0

        merged = False

        for cluster in clusters:
            cluster_y = cluster["y_mean"]
            cluster_x1 = cluster["x1"]
            cluster_x2 = cluster["x2"]

            similar_y = abs(y_mean - cluster_y) <= MERGE_Y_TOLERANCE_PX
            close_or_overlap_x = (
                x1 <= cluster_x2 + MERGE_X_GAP_TOLERANCE_PX
                and x2 >= cluster_x1 - MERGE_X_GAP_TOLERANCE_PX
            )

            if similar_y and close_or_overlap_x:
                cluster["x1"] = min(cluster["x1"], x1)
                cluster["x2"] = max(cluster["x2"], x2)
                cluster["ys"].append(y_mean)
                cluster["source_count"] += 1
                cluster["y_mean"] = float(np.median(cluster["ys"]))
                merged = True
                break

        if not merged:
            clusters.append({
                "x1": x1,
                "x2": x2,
                "ys": [y_mean],
                "y_mean": y_mean,
                "source_count": 1,
            })

    merged_lines = []

    for cluster in clusters:
        x1 = int(cluster["x1"])
        x2 = int(cluster["x2"])
        y = int(round(cluster["y_mean"]))

        pixel_length = float(abs(x2 - x1))

        if pixel_length < MIN_BOUNDARY_PIXEL_LENGTH:
            continue

        merged_lines.append({
            "x1": x1,
            "y1": y,
            "x2": x2,
            "y2": y,
            "pixel_length": pixel_length,
            "postprocess_source_count": int(cluster["source_count"]),
            "postprocess_note": "수평선 필터링 및 유사 경계선 병합 후 생성된 후보선입니다.",
        })

    # 긴 선 우선 정렬 후 개수 제한
    merged_lines.sort(key=lambda l: l["pixel_length"], reverse=True)

    if MAX_BOUNDARY_SEGMENTS is not None and MAX_BOUNDARY_SEGMENTS > 0:
        merged_lines = merged_lines[:MAX_BOUNDARY_SEGMENTS]

    return merged_lines


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

        # 선 위의 평균 depth(깊이) 계산
        sample_count = 30
        xs = np.linspace(x1, x2, sample_count).astype(int)
        ys = np.linspace(y1, y2, sample_count).astype(int)

        depth_values = []
        for x, y in zip(xs, ys):
            z = points_3d[y, x, 2]
            if z > 0.05:
                depth_values.append(float(z))

        mean_depth_m = float(np.mean(depth_values)) if depth_values else None

        edge = {
            "edge_id": f"floor_wall_edge_{i}",
            "type": "floor_wall_boundary_candidate",
            "pixel_start": {"x": int(x1), "y": int(y1)},
            "pixel_end": {"x": int(x2), "y": int(y2)},
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
            "confidence_note": "카메라 FOV 또는 내부파라미터가 없으므로 길이는 추정값입니다.",
        }

        # 후보정 관련 부가 정보가 있으면 JSON에 같이 저장
        if "postprocess_source_count" in line:
            edge["postprocess_source_count"] = int(line["postprocess_source_count"])
        if "postprocess_note" in line:
            edge["postprocess_note"] = str(line["postprocess_note"])

        enriched.append(edge)

    return enriched


# =========================
# 7-1. 후보정: 길이 scale factor 보정
# =========================

def apply_length_scale_correction(edges):
    """
    FOV(시야각) 가정 기반 길이에 scale factor(스케일 계수)를 적용한다.
    실제 기준 길이가 없으면 LENGTH_SCALE_FACTOR = 1.0 유지.
    """
    corrected_edges = []

    for edge in edges:
        edge = dict(edge)

        raw_length = float(edge["length_m_estimated"])
        corrected_length = raw_length * float(LENGTH_SCALE_FACTOR)

        edge["length_m_raw_fov_based"] = raw_length
        edge["length_m_estimated"] = corrected_length
        edge["length_scale_factor"] = float(LENGTH_SCALE_FACTOR)

        if LENGTH_SCALE_FACTOR == 1.0:
            edge["calibration_note"] = "별도 스케일 보정 없이 FOV 가정 기반 추정 길이를 사용했습니다."
        else:
            edge["calibration_note"] = "시연용 기준 길이를 이용해 scale factor를 적용했습니다."

        corrected_edges.append(edge)

    return corrected_edges


# =========================
# 8. 디버그 이미지 저장
# =========================

def save_raw_boundary_overlay(image: Image.Image, raw_line_segments, output_json_path: Path):
    """
    후보정 전 raw boundary line(원본 경계 후보선)을 별도 이미지로 저장한다.
    """
    if raw_line_segments is None:
        return

    base = output_json_path.with_suffix("")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    raw_vis = image_bgr.copy()

    for line in raw_line_segments:
        x1, y1 = int(line["x1"]), int(line["y1"])
        x2, y2 = int(line["x2"]), int(line["y2"])
        cv2.line(raw_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(str(base) + "_boundary_overlay_raw.png", raw_vis)


def save_debug_images(
    image: Image.Image,
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    boundary_mask,
    edges,
    output_json_path: Path,
    raw_line_segments=None,
):
    # 예:
    # output_json_path = ...\base_test\result_1\result_1.json
    # base = ...\base_test\result_1\result_1
    base = output_json_path.with_suffix("")

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # depth(깊이) 시각화
    valid = depth_m > 0.05
    depth_vis = np.zeros_like(depth_m, dtype=np.uint8)

    if np.any(valid):
        d_min = np.percentile(depth_m[valid], 2)
        d_max = np.percentile(depth_m[valid], 98)
        depth_norm = np.clip((depth_m - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(base) + "_depth.png", depth_colormap)

    # floor mask(바닥 마스크)
    floor_vis = image_bgr.copy()
    floor_overlay = np.zeros_like(image_bgr)
    floor_overlay[floor_mask] = (0, 255, 0)
    floor_vis = cv2.addWeighted(floor_vis, 0.75, floor_overlay, 0.25, 0)
    cv2.imwrite(str(base) + "_floor_mask.png", floor_vis)

    # raw boundary overlay(후보정 전 경계 후보선)
    save_raw_boundary_overlay(image, raw_line_segments, output_json_path)

    # boundary overlay(최종 경계 오버레이)
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

        label = f'{edge["length_m_estimated"]:.2f} m'
        cv2.putText(
            boundary_vis,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(base) + "_boundary_overlay.png", boundary_vis)


# =========================
# 9. JSON 저장
# =========================

def build_output_json(
    image_path,
    output_path,
    model_id,
    image,
    depth_m,
    intrinsics,
    floor_plane,
    edges,
    postprocess_info=None,
):
    width, height = image.size
    valid_depth = depth_m[depth_m > 0.05]

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
            "note": "카메라 내부파라미터가 없어서 horizontal_fov_deg 기반으로 3D 좌표와 길이를 추정했습니다.",
            "intrinsics": intrinsics,
        },
        "depth_statistics_m": depth_stats,
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
        "postprocessing": postprocess_info,
        "floor_wall_boundary_edges": edges,
        "llm_placement_context": {
            "summary": "가구 배치 판단에 사용할 수 있는 바닥-벽 경계 후보선과 각 후보선의 추정 길이를 제공합니다.",
            "main_use": "LLM 또는 메인 서버가 벽면 기준 가구 배치 가능 영역을 판단할 때 사용",
            "important_warning": "정확한 실측 길이가 필요하면 카메라 intrinsic parameter 또는 촬영 기기의 FOV가 필요합니다.",
        },
    }

    return result


# =========================
# 10. main
# =========================

def main():
    # result_1, result_2 같은 output(출력) 폴더 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] 입력 이미지 경로:", INPUT_IMAGE_PATH)
    print("[INFO] 출력 폴더 경로:", OUTPUT_DIR)
    print("[INFO] 출력 JSON 경로:", OUTPUT_JSON_PATH)

    print("[1/6] 이미지 로딩 중...")
    image = load_image_rgb(INPUT_IMAGE_PATH)

    print("[2/6] Depth Anything V2 Metric Indoor 추론 중...")
    depth_m = predict_metric_depth(image, MODEL_ID)

    print("[3/6] depth map을 3D point cloud로 변환 중...")
    points_3d, intrinsics = depth_to_point_cloud(depth_m, HORIZONTAL_FOV_DEG)

    print("[4/6] 바닥 평면 추정 중...")
    floor_plane = estimate_floor_plane_ransac(points_3d, depth_m)
    floor_mask_raw = floor_plane["floor_mask"]

    print("[5/6] 바닥-벽 경계 후보선 추출 및 후보정 중...")

    object_exclusion_mask = load_optional_object_exclusion_mask(
        OBJECT_EXCLUSION_MASK_PATH,
        image_size=image.size,
    )

    if USE_DEMO_POSTPROCESSING:
        floor_mask = postprocess_floor_mask_for_demo(
            floor_mask_raw,
            object_exclusion_mask=object_exclusion_mask,
        )
    else:
        floor_mask = floor_mask_raw

    # JSON 및 debug image에도 후보정된 floor_mask가 반영되도록 갱신
    floor_plane["floor_mask"] = floor_mask

    # 1) raw boundary line 추출
    raw_line_segments, boundary_mask = extract_floor_wall_boundary_lines(floor_mask)

    # 2) boundary line 후보정
    if USE_DEMO_POSTPROCESSING:
        line_segments = postprocess_boundary_lines(
            line_segments=raw_line_segments,
            image_width=image.size[0],
            image_height=image.size[1],
        )

        if len(line_segments) == 0 and len(raw_line_segments) > 0:
            print("[WARN] 후보정 후 남은 경계선이 없어 raw 후보선을 사용합니다.")
            line_segments = raw_line_segments
    else:
        line_segments = raw_line_segments

    # 3) 3D 길이 계산
    edges = enrich_lines_with_metric_length(line_segments, points_3d)

    # 4) scale factor 적용
    if USE_DEMO_POSTPROCESSING:
        edges = apply_length_scale_correction(edges)

    postprocess_info = {
        "enabled": bool(USE_DEMO_POSTPROCESSING),
        "floor_mask_postprocessed": bool(USE_DEMO_POSTPROCESSING),
        "object_exclusion_mask_used": OBJECT_EXCLUSION_MASK_PATH is not None,
        "object_exclusion_mask_path": str(OBJECT_EXCLUSION_MASK_PATH) if OBJECT_EXCLUSION_MASK_PATH is not None else None,
        "raw_boundary_line_count": int(len(raw_line_segments)),
        "final_boundary_line_count": int(len(edges)),
        "horizontal_slope_tolerance": float(HORIZONTAL_SLOPE_TOL),
        "boundary_y_min_ratio": float(BOUNDARY_Y_MIN_RATIO),
        "boundary_y_max_ratio": float(BOUNDARY_Y_MAX_RATIO),
        "min_boundary_pixel_length": int(MIN_BOUNDARY_PIXEL_LENGTH),
        "merge_y_tolerance_px": int(MERGE_Y_TOLERANCE_PX),
        "merge_x_gap_tolerance_px": int(MERGE_X_GAP_TOLERANCE_PX),
        "max_boundary_segments": int(MAX_BOUNDARY_SEGMENTS) if MAX_BOUNDARY_SEGMENTS is not None else None,
        "length_scale_factor": float(LENGTH_SCALE_FACTOR),
        "note": "후보정은 모델 재학습이 아니라 floor mask와 boundary line에 대한 후처리입니다.",
    }

    print("[6/6] JSON 저장 중...")
    result_json = build_output_json(
        image_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_JSON_PATH,
        model_id=MODEL_ID,
        image=image,
        depth_m=depth_m,
        intrinsics=intrinsics,
        floor_plane=floor_plane,
        edges=edges,
        postprocess_info=postprocess_info,
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    if SAVE_DEBUG_IMAGES:
        save_debug_images(
            image=image,
            depth_m=depth_m,
            floor_mask=floor_mask,
            boundary_mask=boundary_mask,
            edges=edges,
            output_json_path=OUTPUT_JSON_PATH,
            raw_line_segments=raw_line_segments,
        )

    print("[완료] 저장 완료")
    print(f"[완료] 출력 폴더: {OUTPUT_DIR}")
    print(f"[완료] JSON 저장 위치: {OUTPUT_JSON_PATH}")
    print(f"[완료] raw 경계 후보선 개수: {len(raw_line_segments)}")
    print(f"[완료] 후보정 후 경계 후보선 개수: {len(edges)}")

    print("[완료] 생성 파일:")
    print(f"- {OUTPUT_JSON_PATH}")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_depth.png")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_floor_mask.png")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_boundary_overlay.png")
    print(f"- {OUTPUT_JSON_PATH.with_suffix('')}_boundary_overlay_raw.png")

    for edge in edges:
        print(
            f'- {edge["edge_id"]}: '
            f'{edge["length_m_estimated"]:.3f} m, '
            f'pixel {edge["pixel_start"]} -> {edge["pixel_end"]}'
        )


if __name__ == "__main__":
    main()
