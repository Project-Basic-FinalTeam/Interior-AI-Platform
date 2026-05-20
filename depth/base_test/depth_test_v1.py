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

INPUT_IMAGE_PATH = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Interior-AI-Platform\shared\models\thumb-1920-1174099.jpg")

# 여기 숫자만 그때그때 바꾸면 됩니다.
# 예: 1 -> result_1, 2 -> result_2, 3 -> result_3
RESULT_NUMBER = 5

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

        enriched.append({
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
        })

    return enriched


# =========================
# 8. 디버그 이미지 저장
# =========================

def save_debug_images(
    image: Image.Image,
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    boundary_mask,
    edges,
    output_json_path: Path,
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

    # boundary overlay(경계 오버레이)
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
    floor_mask = floor_plane["floor_mask"]

    print("[5/6] 바닥-벽 경계 후보선 추출 중...")
    line_segments, boundary_mask = extract_floor_wall_boundary_lines(floor_mask)
    edges = enrich_lines_with_metric_length(line_segments, points_3d)

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
        )

    print("[완료] 저장 완료")
    print(f"[완료] 출력 폴더: {OUTPUT_DIR}")
    print(f"[완료] JSON 저장 위치: {OUTPUT_JSON_PATH}")
    print(f"[완료] 추출된 바닥-벽 경계 후보선 개수: {len(edges)}")

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