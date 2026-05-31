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

# 파일 위치: /InteriorPlatform_Workspace/services/ai-perception/modules/
# 파일 명: depth_handler.py

import os
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
# 1. 설정값 (팀원 원본 로직 100% 유지)
# =========================
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
HORIZONTAL_FOV_DEG = 70.0
SAVE_DEBUG_IMAGES = True
USE_DEMO_POSTPROCESSING = True
LENGTH_SCALE_FACTOR = 1.0
HORIZONTAL_SLOPE_TOL = 0.03
BOUNDARY_Y_MIN_RATIO = 0.55
BOUNDARY_Y_MAX_RATIO = 0.82
MIN_BOUNDARY_PIXEL_LENGTH = 200
MERGE_Y_TOLERANCE_PX = 8
MERGE_X_GAP_TOLERANCE_PX = 80
MAX_BOUNDARY_SEGMENTS = 3
POSTPROCESS_FLOOR_UPPER_CUT_RATIO = 0.45
OBJECT_EXCLUSION_MASK_PATH = None
OBJECT_EXCLUSION_DILATE_KERNEL = 21

# =========================
# 팀원 작성 유틸리티 함수 모음 (로직 원본 그대로 유지)
# =========================
def load_image_rgb(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return Image.open(image_path).convert("RGB")

def predict_metric_depth(image: Image.Image, processor, model, device) -> np.ndarray:
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
    return points_3d, {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "horizontal_fov_deg": horizontal_fov_deg}

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
        raise RuntimeError("Not enough floor candidate points.")

    max_sample_points = 12000
    if len(candidate_points) > max_sample_points:
        idx = np.random.choice(len(candidate_points), max_sample_points, replace=False)
        sampled_points = candidate_points[idx]
    else:
        sampled_points = candidate_points

    best_normal, best_d, best_inlier_count = None, None, 0
    median_depth = float(np.median(depth_m[valid_mask]))
    distance_threshold = max(0.04, median_depth * 0.015)
    iterations = 600

    for _ in range(iterations):
        ids = np.random.choice(len(sampled_points), 3, replace=False)
        plane = fit_plane_from_3_points(sampled_points[ids[0]], sampled_points[ids[1]], sampled_points[ids[2]])
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
        raise RuntimeError("Failed to estimate floor plane.")

    all_dist = plane_distance(points_3d.reshape(-1, 3), best_normal, best_d).reshape(height, width)
    floor_mask = (all_dist < distance_threshold) & valid_mask
    upper_cut = int(height * 0.30)
    floor_mask[:upper_cut, :] = False
    
    floor_mask_uint8 = floor_mask.astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    floor_mask_uint8 = cv2.morphologyEx(floor_mask_uint8, cv2.MORPH_OPEN, kernel)
    floor_mask_uint8 = cv2.morphologyEx(floor_mask_uint8, cv2.MORPH_CLOSE, kernel)
    floor_mask = floor_mask_uint8 > 0

    return {"normal": best_normal, "d": float(best_d), "distance_threshold_m": float(distance_threshold), "inlier_count_sampled": int(best_inlier_count), "floor_mask": floor_mask}

def load_optional_object_exclusion_mask(mask_path, image_size):
    if mask_path is None: return None
    mask_path = Path(mask_path)
    if not mask_path.exists(): return None
    width, height = image_size
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    object_mask = mask > 0
    if OBJECT_EXCLUSION_DILATE_KERNEL > 1:
        k = OBJECT_EXCLUSION_DILATE_KERNEL
        k = k + 1 if k % 2 == 0 else k
        kernel = np.ones((k, k), np.uint8)
        object_mask_uint8 = object_mask.astype(np.uint8) * 255
        object_mask_uint8 = cv2.dilate(object_mask_uint8, kernel, iterations=1)
        object_mask = object_mask_uint8 > 0
    return object_mask

def postprocess_floor_mask_for_demo(floor_mask: np.ndarray, object_exclusion_mask=None) -> np.ndarray:
    height, width = floor_mask.shape
    mask = floor_mask.astype(np.uint8) * 255
    mask[:int(height * POSTPROCESS_FLOOR_UPPER_CUT_RATIO), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    if object_exclusion_mask is not None:
        mask[object_exclusion_mask] = 0
    return mask > 0

def extract_floor_wall_boundary_lines(floor_mask: np.ndarray):
    height, width = floor_mask.shape
    boundary_points = []
    for x in range(width):
        ys = np.where(floor_mask[:, x])[0]
        if len(ys) == 0: continue
        y_top = int(np.min(ys))
        if y_top < int(height * 0.25): continue
        boundary_points.append((x, y_top))
    
    if len(boundary_points) < width * 0.15: return [], None
    boundary_points = np.array(boundary_points, dtype=np.int32)
    xs, ys = boundary_points[:, 0], boundary_points[:, 1]
    full_xs = np.arange(xs.min(), xs.max() + 1)
    full_ys = np.interp(full_xs, xs, ys)
    window = max(9, width // 80)
    window = window + 1 if window % 2 == 0 else window
    kernel = np.ones(window) / window
    smooth_ys = np.convolve(full_ys, kernel, mode="same")
    
    boundary_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in zip(full_xs, smooth_ys):
        y_int = int(np.clip(round(y), 0, height - 1))
        boundary_mask[y_int, x] = 255
    
    boundary_mask = cv2.dilate(boundary_mask, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(boundary_mask, rho=1, theta=np.pi/180, threshold=40, minLineLength=max(50, width//6), maxLineGap=max(20, width//25))
    line_segments = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0].tolist()
            pixel_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            line_segments.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "pixel_length": float(pixel_length)})
    
    if len(line_segments) == 0:
        x1, y1 = int(full_xs[0]), int(smooth_ys[0])
        x2, y2 = int(full_xs[-1]), int(smooth_ys[-1])
        pixel_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "pixel_length": float(pixel_length)})
    
    return line_segments, boundary_mask

def normalize_line_segment(line):
    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    if x1 <= x2: return dict(line)
    return {"x1": x2, "y1": y2, "x2": x1, "y2": y1, "pixel_length": float(line["pixel_length"])}

def is_valid_horizontal_boundary(line, image_height: int) -> bool:
    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx < MIN_BOUNDARY_PIXEL_LENGTH: return False
    if dy / (dx + 1e-6) > HORIZONTAL_SLOPE_TOL: return False
    y_mean = (y1 + y2) / 2.0
    if not (image_height * BOUNDARY_Y_MIN_RATIO <= y_mean <= image_height * BOUNDARY_Y_MAX_RATIO): return False
    return True

def postprocess_boundary_lines(line_segments, image_width: int, image_height: int):
    if not line_segments: return []
    valid_lines = [line for line in [normalize_line_segment(l) for l in line_segments] if is_valid_horizontal_boundary(line, image_height)]
    if not valid_lines: return []
    
    valid_lines.sort(key=lambda l: (int(((l["y1"] + l["y2"]) / 2) // MERGE_Y_TOLERANCE_PX), l["x1"]))
    clusters = []
    for line in valid_lines:
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        y_mean = (y1 + y2) / 2.0
        merged = False
        for cluster in clusters:
            if abs(y_mean - cluster["y_mean"]) <= MERGE_Y_TOLERANCE_PX and (x1 <= cluster["x2"] + MERGE_X_GAP_TOLERANCE_PX and x2 >= cluster["x1"] - MERGE_X_GAP_TOLERANCE_PX):
                cluster["x1"], cluster["x2"] = min(cluster["x1"], x1), max(cluster["x2"], x2)
                cluster["ys"].append(y_mean)
                cluster["source_count"] += 1
                cluster["y_mean"] = float(np.median(cluster["ys"]))
                merged = True
                break
        if not merged:
            clusters.append({"x1": x1, "x2": x2, "ys": [y_mean], "y_mean": y_mean, "source_count": 1})
            
    merged_lines = []
    for cluster in clusters:
        x1, x2, y = int(cluster["x1"]), int(cluster["x2"]), int(round(cluster["y_mean"]))
        pixel_length = float(abs(x2 - x1))
        if pixel_length < MIN_BOUNDARY_PIXEL_LENGTH: continue
        merged_lines.append({"x1": x1, "y1": y, "x2": x2, "y2": y, "pixel_length": pixel_length, "postprocess_source_count": int(cluster["source_count"]), "postprocess_note": "Filtered and merged horizontal lines."})
        
    merged_lines.sort(key=lambda l: l["pixel_length"], reverse=True)
    if MAX_BOUNDARY_SEGMENTS is not None and MAX_BOUNDARY_SEGMENTS > 0:
        merged_lines = merged_lines[:MAX_BOUNDARY_SEGMENTS]
    return merged_lines

def get_point_3d_at_pixel(points_3d: np.ndarray, x: int, y: int, patch_size: int = 5):
    height, width, _ = points_3d.shape
    half = patch_size // 2
    x1, x2 = max(0, x - half), min(width, x + half + 1)
    y1, y2 = max(0, y - half), min(height, y + half + 1)
    patch = points_3d[y1:y2, x1:x2].reshape(-1, 3)
    valid = patch[:, 2] > 0.05
    patch = patch[valid]
    if len(patch) == 0: return points_3d[y, x].astype(float)
    return np.median(patch, axis=0).astype(float)

def enrich_lines_with_metric_length(line_segments, points_3d: np.ndarray):
    enriched = []
    for i, line in enumerate(line_segments, start=1):
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        p1 = get_point_3d_at_pixel(points_3d, x1, y1)
        p2 = get_point_3d_at_pixel(points_3d, x2, y2)
        length_m = float(np.linalg.norm(p2 - p1))
        
        xs, ys = np.linspace(x1, x2, 30).astype(int), np.linspace(y1, y2, 30).astype(int)
        depth_values = [float(points_3d[y, x, 2]) for x, y in zip(xs, ys) if points_3d[y, x, 2] > 0.05]
        mean_depth_m = float(np.mean(depth_values)) if depth_values else None
        
        edge = {
            "edge_id": f"floor_wall_edge_{i}", "type": "floor_wall_boundary_candidate",
            "pixel_start": {"x": int(x1), "y": int(y1)}, "pixel_end": {"x": int(x2), "y": int(y2)},
            "point_3d_start_m": {"x": float(p1[0]), "y": float(p1[1]), "z": float(p1[2])},
            "point_3d_end_m": {"x": float(p2[0]), "y": float(p2[1]), "z": float(p2[2])},
            "length_m_estimated": length_m, "pixel_length": float(line["pixel_length"]),
            "mean_depth_m": mean_depth_m, "confidence_note": "Estimated length without camera intrinsic parameters."
        }
        if "postprocess_source_count" in line: edge["postprocess_source_count"] = int(line["postprocess_source_count"])
        if "postprocess_note" in line: edge["postprocess_note"] = str(line["postprocess_note"])
        enriched.append(edge)
    return enriched

def apply_length_scale_correction(edges):
    corrected_edges = []
    for edge in edges:
        edge = dict(edge)
        raw_length = float(edge["length_m_estimated"])
        corrected_length = raw_length * float(LENGTH_SCALE_FACTOR)
        edge["length_m_raw_fov_based"] = raw_length
        edge["length_m_estimated"] = corrected_length
        edge["length_scale_factor"] = float(LENGTH_SCALE_FACTOR)
        edge["calibration_note"] = "Used FOV assumption without extra scale correction." if LENGTH_SCALE_FACTOR == 1.0 else "Applied scale factor based on demo reference length."
        corrected_edges.append(edge)
    return corrected_edges

def save_raw_boundary_overlay(image: Image.Image, raw_line_segments, output_json_path: Path):
    if raw_line_segments is None: return
    base = output_json_path.with_suffix("")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    raw_vis = image_bgr.copy()
    for line in raw_line_segments:
        cv2.line(raw_vis, (int(line["x1"]), int(line["y1"])), (int(line["x2"]), int(line["y2"])), (0, 0, 255), 2)
    cv2.imwrite(str(base) + "_boundary_overlay_raw.png", raw_vis)

def save_debug_images(image, depth_m, floor_mask, boundary_mask, edges, output_json_path: Path, raw_line_segments=None):
    base = output_json_path.with_suffix("")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    valid = depth_m > 0.05
    depth_vis = np.zeros_like(depth_m, dtype=np.uint8)
    if np.any(valid):
        d_min, d_max = np.percentile(depth_m[valid], 2), np.percentile(depth_m[valid], 98)
        depth_norm = np.clip((depth_m - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(str(base) + "_depth.png", cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
    
    floor_vis, floor_overlay = image_bgr.copy(), np.zeros_like(image_bgr)
    floor_overlay[floor_mask] = (0, 255, 0)
    cv2.imwrite(str(base) + "_floor_mask.png", cv2.addWeighted(floor_vis, 0.75, floor_overlay, 0.25, 0))
    
    save_raw_boundary_overlay(image, raw_line_segments, output_json_path)
    
    boundary_vis = image_bgr.copy()
    if boundary_mask is not None: boundary_vis[boundary_mask > 0] = (0, 0, 255)
    for edge in edges:
        x1, y1 = edge["pixel_start"]["x"], edge["pixel_start"]["y"]
        x2, y2 = edge["pixel_end"]["x"], edge["pixel_end"]["y"]
        cv2.line(boundary_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(boundary_vis, (x1, y1), 6, (0, 255, 255), -1)
        cv2.circle(boundary_vis, (x2, y2), 6, (0, 255, 255), -1)
        cv2.putText(boundary_vis, f'{edge["length_m_estimated"]:.2f} m', (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(base) + "_boundary_overlay.png", boundary_vis)

def build_output_json(image_path, output_path, model_id, image, depth_m, intrinsics, floor_plane, edges, postprocess_info=None):
    width, height = image.size
    valid_depth = depth_m[depth_m > 0.05]
    if len(valid_depth) > 0:
        depth_stats = {"min_m": float(np.min(valid_depth)), "max_m": float(np.max(valid_depth)), "mean_m": float(np.mean(valid_depth)), "median_m": float(np.median(valid_depth)), "p10_m": float(np.percentile(valid_depth, 10)), "p90_m": float(np.percentile(valid_depth, 90))}
    else:
        depth_stats = {"min_m": None, "max_m": None, "mean_m": None, "median_m": None, "p10_m": None, "p90_m": None}

    return {
        "input_image": str(image_path), "output_dir": str(output_path.parent), "output_json": str(output_path),
        "model": {"name": model_id, "task": "single_image_metric_depth_estimation", "scene_type": "indoor"},
        "image": {"width": int(width), "height": int(height)},
        "camera_assumption": {"note": "Estimated 3D coordinates based on horizontal_fov_deg.", "intrinsics": intrinsics},
        "depth_statistics_m": depth_stats,
        "floor_plane_estimation": {"plane_normal_camera_coord": [float(floor_plane["normal"][0]), float(floor_plane["normal"][1]), float(floor_plane["normal"][2])], "plane_d": float(floor_plane["d"]), "distance_threshold_m": float(floor_plane["distance_threshold_m"]), "inlier_count_sampled": int(floor_plane["inlier_count_sampled"])},
        "postprocessing": postprocess_info, "floor_wall_boundary_edges": edges,
        "llm_placement_context": {"summary": "Provides boundary lines for furniture placement reasoning.", "main_use": "LLM bounding constraints."}
    }

# =========================
# 핵심 인터페이스 캡슐화 모듈
# =========================
class DepthEstimator:
    def __init__(self, horizontal_fov_deg=70.0):
        self.horizontal_fov_deg = horizontal_fov_deg
        
        # 호출(Init) 파트 최적화: 모델을 여기서 단 한 번만 메모리에 올립니다.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Depth Module] 📐 Initializing Depth Anything V2 on {self.device}...")
        
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        self.cached_mtime = 0
        self.cached_points_3d = None
        self.cached_intrinsics = None

    def _run_team_pipeline(self, image_path_str):
        """팀원의 메인 파이프라인(추론 및 저장)을 도커 볼륨 경로에 맞게 단 한 번 실행합니다."""
        INPUT_IMAGE_PATH = Path(image_path_str)
        OUTPUT_DIR = Path("/app/assets/depth_result")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON_PATH = OUTPUT_DIR / "result.json"

        print("[Depth Module] [1/6] Loading image...")
        image = load_image_rgb(INPUT_IMAGE_PATH)

        print("[Depth Module] [2/6] Inferencing Depth Anything V2 Metric Indoor...")
        depth_m = predict_metric_depth(image, self.processor, self.model, self.device)

        print("[Depth Module] [3/6] Converting depth map to 3D point cloud...")
        points_3d, intrinsics = depth_to_point_cloud(depth_m, self.horizontal_fov_deg)

        # 객체 깊이 조회를 위한 캐싱
        self.cached_points_3d = points_3d
        self.cached_intrinsics = intrinsics

        print("[Depth Module] [4/6] Estimating floor plane...")
        try:
            floor_plane = estimate_floor_plane_ransac(points_3d, depth_m)
            floor_mask_raw = floor_plane["floor_mask"]
        except Exception as e:
            print(f"[WARN] Failed to estimate floor plane: {e}")
            return # 실패 시 저장 건너뜀

        print("[Depth Module] [5/6] Extracting floor-wall boundary lines and post-processing...")
        object_exclusion_mask = load_optional_object_exclusion_mask(OBJECT_EXCLUSION_MASK_PATH, image.size)

        if USE_DEMO_POSTPROCESSING:
            floor_mask = postprocess_floor_mask_for_demo(floor_mask_raw, object_exclusion_mask)
        else:
            floor_mask = floor_mask_raw

        floor_plane["floor_mask"] = floor_mask
        raw_line_segments, boundary_mask = extract_floor_wall_boundary_lines(floor_mask)

        if USE_DEMO_POSTPROCESSING:
            line_segments = postprocess_boundary_lines(raw_line_segments, image.size[0], image.size[1])
            if len(line_segments) == 0 and len(raw_line_segments) > 0:
                print("[WARN] No lines left after post-processing, using raw lines.")
                line_segments = raw_line_segments
        else:
            line_segments = raw_line_segments

        edges = enrich_lines_with_metric_length(line_segments, points_3d)

        if USE_DEMO_POSTPROCESSING:
            edges = apply_length_scale_correction(edges)

        postprocess_info = {
            "enabled": bool(USE_DEMO_POSTPROCESSING),
            "floor_mask_postprocessed": bool(USE_DEMO_POSTPROCESSING),
            "object_exclusion_mask_used": OBJECT_EXCLUSION_MASK_PATH is not None,
            "raw_boundary_line_count": int(len(raw_line_segments)),
            "final_boundary_line_count": int(len(edges)),
        }

        print("[Depth Module] [6/6] Saving JSON and Debug Images...")
        result_json = build_output_json(INPUT_IMAGE_PATH, OUTPUT_JSON_PATH, MODEL_ID, image, depth_m, intrinsics, floor_plane, edges, postprocess_info)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        if SAVE_DEBUG_IMAGES:
            save_debug_images(image, depth_m, floor_mask, boundary_mask, edges, OUTPUT_JSON_PATH, raw_line_segments)
        
        print("[Depth Module] [DONE] Pipeline complete. Assets saved to /app/assets/depth_result")

    def estimate_3d_position_and_scale(self, image_shape, bbox, label="Object"):
        """
        응답 파트: main.py의 수정 없이 가구의 3D X,Y,Z 위치를 즉시 반환합니다.
        가장 정확한 Z값을 얻기 위해 팀원분의 3D Point Cloud(get_point_3d_at_pixel)를 사용합니다.
        """
        target_image = Path("/app/models/test_room.jpg")
        current_mtime = target_image.stat().st_mtime if target_image.exists() else 0

        # 새로운 사진이 들어왔을 때만 팀원분의 무거운 추론 파이프라인을 딱 한 번 돌립니다.
        if self.cached_mtime != current_mtime or self.cached_points_3d is None:
            self._run_team_pipeline(str(target_image))
            self.cached_mtime = current_mtime

        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 3D 포인트 클라우드에서 BBox 중앙의 픽셀 깊이(Z)를 정밀하게 추출
        p3d = get_point_3d_at_pixel(self.cached_points_3d, cx, cy, patch_size=11)
        estimated_x, estimated_y, estimated_z = float(p3d[0]), float(p3d[1]), float(p3d[2])

        # 크기(Scale) 계산
        fx = self.cached_intrinsics["fx"]
        fy = self.cached_intrinsics["fy"]
        bw, bh = x2 - x1, y2 - y1
        physical_w = (bw * estimated_z) / fx
        physical_h = (bh * estimated_z) / fy

        print(f"[Depth Module] 🎯 Detected: {label} | Dist(Z): {estimated_z:.2f}m | Size(WxH): {physical_w:.2f}m x {physical_h:.2f}m")

        return estimated_x, estimated_y, estimated_z, physical_w, physical_h