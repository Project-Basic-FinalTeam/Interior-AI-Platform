"""
Depth Anything V2 Metric Indoor + RF-DETR VisionMessage + Plane Intersection 기반 바닥-벽 경계 추정 코드

핵심 변경점:
기존 방식:
    floor_mask의 위쪽 경계 = 바닥-벽 경계 후보

개선 방식:
    RF-DETR 객체 bbox를 obstacle mask로 제외
    → 바닥 plane 추정
    → 정면 벽 plane 추정
    → 바닥 plane과 벽 plane의 3D 교선 계산
    → 해당 교선을 이미지에 투영하여 바닥-벽 경계로 사용

주의:
- Depth Anything V2에는 마스킹된 이미지가 아니라 원본 이미지를 넣는다.
- RF-DETR 결과는 depth map 추론 이후, 바닥/벽 plane 추정 단계에서 제외 마스크로 사용한다.
- Windows VSCode에서 실행하면 VISION_ENDPOINT는 보통 tcp://127.0.0.1:5556 이다.
- Docker 컨테이너 내부에서 실행하면 VISION_ENDPOINT는 tcp://ai-perception:5556 을 사용한다.
"""

import json
import math
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq
from PIL import Image
import pillow_avif

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# =========================
# 1. 경로 및 설정값
# =========================

PROJECT_ROOT = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Interior-AI-Platform")

INPUT_IMAGE_PATH = PROJECT_ROOT / "shared" / "models" / "thumb-1920-1174099.jpg"

RESULT_NUMBER = 1
BASE_OUTPUT_DIR = PROJECT_ROOT / "depth" / "base_test"
RESULT_NAME = f"v3_result_{RESULT_NUMBER}"
OUTPUT_DIR = BASE_OUTPUT_DIR / RESULT_NAME
OUTPUT_JSON_PATH = OUTPUT_DIR / f"{RESULT_NAME}.json"

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

HORIZONTAL_FOV_DEG = 70.0
SAVE_DEBUG_IMAGES = True

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================
# 2. ZeroMQ / FlatBuffers 설정
# =========================

USE_VISION_MESSAGE = True

# Windows VSCode에서 직접 실행하는 경우
VISION_ENDPOINT = "tcp://127.0.0.1:5556"

# Docker 컨테이너 내부에서 실행하는 경우 아래 값 사용
# VISION_ENDPOINT = "tcp://ai-perception:5556"

ZMQ_TIMEOUT_MS = 5000

VISION_REQUEST_MODE = "image_path_text"

# Depth가 읽는 이미지와 RF-DETR이 추론하는 이미지를 반드시 동일하게 맞춘다.
VISION_IMAGE_PATH_IN_SERVICE = f"shared/models/{INPUT_IMAGE_PATH.name}"

FLATBUFFERS_PYTHON_DIR = PROJECT_ROOT
VISION_MESSAGE_MODULE = "InteriorPlatform.VisionMessage"
VISION_MESSAGE_CLASS = "VisionMessage"

VISION_CONFIDENCE_THRESHOLD = 0.35
BBOX_EXPAND_RATIO = 0.04

# True면 VisionMessage에 들어온 모든 객체를 obstacle로 제외한다.
MASK_ALL_DETECTIONS = True

OBSTACLE_LABEL_KEYWORDS = {
    "chair", "desk", "table", "monitor", "computer", "laptop",
    "printer", "cabinet", "refrigerator", "sofa", "bed",
    "box", "shelf", "bookshelf", "person", "bag", "plant",
    "의자", "책상", "테이블", "모니터", "컴퓨터", "노트북",
    "프린터", "캐비닛", "냉장고", "소파", "침대",
    "박스", "선반", "책장", "사람", "가방", "화분",
}


@dataclass
class VisionObject:
    object_id: int
    label_raw: str
    label_name: str
    asset_name: str | None
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


# =========================
# 3. 이미지 로딩
# =========================

def load_image_rgb(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"입력 이미지가 존재하지 않습니다: {image_path}")

    return Image.open(image_path).convert("RGB")


# =========================
# 4. ZeroMQ REQ/REP 통신
# =========================

def build_vision_request_payload() -> bytes:
    if VISION_REQUEST_MODE == "empty":
        return b""

    if VISION_REQUEST_MODE == "image_path_text":
        return VISION_IMAGE_PATH_IN_SERVICE.encode("utf-8")

    raise ValueError("VISION_REQUEST_MODE은 'empty' 또는 'image_path_text'만 지원합니다.")


def request_vision_message_from_ai_perception() -> bytes | None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)

    socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
    socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)

    try:
        print(f"[INFO] ai-perception 연결 시도: {VISION_ENDPOINT}")
        socket.connect(VISION_ENDPOINT)

        payload = build_vision_request_payload()
        print(f"[INFO] Vision REQ 전송: {payload.decode('utf-8', errors='ignore')}")
        socket.send(payload)

        reply = socket.recv()
        print(f"[INFO] VisionMessage REP 수신 완료: {len(reply)} bytes")

        return reply

    except zmq.error.Again:
        print("[WARN] VisionMessage 수신 timeout. obstacle mask 없이 진행합니다.")
        return None

    except Exception as e:
        print(f"[WARN] VisionMessage 수신 실패: {e}")
        print("[WARN] obstacle mask 없이 진행합니다.")
        return None

    finally:
        socket.close()


# =========================
# 5. FlatBuffers VisionMessage 파싱
# =========================

def setup_flatbuffers_import_path():
    if FLATBUFFERS_PYTHON_DIR.exists():
        sys.path.insert(0, str(FLATBUFFERS_PYTHON_DIR))


def import_vision_message_class():
    setup_flatbuffers_import_path()

    try:
        module = importlib.import_module(VISION_MESSAGE_MODULE)
        return getattr(module, VISION_MESSAGE_CLASS)

    except Exception as e:
        raise ImportError(
            "\n[ERROR] FlatBuffers VisionMessage Python 모듈을 import하지 못했습니다.\n"
            f"- FLATBUFFERS_PYTHON_DIR: {FLATBUFFERS_PYTHON_DIR}\n"
            f"- VISION_MESSAGE_MODULE: {VISION_MESSAGE_MODULE}\n"
            f"- VISION_MESSAGE_CLASS: {VISION_MESSAGE_CLASS}\n"
            "shared/schema/InteriorPlatform.fbs를 flatc --python으로 변환했는지 확인하세요.\n"
        ) from e


def fb_string_to_str(value):
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="ignore")

    return str(value)


def split_label(label_raw: str):
    if "|" in label_raw:
        left, right = label_raw.split("|", 1)
        return left.strip(), right.strip()

    return label_raw.strip(), None


def parse_vision_message_flatbuffers(payload: bytes) -> list[VisionObject]:
    VisionMessage = import_vision_message_class()

    msg = VisionMessage.GetRootAsVisionMessage(payload, 0)

    if not hasattr(msg, "ObjectsLength") or not hasattr(msg, "Objects"):
        raise RuntimeError(
            "VisionMessage 안에서 objects 배열을 찾지 못했습니다. "
            "ObjectsLength(), Objects(i)가 존재하는지 확인하세요."
        )

    object_count = msg.ObjectsLength()
    objects = []

    print(f"[INFO] VisionMessage.objects 개수: {object_count}")

    for i in range(object_count):
        obj = msg.Objects(i)

        try:
            object_id = int(obj.Id())
        except Exception:
            object_id = i

        try:
            label_raw = fb_string_to_str(obj.Label())
        except Exception:
            label_raw = "unknown"

        label_name, asset_name = split_label(label_raw)

        if hasattr(obj, "Confidence"):
            confidence = float(obj.Confidence())
        elif hasattr(obj, "Score"):
            confidence = float(obj.Score())
        else:
            confidence = 1.0

        try:
            x_min = float(obj.XMin())
            y_min = float(obj.YMin())
            x_max = float(obj.XMax())
            y_max = float(obj.YMax())
        except Exception as e:
            print(f"[WARN] object {i} bbox 파싱 실패: {e}")
            continue

        objects.append(
            VisionObject(
                object_id=object_id,
                label_raw=label_raw,
                label_name=label_name,
                asset_name=asset_name,
                confidence=confidence,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )

    return objects


def receive_vision_objects() -> list[VisionObject]:
    if not USE_VISION_MESSAGE:
        print("[INFO] VisionMessage 사용 안 함")
        return []

    payload = request_vision_message_from_ai_perception()

    if payload is None:
        return []

    try:
        objects = parse_vision_message_flatbuffers(payload)
        print(f"[INFO] VisionObject 파싱 완료: {len(objects)}개")
        return objects

    except Exception as e:
        print(f"[WARN] VisionMessage 파싱 실패: {e}")
        print("[WARN] obstacle mask 없이 진행합니다.")
        return []


# =========================
# 6. obstacle mask 생성
# =========================

def should_mask_object(obj: VisionObject) -> bool:
    if obj.confidence < VISION_CONFIDENCE_THRESHOLD:
        return False

    if MASK_ALL_DETECTIONS:
        return True

    label_lower = obj.label_name.lower()
    raw_lower = obj.label_raw.lower()

    for keyword in OBSTACLE_LABEL_KEYWORDS:
        if keyword.lower() in label_lower or keyword.lower() in raw_lower:
            return True

    return False


def create_obstacle_mask_from_vision_objects(
    image_size,
    vision_objects: list[VisionObject],
):
    width, height = image_size
    obstacle_mask = np.zeros((height, width), dtype=bool)

    used_objects = []

    for obj in vision_objects:
        if not should_mask_object(obj):
            continue

        x_min = obj.x_min
        y_min = obj.y_min
        x_max = obj.x_max
        y_max = obj.y_max

        # 절대 pixel 좌표
        if max(x_min, y_min, x_max, y_max) <= 1.5:
            x_min *= width
            x_max *= width
            y_min *= height
            y_max *= height

        x1 = min(x_min, x_max)
        x2 = max(x_min, x_max)
        y1 = min(y_min, y_max)
        y2 = max(y_min, y_max)

        bw = x2 - x1
        bh = y2 - y1

        x1 -= bw * BBOX_EXPAND_RATIO
        x2 += bw * BBOX_EXPAND_RATIO
        y1 -= bh * BBOX_EXPAND_RATIO
        y2 += bh * BBOX_EXPAND_RATIO

        ix1 = int(np.clip(round(x1), 0, width - 1))
        iy1 = int(np.clip(round(y1), 0, height - 1))
        ix2 = int(np.clip(round(x2), 0, width - 1))
        iy2 = int(np.clip(round(y2), 0, height - 1))

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        obstacle_mask[iy1:iy2 + 1, ix1:ix2 + 1] = True

        used_objects.append({
            "id": int(obj.object_id),
            "label_raw": obj.label_raw,
            "label_name": obj.label_name,
            "asset_name": obj.asset_name,
            "confidence": float(obj.confidence),
            "bbox_xyxy_pixel": {
                "x_min": ix1,
                "y_min": iy1,
                "x_max": ix2,
                "y_max": iy2,
            },
        })

    print(f"[INFO] obstacle mask 반영 객체 수: {len(used_objects)}")

    return obstacle_mask, used_objects


# =========================
# 7. Depth Anything V2 Metric Indoor 추론
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


# =========================
# 8. depth map -> 3D point cloud
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

    intrinsics = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "horizontal_fov_deg": float(horizontal_fov_deg),
    }

    return points_3d, intrinsics


# =========================
# 9. 평면 추정 공통 함수
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

    return normal.astype(np.float32), float(d)


def plane_distance(points, normal, d):
    return np.abs(points @ normal + d)


def clean_mask(mask: np.ndarray, kernel_size: int = 7):
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    return mask_uint8 > 0


# =========================
# 10. 바닥 plane 추정
# =========================

def estimate_floor_plane_ransac(
    points_3d: np.ndarray,
    depth_m: np.ndarray,
    obstacle_mask: np.ndarray | None = None,
):
    height, width = depth_m.shape

    valid_mask = depth_m > 0.05

    if obstacle_mask is not None:
        valid_mask = valid_mask & (~obstacle_mask)

    bottom_mask = np.zeros_like(valid_mask, dtype=bool)
    bottom_mask[int(height * 0.45):, :] = True

    candidate_mask = valid_mask & bottom_mask
    candidate_points = points_3d[candidate_mask]

    if len(candidate_points) < 500:
        raise RuntimeError(
            "바닥 후보 점이 너무 적습니다. obstacle mask가 너무 넓거나 바닥이 충분히 보이지 않습니다."
        )

    max_sample_points = 15000

    if len(candidate_points) > max_sample_points:
        idx = np.random.choice(len(candidate_points), max_sample_points, replace=False)
        sampled_points = candidate_points[idx]
    else:
        sampled_points = candidate_points

    median_depth = float(np.median(depth_m[valid_mask]))
    distance_threshold = max(0.04, median_depth * 0.015)

    best_normal = None
    best_d = None
    best_inlier_count = 0
    best_score = -1

    iterations = 800

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

        # 카메라 좌표계 기준으로 바닥은 y축 성분이 큰 평면일 가능성이 높음
        y_alignment = abs(float(normal[1]))

        if y_alignment < 0.35:
            continue

        distances = plane_distance(sampled_points, normal, d)
        inlier_count = int(np.sum(distances < distance_threshold))

        score = inlier_count * (0.7 + 0.3 * y_alignment)

        if score > best_score:
            best_score = score
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None:
        raise RuntimeError("바닥 plane 추정에 실패했습니다.")

    all_dist = plane_distance(
        points_3d.reshape(-1, 3),
        best_normal,
        best_d,
    ).reshape(height, width)

    floor_mask = (all_dist < distance_threshold) & (depth_m > 0.05)

    if obstacle_mask is not None:
        floor_mask = floor_mask & (~obstacle_mask)

    floor_mask[:int(height * 0.30), :] = False
    floor_mask = clean_mask(floor_mask, kernel_size=7)

    return {
        "normal": best_normal,
        "d": float(best_d),
        "distance_threshold_m": float(distance_threshold),
        "inlier_count_sampled": int(best_inlier_count),
        "floor_mask": floor_mask,
        "used_obstacle_mask": obstacle_mask is not None,
    }


# =========================
# 11. 정면 벽 plane 추정
# =========================

def estimate_front_wall_plane_ransac(
    points_3d: np.ndarray,
    depth_m: np.ndarray,
    floor_plane: dict,
    obstacle_mask: np.ndarray | None = None,
):
    height, width = depth_m.shape

    valid_mask = depth_m > 0.05

    if obstacle_mask is not None:
        valid_mask = valid_mask & (~obstacle_mask)

    floor_dist = plane_distance(
        points_3d.reshape(-1, 3),
        floor_plane["normal"],
        floor_plane["d"],
    ).reshape(height, width)

    # 바닥 plane과 가까운 점은 벽 후보에서 제외
    not_floor_mask = floor_dist > (floor_plane["distance_threshold_m"] * 2.0)

    # 벽 후보는 너무 아래쪽 바닥/물체 영역과 너무 위쪽 천장 영역을 일부 제외
    region_mask = np.zeros_like(valid_mask, dtype=bool)
    region_mask[int(height * 0.05):int(height * 0.92), :] = True

    candidate_mask = valid_mask & not_floor_mask & region_mask
    candidate_points = points_3d[candidate_mask]

    if len(candidate_points) < 800:
        raise RuntimeError(
            "정면 벽 후보 점이 너무 적습니다. 벽이 너무 많이 가려졌거나 obstacle mask가 너무 넓을 수 있습니다."
        )

    max_sample_points = 18000

    if len(candidate_points) > max_sample_points:
        idx = np.random.choice(len(candidate_points), max_sample_points, replace=False)
        sampled_points = candidate_points[idx]
    else:
        sampled_points = candidate_points

    median_depth = float(np.median(depth_m[valid_mask]))
    distance_threshold = max(0.05, median_depth * 0.018)

    best_normal = None
    best_d = None
    best_inlier_count = 0
    best_score = -1

    iterations = 1000

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

        # 정면 벽은 카메라 z축 방향과 normal이 어느 정도 정렬됨
        z_alignment = abs(float(normal[2]))
        y_alignment = abs(float(normal[1]))

        # 천장/바닥 계열 plane 제거
        if z_alignment < 0.35:
            continue

        if y_alignment > 0.70:
            continue

        distances = plane_distance(sampled_points, normal, d)
        inlier_count = int(np.sum(distances < distance_threshold))

        score = inlier_count * (0.7 + 0.3 * z_alignment)

        if score > best_score:
            best_score = score
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None:
        raise RuntimeError("정면 벽 plane 추정에 실패했습니다.")

    all_dist = plane_distance(
        points_3d.reshape(-1, 3),
        best_normal,
        best_d,
    ).reshape(height, width)

    wall_mask = (all_dist < distance_threshold) & candidate_mask
    wall_mask = clean_mask(wall_mask, kernel_size=9)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wall_mask.astype(np.uint8) * 255,
        connectivity=8,
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        wall_mask = labels == largest_label

    return {
        "normal": best_normal,
        "d": float(best_d),
        "distance_threshold_m": float(distance_threshold),
        "inlier_count_sampled": int(best_inlier_count),
        "wall_mask": wall_mask,
        "z_alignment": float(abs(best_normal[2])),
        "y_alignment": float(abs(best_normal[1])),
    }


# =========================
# 12. 바닥 plane과 벽 plane의 교선 계산
# =========================

def compute_plane_intersection_line(floor_plane: dict, wall_plane: dict):
    n1 = floor_plane["normal"].astype(np.float64)
    d1 = float(floor_plane["d"])

    n2 = wall_plane["normal"].astype(np.float64)
    d2 = float(wall_plane["d"])

    direction = np.cross(n1, n2)
    direction_norm = np.linalg.norm(direction)

    if direction_norm < 1e-8:
        raise RuntimeError("바닥 plane과 벽 plane이 거의 평행하여 교선을 계산할 수 없습니다.")

    direction = direction / direction_norm

    # 두 평면 위에 있으면서 원점에 가장 가까운 점을 계산
    # n1·x = -d1
    # n2·x = -d2
    # direction·x = 0
    A = np.vstack([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0], dtype=np.float64)

    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        point = np.linalg.lstsq(A, b, rcond=None)[0]

    return point.astype(np.float32), direction.astype(np.float32)


def project_3d_points_to_pixels(points_3d: np.ndarray, intrinsics: dict):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy

    return u, v


def get_projected_intersection_segment(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    intrinsics: dict,
    image_size,
    wall_mask: np.ndarray | None = None,
):
    width, height = image_size

    t_values = np.linspace(-20.0, 20.0, 8000).astype(np.float32)
    sampled_3d = line_point[None, :] + t_values[:, None] * line_direction[None, :]

    z = sampled_3d[:, 2]
    valid = z > 0.05

    u, v = project_3d_points_to_pixels(sampled_3d, intrinsics)

    valid = (
        valid
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    if wall_mask is not None and np.any(wall_mask):
        ys, xs = np.where(wall_mask)

        x_min = int(np.percentile(xs, 1))
        x_max = int(np.percentile(xs, 99))

        # 벽 영역의 좌우 범위 안에 있는 교선만 사용
        valid_with_wall = valid & (u >= x_min) & (u <= x_max)

        if np.sum(valid_with_wall) >= 2:
            valid = valid_with_wall

    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 2:
        raise RuntimeError("이미지 내부에 투영되는 바닥-벽 교선 구간을 찾지 못했습니다.")

    valid_u = u[valid_indices]

    start_idx = valid_indices[np.argmin(valid_u)]
    end_idx = valid_indices[np.argmax(valid_u)]

    p1_3d = sampled_3d[start_idx]
    p2_3d = sampled_3d[end_idx]

    p1_pixel = {
        "x": int(np.clip(round(u[start_idx]), 0, width - 1)),
        "y": int(np.clip(round(v[start_idx]), 0, height - 1)),
    }

    p2_pixel = {
        "x": int(np.clip(round(u[end_idx]), 0, width - 1)),
        "y": int(np.clip(round(v[end_idx]), 0, height - 1)),
    }

    length_m = float(np.linalg.norm(p2_3d - p1_3d))

    return {
        "pixel_start": p1_pixel,
        "pixel_end": p2_pixel,
        "point_3d_start_m": {
            "x": float(p1_3d[0]),
            "y": float(p1_3d[1]),
            "z": float(p1_3d[2]),
        },
        "point_3d_end_m": {
            "x": float(p2_3d[0]),
            "y": float(p2_3d[1]),
            "z": float(p2_3d[2]),
        },
        "length_m_estimated": length_m,
    }


# =========================
# 13. 디버그 이미지 저장
# =========================

def save_debug_images(
    image: Image.Image,
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    obstacle_mask: np.ndarray | None,
    used_objects,
    intersection_segment: dict | None,
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

    if obstacle_mask is not None:
        obstacle_vis = image_bgr.copy()
        obstacle_overlay = np.zeros_like(image_bgr)
        obstacle_overlay[obstacle_mask] = (0, 0, 255)
        obstacle_vis = cv2.addWeighted(obstacle_vis, 0.75, obstacle_overlay, 0.25, 0)

        for obj in used_objects:
            b = obj["bbox_xyxy_pixel"]

            cv2.rectangle(
                obstacle_vis,
                (b["x_min"], b["y_min"]),
                (b["x_max"], b["y_max"]),
                (0, 255, 255),
                2,
            )

            label = f'{obj["label_name"]} {obj["confidence"]:.2f}'

            cv2.putText(
                obstacle_vis,
                label,
                (b["x_min"], max(20, b["y_min"] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(base) + "_vision_obstacle_mask.png", obstacle_vis)

    floor_vis = image_bgr.copy()
    floor_overlay = np.zeros_like(image_bgr)
    floor_overlay[floor_mask] = (0, 255, 0)
    floor_vis = cv2.addWeighted(floor_vis, 0.75, floor_overlay, 0.25, 0)
    cv2.imwrite(str(base) + "_floor_mask.png", floor_vis)

    wall_vis = image_bgr.copy()
    wall_overlay = np.zeros_like(image_bgr)
    wall_overlay[wall_mask] = (255, 0, 0)
    wall_vis = cv2.addWeighted(wall_vis, 0.75, wall_overlay, 0.25, 0)
    cv2.imwrite(str(base) + "_wall_mask.png", wall_vis)

    intersection_vis = image_bgr.copy()

    if obstacle_mask is not None:
        obstacle_overlay = np.zeros_like(image_bgr)
        obstacle_overlay[obstacle_mask] = (0, 0, 255)
        intersection_vis = cv2.addWeighted(intersection_vis, 0.90, obstacle_overlay, 0.10, 0)

    floor_overlay = np.zeros_like(image_bgr)
    floor_overlay[floor_mask] = (0, 255, 0)
    intersection_vis = cv2.addWeighted(intersection_vis, 0.90, floor_overlay, 0.10, 0)

    wall_overlay = np.zeros_like(image_bgr)
    wall_overlay[wall_mask] = (255, 0, 0)
    intersection_vis = cv2.addWeighted(intersection_vis, 0.90, wall_overlay, 0.10, 0)

    if intersection_segment is not None:
        p1 = intersection_segment["pixel_start"]
        p2 = intersection_segment["pixel_end"]

        cv2.line(
            intersection_vis,
            (p1["x"], p1["y"]),
            (p2["x"], p2["y"]),
            (0, 255, 255),
            5,
        )

        label = f'{intersection_segment["length_m_estimated"]:.2f} m'

        cv2.putText(
            intersection_vis,
            label,
            (p1["x"], max(25, p1["y"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(base) + "_plane_intersection_overlay.png", intersection_vis)


# =========================
# 14. JSON 저장
# =========================

def build_output_json(
    image_path,
    output_path,
    model_id,
    image,
    depth_m,
    intrinsics,
    used_objects,
    obstacle_mask,
    floor_plane,
    wall_plane,
    intersection_line,
    intersection_segment,
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

    obstacle_pixel_count = int(np.sum(obstacle_mask)) if obstacle_mask is not None else 0
    obstacle_area_ratio = float(obstacle_pixel_count / (width * height))

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
        "vision_message_input": {
            "used": bool(USE_VISION_MESSAGE),
            "transport": "ZeroMQ REQ/REP",
            "serialization": "FlatBuffers",
            "endpoint": VISION_ENDPOINT,
            "request_image_path": VISION_IMAGE_PATH_IN_SERVICE,
            "objects_field": "objects",
            "bbox_format": "x_min, y_min, x_max, y_max",
            "bbox_unit": "absolute_pixel",
            "used_object_count": len(used_objects),
            "used_objects": used_objects,
            "obstacle_pixel_count": obstacle_pixel_count,
            "obstacle_area_ratio": obstacle_area_ratio,
        },
        "camera_assumption": {
            "note": "horizontal_fov_deg 기반으로 3D 좌표와 길이를 추정했습니다.",
            "intrinsics": intrinsics,
        },
        "depth_statistics_m": depth_stats,
        "floor_plane": {
            "normal": [
                float(floor_plane["normal"][0]),
                float(floor_plane["normal"][1]),
                float(floor_plane["normal"][2]),
            ],
            "d": float(floor_plane["d"]),
            "distance_threshold_m": float(floor_plane["distance_threshold_m"]),
            "inlier_count_sampled": int(floor_plane["inlier_count_sampled"]),
        },
        "front_wall_plane": {
            "normal": [
                float(wall_plane["normal"][0]),
                float(wall_plane["normal"][1]),
                float(wall_plane["normal"][2]),
            ],
            "d": float(wall_plane["d"]),
            "distance_threshold_m": float(wall_plane["distance_threshold_m"]),
            "inlier_count_sampled": int(wall_plane["inlier_count_sampled"]),
            "z_alignment": float(wall_plane["z_alignment"]),
            "y_alignment": float(wall_plane["y_alignment"]),
        },
        "floor_wall_intersection_line": {
            "line_point_m": {
                "x": float(intersection_line["point"][0]),
                "y": float(intersection_line["point"][1]),
                "z": float(intersection_line["point"][2]),
            },
            "line_direction": {
                "x": float(intersection_line["direction"][0]),
                "y": float(intersection_line["direction"][1]),
                "z": float(intersection_line["direction"][2]),
            },
            "projected_segment": intersection_segment,
            "note": "바닥 plane과 정면 벽 plane의 3D 교선을 이미지에 투영한 결과입니다.",
        },
        "llm_placement_context": {
            "summary": "RF-DETR 객체 영역을 제외한 뒤, 바닥 plane과 정면 벽 plane의 교선을 계산하여 바닥-벽 경계선을 추정했습니다.",
            "main_use": "가구 배치 가능 벽면과 바닥 경계 기준 좌표를 LLM 또는 메인 서버에 전달할 때 사용합니다.",
            "important_warning": "벽과 바닥 plane 자체가 충분히 보이지 않으면 교선 추정도 불안정할 수 있습니다.",
        },
    }

    return result


# =========================
# 15. main
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] 입력 이미지:", INPUT_IMAGE_PATH)
    print("[INFO] RF-DETR 요청 이미지:", VISION_IMAGE_PATH_IN_SERVICE)
    print("[INFO] 출력 폴더:", OUTPUT_DIR)
    print("[INFO] Vision endpoint:", VISION_ENDPOINT)

    print("[1/9] 이미지 로딩 중...")
    image = load_image_rgb(INPUT_IMAGE_PATH)

    print("[2/9] ai-perception에 REQ 요청 후 VisionMessage 수신 중...")
    vision_objects = receive_vision_objects()

    print("[3/9] VisionMessage.objects 기반 obstacle mask 생성 중...")
    obstacle_mask, used_objects = create_obstacle_mask_from_vision_objects(
        image_size=image.size,
        vision_objects=vision_objects,
    )

    if len(used_objects) == 0:
        print("[WARN] obstacle mask에 사용된 객체가 없습니다. 기존 depth-only 방식과 거의 동일하게 진행됩니다.")

    print("[4/9] Depth Anything V2 Metric Indoor 추론 중...")
    depth_m = predict_metric_depth(image, MODEL_ID)

    print("[5/9] depth map을 3D point cloud로 변환 중...")
    points_3d, intrinsics = depth_to_point_cloud(depth_m, HORIZONTAL_FOV_DEG)

    print("[6/9] obstacle mask를 제외하고 바닥 plane 추정 중...")
    floor_plane = estimate_floor_plane_ransac(
        points_3d=points_3d,
        depth_m=depth_m,
        obstacle_mask=obstacle_mask,
    )

    print("[7/9] obstacle mask와 바닥 plane을 제외하고 정면 벽 plane 추정 중...")
    wall_plane = estimate_front_wall_plane_ransac(
        points_3d=points_3d,
        depth_m=depth_m,
        floor_plane=floor_plane,
        obstacle_mask=obstacle_mask,
    )

    print("[8/9] 바닥 plane과 정면 벽 plane의 3D 교선 계산 중...")
    line_point, line_direction = compute_plane_intersection_line(
        floor_plane=floor_plane,
        wall_plane=wall_plane,
    )

    intersection_segment = get_projected_intersection_segment(
        line_point=line_point,
        line_direction=line_direction,
        intrinsics=intrinsics,
        image_size=image.size,
        wall_mask=wall_plane["wall_mask"],
    )

    intersection_line = {
        "point": line_point,
        "direction": line_direction,
    }

    print("[9/9] JSON 및 디버그 이미지 저장 중...")
    result_json = build_output_json(
        image_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_JSON_PATH,
        model_id=MODEL_ID,
        image=image,
        depth_m=depth_m,
        intrinsics=intrinsics,
        used_objects=used_objects,
        obstacle_mask=obstacle_mask,
        floor_plane=floor_plane,
        wall_plane=wall_plane,
        intersection_line=intersection_line,
        intersection_segment=intersection_segment,
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    if SAVE_DEBUG_IMAGES:
        save_debug_images(
            image=image,
            depth_m=depth_m,
            floor_mask=floor_plane["floor_mask"],
            wall_mask=wall_plane["wall_mask"],
            obstacle_mask=obstacle_mask,
            used_objects=used_objects,
            intersection_segment=intersection_segment,
            output_json_path=OUTPUT_JSON_PATH,
        )

    print("\n[완료] 저장 완료")
    print(f"- JSON: {OUTPUT_JSON_PATH}")
    print(f"- depth: {OUTPUT_JSON_PATH.with_suffix('')}_depth.png")
    print(f"- obstacle mask: {OUTPUT_JSON_PATH.with_suffix('')}_vision_obstacle_mask.png")
    print(f"- floor mask: {OUTPUT_JSON_PATH.with_suffix('')}_floor_mask.png")
    print(f"- wall mask: {OUTPUT_JSON_PATH.with_suffix('')}_wall_mask.png")
    print(f"- plane intersection overlay: {OUTPUT_JSON_PATH.with_suffix('')}_plane_intersection_overlay.png")

    print("\n[SUMMARY]")
    print(f"- RF-DETR 객체 반영 수: {len(used_objects)}")
    print(f"- 바닥 plane inlier 수: {floor_plane['inlier_count_sampled']}")
    print(f"- 정면 벽 plane inlier 수: {wall_plane['inlier_count_sampled']}")
    print(
        "- 교선 투영 픽셀:",
        intersection_segment["pixel_start"],
        "->",
        intersection_segment["pixel_end"],
    )
    print(f"- 교선 추정 길이: {intersection_segment['length_m_estimated']:.3f} m")


if __name__ == "__main__":
    main()