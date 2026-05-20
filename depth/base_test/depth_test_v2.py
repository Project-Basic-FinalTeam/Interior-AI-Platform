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
import pillow_avif  # AVIF 이미지 로딩용 등록 플러그인

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# =========================
# 1. 경로 및 설정값
# =========================

# 현재 shared > models > test_room.jpg 기준
# Windows VSCode에서 실행하는 경우
INPUT_IMAGE_PATH = Path(r"C:\Users\khrha\Desktop\SWPJ_4\shared\models\test_room.jpg")

# result 뒤 숫자만 변경하면 됩니다.
RESULT_NUMBER = 1

BASE_OUTPUT_DIR = Path(r"C:\Users\khrha\Desktop\SWPJ_4\Depth\base_test")
RESULT_NAME = f"v2_result_{RESULT_NUMBER}"
OUTPUT_DIR = BASE_OUTPUT_DIR / RESULT_NAME
OUTPUT_JSON_PATH = OUTPUT_DIR / f"{RESULT_NAME}.json"

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

HORIZONTAL_FOV_DEG = 70

SAVE_DEBUG_IMAGES = True


# =========================
# 2. ZeroMQ / FlatBuffers 설정
# =========================

USE_VISION_MESSAGE = True

# Docker 내부 컨테이너에서 실행하면 이 값 사용
VISION_ENDPOINT = "tcp://ai-perception:5556"

# Windows VSCode에서 직접 실행하고 docker-compose에서 5556 포트가 host로 열려 있으면 아래로 변경
# VISION_ENDPOINT = "tcp://127.0.0.1:5556"

ZMQ_TIMEOUT_MS = 5000

# REQ/REP 구조에서는 요청을 한 번 보내야 응답을 받을 수 있습니다.
# ai-perception이 요청 내용을 사용하지 않고 test_room.jpg를 내부에서 읽는 구조라면 "empty"로 둬도 됩니다.
# 만약 image path 문자열을 요구하면 "image_path_text"로 바꾸세요.
VISION_REQUEST_MODE = "image_path_text"  # "empty" 또는 "image_path_text"
VISION_IMAGE_PATH_IN_SERVICE = "shared/models/test_room.jpg"

# FlatBuffers Python 생성 코드 위치
# flatc --python으로 생성된 InteriorPlatform 폴더가 있는 상위 경로를 넣으세요.
FLATBUFFERS_PYTHON_DIR = Path(r"C:\Users\khrha\Desktop\SWPJ_4")

# 실제 생성된 모듈 경로가 다르면 여기만 수정
VISION_MESSAGE_MODULE = "InteriorPlatform.VisionMessage"
VISION_MESSAGE_CLASS = "VisionMessage"

# RF-DETR confidence 기준
VISION_CONFIDENCE_THRESHOLD = 0.35

# bbox 주변까지 조금 넓게 제외
BBOX_EXPAND_RATIO = 0.04

# True면 VisionMessage에 들어온 모든 객체를 obstacle로 제외합니다.
# RF-DETR 결과가 가구 중심이면 이게 가장 안전합니다.
MASK_ALL_DETECTIONS = True

# MASK_ALL_DETECTIONS=False일 때만 아래 label을 기준으로 필터링합니다.
OBSTACLE_LABEL_KEYWORDS = {
    "chair", "desk", "table", "monitor", "computer", "laptop",
    "printer", "cabinet", "refrigerator", "sofa", "bed",
    "box", "shelf", "bookshelf", "person", "bag",
    "의자", "책상", "테이블", "모니터", "컴퓨터", "노트북",
    "프린터", "캐비닛", "냉장고", "소파", "침대",
    "박스", "선반", "책장", "사람", "가방",
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

    image = Image.open(image_path).convert("RGB")
    return image


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
        print(f"[INFO] Vision REQ 전송: {len(payload)} bytes")
        socket.send(payload)

        # REP 응답 수신
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
        vision_cls = getattr(module, VISION_MESSAGE_CLASS)
        return vision_cls

    except Exception as e:
        raise ImportError(
            "\n[ERROR] FlatBuffers VisionMessage Python 모듈을 import하지 못했습니다.\n"
            f"- 현재 설정 FLATBUFFERS_PYTHON_DIR: {FLATBUFFERS_PYTHON_DIR}\n"
            f"- 현재 설정 VISION_MESSAGE_MODULE: {VISION_MESSAGE_MODULE}\n"
            f"- 현재 설정 VISION_MESSAGE_CLASS: {VISION_MESSAGE_CLASS}\n"
            "shared/schema/InteriorPlatform.fbs를 flatc --python으로 변환했는지,\n"
            "생성된 Python 모듈 경로가 위 설정과 맞는지 확인하세요.\n"
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
    """
    label 예시:
    '의자|furniture_0.ply'

    반환:
    label_name='의자'
    asset_name='furniture_0.ply'
    """
    if "|" in label_raw:
        left, right = label_raw.split("|", 1)
        return left.strip(), right.strip()

    return label_raw.strip(), None


def parse_vision_message_flatbuffers(payload: bytes) -> list[VisionObject]:
    """
    shared/schema/InteriorPlatform 내 VisionMessage 기준:
    - VisionMessage.objects 배열
    - object.x_min
    - object.y_min
    - object.x_max
    - object.y_max
    - object.label
    - object.id

    FlatBuffers Python 생성 메서드는 보통 다음처럼 PascalCase로 생성됩니다.
    - ObjectsLength()
    - Objects(i)
    - XMin()
    - YMin()
    - XMax()
    - YMax()
    - Label()
    - Id()
    """

    VisionMessage = import_vision_message_class()

    msg = VisionMessage.GetRootAsVisionMessage(payload, 0)

    if not hasattr(msg, "ObjectsLength") or not hasattr(msg, "Objects"):
        raise RuntimeError(
            "VisionMessage 안에서 objects 배열을 찾지 못했습니다. "
            "FlatBuffers 생성 코드에서 ObjectsLength(), Objects(i)가 존재하는지 확인하세요."
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

        # confidence 필드가 없을 수도 있으므로 optional 처리
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
            print(f"[WARN] object {i}의 bbox 파싱 실패: {e}")
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

        # 선배님 답변 기준: pixel 절대 좌표
        # 그래도 혹시 0~1 값이 들어올 경우에 대비
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

    print(f"[INFO] obstacle mask에 반영된 객체 수: {len(used_objects)}")

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
# 8. depth map을 3D point cloud로 변환
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
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "horizontal_fov_deg": float(horizontal_fov_deg),
    }


# =========================
# 9. 바닥 plane RANSAC 추정
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


def estimate_floor_plane_ransac(
    points_3d: np.ndarray,
    depth_m: np.ndarray,
    obstacle_mask: np.ndarray | None = None,
):
    height, width = depth_m.shape

    valid_mask = depth_m > 0.05

    if obstacle_mask is not None:
        if obstacle_mask.shape != valid_mask.shape:
            raise ValueError("obstacle_mask 크기가 depth_m 크기와 다릅니다.")

        # 객체 영역 제외
        valid_mask = valid_mask & (~obstacle_mask)

    # 바닥은 이미지 하단부에 많이 보인다고 가정
    y_start = int(height * 0.45)

    bottom_mask = np.zeros_like(valid_mask, dtype=bool)
    bottom_mask[y_start:, :] = True

    candidate_mask = valid_mask & bottom_mask
    candidate_points = points_3d[candidate_mask]

    if len(candidate_points) < 500:
        raise RuntimeError(
            "바닥 후보 점이 너무 적습니다. "
            "VisionMessage obstacle mask가 너무 넓거나 이미지 하단 바닥이 충분히 보이지 않을 수 있습니다."
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

    valid_depth_values = depth_m[valid_mask]

    if len(valid_depth_values) == 0:
        raise RuntimeError("유효한 depth 값이 없습니다.")

    median_depth = float(np.median(valid_depth_values))
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

    floor_mask = (all_dist < distance_threshold) & (depth_m > 0.05)

    if obstacle_mask is not None:
        floor_mask = floor_mask & (~obstacle_mask)

    # 이미지 상단의 잘못된 바닥 후보 제거
    floor_mask[:int(height * 0.30), :] = False

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
        "used_obstacle_mask": obstacle_mask is not None,
    }


# =========================
# 10. 바닥-벽 경계선 후보 추출
# =========================

def extract_floor_wall_boundary_lines(
    floor_mask: np.ndarray,
    obstacle_mask: np.ndarray | None = None,
):
    height, width = floor_mask.shape

    usable_floor_mask = floor_mask.copy()

    if obstacle_mask is not None:
        usable_floor_mask = usable_floor_mask & (~obstacle_mask)

    boundary_points = []

    for x in range(width):
        ys = np.where(usable_floor_mask[:, x])[0]

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

    if obstacle_mask is not None:
        boundary_mask[obstacle_mask] = 0

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
# 11. 선분의 3D 길이 계산
# =========================

def get_point_3d_at_pixel(
    points_3d: np.ndarray,
    x: int,
    y: int,
    patch_size: int = 5,
):
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
            "confidence_note": "VisionMessage objects 영역을 obstacle mask로 제외한 뒤 추정한 바닥-벽 경계 후보입니다.",
        })

    return enriched


# =========================
# 12. 디버그 이미지 저장
# =========================

def save_debug_images(
    image: Image.Image,
    depth_m: np.ndarray,
    floor_mask: np.ndarray,
    boundary_mask,
    edges,
    output_json_path: Path,
    obstacle_mask: np.ndarray | None = None,
    used_objects=None,
):
    base = output_json_path.with_suffix("")

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # depth 시각화
    valid = depth_m > 0.05
    depth_vis = np.zeros_like(depth_m, dtype=np.uint8)

    if np.any(valid):
        d_min = np.percentile(depth_m[valid], 2)
        d_max = np.percentile(depth_m[valid], 98)
        depth_norm = np.clip((depth_m - d_min) / (d_max - d_min + 1e-6), 0, 1)
        depth_vis = (depth_norm * 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(base) + "_depth.png", depth_colormap)

    # obstacle mask 시각화
    if obstacle_mask is not None:
        obstacle_vis = image_bgr.copy()
        obstacle_overlay = np.zeros_like(image_bgr)
        obstacle_overlay[obstacle_mask] = (0, 0, 255)
        obstacle_vis = cv2.addWeighted(obstacle_vis, 0.75, obstacle_overlay, 0.25, 0)

        if used_objects:
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

    # floor mask 시각화
    floor_vis = image_bgr.copy()
    floor_overlay = np.zeros_like(image_bgr)
    floor_overlay[floor_mask] = (0, 255, 0)
    floor_vis = cv2.addWeighted(floor_vis, 0.75, floor_overlay, 0.25, 0)
    cv2.imwrite(str(base) + "_floor_mask.png", floor_vis)

    # boundary overlay 시각화
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
# 13. JSON 저장
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
    used_objects,
    obstacle_mask,
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

    if obstacle_mask is not None:
        obstacle_pixel_count = int(np.sum(obstacle_mask))
        obstacle_area_ratio = float(obstacle_pixel_count / (width * height))
    else:
        obstacle_pixel_count = 0
        obstacle_area_ratio = 0.0

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
            "message_type": "InteriorPlatform.VisionMessage",
            "objects_field": "objects",
            "bbox_format": "x_min, y_min, x_max, y_max",
            "bbox_unit": "absolute_pixel",
            "label_field": "label",
            "id_field": "id",
            "confidence_threshold": float(VISION_CONFIDENCE_THRESHOLD),
            "bbox_expand_ratio": float(BBOX_EXPAND_RATIO),
            "mask_all_detections": bool(MASK_ALL_DETECTIONS),
            "used_object_count": len(used_objects),
            "used_objects": used_objects,
            "obstacle_pixel_count": obstacle_pixel_count,
            "obstacle_area_ratio": obstacle_area_ratio,
            "note": "VisionMessage.objects 배열의 bbox를 obstacle mask로 변환한 뒤, 바닥 plane 및 바닥-벽 경계 추정에서 제외했습니다.",
        },
        "camera_assumption": {
            "note": "horizontal_fov_deg 기반으로 3D 좌표와 길이를 추정했습니다.",
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
            "used_obstacle_mask": bool(floor_plane.get("used_obstacle_mask", False)),
        },
        "floor_wall_boundary_edges": edges,
        "llm_placement_context": {
            "summary": "RF-DETR 기반 VisionMessage 객체 영역을 제외한 뒤, 가구 배치 판단에 사용할 수 있는 바닥-벽 경계 후보선과 각 후보선의 추정 길이를 제공합니다.",
            "main_use": "LLM 또는 메인 서버가 벽면 기준 가구 배치 가능 영역을 판단할 때 사용",
            "important_warning": "현재는 bbox 기반 obstacle mask이므로 실제 객체보다 넓은 영역이 제외될 수 있습니다. 정밀도가 더 필요하면 bbox 대신 segmentation mask 연동이 필요합니다.",
        },
    }

    return result


# =========================
# 14. main
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] 입력 이미지 경로:", INPUT_IMAGE_PATH)
    print("[INFO] 출력 폴더 경로:", OUTPUT_DIR)
    print("[INFO] 출력 JSON 경로:", OUTPUT_JSON_PATH)
    print("[INFO] Vision endpoint:", VISION_ENDPOINT)

    print("[1/8] 이미지 로딩 중...")
    image = load_image_rgb(INPUT_IMAGE_PATH)

    print("[2/8] ai-perception에 REQ 요청 후 VisionMessage 수신 중...")
    vision_objects = receive_vision_objects()

    print("[3/8] VisionMessage.objects 기반 obstacle mask 생성 중...")
    obstacle_mask, used_objects = create_obstacle_mask_from_vision_objects(
        image_size=image.size,
        vision_objects=vision_objects,
    )

    if len(used_objects) == 0:
        print("[WARN] obstacle mask에 사용된 객체가 없습니다. 기존 방식과 거의 동일하게 진행됩니다.")

    print("[4/8] Depth Anything V2 Metric Indoor 추론 중...")
    depth_m = predict_metric_depth(image, MODEL_ID)

    print("[5/8] depth map을 3D point cloud로 변환 중...")
    points_3d, intrinsics = depth_to_point_cloud(depth_m, HORIZONTAL_FOV_DEG)

    print("[6/8] obstacle mask를 제외하고 바닥 평면 추정 중...")
    floor_plane = estimate_floor_plane_ransac(
        points_3d=points_3d,
        depth_m=depth_m,
        obstacle_mask=obstacle_mask,
    )
    floor_mask = floor_plane["floor_mask"]

    print("[7/8] obstacle mask를 제외하고 바닥-벽 경계 후보선 추출 중...")
    line_segments, boundary_mask = extract_floor_wall_boundary_lines(
        floor_mask=floor_mask,
        obstacle_mask=obstacle_mask,
    )
    edges = enrich_lines_with_metric_length(line_segments, points_3d)

    print("[8/8] JSON 및 디버그 이미지 저장 중...")
    result_json = build_output_json(
        image_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_JSON_PATH,
        model_id=MODEL_ID,
        image=image,
        depth_m=depth_m,
        intrinsics=intrinsics,
        floor_plane=floor_plane,
        edges=edges,
        used_objects=used_objects,
        obstacle_mask=obstacle_mask,
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
            obstacle_mask=obstacle_mask,
            used_objects=used_objects,
        )

    print("\n[완료] 저장 완료")
    print(f"- JSON: {OUTPUT_JSON_PATH}")
    print(f"- depth: {OUTPUT_JSON_PATH.with_suffix('')}_depth.png")
    print(f"- obstacle mask: {OUTPUT_JSON_PATH.with_suffix('')}_vision_obstacle_mask.png")
    print(f"- floor mask: {OUTPUT_JSON_PATH.with_suffix('')}_floor_mask.png")
    print(f"- boundary overlay: {OUTPUT_JSON_PATH.with_suffix('')}_boundary_overlay.png")

    print(f"\n[완료] VisionMessage에서 obstacle mask에 반영된 객체 수: {len(used_objects)}")
    print(f"[완료] 추출된 바닥-벽 경계 후보선 개수: {len(edges)}")

    for edge in edges:
        print(
            f'- {edge["edge_id"]}: '
            f'{edge["length_m_estimated"]:.3f} m, '
            f'pixel {edge["pixel_start"]} -> {edge["pixel_end"]}'
        )


if __name__ == "__main__":
    main()