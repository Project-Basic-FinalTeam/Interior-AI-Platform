# 파일 위치: /InteriorPlatform_Workspace/services/ai-perception/main.py

import os
import time
import zmq
import sys
import cv2
import shutil
import flatbuffers
import numpy as np
import torch
import json # 추가됨

from modules.rf_detr_handler import RFDETRHandler
from modules.depth_handler import DepthEstimator

sys.path.append('/app/schema')
import InteriorPlatform.VisionMessage as VisionMessage
import InteriorPlatform.DetectedObject as DetectedObject
import InteriorPlatform.BoundingBox as BoundingBox
import InteriorPlatform.Vec3 as Vec3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_3dgs_dual_track(cropped_img, save_path, asset_dir):
    if DEVICE == "cuda":
        pass 
    else:
        sample_source = os.path.join(asset_dir, "sample_plant.ply")
        if os.path.exists(sample_source):
            shutil.copy(sample_source, save_path)
        else:
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n"
            with open(save_path, 'wb') as f:
                f.write(header.encode('utf-8'))
                f.write(np.zeros(17, dtype=np.float32).tobytes())

def send_empty_response(socket):
    """오류 시 통신 단절 방지를 위한 빈 메시지 전송"""
    builder = flatbuffers.Builder(64)
    VisionMessage.VisionMessageStart(builder)
    VisionMessage.VisionMessageAddTimestamp(builder, int(time.time()))
    msg_offset = VisionMessage.VisionMessageEnd(builder)
    builder.Finish(msg_offset)
    socket.send(bytes(builder.Output()))

def main():
    print("======================================")
    print("🧠 [AI Perception] Roboflow RF-DETR Pipeline Initialized")
    print("======================================")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556")

    model_dir = "/app/models"
    asset_dir = "/app/assets" 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    image_path = os.path.join(model_dir, "test_room.jpg")
    weights_path = os.path.join(model_dir, "best.pt")
    
    vision_handler = RFDETRHandler(model_path=weights_path, device=DEVICE, confidence=0.05) 
    depth_estimator = DepthEstimator()

    while True:
        try:
            # 1. 요청 대기
            request = socket.recv()
            print(f"[AI Perception] Command received: Processing image...")

            original_img = cv2.imread(image_path)
            if original_img is None:
                raise FileNotFoundError(f"Image file not found: {image_path}")

            img_h, img_w = original_img.shape[:2]
            detected_objects = vision_handler.detect(original_img, threshold=0.05)
            
            builder = flatbuffers.Builder(1024)
            object_offsets = []
            valid_id = 0 
            debug_img = original_img.copy()

            for obj in detected_objects:
                orig_x1, orig_y1, orig_x2, orig_y2 = map(int, obj["bbox"])
                class_name = obj["label"]
                conf = obj["confidence"]

                orig_w = orig_x2 - orig_x1
                orig_h = orig_y2 - orig_y1
                img_area = img_w * img_h
                
                if orig_w < 30 or orig_h < 30: continue 
                if (orig_w * orig_h) > img_area * 0.8: continue 

                margin_x = int(orig_w * 0.08)
                margin_y = int(orig_h * 0.08)
                
                crop_x1 = max(0, orig_x1 - margin_x)
                crop_y1 = max(0, orig_y1 - margin_y)
                crop_x2 = min(img_w, orig_x2 + margin_x)
                crop_y2 = min(img_h, orig_y2 + margin_y)

                cv2.rectangle(debug_img, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                cv2.putText(debug_img, f"{class_name}_{valid_id}", (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cropped_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
                yolo_crop_path = os.path.join(asset_dir, f"yolo_crop_{valid_id}.jpg")
                cv2.imwrite(yolo_crop_path, cropped_img)
                
                safe_class_name = class_name.replace(' ', '_')
                ply_filename = f"asset_{safe_class_name}_{valid_id}.ply"
                ply_filepath = os.path.join(asset_dir, ply_filename)

                generate_3dgs_dual_track(cropped_img, ply_filepath, asset_dir)

                # 기존 1차 추정 (관통 오차가 포함되어 있을 수 있음)
                # depth_estimator가 계산한 실제 가구 크기(scale_w, scale_h)를 그대로 사용합니다.
                pos_x, pos_y, pos_z, scale_w, scale_h = depth_estimator.estimate_3d_position_and_scale(
                    original_img.shape, (orig_x1, orig_y1, orig_x2, orig_y2), label=class_name
                )
                
                # 🔥 핵심: JSON의 바닥 평면 방정식(RANSAC)을 이용한 Raycast 위치 정밀 보정
                result_json_path = os.path.join(asset_dir, "depth_result", "result.json")
                if os.path.exists(result_json_path):
                    try:
                        with open(result_json_path, 'r', encoding='utf-8') as f:
                            depth_data = json.load(f)
                            
                        intrinsics = depth_data.get("camera_assumption", {}).get("intrinsics", {})
                        plane = depth_data.get("floor_plane_estimation", {})

                        if intrinsics and plane:
                            fx, fy = intrinsics["fx"], intrinsics["fy"]
                            cx_img, cy_img = intrinsics["cx"], intrinsics["cy"]
                            
                            normal = plane.get("plane_normal_camera_coord", [0, 1, 0])
                            d = plane.get("plane_d", 0)
                            a, b, c = normal[0], normal[1], normal[2]

                            # 1) BBox의 맨 아랫부분(바닥과 닿는 지점 y2)을 향하는 카메라 Ray 벡터 계산
                            bx = (orig_x1 + orig_x2) / 2.0
                            ray_x = (bx - cx_img) / fx
                            ray_y = (orig_y2 - cy_img) / fy 

                            # 2) Ray와 바닥 평면이 만나는 교차점의 정확한 Z(깊이) 역산
                            denominator = (a * ray_x) + (b * ray_y) + c
                            if abs(denominator) > 1e-6:
                                new_z = -d / denominator
                                
                                # 계산된 깊이가 현실적일 경우에만 100% 덮어쓰기
                                if 0.5 < new_z < 10.0:
                                    pos_z = new_z
                                    
                                    # 3) 새로 찾은 Z값에 비례하여 X, Y 위치만 재계산 (원근법 보정)
                                    # 🚨 가구 크기(scale_w, scale_h)는 팀원분이 계산한 depth_estimator 원본 값을 유지합니다!
                                    pos_x = ray_x * pos_z
                                    by = (orig_y1 + orig_y2) / 2.0
                                    
                                    # depth_handler 로직과 동일하게 Y축 반전 (유니티 기준)
                                    pos_y = -(by - cy_img) * pos_z / fy 
                                    
                                    print(f"[AI Perception] 🛠️ {class_name}_{valid_id} Raycast pos updated: Z={pos_z:.2f}m (Scale kept: {scale_w:.2f}x{scale_h:.2f}m)")
                    except Exception as e:
                        print(f"[WARN] Raycast correction failed (using original estimation): {e}")

                combined_label = f"{class_name}|{ply_filename}|{scale_w:.2f}|{scale_h:.2f}"
                label_offset = builder.CreateString(combined_label)

                BoundingBox.BoundingBoxStart(builder)
                BoundingBox.BoundingBoxAddXMin(builder, float(orig_x1))
                BoundingBox.BoundingBoxAddYMin(builder, float(orig_y1))
                BoundingBox.BoundingBoxAddXMax(builder, float(orig_x2))
                BoundingBox.BoundingBoxAddYMax(builder, float(orig_y2))
                bbox_offset = BoundingBox.BoundingBoxEnd(builder)

                DetectedObject.DetectedObjectStart(builder)
                DetectedObject.DetectedObjectAddId(builder, valid_id)
                DetectedObject.DetectedObjectAddLabel(builder, label_offset)
                DetectedObject.DetectedObjectAddConfidence(builder, conf)
                DetectedObject.DetectedObjectAddBbox(builder, bbox_offset)
                DetectedObject.DetectedObjectAddPosition3d(
                    builder,
                    Vec3.CreateVec3(builder, float(pos_x), float(pos_y), float(pos_z))
                )
                obj_offset = DetectedObject.DetectedObjectEnd(builder)
                object_offsets.append(obj_offset)
                valid_id += 1

            VisionMessage.VisionMessageStartObjectsVector(builder, len(object_offsets))
            for obj in reversed(object_offsets):
                builder.PrependUOffsetTRelative(obj)
            objects_vector = builder.EndVector()

            VisionMessage.VisionMessageStart(builder)
            VisionMessage.VisionMessageAddTimestamp(builder, int(time.time()))
            VisionMessage.VisionMessageAddObjects(builder, objects_vector)
            msg_offset = VisionMessage.VisionMessageEnd(builder)
            builder.Finish(msg_offset)

            debug_save_path = os.path.join(asset_dir, "debug_rfdetr_boxes.jpg")
            cv2.imwrite(debug_save_path, debug_img)
            
            # 🔥 핵심: 반드시 send 호출
            socket.send(bytes(builder.Output()))
            print(f"[AI Perception] Inference complete! (Detected {valid_id} objects)")

        except Exception as e:
            print(f"[AI Perception] ❌ Fatal error during processing: {e}")
            send_empty_response(socket)

if __name__ == "__main__":
    main()