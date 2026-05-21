import os
import time
import zmq
import sys
import cv2
import shutil
import flatbuffers
import numpy as np
import torch

from ultralytics import YOLO 
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

def main():
    print("======================================")
    print("🧠 [AI Perception] 완벽 분리 아키텍처 (중복 제거 + 정확한 3D Depth 복구)")
    print("======================================")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556")

    model_dir = "/app/models"
    asset_dir = "/app/assets" 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    image_path = os.path.join(model_dir, "test_room.jpg")
    
    print("[AI Perception] YOLO-World 가중치 로드 중...")
    model = YOLO("yolov8s-worldv2.pt")  
    
    MASTER_CLASSES = [
        "bed", "sofa", "dining table", "coffee table", "desk", "chair", "armchair", 
        "couch", "potted plant", "houseplant", "cabinet", "closet", "tv", "monitor", 
        "lamp", "rug", "shelf", "refrigerator", "drawer", "nightstand", "trash can", 
        "painting", "mirror", "bookshelf", "stool", "washing machine", "microwave"
    ]
    model.set_classes(MASTER_CLASSES)
    print(f"[AI Perception] ✅ 마스터 사전 주입 완료!")

    depth_estimator = DepthEstimator()

    while True:
        print("\n[AI Perception] 유니티의 스캔 명령 대기 중...")
        request = socket.recv()
        print(f"[AI Perception] 명령 수신: 지능형 파이프라인 가동!")

        original_img = cv2.imread(image_path)
        img_h, img_w = original_img.shape[:2]

        # 🔥 [해결 1] agnostic_nms=True 추가: 이름이 달라도 겹친 박스는 무조건 제거!
        results = model.predict(original_img, conf=0.05, iou=0.3, agnostic_nms=True)
        
        builder = flatbuffers.Builder(1024)
        object_offsets = []

        valid_id = 0 
        debug_img = original_img.copy()

        for box in results[0].boxes:
            # 🔥 [해결 2] 1. 유니티 전송 및 거리 계산용 '원본 타이트한 박스' (잘 보이던 코드와 100% 동일)
            orig_x1, orig_y1, orig_x2, orig_y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            class_name = MASTER_CLASSES[cls_id]
            conf = float(box.conf[0].item())

            orig_w = orig_x2 - orig_x1
            orig_h = orig_y2 - orig_y1
            
            img_area = img_w * img_h
            
            if orig_w < 30 or orig_h < 30: continue 
            if (orig_w * orig_h) > img_area * 0.8: continue 

            # 2. SAM 누끼용 8% 여백 추가 박스 (파일을 자르고 저장할 때만 사용!)
            margin_x = int(orig_w * 0.08)
            margin_y = int(orig_h * 0.08)
            
            crop_x1 = max(0, orig_x1 - margin_x)
            crop_y1 = max(0, orig_y1 - margin_y)
            crop_x2 = min(img_w, orig_x2 + margin_x)
            crop_y2 = min(img_h, orig_y2 + margin_y)

            cv2.rectangle(debug_img, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{class_name}_{valid_id}", (orig_x1, orig_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 여백을 포함하여 오려낸 이미지를 SAM 전용 폴더에 저장
            cropped_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
            yolo_crop_path = os.path.join(asset_dir, f"yolo_crop_{valid_id}.jpg")
            cv2.imwrite(yolo_crop_path, cropped_img)
            print(f"   -> ✂️ SAM 전용 누끼 소스(여백 포함) 저장 완료: {yolo_crop_path}")

            safe_class_name = class_name.replace(' ', '_')
            ply_filename = f"asset_{safe_class_name}_{valid_id}.ply"
            ply_filepath = os.path.join(asset_dir, ply_filename)

            generate_3dgs_dual_track(cropped_img, ply_filepath, asset_dir)

            # 🔥 거리 계산은 무조건 "원본 타이트한 박스(orig)"로 수행 (우주로 날아가는 버그 방지)
            pos_x, pos_y, pos_z, scale_w, scale_h = depth_estimator.estimate_3d_position_and_scale(
                original_img.shape, (orig_x1, orig_y1, orig_x2, orig_y2)
            )

            combined_label = f"{class_name}|{ply_filename}|{scale_w:.2f}|{scale_h:.2f}"
            label_offset = builder.CreateString(combined_label)

            BoundingBox.BoundingBoxStart(builder)
            # 유니티 BBox 전송도 무조건 "원본 타이트한 박스(orig)" 사용
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

        debug_save_path = os.path.join(asset_dir, "debug_yoloworld_boxes.jpg")
        cv2.imwrite(debug_save_path, debug_img)
        print(f"[AI Perception] 📸 디버그 이미지 저장 완료: {debug_save_path}")

        binary_data = builder.Output()
        socket.send(binary_data)
        print(f"[AI Perception] 추론 완료! (최종 {valid_id}개 객체 탐지)")

if __name__ == "__main__":
    main()