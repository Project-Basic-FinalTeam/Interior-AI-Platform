import os
import time
import zmq
import sys
import cv2
import shutil
import flatbuffers
import numpy as np
import torch

# 🔥 기존 RF-DETR 임포트 삭제 및 YOLO 임포트 추가
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
    print("🧠 [AI Perception] YOLO-World (Master Dictionary) 아키텍처 가동")
    print("======================================")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556")

    model_dir = "/app/models"
    asset_dir = "/app/assets" 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    image_path = os.path.join(model_dir, "test_room.jpg")
    
    # ================================================================
    # 🔥 [수정 1] YOLO-World 모델 로드 및 마스터 사전 주입
    # ================================================================
    print("[AI Perception] YOLO-World 가중치 다운로드 및 로드 중 (최초 1회 소요)...")
    # 가장 가볍고 빠른 Small 버전을 사용하여 렉을 최소화합니다.
    model = YOLO("yolov8s-worldv2.pt")  
    
    # 우리의 디지털 트윈 플랫폼이 찾아내야 할 실내 가구/소품 마스터 리스트
    MASTER_CLASSES = [
        "bed", "sofa", "dining table", "coffee table", "desk", "chair", "armchair", 
        "couch", "potted plant", "houseplant", "cabinet", "closet", "tv", "monitor", 
        "lamp", "rug", "shelf", "refrigerator", "drawer", "nightstand", "trash can", 
        "painting", "mirror", "bookshelf", "stool", "washing machine", "microwave"
    ]
    # 모델에 우리가 정의한 단어들만 찾도록 강제 주입
    model.set_classes(MASTER_CLASSES)
    print(f"[AI Perception] ✅ 마스터 사전 주입 완료: 총 {len(MASTER_CLASSES)}개 가구/소품 실시간 인식 준비 끝!")
    # ================================================================

    depth_estimator = DepthEstimator()

    while True:
        print("\n[AI Perception] 유니티의 스캔 명령 대기 중... (웹캠은 유니티가 제어합니다)")
        request = socket.recv()
        print(f"[AI Perception] 명령 수신: {request.decode('utf-8')} -> 지능형 파이프라인 가동!")

        original_img = cv2.imread(image_path)
        img_h, img_w = original_img.shape[:2]

        # 🔥 [수정 2] YOLO-World 추론 실행 (conf 0.05로 지정된 단어만 귀신같이 찾아냅니다)
        results = model.predict(original_img, conf=0.05, iou=0.3)
        
        builder = flatbuffers.Builder(1024)
        object_offsets = []

        valid_id = 0  # 크기가 통과된 객체만 세기 위한 번호표
        debug_img = original_img.copy()

        # 🔥 [수정 3] 결과 파싱 루프 (YOLO의 박스 객체에서 데이터 추출)
        for box in results[0].boxes:
            # 좌표, 클래스 ID, 확신도 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            class_name = MASTER_CLASSES[cls_id]
            conf = float(box.conf[0].item())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            w = x2 - x1
            h = y2 - y1
            
            img_area = img_w * img_h
            box_area = w * h
            
            # 크기 필터링 로직 유지 (먼지 및 방 전체 컷)
            if w < 30 or h < 30:
                continue 
                
            if box_area > img_area * 0.8:
                print(f"[필터링] '{class_name}' 객체가 방 전체를 덮고 있어 무시합니다.")
                continue 

            # 🔥 [디버그] 통과된 객체들의 네모 박스와 이름을 이미지에 그립니다.
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{class_name}_{valid_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cropped_img = original_img[y1:y2, x1:x2]

            # 파일명에 띄어쓰기가 있으면 버그가 날 수 있으므로 언더스코어(_)로 치환
            safe_class_name = class_name.replace(' ', '_')
            ply_filename = f"asset_{safe_class_name}_{valid_id}.ply"
            ply_filepath = os.path.join(asset_dir, ply_filename)

            generate_3dgs_dual_track(cropped_img, ply_filepath, asset_dir)

            # 카메라 하향각 및 물리적 크기(Scale) 동기화
            pos_x, pos_y, pos_z, scale_w, scale_h = depth_estimator.estimate_3d_position_and_scale(original_img.shape, (x1, y1, x2, y2))

            # 유니티로 스케일을 넘겨주기 위해 Label에 크기 정보를 합칩니다.
            combined_label = f"{class_name}|{ply_filename}|{scale_w:.2f}|{scale_h:.2f}"
            label_offset = builder.CreateString(combined_label)

            BoundingBox.BoundingBoxStart(builder)
            BoundingBox.BoundingBoxAddXMin(builder, float(x1))
            BoundingBox.BoundingBoxAddYMin(builder, float(y1))
            BoundingBox.BoundingBoxAddXMax(builder, float(x2))
            BoundingBox.BoundingBoxAddYMax(builder, float(y2))
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

        # 🔥 [디버그] 파일명을 구분하기 위해 _yoloworld 로 변경 저장
        debug_save_path = os.path.join(asset_dir, "debug_yoloworld_boxes.jpg")
        cv2.imwrite(debug_save_path, debug_img)
        print(f"[AI Perception] 📸 디버그용 바운딩 박스 이미지 저장 완료: {debug_save_path}")

        binary_data = builder.Output()
        socket.send(binary_data)
        print(f"[AI Perception] 추론 및 3DGS 배포 완료! (최종 {valid_id}개 객체 탐지)")

if __name__ == "__main__":
    main()