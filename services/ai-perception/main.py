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

sys.path.append('/app/schema')
import InteriorPlatform.VisionMessage as VisionMessage
import InteriorPlatform.DetectedObject as DetectedObject
import InteriorPlatform.BoundingBox as BoundingBox
import InteriorPlatform.Vec3 as Vec3
import InteriorPlatform.HandTracking as HandTracking

# --- 투트랙 하드웨어 자동 감지 시스템 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[System] 현재 감지된 하드웨어: {DEVICE.upper()}")

def generate_3dgs_dual_track(cropped_img, save_path, asset_dir):
    if DEVICE == "cuda":
        # 트랙 A: 고사양 GPU 유저 (진짜 LGM 가동)
        print("  -> 🚀 [GPU 모드] LGM 엔진을 가동하여 3DGS를 실시간 생성합니다...")
        # TODO: 향후 실제 LGM 추론 코드가 여기에 들어갑니다.
        # lgm_model.generate(cropped_img, save_path)
    else:
        # 트랙 B: 일반 PC 유저 (안전 모드 - 고품질 샘플 파일 제공)
        print("  -> 🛡️ [CPU 모드] 하드웨어 보호를 위해 LGM 대신 샘플 에셋을 할당합니다.")
        
        # 테스트용 고품질 .ply 파일이 있는지 확인
        sample_source = os.path.join(asset_dir, "sample_plant.ply")
        
        if os.path.exists(sample_source):
            shutil.copy(sample_source, save_path)
        else:
            # 샘플이 없으면 임시 더미 파일 생성 (유니티 에러 방지용 껍데기)
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n"
            with open(save_path, 'wb') as f:
                f.write(header.encode('utf-8'))
                f.write(np.zeros(17, dtype=np.float32).tobytes())


def main():
    print("======================================")
    print("[AI Perception] 비전 + 3DGS 서비스 대기 모드")
    print("======================================")

    # 1. ZMQ 설정 (REP: 요청이 올 때만 반응함)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556")

    model_dir = "/app/models"
    asset_dir = "/app/assets" # 3DGS 파일이 저장될 공유 폴더
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    image_path = os.path.join(model_dir, "test_room.jpg")
    model = YOLO(os.path.join(model_dir, "yolo11n.pt") if os.path.exists(os.path.join(model_dir, "yolo11n.pt")) else "yolo11n.pt")

    while True:
        # 2. C++의 요청 대기
        print("\n[AI Perception] C++의 추론 명령을 기다립니다...")
        request = socket.recv()
        print(f"[AI Perception] 명령 수신: {request.decode('utf-8')} -> 3DGS 파이프라인 가동!")

        if not os.path.exists(image_path):
            socket.send(b"ERROR: NO_IMAGE")
            continue

        # 원본 이미지 불러오기 (자르기 용도)
        original_img = cv2.imread(image_path)
        results = model(image_path, verbose=False)
        builder = flatbuffers.Builder(1024)

        object_offsets = []
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes):
                class_name = model.names[int(box.cls[0])]
                
                # 3. Bounding Box 좌표 추출 및 가구 이미지 Crop
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 이미지 경계 초과 방지
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(original_img.shape[1], x2), min(original_img.shape[0], y2)
                cropped_img = original_img[y1:y2, x1:x2]
                
                # 4. 투트랙 3DGS 생성 가동
                ply_filename = f"asset_{class_name}_{j}.ply"
                ply_filepath = os.path.join(asset_dir, ply_filename)
                
                generate_3dgs_dual_track(cropped_img, ply_filepath, asset_dir)
                print(f"  -> [{class_name}] 3DGS 에셋 준비 완료: {ply_filename}")

                # 5. 유니티 해독용 특수 라벨 생성 (이름|파일명)
                combined_label = f"{class_name}|{ply_filename}"
                label_offset = builder.CreateString(combined_label)

                # FlatBuffers 패킹
                BoundingBox.BoundingBoxStart(builder)
                BoundingBox.BoundingBoxAddXMin(builder, float(x1))
                BoundingBox.BoundingBoxAddYMin(builder, float(y1))
                BoundingBox.BoundingBoxAddXMax(builder, float(x2))
                BoundingBox.BoundingBoxAddYMax(builder, float(y2))
                bbox_offset = BoundingBox.BoundingBoxEnd(builder)

                DetectedObject.DetectedObjectStart(builder)
                DetectedObject.DetectedObjectAddId(builder, j) # 고유 ID
                DetectedObject.DetectedObjectAddLabel(builder, label_offset)
                DetectedObject.DetectedObjectAddConfidence(builder, float(box.conf[0]))
                DetectedObject.DetectedObjectAddBbox(builder, bbox_offset)

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                DetectedObject.DetectedObjectAddPosition3d(
                    builder,
                    Vec3.CreateVec3(builder, float(center_x), float(center_y), 2.5)
                )
                obj_offset = DetectedObject.DetectedObjectEnd(builder)
                object_offsets.append(obj_offset)

        VisionMessage.VisionMessageStartObjectsVector(builder, len(object_offsets))
        for obj in reversed(object_offsets):
            builder.PrependUOffsetTRelative(obj)
        objects_vector = builder.EndVector()

        HandTracking.HandTrackingStart(builder)
        HandTracking.HandTrackingAddIsPinching(builder, False)
        hand_offset = HandTracking.HandTrackingEnd(builder)

        VisionMessage.VisionMessageStart(builder)
        VisionMessage.VisionMessageAddTimestamp(builder, int(time.time()))
        VisionMessage.VisionMessageAddObjects(builder, objects_vector)
        VisionMessage.VisionMessageAddHands(builder, hand_offset)
        msg_offset = VisionMessage.VisionMessageEnd(builder)
        builder.Finish(msg_offset)

        # 6. C++로 압축된 결과 전송
        binary_data = builder.Output()
        socket.send(binary_data)
        print(f"[AI Perception] 추론 및 3DGS 배포 완료! ({len(object_offsets)}개 객체)")

if __name__ == "__main__":
    main()