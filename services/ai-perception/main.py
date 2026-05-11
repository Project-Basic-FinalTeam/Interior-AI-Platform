# services/ai-perception/main.py

import os
import time
import zmq
import cv2
import sys
import flatbuffers
from ultralytics import YOLO

# 컴파일된 FlatBuffers 파이썬 클래스 경로 추가
sys.path.append('/app/schema')
import InteriorPlatform.VisionMessage as VisionMessage
import InteriorPlatform.DetectedObject as DetectedObject
import InteriorPlatform.BoundingBox as BoundingBox
import InteriorPlatform.Vec3 as Vec3
import InteriorPlatform.HandTracking as HandTracking

def main():
    print("======================================")
    print("[AI Perception] 비전 서비스 기동 중...")
    print("======================================")

    # 1. ZMQ 설정
    zmq_url = os.getenv("ZMQ_TARGET_URL", "tcp://logic-cpp:5556")
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(zmq_url)

    # 2. 모델 및 이미지 경로
    model_dir = "/app/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "yolo11n.pt")
    image_path = os.path.join(model_dir, "test_room.jpg")

    model = YOLO(model_path if os.path.exists(model_path) else "yolo11n.pt")
    if not os.path.exists(model_path): model.save(model_path)

    print("[AI Perception] 추론 루프 시작...")

    while True:
        if not os.path.exists(image_path):
            print(f"[경고] {image_path} 사진 파일이 없습니다. 사진을 넣어주세요!")
            time.sleep(3)
            continue

        # YOLO 추론
        results = model(image_path, verbose=False)
        builder = flatbuffers.Builder(1024)

        object_offsets = []
        for result in results:
            for box in result.boxes:
                # 1) String 생성 (테이블 시작 전 가능)
                label_name = model.names[int(box.cls[0])]
                label_offset = builder.CreateString(label_name)

                # 2) BoundingBox (Table) 생성 및 Offset 확보
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                BoundingBox.BoundingBoxStart(builder)
                BoundingBox.BoundingBoxAddXMin(builder, float(x1))
                BoundingBox.BoundingBoxAddYMin(builder, float(y1))
                BoundingBox.BoundingBoxAddXMax(builder, float(x2))
                BoundingBox.BoundingBoxAddYMax(builder, float(y2))
                bbox_offset = BoundingBox.BoundingBoxEnd(builder)

                # --- 3) DetectedObject (Table) 조립 시작 ---
                DetectedObject.DetectedObjectStart(builder)
                DetectedObject.DetectedObjectAddId(builder, 1)
                DetectedObject.DetectedObjectAddLabel(builder, label_offset)
                DetectedObject.DetectedObjectAddConfidence(builder, float(box.conf[0]))
                DetectedObject.DetectedObjectAddBbox(builder, bbox_offset)

                # [핵심 수정] Struct(Vec3)는 반드시 Table을 만들고 있는 도중에 "In-line"으로 넣어야 함
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                # CreateVec3를 별도의 변수에 담지 않고 Add 함수 안에서 즉시 호출
                DetectedObject.DetectedObjectAddPosition3d(
                    builder, 
                    Vec3.CreateVec3(builder, float(center_x), float(center_y), 2.5)
                )

                obj_offset = DetectedObject.DetectedObjectEnd(builder)
                object_offsets.append(obj_offset)

        # 4) Objects Vector 생성
        VisionMessage.VisionMessageStartObjectsVector(builder, len(object_offsets))
        for obj in reversed(object_offsets):
            builder.PrependUOffsetTRelative(obj)
        objects_vector = builder.EndVector()

        # 5) HandTracking (Table) 생성
        HandTracking.HandTrackingStart(builder)
        HandTracking.HandTrackingAddIsPinching(builder, False)
        hand_offset = HandTracking.HandTrackingEnd(builder)

        # 6) 최종 VisionMessage 조립
        VisionMessage.VisionMessageStart(builder)
        VisionMessage.VisionMessageAddTimestamp(builder, int(time.time()))
        VisionMessage.VisionMessageAddObjects(builder, objects_vector)
        VisionMessage.VisionMessageAddHands(builder, hand_offset)
        msg_offset = VisionMessage.VisionMessageEnd(builder)

        builder.Finish(msg_offset)

        # 전송
        binary_data = builder.Output()
        socket.send(binary_data)

        print(f"[AI Perception] 사진 분석 완료: {len(object_offsets)}개 객체 전송됨 -> C++")
        time.sleep(3)

if __name__ == "__main__":
    main()