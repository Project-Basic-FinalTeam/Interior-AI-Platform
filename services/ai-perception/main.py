import os
import time
import zmq
import sys
import flatbuffers
from ultralytics import YOLO

sys.path.append('/app/schema')
import InteriorPlatform.VisionMessage as VisionMessage
import InteriorPlatform.DetectedObject as DetectedObject
import InteriorPlatform.BoundingBox as BoundingBox
import InteriorPlatform.Vec3 as Vec3
import InteriorPlatform.HandTracking as HandTracking

def main():
    print("======================================")
    print("[AI Perception] 비전 서비스 대기 모드 (REQ-REP)")
    print("======================================")

    # 1. ZMQ 설정 (REP: 요청이 올 때만 반응함)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556") # 서버로서 바인딩

    model_dir = "/app/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "yolo11n.pt")
    image_path = os.path.join(model_dir, "test_room.jpg")

    model = YOLO(model_path if os.path.exists(model_path) else "yolo11n.pt")
    if not os.path.exists(model_path): model.save(model_path)

    while True:
        # 2. C++의 요청 대기 (여기서 멈춰있으므로 CPU/GPU 점유율 0%)
        print("\n[AI Perception] C++의 추론 명령을 기다립니다...")
        request = socket.recv()
        print(f"[AI Perception] 명령 수신: {request.decode('utf-8')} -> YOLO 추론 시작!")

        if not os.path.exists(image_path):
            socket.send(b"ERROR: NO_IMAGE")
            continue

        # 3. 요청이 들어왔을 때만 딱 한 번 추론
        results = model(image_path, verbose=False)
        builder = flatbuffers.Builder(1024)

        object_offsets = []
        for result in results:
            for box in result.boxes:
                label_name = model.names[int(box.cls[0])]
                label_offset = builder.CreateString(label_name)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                BoundingBox.BoundingBoxStart(builder)
                BoundingBox.BoundingBoxAddXMin(builder, float(x1))
                BoundingBox.BoundingBoxAddYMin(builder, float(y1))
                BoundingBox.BoundingBoxAddXMax(builder, float(x2))
                BoundingBox.BoundingBoxAddYMax(builder, float(y2))
                bbox_offset = BoundingBox.BoundingBoxEnd(builder)

                DetectedObject.DetectedObjectStart(builder)
                DetectedObject.DetectedObjectAddId(builder, 1)
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

        # 4. C++로 압축된 결과 전송
        binary_data = builder.Output()
        socket.send(binary_data)
        print(f"[AI Perception] 추론 결과({len(object_offsets)}개 객체) 반환 완료.")

if __name__ == "__main__":
    main()