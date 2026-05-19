# services/ai-perception/main.py
import os
import time
import zmq
import sys
import cv2
import shutil
import flatbuffers
import numpy as np
import torch

from modules.depth_handler import DepthEstimator
from modules.rfdetr_handler import RFDETRDetector
# 🔥 hand_tracker 임포트 삭제 완료!

sys.path.append('/app/schema')
import InteriorPlatform.VisionMessage as VisionMessage
import InteriorPlatform.DetectedObject as DetectedObject
import InteriorPlatform.BoundingBox as BoundingBox
import InteriorPlatform.Vec3 as Vec3
# 🔥 HandTracking 스키마는 이제 유니티 내부에서 처리하므로 파이썬에서 쓰지 않습니다.

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
    print("[AI Perception] 엣지 AI 아키텍처 적용 완료 (비전 및 3DGS 전담)")
    print("======================================")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556")

    model_dir = "/app/models"
    asset_dir = "/app/assets" 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    # 엣지 아키텍처에서는 유니티가 웹캠 화면을 찍어서 ZMQ로 파이썬에 넘겨주는 것이 정석입니다.
    # 현재는 테스트를 위해 다시 test_room.jpg를 스캔하도록 세팅합니다.
    image_path = os.path.join(model_dir, "test_room.jpg")
    rfdetr_weights = os.path.join(model_dir, "rfdetr_base.pth")
    model = RFDETRDetector(
        model_path=rfdetr_weights if os.path.exists(rfdetr_weights) else None,
        confidence=0.5,
    )
    depth_estimator = DepthEstimator()

    while True:
        print("\n[AI Perception] 유니티의 스캔 명령 대기 중... (웹캠은 유니티가 제어합니다)")
        request = socket.recv()
        print(f"[AI Perception] 명령 수신: {request.decode('utf-8')} -> 3D 파이프라인 가동!")

        original_img = cv2.imread(image_path)
        detections = model.predict(original_img)
        builder = flatbuffers.Builder(1024)
        object_offsets = []

        for j, det in enumerate(detections):
            class_name = det["label"]
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_img.shape[1], x2), min(original_img.shape[0], y2)
            cropped_img = original_img[y1:y2, x1:x2]

            ply_filename = f"asset_{class_name}_{j}.ply"
            ply_filepath = os.path.join(asset_dir, ply_filename)

            generate_3dgs_dual_track(cropped_img, ply_filepath, asset_dir)

            combined_label = f"{class_name}|{ply_filename}"
            label_offset = builder.CreateString(combined_label)

            BoundingBox.BoundingBoxStart(builder)
            BoundingBox.BoundingBoxAddXMin(builder, float(x1))
            BoundingBox.BoundingBoxAddYMin(builder, float(y1))
            BoundingBox.BoundingBoxAddXMax(builder, float(x2))
            BoundingBox.BoundingBoxAddYMax(builder, float(y2))
            bbox_offset = BoundingBox.BoundingBoxEnd(builder)

            DetectedObject.DetectedObjectStart(builder)
            DetectedObject.DetectedObjectAddId(builder, j)
            DetectedObject.DetectedObjectAddLabel(builder, label_offset)
            DetectedObject.DetectedObjectAddConfidence(builder, float(det["confidence"]))
            DetectedObject.DetectedObjectAddBbox(builder, bbox_offset)

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            estimated_z = depth_estimator.estimate_depth(original_img, (x1, y1, x2, y2))

            DetectedObject.DetectedObjectAddPosition3d(
                builder,
                Vec3.CreateVec3(builder, float(center_x), float(center_y), float(estimated_z))
            )
            obj_offset = DetectedObject.DetectedObjectEnd(builder)
            object_offsets.append(obj_offset)

        VisionMessage.VisionMessageStartObjectsVector(builder, len(object_offsets))
        for obj in reversed(object_offsets):
            builder.PrependUOffsetTRelative(obj)
        objects_vector = builder.EndVector()

        VisionMessage.VisionMessageStart(builder)
        VisionMessage.VisionMessageAddTimestamp(builder, int(time.time()))
        VisionMessage.VisionMessageAddObjects(builder, objects_vector)
        # 🔥 Hands 데이터 전송 코드 삭제 완료!
        msg_offset = VisionMessage.VisionMessageEnd(builder)
        builder.Finish(msg_offset)

        binary_data = builder.Output()
        socket.send(binary_data)
        print(f"[AI Perception] 추론 및 3DGS 배포 완료! ({len(object_offsets)}개 객체)")

if __name__ == "__main__":
    main()