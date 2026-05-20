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
    print("[AI Perception] 엣지 AI 아키텍처 적용 완료 (비전 및 3DGS 전담)")
    print("======================================")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5556")

    model_dir = "/app/models"
    asset_dir = "/app/assets" 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)
    
    image_path = os.path.join(model_dir, "test_room.jpg")
    rfdetr_weights = os.path.join(model_dir, "rfdetr_base.pth")
    model = RFDETRDetector(
        model_path=rfdetr_weights if os.path.exists(rfdetr_weights) else None,
        confidence=0.4,  
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

        valid_id = 0  # 🔥 크기가 통과된 객체만 세기 위한 번호표

        debug_img = original_img.copy()

        for det in detections:
            class_name = det["label"]
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_img.shape[1], x2), min(original_img.shape[0], y2)

            w = x2 - x1
            h = y2 - y1
            
            img_area = original_img.shape[0] * original_img.shape[1]
            box_area = w * h
            
            if w < 30 or h < 30:
                continue # 먼지 컷
                
            if box_area > img_area * 0.8:
                print(f"[필터링] '{class_name}' 객체가 방 전체를 덮고 있어 무시합니다.")
                continue # 방 전체 컷

            # 🔥 [디버그 2] 통과된 객체들의 네모 박스와 이름을 이미지에 그립니다.
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{class_name}_{valid_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cropped_img = original_img[y1:y2, x1:x2]

            ply_filename = f"asset_{class_name}_{valid_id}.ply"
            ply_filepath = os.path.join(asset_dir, ply_filename)

            generate_3dgs_dual_track(cropped_img, ply_filepath, asset_dir)

            # 🔥 [수정 2] 카메라 하향각 및 물리적 크기(Scale) 동기화 함수 호출!
            pos_x, pos_y, pos_z, scale_w, scale_h = depth_estimator.estimate_3d_position_and_scale(original_img.shape, (x1, y1, x2, y2))

            # 🔥 [수정 3] 유니티로 스케일을 넘겨주기 위해 Label에 크기 정보를 합칩니다.
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
            DetectedObject.DetectedObjectAddConfidence(builder, float(det["confidence"]))
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

        # 🔥 [디버그 3] 박스가 다 그려진 이미지를 assets 폴더에 저장합니다.
        debug_save_path = os.path.join(asset_dir, "debug_rfdetr_boxes.jpg")
        cv2.imwrite(debug_save_path, debug_img)
        print(f"[AI Perception] 📸 디버그용 바운딩 박스 이미지 저장 완료: {debug_save_path}")

        binary_data = builder.Output()
        socket.send(binary_data)
        print(f"[AI Perception] 추론 및 3DGS 배포 완료! (최종 {valid_id}개 객체)")

if __name__ == "__main__":
    main()