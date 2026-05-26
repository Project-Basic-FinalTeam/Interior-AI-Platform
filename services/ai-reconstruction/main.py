import sys
import os
import zmq
import uvicorn
import time
import threading
import subprocess
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  
from PIL import Image 

from ultralytics import SAM

# ONNX 단순 경고 숨기기
os.environ["ONNXRUNTIME_PROVIDER"] = "CPUExecutionProvider"
os.environ["ORT_ENV_DISABLE_CUDA"] = "1"

SHARED_DIR = "/app/output_assets"
MODEL_PATH = "/app/models/model.safetensors" 
os.makedirs(SHARED_DIR, exist_ok=True)

app = FastAPI()
app.mount("/assets", StaticFiles(directory=SHARED_DIR), name="assets")

print("[3DGS] 🚀 서버 초기화 완료 (YOLO 사전 Crop + SAM 무조건 강제 반전 누끼 모드)")

print("[3DGS] Meta SAM 2.1 Base+ 가중치 로드 중...")
sam_model = SAM("sam2.1_b.pt")
print("[3DGS] ✅ SAM 2.1 Base+ 로드 완료!")

def generate_3dgs_ply(object_id: str) -> str:
    print(f"\n[3DGS] 🚀 LGM 엔진 가동 시작! (가구 ID: {object_id})")
    
    yolo_crop_path = os.path.join(SHARED_DIR, f"yolo_crop_{object_id}.jpg")
    target_image_path = yolo_crop_path
    
    if os.path.exists(yolo_crop_path):
        try:
            orig_bgr = cv2.imread(yolo_crop_path)
            orig_rgba = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGBA)
            img_h, img_w = orig_bgr.shape[:2]
            
            inset_x = max(2, int(img_w * 0.01))
            inset_y = max(2, int(img_h * 0.01))
            sam_bbox = [inset_x, inset_y, max(inset_x + 1, img_w - inset_x), max(inset_y + 1, img_h - inset_y)]

            print(f"[3DGS] 🧹 {yolo_crop_path} 파일 내부에서 집중 누끼를 따는 중...")
            results = sam_model(yolo_crop_path, bboxes=[sam_bbox], retina_masks=True, verbose=False)
            
            if results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                if mask.ndim == 3 and mask.shape[0] == 1: mask = mask[0]
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask = (mask * 255).astype(np.uint8)

                # 🔥 1. 책장 내부 파먹힘 방지 (외곽선만 찾아서 안쪽을 꽉 채움)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled_mask = np.zeros_like(mask)
                cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
                mask = filled_mask

                # 2. 끊어진 틈새 부드럽게 연결
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # 3. 자잘한 노이즈 제거
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                min_area_threshold = max(50, (img_w * img_h) * 0.001) 
                clean_mask = np.zeros_like(mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                        clean_mask[labels == i] = 255
                mask = clean_mask
                mask = cv2.GaussianBlur(mask, (3, 3), 0)

                # 4. 투명도 및 크롭 처리
                orig_rgba[:, :, 3] = mask
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    cx, cy, cw, ch = cv2.boundingRect(coords)
                    final_rgba_array = orig_rgba[cy:cy+ch, cx:cx+cw]
                else:
                    final_rgba_array = orig_rgba

                isolated_rgba = Image.fromarray(final_rgba_array)
                bbox_tight = isolated_rgba.getbbox()
                if bbox_tight: isolated_rgba = isolated_rgba.crop(bbox_tight)
                
                # 5. LGM 전용 프레이밍
                CANVAS_SIZE = 512
                scale = (CANVAS_SIZE * 0.85) / max(isolated_rgba.size)
                new_w, new_h = int(isolated_rgba.size[0] * scale), int(isolated_rgba.size[1] * scale)
                resized_rgba = isolated_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
                white_bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
                white_bg.paste(resized_rgba, ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2), mask=resized_rgba)
                final_rgb_img = white_bg.convert("RGB")
                
                # 🔥 6. [복구됨] 이미지 저장 로직
                target_image_path = os.path.join(SHARED_DIR, f"crop_sam_{object_id}.jpg")
                final_rgb_img.save(target_image_path, format="JPEG")
                debug_sam_path = os.path.join(SHARED_DIR, f"debug_sam_{object_id}.jpg")
                final_rgb_img.save(debug_sam_path, format="JPEG")
                print(f"[3DGS] ✅ 누끼 및 이미지 저장 완료: {debug_sam_path}")
            
        except Exception as e:
            print(f"[3DGS] 🚨 작업 중 에러 발생: {e}")

    # LGM 추론 엔진 실행
    cmd = [sys.executable, "/app/lgm_core/infer.py", "big", "--resume", MODEL_PATH, 
           "--test_path", target_image_path, "--workspace", SHARED_DIR]
    subprocess.run(cmd, check=True)
    
    # 결과 PLY 정리
    base_name = os.path.splitext(os.path.basename(target_image_path))[0]
    expected_ply_path = os.path.join(SHARED_DIR, f"{base_name}.ply")
    output_path = os.path.join(SHARED_DIR, f"furniture_{object_id}.ply")
    
    if os.path.exists(expected_ply_path):
        if os.path.exists(output_path): os.remove(output_path)
        os.rename(expected_ply_path, output_path)
    
    if target_image_path != yolo_crop_path and os.path.exists(target_image_path):
        os.remove(target_image_path)
        
    return f"furniture_{object_id}.ply"

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

def run_zmq_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5557") 
    print("[AI Reconstruction] 🟢 유니티/C++ 통신 대기 중...")
    
    while True:
        try:
            msg = socket.recv_json()
            obj_id = msg.get("obj_id", "001")
            
            print(f"[AI Reconstruction] 📩 C++ 지시 수신: 가구 ID {obj_id} 변환 요청")
            start_time = time.time()
            
            ply_filename = generate_3dgs_ply(obj_id)
            elapsed = time.time() - start_time
            
            print(f"[AI Reconstruction] ⏱️ 렌더링 소요 시간: {elapsed:.2f}초")
            
            download_url = f"http://100.118.177.19:8000/assets/{ply_filename}?t={int(time.time())}"
            socket.send_json({"status": "success", "obj_id": obj_id, "ply_url": download_url})
            
        except Exception as e:
            print(f"[AI Reconstruction] 🚨 에러: {e}")
            socket.send_json({"status": "error", "message": str(e)})

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    run_zmq_server()