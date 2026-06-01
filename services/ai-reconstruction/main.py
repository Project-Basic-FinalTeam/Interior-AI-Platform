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

# 🔥 rembg 라이브러리 임포트
from rembg import remove, new_session 

# ONNX 단순 경고 숨기기
os.environ["ONNXRUNTIME_PROVIDER"] = "CPUExecutionProvider"
os.environ["ORT_ENV_DISABLE_CUDA"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

SHARED_DIR = "/app/output_assets"
# MODEL_PATH = "/app/models/model.safetensors" # TRELLIS는 자체 로더를 사용하므로 제거 무방
os.makedirs(SHARED_DIR, exist_ok=True)

app = FastAPI()
app.mount("/assets", StaticFiles(directory=SHARED_DIR), name="assets")

print("[3DGS] 🚀 서버 초기화 완료 (YOLO 사전 Crop + Rembg 고품질 누끼 모드)")

# 🔥 고품질 배경 제거 엔진(Rembg) 로드
print("[3DGS] 고품질 배경 제거 엔진(Rembg) 로드 중...")
rembg_session = new_session("u2net")
print("[3DGS] ✅ 누끼 엔진 로드 완료!")

def generate_3dgs_ply(object_id: str) -> str:
    print(f"\n[3DGS] 🚀 TRELLIS 엔진 가동 시작! (가구 ID: {object_id})")
    
    yolo_crop_path = os.path.join(SHARED_DIR, f"yolo_crop_{object_id}.jpg")
    target_image_path = yolo_crop_path
    
    if os.path.exists(yolo_crop_path):
        try:
            # 1. OpenCV로 이미지를 읽어 PIL 포맷으로 변환 (RGB)
            orig_bgr = cv2.imread(yolo_crop_path)
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(orig_rgb)

            print(f"[3DGS] 🧹 {yolo_crop_path} 배경 제거 및 정밀 누끼 추출 중...")
            
            # 2. Rembg를 이용한 완벽한 배경 제거
            isolated_rgba = remove(pil_img, session=rembg_session, post_process_mask=True)

            # 3. 빈 공간 크롭 (Tight BBox)
            bbox_tight = isolated_rgba.getbbox()
            if bbox_tight: 
                isolated_rgba = isolated_rgba.crop(bbox_tight)
                
            # 4. TRELLIS 전용 프레이밍 (흰색 배경에 85% 비율로 중앙 배치)
            CANVAS_SIZE = 512
            scale = (CANVAS_SIZE * 0.85) / max(isolated_rgba.size)
            new_w, new_h = int(isolated_rgba.size[0] * scale), int(isolated_rgba.size[1] * scale)
            resized_rgba = isolated_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            white_bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
            white_bg.paste(resized_rgba, ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2), mask=resized_rgba)
            final_rgb_img = white_bg.convert("RGB")
            
            # 5. 이미지 저장
            target_image_path = os.path.join(SHARED_DIR, f"crop_sam_{object_id}.jpg")
            final_rgb_img.save(target_image_path, format="JPEG")
            
            debug_sam_path = os.path.join(SHARED_DIR, f"debug_sam_{object_id}.jpg")
            final_rgb_img.save(debug_sam_path, format="JPEG")
            print(f"[3DGS] ✅ 누끼 및 이미지 저장 완료: {debug_sam_path}")
            
        except Exception as e:
            print(f"[3DGS] 🚨 작업 중 에러 발생: {e}")

    # 🔥 [핵심 변경] LGM 대신 TRELLIS 추론 엔진 실행
    print(f"[3DGS] 마이크로소프트 TRELLIS를 통한 3D 구조 추론 중...")
    cmd = [
        sys.executable, "/app/trellis_core/infer.py", 
        "--test_path", target_image_path, 
        "--workspace", SHARED_DIR
    ]
    subprocess.run(cmd, check=True)
    
    # 결과 PLY 정리
    base_name = os.path.splitext(os.path.basename(target_image_path))[0]
    expected_ply_path = os.path.join(SHARED_DIR, f"{base_name}.ply")
    output_path = os.path.join(SHARED_DIR, f"furniture_{object_id}.ply")
    
    if os.path.exists(expected_ply_path):
        os.replace(expected_ply_path, output_path)
    
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