# 파일 위치: /InteriorPlatform_Workspace/services/ai-reconstruction/main.py
# 소스 코드 맨 위에는 항상 파일 위치와 파일 명을 함께 주석으로 삽입할 것

import sys
import os
import zmq
import uvicorn
import time
import threading
import subprocess
import cv2
import numpy as np
import re               # 🔥 추가: 파일명에서 숫자만 추적하기 위한 정규식 라이브러리
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  
from PIL import Image 
import urllib.request   # URL 다운로드용
import urllib.parse     # URL 파싱용

from rembg import remove, new_session 

os.environ["ONNXRUNTIME_PROVIDER"] = "CPUExecutionProvider"
os.environ["ORT_ENV_DISABLE_CUDA"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

SHARED_DIR = "/app/output_assets"
os.makedirs(SHARED_DIR, exist_ok=True)

app = FastAPI()
app.mount("/assets", StaticFiles(directory=SHARED_DIR), name="assets")

print("[3DGS] 🚀 서버 초기화 완료 (YOLO 사전 Crop + Rembg 고품질 누끼 모드)")

print("[3DGS] 고품질 배경 제거 엔진(Rembg) 로드 중...")
rembg_session = new_session("u2net")
print("[3DGS] ✅ 누끼 엔진 로드 완료!")

def generate_3dgs_ply(object_id: str) -> str:
    print(f"\n[3DGS] 🚀 TRELLIS 엔진 가동 시작! (가구 ID: {object_id})")
    yolo_crop_path = os.path.join(SHARED_DIR, f"yolo_crop_{object_id}.jpg")
    target_image_path = yolo_crop_path
    
    if os.path.exists(yolo_crop_path):
        try:
            orig_bgr = cv2.imread(yolo_crop_path)
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(orig_rgb)

            print(f"[3DGS] 🧹 {yolo_crop_path} 배경 제거 및 정밀 누끼 추출 중...")
            isolated_rgba = remove(pil_img, session=rembg_session, post_process_mask=True)

            bbox_tight = isolated_rgba.getbbox()
            if bbox_tight: 
                isolated_rgba = isolated_rgba.crop(bbox_tight)
                
            CANVAS_SIZE = 512
            scale = (CANVAS_SIZE * 0.85) / max(isolated_rgba.size)
            new_w, new_h = int(isolated_rgba.size[0] * scale), int(isolated_rgba.size[1] * scale)
            resized_rgba = isolated_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            white_bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
            white_bg.paste(resized_rgba, ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2), mask=resized_rgba)
            final_rgb_img = white_bg.convert("RGB")
            
            target_image_path = os.path.join(SHARED_DIR, f"crop_sam_{object_id}.jpg")
            final_rgb_img.save(target_image_path, format="JPEG")
            
            debug_sam_path = os.path.join(SHARED_DIR, f"debug_sam_{object_id}.jpg")
            final_rgb_img.save(debug_sam_path, format="JPEG")
            print(f"[3DGS] ✅ 누끼 및 이미지 저장 완료: {debug_sam_path}")
            
        except Exception as e:
            print(f"[3DGS] 🚨 작업 중 에러 발생: {e}")

    if not os.path.exists(target_image_path):
        print(f"[3DGS] 🚨 에러: 3D 변환을 위한 원본 이미지가 존재하지 않습니다! ({target_image_path})")
        return "asset_unknown.ply"

    # ====================================================================
    # 🔥 [회원님 아이디어 적용] 원본 파일 대피 로직 (덮어쓰기 방지)
    # ====================================================================
    num_match = re.search(r'\d+', object_id)
    original_ply = ""
    backup_ply = ""
    
    # RAG 교체 요청("rag_" 포함)일 때만 기존 원본 파일(furniture_2.ply)을 숨깁니다.
    if num_match and "rag" in object_id:
        original_ply = os.path.join(SHARED_DIR, f"furniture_{num_match.group()}.ply")
        backup_ply = os.path.join(SHARED_DIR, f"backup_temp_{num_match.group()}.ply")
        if os.path.exists(original_ply):
            os.rename(original_ply, backup_ply)
            print(f"[3DGS] 🛡️ 원본 가구 보호: {original_ply} -> 백업 완료")

    print(f"[3DGS] 마이크로소프트 TRELLIS를 통한 3D 구조 추론 중...")
    cmd = [
        sys.executable, "/app/trellis_core/infer.py", 
        "--test_path", target_image_path, 
        "--workspace", SHARED_DIR
    ]
    subprocess.run(cmd, check=True)
    
    # ====================================================================
    # 🔥 출력물 이름 강제 교체 및 대피했던 원본 복구
    # ====================================================================
    base_name = os.path.splitext(os.path.basename(target_image_path))[0]
    expected_ply_path_1 = os.path.join(SHARED_DIR, f"{base_name}.ply")
    
    expected_ply_path_2 = ""
    if num_match:
        expected_ply_path_2 = os.path.join(SHARED_DIR, f"furniture_{num_match.group()}.ply")

    # 유니티에게 줄 최종 이름 (예: furniture_rag_2.ply)
    output_path = os.path.join(SHARED_DIR, f"furniture_{object_id}.ply") 
    
    if os.path.exists(expected_ply_path_1):
        os.replace(expected_ply_path_1, output_path)
    elif expected_ply_path_2 and os.path.exists(expected_ply_path_2):
        os.replace(expected_ply_path_2, output_path)
        print(f"[3DGS] 🔧 TRELLIS 출력물 이름 강제 보정 완료: {output_path}")

    # 🚨 대피시켜 둔 진짜 원본 파일(furniture_2.ply) 복구!
    if backup_ply and os.path.exists(backup_ply):
        os.rename(backup_ply, original_ply)
        print(f"[3DGS] 🛡️ 원본 가구 파일 안전하게 복구 완료: {original_ply}")

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
            image_url = msg.get("image_url", "") 
            
            print(f"[AI Reconstruction] 📩 C++ 지시 수신: 가구 ID {obj_id} 변환 요청")
            start_time = time.time()
            
            if image_url and image_url.startswith("gs://"):
                try:
                    parts = image_url[5:].split('/', 1)
                    bucket = parts[0]
                    path = urllib.parse.quote(parts[1], safe='')
                    http_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{path}?alt=media"
                    
                    save_path = os.path.join(SHARED_DIR, f"yolo_crop_{obj_id}.jpg")
                    print(f"[AI Reconstruction] 🌐 RAG 추천 이미지 다운로드 중... ({http_url})")
                    
                    req = urllib.request.Request(http_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
                        out_file.write(response.read())
                    print("[AI Reconstruction] 📥 이미지 다운로드 완료!")
                except Exception as dl_e:
                    print(f"[AI Reconstruction] 🚨 이미지 다운로드 실패: {dl_e}")
            
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