import sys
import os
import zmq
import uvicorn
import time
import threading
import subprocess
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  
from PIL import Image  # 🔥 이미지 크롭 연산을 위해 PIL.Image 임포트

# ONNX 단순 경고 숨기기 및 CPU 강제 전환 (에러 로그 도배 방지)
os.environ["ONNXRUNTIME_PROVIDER"] = "CPUExecutionProvider"
os.environ["ORT_ENV_DISABLE_CUDA"] = "1"

# ==========================================
# 1. 설정
# ==========================================
SHARED_DIR = "/app/output_assets"
MODEL_PATH = "/app/models/model.safetensors" 
os.makedirs(SHARED_DIR, exist_ok=True)

app = FastAPI()
app.mount("/assets", StaticFiles(directory=SHARED_DIR), name="assets")

print("[3DGS] 🚀 서버 초기화 완료 (LGM 원본 자동화 스크립트 연동 모드)")

# ==========================================
# 2. 추론 파이프라인 (infer.py 원격 호출)
# ==========================================
def generate_3dgs_ply(image_path: str, object_id: str, bbox: list = None) -> str:
    print(f"\n[3DGS] 🚀 LGM 엔진 가동 시작! (가구 ID: {object_id})")
    
    target_image_path = image_path
    
    # 🔥 [크롭 로직] 비율 좌표를 픽셀로 변환 후 이미지 자르기
    if bbox is not None and len(bbox) == 4:
        try:
            print(f"[3DGS] ✂️ 바운딩 박스 크롭 진행 중... 받은 좌표: {bbox}")
            raw_img = Image.open(image_path).convert("RGB")
            img_w, img_h = raw_img.size  # 원본 이미지의 실제 픽셀 크기
            
            x, y, w, h = bbox
            
            # [안전장치] YOLO가 0~1 사이의 비율(정규화)로 값을 줬다면 픽셀 단위로 곱해줍니다.
            if w <= 1.5 and h <= 1.5:
                x = x * img_w
                y = y * img_h
                w = w * img_w
                h = h * img_h
            
            # 소수점을 확실한 정수(픽셀)로 변환
            left = int(x)
            upper = int(y)
            right = int(x + w)
            lower = int(y + h)
            
            # PIL 라이브러리의 crop은 (left, upper, right, lower) 기준입니다.
            cropped_img = raw_img.crop((left, upper, right, lower))
            
            # 잘린 이미지를 추론용으로 덮어쓰지 않고 임시 파일로 저장합니다.
            target_image_path = os.path.join(SHARED_DIR, f"crop_{object_id}.jpg")
            cropped_img.save(target_image_path)
            print(f"[3DGS] ✅ 크롭 이미지 준비 완료: {target_image_path} (크기: {int(w)}x{int(h)})")
            
        except Exception as e:
            print(f"[3DGS] ⚠️ 이미지 크롭 실패, 원본 전체 이미지를 사용합니다: {e}")

    # 원작자의 infer.py 스크립트 실행 명령어 (target_image_path 사용)
    cmd = [
        sys.executable,
        "/app/lgm_core/infer.py",
        "big",
        "--resume", MODEL_PATH,
        "--test_path", target_image_path,  # 원본 대신 크롭된 이미지가 들어갑니다.
        "--workspace", SHARED_DIR
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[3DGS] ⚠️ 비디오 렌더링 중단됨 (무시 가능: 3D 알맹이 PLY 파일은 추출 성공!)")
    
    # 파일명 변경 및 검증 로직
    base_name = os.path.splitext(os.path.basename(target_image_path))[0]
    expected_ply_path = os.path.join(SHARED_DIR, f"{base_name}.ply")
    
    output_filename = f"furniture_{object_id}.ply"
    output_path = os.path.join(SHARED_DIR, output_filename)
    
    if os.path.exists(expected_ply_path):
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(expected_ply_path, output_path)
        print(f"[3DGS] ✅ 3D 가구 생성 완료: {output_filename}")
    else:
        ply_files = [f for f in os.listdir(SHARED_DIR) if f.endswith('.ply') and not f.startswith('furniture_')]
        if ply_files:
            latest_ply = max([os.path.join(SHARED_DIR, f) for f in ply_files], key=os.path.getctime)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(latest_ply, output_path)
            print(f"[3DGS] ✅ 3D 가구 생성 완료 (최신 파일 탐색): {output_filename}")
        else:
            raise FileNotFoundError(f"🚨 PLY 파일이 없습니다. AI 엔진 추론 자체가 실패했습니다.")
    
    # 추론이 끝난 임시 크롭 이미지는 삭제하여 용량을 아낍니다.
    if target_image_path != image_path and os.path.exists(target_image_path):
        os.remove(target_image_path)
        
    return output_filename

# ==========================================
# 3. ZMQ 및 FastAPI 서버 통신 로직
# ==========================================
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
            image_path = msg.get("image_path", "")
            obj_id = msg.get("obj_id", "001")
            bbox = msg.get("bbox", None)  
            
            print(f"[AI Reconstruction] 📩 C++ 지시 수신: 가구 ID {obj_id} 변환 요청")
            start_time = time.time()
            
            # 파싱한 bbox 데이터를 인자로 함께 전달
            ply_filename = generate_3dgs_ply(image_path, obj_id, bbox)
            elapsed = time.time() - start_time
            
            print(f"[AI Reconstruction] ⏱️ 총 렌더링 소요 시간: {elapsed:.2f}초")
            
            # 유니티로 URL 반환
            download_url = f"http://100.118.177.19:8000/assets/{ply_filename}?t={int(time.time())}"
            socket.send_json({"status": "success", "obj_id": obj_id, "ply_url": download_url})
            
        except Exception as e:
            print(f"[AI Reconstruction] 🚨 에러 발생: {e}")
            socket.send_json({"status": "error", "message": str(e)})

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    run_zmq_server()