import sys
import os
import zmq
import uvicorn
import time
import threading
import subprocess
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  
from PIL import Image 
# 🔥 배경 제거 라이브러리 임포트 완료!
from rembg import remove

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

print("[3DGS] 🚀 서버 초기화 완료 (LGM 원본 + 배경 제거 누끼 모드 통합)")

# ==========================================
# 2. 추론 파이프라인 (infer.py 원격 호출)
# ==========================================
def generate_3dgs_ply(image_path: str, object_id: str, bbox: list = None) -> str:
    print(f"\n[3DGS] 🚀 LGM 엔진 가동 시작! (가구 ID: {object_id})")
    
    target_image_path = image_path
    
    if bbox is not None and len(bbox) == 4:
        try:
            print(f"[3DGS] ✂️ 바운딩 박스 크롭 및 누끼 따기 시작... 좌표: {bbox}")
            raw_img = Image.open(image_path).convert("RGB")
            img_w, img_h = raw_img.size
            
            x, y, w, h = bbox
            
            if w <= 1.5 and h <= 1.5:
                x = x * img_w
                y = y * img_h
                w = w * img_w
                h = h * img_h
            
            left = int(x)
            upper = int(y)
            right = int(x + w)
            lower = int(y + h)
            
            # 1. 이미지 자르기
            cropped_img = raw_img.crop((left, upper, right, lower))

            # 2. 투명하게 배경 제거
            print("[3DGS] 🧹 가구만 남기고 배경 날리는 중 (누끼 따기)...")
            isolated_rgba = remove(cropped_img)
            
            # 3. 투명 배경 밑에 '순백색(White)' 도화지를 깔아줍니다.
            white_bg = Image.new("RGBA", isolated_rgba.size, (255, 255, 255, 255))
            white_bg.paste(isolated_rgba, mask=isolated_rgba)
            final_rgb_img = white_bg.convert("RGB")
            
            # 4. 저장
            target_image_path = os.path.join(SHARED_DIR, f"crop_{object_id}.jpg")
            final_rgb_img.save(target_image_path, format="JPEG")
            
            print(f"[3DGS] ✅ 누끼 추출 및 흰색 배경 합성 완료: {target_image_path} (크기: {int(w)}x{int(h)})")
            
        except Exception as e:
            print(f"[3DGS] ⚠️ 이미지 크롭/합성 실패, 원본 전체 이미지를 사용합니다: {e}")

    # 원작자의 infer.py 스크립트 실행 명령어 (투명 배경의 PNG 파일이 들어갑니다)
    cmd = [
        sys.executable,
        "/app/lgm_core/infer.py",
        "big",
        "--resume", MODEL_PATH,
        "--test_path", target_image_path, 
        "--workspace", SHARED_DIR
    ]
    
    try:
        # 비디오 렌더링 에러는 무시하도록 세팅 (LGM infer.py 내부의 비디오 생성 과정이 CPU에서 에러나는 경우가 많음)
        subprocess.run(cmd, check=False)
        print("[3DGS] 엔진 추론 프로세스 완료 (PLY 파일 확인)")
    except subprocess.CalledProcessError as e:
        print(f"[3DGS] ❌ AI 엔진 실행 중 치명적 오류: {e}")
    
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
            print(f"[3DGS] 🚨 에러 발생: PLY 파일이 없습니다. AI 엔진 추론 자체가 실패했습니다.")
    
    # 추론이 끝난 임시 크롭 이미지(.png)는 삭제하여 용량을 아낍니다.
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
            print(f"[AI Reconstruction] 🚨 ZMQ 루프 내부 에러 발생: {e}")
            socket.send_json({"status": "error", "message": str(e)})

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    run_zmq_server()