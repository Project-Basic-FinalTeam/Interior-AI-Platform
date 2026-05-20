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

# 🔥 기존 rembg를 과감히 버리고, 초경량/초정밀 MobileSAM을 도입합니다!
from ultralytics import SAM

# ONNX 단순 경고 숨기기 및 CPU 강제 전환 (에러 로그 도배 방지)
os.environ["ONNXRUNTIME_PROVIDER"] = "CPUExecutionProvider"
os.environ["ORT_ENV_DISABLE_CUDA"] = "1"

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
SHARED_DIR = "/app/output_assets"
MODEL_PATH = "/app/models/model.safetensors" 
os.makedirs(SHARED_DIR, exist_ok=True)

app = FastAPI()
app.mount("/assets", StaticFiles(directory=SHARED_DIR), name="assets")

print("[3DGS] 🚀 서버 초기화 완료 (LGM 원본 + MobileSAM 정밀 누끼 모드 통합)")

# 🔥 [핵심 1] MobileSAM 모델 전역 로드 (서버 켤 때 한 번만 로드하여 딜레이 최소화)
print("[3DGS] MobileSAM 가중치 로드 중...")
sam_model = SAM("mobile_sam.pt")
print("[3DGS] ✅ MobileSAM 로드 완료!")

# ==========================================
# 2. 추론 파이프라인 (infer.py 원격 호출)
# ==========================================
def generate_3dgs_ply(image_path: str, object_id: str, bbox: list = None) -> str:
    print(f"\n[3DGS] 🚀 LGM 엔진 가동 시작! (가구 ID: {object_id})")
    
    target_image_path = image_path
    
    if bbox is not None and len(bbox) == 4:
        try:
            print(f"[3DGS] ✂️ MobileSAM에게 바운딩 박스 힌트 전달 중... 좌표: {bbox}")
            
            # 1. 원본 이미지를 읽고 투명도(Alpha)를 넣을 수 있는 RGBA로 변환
            orig_img = cv2.imread(image_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGBA)
            img_h, img_w = orig_img.shape[:2]
            
            x, y, w, h = bbox
            
            # YOLO 정규화 좌표 방어 로직
            if w <= 1.5 and h <= 1.5:
                x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
            
            # 화면 밖으로 나가지 않게 안전장치
            left = max(0, int(x))
            upper = max(0, int(y))
            right = min(img_w, int(x + w))
            lower = min(img_h, int(y + h))
            
            # 🔥 [핵심 2] SAM 추론: 박스 영역 안에서 진짜 가구 외곽선만 픽셀 단위로 찾기
            print("[3DGS] 🧹 SAM이 가구 외곽선을 따라 정밀 누끼를 따는 중...")
            # retina_masks=True 옵션으로 원본 화질을 유지하며 마스크를 땁니다.
            results = sam_model(image_path, bboxes=[left, upper, right, lower], retina_masks=True, verbose=False)
            
            if results[0].masks is not None:
                # SAM이 찾은 흑백 마스크 (가구=1, 배경=0)
                mask = results[0].masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                
                # 🔥 원본 이미지의 투명도(Alpha) 채널에 SAM 마스크를 덮어씌움 (배경 즉시 투명화!)
                orig_img[:, :, 3] = mask
                
                # 투명해진 이미지에서 바운딩 박스 영역만 잘라내기
                cropped_rgba_array = orig_img[upper:lower, left:right]
                isolated_rgba = Image.fromarray(cropped_rgba_array)
                
                # 🔥 LGM 엔진이 입체감을 잘 잡도록 '순백색(White)' 도화지를 밑에 깔아줌
                white_bg = Image.new("RGBA", isolated_rgba.size, (255, 255, 255, 255))
                white_bg.paste(isolated_rgba, mask=isolated_rgba)
                final_rgb_img = white_bg.convert("RGB")
                
                target_image_path = os.path.join(SHARED_DIR, f"crop_{object_id}.jpg")
                final_rgb_img.save(target_image_path, format="JPEG")
                
                print(f"[3DGS] ✅ SAM 정밀 누끼 추출 및 흰색 배경 합성 완료 (크기: {int(w)}x{int(h)})")
            else:
                # 혹시라도 SAM이 물체를 못 찾았을 경우 일반 크롭으로 백업(Fallback)
                print("[3DGS] ⚠️ SAM이 마스크를 찾지 못했습니다. 일반 사각형 크롭으로 대체합니다.")
                raw_img = Image.open(image_path).convert("RGB")
                cropped_img = raw_img.crop((left, upper, right, lower))
                target_image_path = os.path.join(SHARED_DIR, f"crop_{object_id}.jpg")
                cropped_img.save(target_image_path)
            
        except Exception as e:
            print(f"[3DGS] 🚨 SAM 누끼 작업 중 치명적 에러 발생: {e}")

    # 원작자의 infer.py 스크립트 실행 명령어 (투명 배경 처리된 PNG가 들어갑니다)
    cmd = [
        sys.executable,
        "/app/lgm_core/infer.py",
        "big",
        "--resume", MODEL_PATH,
        "--test_path", target_image_path, 
        "--workspace", SHARED_DIR
    ]
    
    try:
        # 비디오 렌더링 에러는 무시하도록 세팅
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
    
    # 추론이 끝난 임시 크롭 이미지(.jpg)는 삭제하여 용량을 아낍니다.
    if target_image_path != image_path and os.path.exists(target_image_path):
        os.remove(target_image_path)
        
    return output_filename

# ==========================================
# 3. ZMQ 및 FastAPI 서버 통신 로직 (변경 없음)
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