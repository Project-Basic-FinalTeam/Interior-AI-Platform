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
            
            # 오려진 사진 안에서 "가장자리 1%를 제외한 안쪽 전부"를 탐색하라고 SAM에게 지시
            inset_x = max(2, int(img_w * 0.01))
            inset_y = max(2, int(img_h * 0.01))
            sam_bbox = [inset_x, inset_y, max(inset_x + 1, img_w - inset_x), max(inset_y + 1, img_h - inset_y)]

            print(f"[3DGS] 🧹 {yolo_crop_path} 파일 내부에서 집중 누끼를 따는 중...")
            results = sam_model(yolo_crop_path, bboxes=[sam_bbox], retina_masks=True, verbose=False)
            
            if results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    
                mask = (mask * 255).astype(np.uint8)
                
                # 🔥 [사용자 명령 적용] 묻지도 따지지도 않고 마스크를 완벽하게 뒤집어버립니다!
                print("[3DGS] 🔄 마스크 강제 반전(Invert) 실행!")
                mask = 255 - mask
                
                # 끊어진 틈새 부드럽게 연결
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # 면적 0.1% 이하의 자잘한 허공 먼지만 제거
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                min_area_threshold = max(50, (img_w * img_h) * 0.001) 
                
                clean_mask = np.zeros_like(mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                        clean_mask[labels == i] = 255
                mask = clean_mask

                mask = cv2.GaussianBlur(mask, (3, 3), 0)

                # 투명도 적용
                orig_rgba[:, :, 3] = mask
                
                # 스마트 크롭 (진짜 픽셀 기준 타이트하게)
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    cx, cy, cw, ch = cv2.boundingRect(coords)
                    final_rgba_array = orig_rgba[cy:cy+ch, cx:cx+cw]
                else:
                    final_rgba_array = orig_rgba

                isolated_rgba = Image.fromarray(final_rgba_array)
                
                # LGM 전용 프레이밍 (정중앙 512x512 배치)
                bbox_tight = isolated_rgba.getbbox()
                if bbox_tight:
                    isolated_rgba = isolated_rgba.crop(bbox_tight)
                
                CANVAS_SIZE = 512
                scale = (CANVAS_SIZE * 0.85) / max(isolated_rgba.size)
                new_w = int(isolated_rgba.size[0] * scale)
                new_h = int(isolated_rgba.size[1] * scale)
                
                resized_rgba = isolated_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                white_bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
                paste_x = (CANVAS_SIZE - new_w) // 2
                paste_y = (CANVAS_SIZE - new_h) // 2
                white_bg.paste(resized_rgba, (paste_x, paste_y), mask=resized_rgba)
                
                final_rgb_img = white_bg.convert("RGB")
                
                target_image_path = os.path.join(SHARED_DIR, f"crop_sam_{object_id}.jpg")
                final_rgb_img.save(target_image_path, format="JPEG")
                
                # 디버그용 파일 저장
                debug_sam_path = os.path.join(SHARED_DIR, f"debug_sam_{object_id}.jpg")
                final_rgb_img.save(debug_sam_path, format="JPEG")
                
                print(f"[3DGS] ✅ 독립 파일 전용 누끼 완료! 디버그 파일: debug_sam_{object_id}.jpg")
            else:
                print("[3DGS] ⚠️ SAM 마스크 실패. 원본 Crop을 그대로 사용합니다.")
                isolated_rgba = Image.fromarray(orig_rgba).convert("RGB")
                target_image_path = os.path.join(SHARED_DIR, f"crop_sam_{object_id}.jpg")
                isolated_rgba.save(target_image_path)
            
        except Exception as e:
            print(f"[3DGS] 🚨 작업 중 에러 발생: {e}")
    else:
        print(f"[3DGS] 🚨 에러: {yolo_crop_path} 파일이 폴더에 존재하지 않습니다! YOLO가 저장을 실패했습니다.")

    cmd = [
        sys.executable,
        "/app/lgm_core/infer.py",
        "big",
        "--resume", MODEL_PATH,
        "--test_path", target_image_path, 
        "--workspace", SHARED_DIR
    ]
    
    try:
        # 🔥 [필수 안전장치 복구] check=True로 설정하여 에러 발생 시 즉시 유니티로 예외를 던집니다. (70초 타임아웃 방지)
        subprocess.run(cmd, check=True)
        print("[3DGS] 엔진 추론 프로세스 완료 (PLY 파일 확인)")
    except subprocess.CalledProcessError as e:
        print(f"[3DGS] ❌ AI 엔진 실행 중 오류: {e}")
        raise RuntimeError("LGM 3D 렌더링 엔진에서 치명적인 에러가 발생했습니다.")
    
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
            print(f"[3DGS] ✅ 3D 가구 생성 완료: {output_filename}")
        else:
            print(f"[3DGS] 🚨 에러 발생: PLY 파일이 없습니다.")
    
    if target_image_path != yolo_crop_path and os.path.exists(target_image_path):
        os.remove(target_image_path)
        
    return output_filename

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