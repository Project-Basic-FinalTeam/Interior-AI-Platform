# FILE PATH: /InteriorPlatform_Workspace/services/ai-reconstruction/trellis_core/
# FILE NAME: infer.py

import argparse
import os
import torch
from PIL import Image
import sys

from trellis.pipelines import TrellisImageTo3DPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True, help="Input image path")
    parser.add_argument("--workspace", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    print("[TRELLIS] Starting Microsoft TRELLIS 3DGS Engine...")

    model_id = "JeffreyXiang/TRELLIS-image-large"
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    
    image = Image.open(args.test_path).convert("RGB")
    
    print(f"[TRELLIS] Imagining 3D geometric structure for '{args.test_path}'...")
    with torch.no_grad():
        outputs = pipeline.run(
            image,
            seed=42, 
            formats=["gaussian"], 
        )
    
    base_name = os.path.splitext(os.path.basename(args.test_path))[0]
    
    # 🔥 [고유 ID 추출 로직] crop_sam_0 에서 숫자 0을 분리해냅니다.
    try:
        obj_id = base_name.split('_')[-1]
        int(obj_id) # 숫자인지 검증
    except Exception:
        obj_id = "0"
        
    # 🔥 유니티가 애타게 찾고 있는 'furniture_X.ply' 이름으로 공유 볼륨에 정확히 저장합니다.
    output_ply_path = os.path.join(args.workspace, f"furniture_{obj_id}.ply")
    
    print(f"[TRELLIS] Extracting 3DGS data and saving to PLY format...")
    
    outputs['gaussian'][0].save_ply(output_ply_path)
    
    print(f"[TRELLIS] Successfully generated: {output_ply_path}")

if __name__ == "__main__":
    main()