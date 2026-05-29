"""
generate_trellis_ply_from_images.py

로컬 이미지 폴더의 이미지들을 TRELLIS에 넣어서 3D Gaussian PLY를 생성하고,
InteriorPlatform/shared/assets_3dgs 폴더에 저장합니다.

실행 전제:
1. TRELLIS repo가 로컬에 clone되어 있어야 합니다.
2. 이 스크립트는 TRELLIS가 설치된 Python/Conda 환경에서 실행해야 합니다.
3. NVIDIA CUDA GPU가 있는 PC에서 실행해야 합니다.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PROJECT_ROOT = Path(r"C:\SWPJ_____4\Interior-AI-Platform")
DEFAULT_IMAGE_DIR = DEFAULT_PROJECT_ROOT / "infrastructure" / "product_db" / "data" / "images"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "InteriorPlatform" / "shared" / "assets_3dgs"
DEFAULT_TRELLIS_REPO = Path(r"C:\SWPJ_____4\TRELLIS")
DEFAULT_LOG_DIR = DEFAULT_PROJECT_ROOT / "logs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name


def collect_images(image_dir: Path, limit: int | None = None) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: str(p).lower())

    if limit is not None and limit > 0:
        images = images[:limit]

    return images


def make_output_path(
    image_path: Path,
    output_dir: Path,
    used_names: set[str],
) -> Path:
    base_name = sanitize_filename(image_path.stem) + ".ply"

    if base_name not in used_names:
        used_names.add(base_name)
        return output_dir / base_name

    parent_name = sanitize_filename(image_path.parent.name)
    final_name = f"{parent_name}_{base_name}"
    counter = 2

    while final_name in used_names:
        final_name = sanitize_filename(f"{parent_name}_{image_path.stem}_{counter}.ply")
        counter += 1

    used_names.add(final_name)
    return output_dir / final_name


def add_trellis_to_path(trellis_repo: Path) -> None:
    if not trellis_repo.exists():
        raise FileNotFoundError(f"TRELLIS repo does not exist: {trellis_repo}")

    trellis_pkg = trellis_repo / "trellis"
    if not trellis_pkg.exists():
        raise FileNotFoundError(f"Invalid TRELLIS repo. Missing folder: {trellis_pkg}")

    sys.path.insert(0, str(trellis_repo))


def load_trellis_pipeline(trellis_repo: Path, model_name: str):
    """
    TRELLIS pipeline을 한 번만 로드합니다.
    이미지마다 모델을 다시 로드하면 너무 느립니다.
    """
    os.environ.setdefault("SPCONV_ALGO", "native")
    add_trellis_to_path(trellis_repo)

    from trellis.pipelines import TrellisImageTo3DPipeline

    print(f"[INFO] Loading TRELLIS model: {model_name}")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
    pipeline.cuda()
    print("[INFO] TRELLIS pipeline loaded on CUDA")

    return pipeline


def generate_gaussian_ply(
    pipeline,
    image_path: Path,
    output_ply_path: Path,
    seed: int,
    sparse_steps: int | None = None,
    sparse_cfg: float | None = None,
    slat_steps: int | None = None,
    slat_cfg: float | None = None,
) -> None:
    """
    이미지 1장을 TRELLIS에 넣고 3D Gaussian PLY를 저장합니다.
    출력 PLY는 mesh PLY가 아니라 3DGS/Gaussian PLY입니다.
    """
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")

    run_kwargs = {"seed": seed}

    sparse_params = {}
    if sparse_steps is not None:
        sparse_params["steps"] = sparse_steps
    if sparse_cfg is not None:
        sparse_params["cfg_strength"] = sparse_cfg
    if sparse_params:
        run_kwargs["sparse_structure_sampler_params"] = sparse_params

    slat_params = {}
    if slat_steps is not None:
        slat_params["steps"] = slat_steps
    if slat_cfg is not None:
        slat_params["cfg_strength"] = slat_cfg
    if slat_params:
        run_kwargs["slat_sampler_params"] = slat_params

    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        outputs = pipeline.run(image, **run_kwargs)

    if "gaussian" not in outputs or not outputs["gaussian"]:
        raise RuntimeError("TRELLIS output does not contain gaussian result")

    outputs["gaussian"][0].save_ply(str(output_ply_path))

    if not output_ply_path.exists():
        raise RuntimeError(f"PLY was not created: {output_ply_path}")


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TRELLIS 3D Gaussian PLY files from local product images."
    )

    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trellis_repo", default=str(DEFAULT_TRELLIS_REPO))
    parser.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--model_name", default="microsoft/TRELLIS-image-large")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # 속도/품질 조절용. 지정 안 하면 TRELLIS 기본값 사용.
    parser.add_argument("--sparse_steps", type=int, default=None)
    parser.add_argument("--sparse_cfg", type=float, default=None)
    parser.add_argument("--slat_steps", type=int, default=None)
    parser.add_argument("--slat_cfg", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    trellis_repo = Path(args.trellis_repo)
    log_dir = Path(args.log_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir, args.limit)

    if not images:
        print(f"[WARN] No images found in: {image_dir}")
        return

    print(f"[INFO] image_dir: {image_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] trellis_repo: {trellis_repo}")
    print(f"[INFO] found images: {len(images)}")

    used_names: set[str] = set()
    tasks: list[tuple[Path, Path]] = []

    for image_path in images:
        output_ply_path = make_output_path(image_path, output_dir, used_names)
        tasks.append((image_path, output_ply_path))

    if args.dry_run:
        print("[DRY_RUN] TRELLIS will not run.")
        for image_path, output_ply_path in tasks:
            print(f"[DRY_RUN] {image_path} -> {output_ply_path}")
        return

    pipeline = load_trellis_pipeline(trellis_repo, args.model_name)

    generated_rows: list[dict] = []
    failed_rows: list[dict] = []

    for index, (image_path, output_ply_path) in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {image_path.name}")

        if output_ply_path.exists() and not args.overwrite:
            print(f"[SKIP] Already exists: {output_ply_path}")
            generated_rows.append({
                "image_path": str(image_path),
                "output_ply_path": str(output_ply_path),
                "output_type": "3dgs",
                "status": "skipped_exists",
            })
            continue

        try:
            generate_gaussian_ply(
                pipeline=pipeline,
                image_path=image_path,
                output_ply_path=output_ply_path,
                seed=args.seed,
                sparse_steps=args.sparse_steps,
                sparse_cfg=args.sparse_cfg,
                slat_steps=args.slat_steps,
                slat_cfg=args.slat_cfg,
            )

            print(f"[OK] Saved: {output_ply_path}")
            generated_rows.append({
                "image_path": str(image_path),
                "output_ply_path": str(output_ply_path),
                "output_type": "3dgs",
                "status": "generated",
            })

        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {reason}")
            traceback.print_exc()

            failed_rows.append({
                "image_path": str(image_path),
                "reason": reason,
            })

    write_csv(
        log_dir / "generated_trellis_ply.csv",
        generated_rows,
        ["image_path", "output_ply_path", "output_type", "status"],
    )
    write_csv(
        log_dir / "failed_trellis_ply.csv",
        failed_rows,
        ["image_path", "reason"],
    )

    print("\n[DONE]")
    print(f"- generated/skipped: {len(generated_rows)}")
    print(f"- failed: {len(failed_rows)}")
    print(f"- output_dir: {output_dir}")
    print(f"- logs: {log_dir}")


if __name__ == "__main__":
    main()
