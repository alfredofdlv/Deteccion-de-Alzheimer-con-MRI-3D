#!/usr/bin/env python3
"""
benchmark_mni_n4_preprocess.py — Piloto de tiempos para variante mni_n4.

Procesa N imágenes y reporta desglose por etapa + ETA extrapolada.

Uso:
    python scripts/benchmark_mni_n4_preprocess.py \
        --dataset oasis3 --limit 10 --hdbet-device cuda --gpus 0
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_to_pt import (
    _bind_cuda_device,
    _set_thread_limits,
    derive_pt_filename,
)
from src.config import cfg
from src.spatial_preprocess import process_image_mni_pipeline

DATASET_COUNTS = {
    "oasis3": 2450,
    "adni": 3877,
    "total": 6327,
}


def _load_unique_paths(dataset: str, limit: int) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        csv_path = cfg.DATA_SPLITS_DIR / f"{dataset}_{split}.csv"
        if not csv_path.exists():
            continue
        for img_path in pd.read_csv(csv_path)["image_path"]:
            if img_path not in seen:
                seen.add(img_path)
                paths.append(img_path)
            if len(paths) >= limit:
                return paths
    return paths


def _fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mni_n4 preprocess")
    parser.add_argument("--dataset", type=str, default="oasis3", choices=["oasis1", "oasis3", "adni"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--hdbet-device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--compare-mni", action="store_true", help="También mide mni sin N4 en mismas rutas")
    args = parser.parse_args()

    _set_thread_limits(1)
    device = "cuda"
    if args.gpus:
        device = _bind_cuda_device(f"cuda:{args.gpus.split(',')[0].strip()}")
    elif args.hdbet_device == "cpu":
        device = "cpu"
    else:
        device = _bind_cuda_device("cuda")

    paths = _load_unique_paths(args.dataset, args.limit)
    if not paths:
        print("[ERROR] No hay rutas NIfTI en splits")
        sys.exit(1)

    out_dir = cfg.preprocessed_dir(args.dataset, "mni_n4")
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate = cfg.INTERMEDIATE_DIR / cfg.preprocessed_subdir(args.dataset, "mni_n4") / "_benchmark"
    intermediate.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: {args.dataset} | imgs: {len(paths)} | HD-BET: {device}")
    print(f"[INFO] N4: antspy (CPU) | destino: {out_dir}\n")

    rows: list[dict] = []
    t_global = time.perf_counter()

    for i, img_path in enumerate(paths, 1):
        stem = Path(img_path).stem.replace(".nii", "")
        work_dir = intermediate / stem
        work_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        tensor, meta = process_image_mni_pipeline(
            img_path,
            work_dir=work_dir,
            keep_intermediate=False,
            device=device,
            apply_n4=True,
        )
        wall = time.perf_counter() - t0
        timings = meta.get("timings_s", {})
        pt_path = out_dir / derive_pt_filename(img_path)
        import torch
        torch.save(tensor, pt_path)

        row = {
            "i": i,
            "wall_s": wall,
            "skull_strip_s": timings.get("skull_strip_s", float("nan")),
            "n4_s": timings.get("n4_s", float("nan")),
            "registration_s": timings.get("registration_s", float("nan")),
        }
        rows.append(row)
        print(
            f"  [{i}/{len(paths)}] wall={wall:.1f}s | "
            f"strip={row['skull_strip_s']:.1f}s n4={row['n4_s']:.1f}s reg={row['registration_s']:.1f}s"
        )

    total_wall = time.perf_counter() - t_global
    df = pd.DataFrame(rows)
    avg_wall = df["wall_s"].mean()
    avg_n4 = df["n4_s"].mean()
    avg_strip = df["skull_strip_s"].mean()
    avg_reg = df["registration_s"].mean()

    print("\n" + "=" * 60)
    print("BENCHMARK mni_n4")
    print(f"  Imágenes:     {len(paths)}")
    print(f"  Tiempo total: {_fmt_eta(total_wall)}")
    print(f"  Promedio/img: {avg_wall:.2f}s (strip {avg_strip:.1f} + n4 {avg_n4:.1f} + reg {avg_reg:.1f})")
    print("=" * 60)

    print("\nETA extrapolada (mni_n4):")
    for workers in (1, 4, 8):
        for label, n in DATASET_COUNTS.items():
            eta_s = n * avg_wall / workers
            print(f"  {label:8} n={n:5} workers={workers} → {_fmt_eta(eta_s)}")

    if args.compare_mni:
        print("\n[INFO] Comparando mni (sin N4) en mismas rutas...")
        mni_rows = []
        for img_path in paths:
            work_dir = intermediate / f"mni_{Path(img_path).stem}"
            work_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            _, meta = process_image_mni_pipeline(
                img_path, work_dir=work_dir, device=device, apply_n4=False,
            )
            mni_rows.append(time.perf_counter() - t0)
        avg_mni = sum(mni_rows) / len(mni_rows)
        print(f"  mni avg: {avg_mni:.2f}s | mni_n4 avg: {avg_wall:.2f}s | delta N4: +{avg_wall - avg_mni:.2f}s")


if __name__ == "__main__":
    main()
