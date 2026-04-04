"""
preprocess_to_pt.py — Preprocesar imagenes NIfTI/ANALYZE a tensores .pt.

Aplica las transforms determinísticas (Load, ChannelFirst, Orientation RAS,
ScaleIntensity percentiles, Resize 96³) una sola vez y guarda el resultado
como tensor PyTorch. Esto elimina el cuello de botella de I/O durante el
entrenamiento (~1.4s/batch -> ~0.01s/batch).

Genera tambien CSVs de splits con rutas a los .pt.

OASIS-3 (Linux): los CSV NIfTI con rutas validas se generan con
`prepare_oasis3_nifti_splits.py` (entrada: cfg.OASIS3_RAW_DIR, p. ej.
data/raw/OASIS-3/). Luego este script lee data/splits/oasis3_{train,val,test}.csv.

Uso:
    python preprocess_to_pt.py                     # OASIS-3 (default)
    python preprocess_to_pt.py --dataset oasis1    # OASIS-1
"""

import argparse
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
from monai.transforms import (
    Compose,
    CropForeground,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    Resize,
    ScaleIntensityRangePercentiles,
)

from src.config import cfg

APPROX_MB_PER_IMAGE = 3.4


def build_preprocess_pipeline() -> Compose:
    """Transforms determinísticas offline: Load → ChannelFirst → Orientation RAS
    → ScaleIntensity → CropForeground → Resize 96³."""
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRangePercentiles(
            lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForeground(
            select_fn=lambda x: x > 0.1,
            margin=2,
        ),
        Resize(spatial_size=cfg.IMAGE_SIZE),
    ])


def derive_pt_filename(image_path: str) -> str:
    """Genera un nombre unico .pt a partir de la ruta original (.nii.gz / .nii)."""
    p = Path(image_path)
    # p.stem de foo.nii.gz es foo.nii; quitar sufijo .nii para alinear con nombres previos
    return p.stem.replace(".nii", "") + ".pt"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesar imagenes a tensores .pt"
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=["oasis1", "oasis3"],
    )
    args = parser.parse_args()

    out_dir = cfg.DATA_DIR / "preprocessed" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_preprocess_pipeline()

    splits = ["train", "val", "test"]
    if args.dataset == "oasis1":
        csv_paths = {s: cfg.DATA_SPLITS_DIR / f"{s}.csv" for s in splits}
    else:
        csv_paths = {s: cfg.DATA_SPLITS_DIR / f"{args.dataset}_{s}.csv" for s in splits}

    all_images = {}
    for split, csv_path in csv_paths.items():
        if not csv_path.exists():
            print(f"[WARN] No existe {csv_path}, saltando split '{split}'")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = row["image_path"]
            if img_path not in all_images:
                all_images[img_path] = {
                    "subject_id": row["subject_id"],
                    "label": row["label"],
                    "splits": [],
                }
            all_images[img_path]["splits"].append(split)

    unique_paths = list(all_images.keys())
    total = len(unique_paths)
    print(f"[INFO] {total} imagenes unicas a preprocesar para '{args.dataset}'")
    print(f"[INFO] Destino: {out_dir}")
    print(f"[INFO] Tamaño estimado: ~{total * APPROX_MB_PER_IMAGE / 1024:.1f} GB\n")

    t_start = time.time()
    success = 0
    errors = []
    pt_map = {}

    for i, img_path in enumerate(unique_paths, 1):
        pt_name = derive_pt_filename(img_path)
        pt_path = out_dir / pt_name

        if pt_path.exists():
            pt_map[img_path] = str(pt_path)
            success += 1
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t_start
                eta = (elapsed / i) * (total - i)
                print(
                    f"\r  [{i}/{total}] {i/total:>6.1%} | "
                    f"{elapsed:.0f}s / ETA {eta:.0f}s | "
                    f"(skip, ya existe)",
                    end="", flush=True,
                )
            continue

        try:
            tensor = pipeline(img_path)
            torch.save(tensor.contiguous(), pt_path)
            pt_map[img_path] = str(pt_path)
            success += 1
        except Exception as e:
            errors.append((img_path, str(e)))
            print(f"\n  [ERROR] {img_path}: {e}")

        elapsed = time.time() - t_start
        eta = (elapsed / i) * (total - i)
        avg = elapsed / i
        print(
            f"\r  [{i}/{total}] {i/total:>6.1%} | "
            f"{elapsed:.0f}s / ETA {timedelta(seconds=int(eta))} | "
            f"avg: {avg:.2f}s/img | ok: {success} err: {len(errors)}",
            end="", flush=True,
        )

    elapsed_total = time.time() - t_start
    print(f"\n\n{'='*60}")
    print(f"PREPROCESAMIENTO COMPLETADO")
    print(f"  Exitosas:  {success}/{total}")
    print(f"  Errores:   {len(errors)}")
    print(f"  Tiempo:    {timedelta(seconds=int(elapsed_total))}")
    print(f"  Destino:   {out_dir}")
    print(f"{'='*60}\n")

    print("[INFO] Generando CSVs de splits con rutas .pt...")
    for split, csv_path in csv_paths.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        new_rows = []
        for _, row in df.iterrows():
            orig = row["image_path"]
            if orig in pt_map:
                new_rows.append({
                    "subject_id": row["subject_id"],
                    "image_path": pt_map[orig],
                    "label": row["label"],
                })
        new_df = pd.DataFrame(new_rows)

        if args.dataset == "oasis1":
            out_csv = cfg.DATA_SPLITS_DIR / f"{split}_pt.csv"
        else:
            out_csv = cfg.DATA_SPLITS_DIR / f"{args.dataset}_{split}_pt.csv"

        new_df.to_csv(out_csv, index=False)
        print(f"  {out_csv.name}: {len(new_df)} samples")

    if errors:
        print(f"\n[WARN] Imagenes con errores:")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
