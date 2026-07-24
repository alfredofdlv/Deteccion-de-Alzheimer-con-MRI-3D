"""
prepare_adni_splits.py — Genera adni_*_{train,val,test}_pt.csv apuntando a .pt existentes.

Cuando los tensores ya están en data/preprocessed/adni/ (p. ej. tras preprocess_to_pt),
este script mapea rutas NIfTI del manifest a rutas .pt sin reprocesar imágenes.

Uso:
    python scripts/prepare_adni_splits.py --label-mode baseline_sc
    python scripts/prepare_adni_splits.py --label-mode union
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_oasis3_splits import print_split_summary
from src.config import cfg

LABEL_MODE_DATASET = {
    "baseline_sc": "adni_baseline_sc",
    "union": "adni",
    "baseline": "adni_baseline",
    "cdr_longitudinal": "adni_longitudinal",
}


def nifti_to_pt_path(nifti_path: str, pt_dir: Path) -> Path | None:
    """Convierte ruta NIfTI ADNI a ruta .pt preprocesada."""
    stem = Path(nifti_path).stem.replace(".nii", "")
    pt_path = pt_dir / f"{stem}.pt"
    return pt_path if pt_path.exists() else None


def build_pt_split(
    split: str,
    dataset: str,
    pt_dir: Path,
    variant: str,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    manifest_prefix = dataset if dataset != "adni" else "adni"
    csv_path = cfg.DATA_SPLITS_DIR / f"{manifest_prefix}_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe manifest: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = []
    missing: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        pt_path = nifti_to_pt_path(row["image_path"], pt_dir)
        if pt_path is None:
            missing.append((row["subject_id"], row["image_path"]))
            continue
        entry = {
            "subject_id": row["subject_id"],
            "image_path": str(pt_path.resolve()),
            "label": int(row["label"]),
        }
        if "GENDER" in row.index:
            entry["GENDER"] = row["GENDER"]
        if "EDUC" in row.index:
            entry["EDUC"] = row["EDUC"]
        rows.append(entry)

    out_df = pd.DataFrame(rows)
    out_csv = cfg.DATA_SPLITS_DIR / cfg.split_csv_name(dataset, split, variant)
    out_df.to_csv(out_csv, index=False)
    return out_df, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar splits ADNI *_pt.csv")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="baseline_sc",
        choices=list(LABEL_MODE_DATASET.keys()),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=cfg.PREPROCESS_VARIANT_DEFAULT,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset = LABEL_MODE_DATASET[args.label_mode]
    pt_dir = cfg.preprocessed_dir("adni", args.variant)
    if not pt_dir.exists():
        print(f"[ERROR] No existe directorio .pt: {pt_dir}")
        sys.exit(1)

    print(f"[INFO] label-mode={args.label_mode} → dataset={dataset}")
    print(f"[INFO] .pt dir: {pt_dir}")

    all_missing: list[tuple[str, str]] = []
    for split in ("train", "val", "test"):
        out_df, missing = build_pt_split(split, dataset, pt_dir, args.variant)
        all_missing.extend(missing)
        print_split_summary(split, out_df)
        out_name = cfg.split_csv_name(dataset, split, args.variant)
        print(f"  → {out_name}: {len(out_df)} muestras")

    if all_missing:
        print(f"\n[WARN] {len(all_missing)} imágenes sin .pt correspondiente")
        if args.verbose:
            for sid, path in all_missing[:20]:
                print(f"  {sid}: {path}")
        if len(all_missing) > 20:
            print(f"  ... y {len(all_missing) - 20} más")

    print("\n[OK] Splits *_pt.csv generados.")


if __name__ == "__main__":
    main()
