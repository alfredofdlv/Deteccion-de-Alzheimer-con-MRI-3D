#!/usr/bin/env python3
"""Compara eTIV_norm ADNI (EICV UCSDVOL) vs proxy OASIS (voxels foreground) en splits MNI."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg


def main() -> None:
    csv_path = cfg.DATA_SPLITS_DIR / cfg.split_csv_name("oasis3_adni", "train", "mni")
    if not csv_path.is_file():
        print(f"Falta {csv_path}. Ejecuta: python scripts/enrich_splits_etiv.py --dataset oasis3_adni --variant mni")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "eTIV_norm" not in df.columns:
        print("Columna eTIV_norm ausente. Ejecuta enrich_splits_etiv.py primero.")
        sys.exit(1)

    src = df.get("source", pd.Series(["?"] * len(df))).astype(str).str.lower()
    adni = df[src == "adni"]
    oasis = df[src == "oasis3"]

    print(f"=== eTIV_norm en {csv_path.name} (train) ===\n")
    for label, part in (("ADNI (EICV UCSDVOL)", adni), ("OASIS-3 (proxy voxels)", oasis)):
        if part.empty:
            print(f"{label}: sin filas")
            continue
        s = part["eTIV_norm"]
        print(
            f"{label}: n={len(part)}  mean={s.mean():.4f}  std={s.std():.4f}  "
            f"min={s.min():.4f}  max={s.max():.4f}  median={s.median():.4f}"
        )
    print(
        "\nSi LF+eTIV empeora en union pero mejora con --source-filter adni, "
        "el proxy OASIS es el sospechoso principal."
    )


if __name__ == "__main__":
    main()
