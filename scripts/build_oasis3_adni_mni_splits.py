#!/usr/bin/env python3
"""Fusiona splits MNI OASIS-3 + ADNI en oasis3_adni_{split}_pt_mni.csv.

DEPRECADO: usar scripts/build_oasis3_adni_union_global_splits.py (re-split global).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

SOURCES = (("oasis3", "oas"), ("adni", "adni"))


def build_split(split: str) -> int:
    frames: list[pd.DataFrame] = []
    for dataset, prefix in SOURCES:
        csv_path = cfg.DATA_SPLITS_DIR / f"{dataset}_{split}_pt_mni.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Falta {csv_path}")
        df = pd.read_csv(csv_path).copy()
        df["subject_id"] = df["subject_id"].astype(str).map(
            lambda s: f"{prefix}::{s}" if not str(s).startswith(f"{prefix}::") else s
        )
        df["source"] = dataset
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    out_csv = cfg.DATA_SPLITS_DIR / f"oasis3_adni_{split}_pt_mni.csv"
    merged.to_csv(out_csv, index=False)
    print(f"  {out_csv.name}: {len(merged)} muestras")
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    total = sum(build_split(s) for s in ("train", "val", "test"))
    print(f"[OK] Total muestras fusionadas MNI: {total}")


if __name__ == "__main__":
    main()
