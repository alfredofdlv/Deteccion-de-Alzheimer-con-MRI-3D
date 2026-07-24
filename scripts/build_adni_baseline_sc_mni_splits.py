#!/usr/bin/env python3
"""Genera adni_baseline_sc_*_pt_mni.csv cruzando stems con tensores en adni_mni/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg


def _pt_stem(path: str) -> str:
    return Path(path).stem


def _scan_mni_dir() -> dict[str, str]:
    mni_dir = cfg.preprocessed_dir("adni", "mni")
    if not mni_dir.is_dir():
        return {}
    return {p.stem: str(p.resolve()) for p in mni_dir.glob("*.pt")}


def build_split(split: str, mni_scan: dict[str, str]) -> int:
    baseline_csv = cfg.DATA_SPLITS_DIR / f"adni_baseline_sc_{split}_pt.csv"
    out_csv = cfg.DATA_SPLITS_DIR / f"adni_baseline_sc_{split}_pt_mni.csv"
    if not baseline_csv.exists():
        print(f"[WARN] No existe {baseline_csv}, saltando")
        return 0

    baseline_df = pd.read_csv(baseline_csv)
    rows: list[dict] = []
    missing = 0
    for _, row in baseline_df.iterrows():
        stem = _pt_stem(str(row["image_path"]))
        mni_path = mni_scan.get(stem)
        if mni_path is None:
            missing += 1
            continue
        out_row = {
            "subject_id": row["subject_id"],
            "image_path": mni_path,
            "label": row["label"],
        }
        for col in ("GENDER", "EDUC"):
            if col in row.index:
                out_row[col] = row[col]
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(
        f"  {out_csv.name}: {len(out_df)} muestras "
        f"(baseline_sc={len(baseline_df)}, sin MNI={missing})"
    )
    return len(out_df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    mni_scan = _scan_mni_dir()
    print(f"[INFO] Tensores adni_mni indexados: {len(mni_scan)}")
    total = sum(build_split(s, mni_scan) for s in ("train", "val", "test"))
    print(f"[OK] Total muestras baseline_sc MNI: {total}")


if __name__ == "__main__":
    main()
