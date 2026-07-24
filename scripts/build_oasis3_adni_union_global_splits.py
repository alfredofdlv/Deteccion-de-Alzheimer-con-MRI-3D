#!/usr/bin/env python3
"""OASIS-3 (todas las sesiones MNI) + ADNI union MNI → re-split global oasis3_adni."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _merge_global_split_utils import (
    build_and_write,
    load_adni_union_pt_pool,
    load_oasis_pt_pool,
    merge_pools,
    prefix_subjects,
)
from src.config import cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=365)
    parser.add_argument(
        "--variant", type=str, default="mni",
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado (mni, mni_n4, ...)",
    )
    args = parser.parse_args()

    print(f"[1/3] Pool OASIS {args.variant} (todas las sesiones)...")
    oasis = load_oasis_pt_pool(variant=args.variant, window=args.window)
    print(f"      {len(oasis)} sesiones, {oasis['subject_id'].nunique()} sujetos")

    print(f"[2/3] Pool ADNI union {args.variant}...")
    adni = load_adni_union_pt_pool(variant=args.variant)
    print(f"      {len(adni)} sesiones, {adni['subject_id'].nunique()} sujetos")

    print("[3/3] Merge + re-split global...")
    pool = merge_pools(
        prefix_subjects(oasis, "oas", "oasis3"),
        prefix_subjects(adni, "adni", "adni"),
    )
    total = build_and_write(pool, "oasis3_adni", variant=args.variant)
    print(f"[OK] Total muestras fusionadas union {args.variant}: {total}")


if __name__ == "__main__":
    main()
