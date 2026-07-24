#!/usr/bin/env python3
"""
enrich_splits_etiv.py — Añade columna eTIV_norm a los CSV de split.

- ADNI: EICV (eTIV) desde UCSDVOL unido por (PTID, VISCODE) vía RID.
- OASIS-3: proxy = conteo de voxels de foreground en tensor .pt / ETIV_NORM_DIVISOR.

Uso:
    python scripts/enrich_splits_etiv.py --dataset oasis3_adni --variant mni
    python scripts/enrich_splits_etiv.py --dataset oasis3 --variant mni
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

ADNI_PT_VIS_RE = re.compile(r"^(?P<ptid>\d+_S_\d+)_(?P<viscode>[a-z0-9]+)_(?P<imgid>\d+)$")


def _latest_csv(folder: Path, prefix: str) -> Path:
    matches = sorted(folder.glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No se encontro {prefix}_*.csv en {folder}")
    return matches[-1]


def build_ptid_rid_map() -> pd.DataFrame:
    apoe_path = _latest_csv(
        cfg.ADNI_DOCS_DIR / "02_study_files" / "subject_characteristics",
        "APOERES",
    )
    df = pd.read_csv(apoe_path, usecols=["PTID", "RID"])
    df["PTID"] = df["PTID"].astype(str)
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    return df.dropna(subset=["RID"]).drop_duplicates(subset=["PTID"], keep="first")


def build_adni_eicv_table() -> pd.DataFrame:
    ucsd_path = _latest_csv(
        cfg.ADNI_DOCS_DIR / "03_imaging" / "analysis",
        "UCSDVOL",
    )
    df = pd.read_csv(ucsd_path, usecols=["RID", "VISCODE", "EICV"])
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df["VISCODE"] = df["VISCODE"].astype(str).str.lower()
    df["EICV"] = pd.to_numeric(df["EICV"], errors="coerce")
    df = df.dropna(subset=["RID", "EICV"])
    return df.drop_duplicates(subset=["RID", "VISCODE"], keep="last")


def parse_adni_ptid_viscode(subject_id: str, image_path: str) -> tuple[str, str]:
    sid = str(subject_id)
    if "::" in sid:
        sid = sid.split("::", 1)[1]
    stem = Path(image_path).stem
    m = ADNI_PT_VIS_RE.match(stem)
    if m:
        return m.group("ptid"), m.group("viscode").lower()
    return sid, "sc"


def load_etiv_cache() -> dict[str, float]:
    path = cfg.ETIV_CACHE_CSV
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["image_path"].astype(str), df["eTIV_norm"].astype(float)))


def save_etiv_cache(cache: dict[str, float]) -> None:
    cfg.ETIV_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"image_path": list(cache.keys()), "eTIV_norm": list(cache.values())},
    ).to_csv(cfg.ETIV_CACHE_CSV, index=False)


def compute_oasis_proxy_etiv(image_path: str) -> float:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "image" in obj:
            obj = obj["image"]
        elif "data" in obj:
            obj = obj["data"]
    arr = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
    arr = np.squeeze(arr)
    fg_count = float((arr > cfg.ETIV_FOREGROUND_THRESHOLD).sum())
    return fg_count / cfg.ETIV_NORM_DIVISOR


def enrich_dataframe(
    df: pd.DataFrame,
    adni_eicv: pd.DataFrame,
    ptid_rid: pd.DataFrame,
    cache: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    etiv_vals: list[float] = []
    adni_lookup = adni_eicv.merge(ptid_rid, on="RID", how="left")
    adni_lookup = adni_lookup.dropna(subset=["PTID", "EICV"])
    adni_map = adni_lookup.set_index(["PTID", "VISCODE"])["EICV"].to_dict()
    adni_median = float(adni_lookup["EICV"].median()) / cfg.ETIV_NORM_DIVISOR

    for _, row in out.iterrows():
        img_path = str(row["image_path"])
        if img_path in cache:
            etiv_vals.append(cache[img_path])
            continue

        source = str(row.get("source", "")).lower()
        is_adni = source == "adni" or str(row.get("subject_id", "")).startswith("adni::")

        if is_adni:
            ptid, viscode = parse_adni_ptid_viscode(row["subject_id"], img_path)
            eicv = adni_map.get((ptid, viscode))
            if eicv is None:
                # fallback: cualquier visita del sujeto
                matches = adni_lookup[adni_lookup["PTID"] == ptid]["EICV"]
                eicv = float(matches.iloc[-1]) if len(matches) else adni_median * cfg.ETIV_NORM_DIVISOR
            val = float(eicv) / cfg.ETIV_NORM_DIVISOR
        else:
            val = compute_oasis_proxy_etiv(img_path)

        cache[img_path] = val
        etiv_vals.append(val)

    out["eTIV_norm"] = etiv_vals
    return out


def enrich_dataset_splits(dataset: str, variant: str) -> None:
    adni_eicv = build_adni_eicv_table()
    ptid_rid = build_ptid_rid_map()
    cache = load_etiv_cache()

    for split in ("train", "val", "test"):
        csv_path = cfg.DATA_SPLITS_DIR / cfg.split_csv_name(dataset, split, variant)
        if not csv_path.is_file():
            print(f"[SKIP] No existe {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        enriched = enrich_dataframe(df, adni_eicv, ptid_rid, cache)
        enriched.to_csv(csv_path, index=False)
        n_adni = (enriched.get("source", pd.Series(dtype=str)).astype(str).str.lower() == "adni").sum()
        if "source" not in enriched.columns:
            n_adni = 0
        print(
            f"[OK] {csv_path.name}: n={len(enriched)} "
            f"eTIV_norm mean={enriched['eTIV_norm'].mean():.4f} "
            f"min={enriched['eTIV_norm'].min():.4f} max={enriched['eTIV_norm'].max():.4f}"
            + (f" adni_rows={n_adni}" if n_adni else "")
        )

    save_etiv_cache(cache)
    print(f"[OK] Cache: {cfg.ETIV_CACHE_CSV} ({len(cache)} entradas)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enriquecer splits con eTIV_norm")
    parser.add_argument(
        "--dataset", type=str, default="oasis3_adni",
        help="Dataset de splits (default: oasis3_adni)",
    )
    parser.add_argument(
        "--variant", type=str, default="mni",
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
    )
    args = parser.parse_args()
    enrich_dataset_splits(args.dataset, args.variant)


if __name__ == "__main__":
    main()
