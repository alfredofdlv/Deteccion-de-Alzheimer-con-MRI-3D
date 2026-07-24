"""Utilidades para fusionar OASIS-3 + ADNI y re-split global estratificado."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

PATTERN = re.compile(r"sub-(OAS3\d+)_sess?-d(\d+)")


def scan_oasis_pt_files(variant: str = "mni") -> pd.DataFrame:
    pt_dir = cfg.preprocessed_dir("oasis3", variant)
    records = []
    for pt_file in sorted(pt_dir.glob("*.pt")):
        m = PATTERN.match(pt_file.name)
        if m is None:
            continue
        records.append({
            "subject_id": m.group(1),
            "scan_day": int(m.group(2)),
            "image_path": str(pt_file.resolve()),
        })
    return pd.DataFrame(records)


def scan_oasis_mni_pt_files() -> pd.DataFrame:
    """Compat: escanea oasis3_mni."""
    return scan_oasis_pt_files("mni")


def load_oasis_pt_pool(variant: str = "mni", window: int = 365) -> pd.DataFrame:
    scans = scan_oasis_pt_files(variant)
    clinical = pd.read_csv(cfg.OASIS3_CLINICAL_CSV)
    return match_oasis_labels(scans, clinical, window=window)


def load_oasis_mni_pool(window: int = 365) -> pd.DataFrame:
    return load_oasis_pt_pool("mni", window=window)


def match_oasis_labels(scans: pd.DataFrame, clinical: pd.DataFrame, window: int) -> pd.DataFrame:
    """Match scan ↔ CDR; conserva scan_day y distancia clínica para dedup."""
    clin = clinical[
        ["OASISID", "days_to_visit", "label", "age at visit", "GENDER", "EDUC", "APOE_e4"]
    ].dropna(subset=["label"]).copy()
    clin["days_to_visit"] = clin["days_to_visit"].astype(int)
    clin["label"] = clin["label"].astype(int)

    matched = []
    for _, row in scans.iterrows():
        subj_rows = clin[clin["OASISID"] == row["subject_id"]]
        if subj_rows.empty:
            continue
        diffs = (subj_rows["days_to_visit"] - row["scan_day"]).abs()
        min_diff = int(diffs.min())
        if min_diff > window:
            continue
        best = subj_rows.loc[diffs.idxmin()]
        matched.append({
            "subject_id": row["subject_id"],
            "scan_day": row["scan_day"],
            "clinical_distance": min_diff,
            "image_path": row["image_path"],
            "label": int(best["label"]),
            "age_at_visit": float(best["age at visit"]),
            "GENDER": str(best["GENDER"]),
            "EDUC": float(best["EDUC"]) if pd.notna(best["EDUC"]) else 12.0,
            "APOE_e4": float(best["APOE_e4"]) if pd.notna(best["APOE_e4"]) else 0.0,
        })
    return pd.DataFrame(matched)


def dedup_oasis_one_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Un scan por sujeto: visita clínica más cercana; empate → menor scan_day."""
    if df.empty:
        return df
    df = df.sort_values(["subject_id", "clinical_distance", "scan_day"])
    return df.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)


def _concat_split_csvs(prefix: str, variant: str = "mni") -> pd.DataFrame:
    variant_suffix = "" if variant == "cropped" else f"_{variant}"
    frames = []
    for split in ("train", "val", "test"):
        path = cfg.DATA_SPLITS_DIR / f"{prefix}_{split}_pt{variant_suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}")
        frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def load_adni_union_pt_pool(variant: str = "mni") -> pd.DataFrame:
    return _concat_split_csvs("adni", variant=variant)


def load_adni_baseline_pt_pool(variant: str = "mni") -> pd.DataFrame:
    return _concat_split_csvs("adni_baseline_sc", variant=variant)


def load_adni_union_mni_pool() -> pd.DataFrame:
    return load_adni_union_pt_pool("mni")


def load_adni_baseline_mni_pool() -> pd.DataFrame:
    return load_adni_baseline_pt_pool("mni")


def prefix_subjects(df: pd.DataFrame, prefix: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = out["subject_id"].astype(str).map(
        lambda s: s if s.startswith(f"{prefix}::") else f"{prefix}::{s}"
    )
    out["source"] = source
    return out


def merge_pools(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def subject_majority_label(df: pd.DataFrame) -> pd.Series:
    return df.groupby("subject_id")["label"].agg(lambda x: x.value_counts().idxmax())


def stratified_subject_split(
    df: pd.DataFrame,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio = cfg.TRAIN_RATIO if train_ratio is None else train_ratio
    val_ratio = cfg.VAL_RATIO if val_ratio is None else val_ratio
    seed = cfg.RANDOM_SEED if seed is None else seed

    subjects = subject_majority_label(df).reset_index()
    subjects.columns = ["subject_id", "strat_label"]

    test_ratio = 1.0 - train_ratio - val_ratio
    train_subj, temp_subj = train_test_split(
        subjects,
        test_size=(val_ratio + test_ratio),
        stratify=subjects["strat_label"],
        random_state=seed,
    )
    val_relative = val_ratio / (val_ratio + test_ratio)
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=(1.0 - val_relative),
        stratify=temp_subj["strat_label"],
        random_state=seed,
    )

    def filter_df(subj_df: pd.DataFrame) -> pd.DataFrame:
        return df[df["subject_id"].isin(subj_df["subject_id"])].reset_index(drop=True)

    return filter_df(train_subj), filter_df(val_subj), filter_df(test_subj)


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    n_subj = df["subject_id"].nunique()
    counts = df["label"].value_counts().sort_index()
    dist = "  ".join(f"{label_names.get(int(l), l)}={c}" for l, c in counts.items())
    src = ""
    if "source" in df.columns:
        src_counts = df["source"].value_counts().to_dict()
        src = f"  |  sources={src_counts}"
    print(f"  {name:<6}: {len(df):>5} muestras, {n_subj:>4} sujetos  |  {dist}{src}")


def write_global_splits(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str = "mni",
) -> None:
    cfg.DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["subject_id", "image_path", "label", "source"]
    for optional in ("age_at_visit", "GENDER", "EDUC", "APOE_e4"):
        if optional in train_df.columns:
            cols.append(optional)

    suffix = "" if variant == "cropped" else f"_{variant}"
    for split_name, split_df in (
        ("train", train_df), ("val", val_df), ("test", test_df),
    ):
        out_path = cfg.DATA_SPLITS_DIR / f"{dataset_name}_{split_name}_pt{suffix}.csv"
        split_df[cols].to_csv(out_path, index=False)
        print(f"  → {out_path.name}: {len(split_df)} muestras")


def verify_no_subject_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
    if overlap:
        raise RuntimeError(
            f"Leakage: {len(overlap)} sujetos en train y test (ej. {list(overlap)[:3]})"
        )
    print(f"[OK] Sin solapamiento train/test ({train_df['subject_id'].nunique()} / "
          f"{test_df['subject_id'].nunique()} sujetos)")


def build_and_write(
    pool: pd.DataFrame,
    dataset_name: str,
    variant: str = "mni",
) -> int:
    train_df, val_df, test_df = stratified_subject_split(pool)
    verify_no_subject_leakage(train_df, test_df)
    print(f"\n[OK] Re-split global — {dataset_name}")
    print_split_summary("train", train_df)
    print_split_summary("val", val_df)
    print_split_summary("test", test_df)
    write_global_splits(dataset_name, train_df, val_df, test_df, variant=variant)
    return len(train_df) + len(val_df) + len(test_df)
