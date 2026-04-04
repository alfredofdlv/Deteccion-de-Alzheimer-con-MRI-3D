"""
prepare_oasis3_nifti_splits.py — Genera oasis3_{train,val,test}.csv con rutas NIfTI
reales en Linux a partir del arbol descargado en cfg.OASIS3_RAW_DIR.

Flujo recomendado (pipeline OASIS-3 en Linux):
  1. Este script  →  data/splits/oasis3_{train,val,test}.csv
  2. preprocess_to_pt.py  →  data/preprocessed/oasis3/*.pt + oasis3_*_pt.csv

Por cada carpeta de sesion (OAS3XXXX_MR_dYYYY) se elige un unico T1w:
  preferencia por menor numero de run (_run-01_ antes que _run-02_; sin _run-_ al final).

Matching escaneo <-> etiqueta: misma logica que prepare_oasis3_splits.py (visita
clinica mas cercana en oasis3_master_clinical.csv, ventana configurable).

Split sujeto-nivel estratificado 70/15/15 (cfg.TRAIN_RATIO / VAL_RATIO / TEST_RATIO).

Uso:
    python prepare_oasis3_nifti_splits.py
    python prepare_oasis3_nifti_splits.py --window 180 --verbose
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from prepare_oasis3_splits import match_labels, print_split_summary, stratified_subject_split
from src.config import cfg

# Mismo patron que prepare_oasis3_splits.py (nombres BIDS en .pt / .nii.gz)
PATTERN = re.compile(r"sub-(OAS3\d+)_sess?-d(\d+)")
SESSION_DIR_PATTERN = re.compile(r"^(OAS3\d+)_MR_d(\d+)$")
RUN_PATTERN = re.compile(r"_run-(\d+)_")


def _run_sort_key(path: Path) -> tuple[int, str]:
    """Menor numero de run primero; sin sufijo _run-_ al final (orden estable por nombre)."""
    m = RUN_PATTERN.search(path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (9999, path.name)


def pick_one_t1w_per_session(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=_run_sort_key)[0]


def scan_nifti_sessions(raw_dir: Path) -> pd.DataFrame:
    """
    Recorre raw_dir (una carpeta por MR session), elige un T1w por sesion,
    devuelve subject_id, scan_day, image_path (absoluto POSIX).
    """
    records: list[dict] = []
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio raw OASIS-3: {raw_dir}")

    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        if SESSION_DIR_PATTERN.match(child.name) is None:
            continue
        niftis = [
            p for p in child.rglob("*.nii.gz")
            if p.is_file() and "T1w" in p.name
        ]
        if not niftis:
            continue
        chosen = pick_one_t1w_per_session(niftis)
        if chosen is None:
            continue
        m = PATTERN.match(chosen.name)
        if m is None:
            print(f"[WARN] No se pudo parsear subject/sesion desde: {chosen}")
            continue
        subject_id = m.group(1)
        scan_day = int(m.group(2))
        records.append({
            "subject_id": subject_id,
            "scan_day": scan_day,
            "image_path": str(chosen.resolve()),
        })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar splits NIfTI OASIS-3 (rutas Linux) desde data/raw/OASIS-3/",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=365,
        help="Ventana maxima en dias entre escaner y visita clinica (default: 365)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles de scans descartados",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=f"Override de directorio raw (default: {cfg.OASIS3_RAW_DIR})",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir if args.raw_dir is not None else cfg.OASIS3_RAW_DIR
    clinical_csv = cfg.OASIS3_CLINICAL_CSV
    splits_dir = cfg.DATA_SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Escaneando T1w en: {raw_dir}")
    scans = scan_nifti_sessions(raw_dir)
    print(f"      {len(scans)} sesiones con 1 T1w ({scans['subject_id'].nunique()} sujetos)")

    print(f"[2/4] Cargando datos clinicos: {clinical_csv}")
    clinical = pd.read_csv(clinical_csv)
    print(f"      {len(clinical)} visitas clinicas ({clinical['OASISID'].nunique()} sujetos)")

    print(f"[3/4] Matching NIfTI <-> etiqueta clinica (ventana +/-{args.window}d)...")
    matched, discarded = match_labels(scans, clinical, window=args.window)
    print(f"      Matcheados: {len(matched)}  |  Descartados: {len(discarded)}")

    if args.verbose and discarded:
        print("      Scans descartados:")
        for subj, day, reason in discarded[:20]:
            print(f"        {subj} d{day}: {reason}")
        if len(discarded) > 20:
            print(f"        ... y {len(discarded) - 20} mas")

    if matched.empty:
        print("[ERROR] No hay muestras tras el matching. Revisa el CSV clinico y el arbol raw.")
        return

    print("\n      Distribucion de clases (total):")
    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    for lbl, cnt in matched["label"].value_counts().sort_index().items():
        print(f"        {label_names.get(lbl, lbl)}: {cnt} ({cnt/len(matched):.1%})")

    print(f"\n[4/4] Generando splits sujeto-nivel estratificados "
          f"({cfg.TRAIN_RATIO:.0%}/{cfg.VAL_RATIO:.0%}/{cfg.TEST_RATIO:.0%})...")
    train_df, val_df, test_df = stratified_subject_split(
        matched, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.RANDOM_SEED
    )

    out_cols = [
        "subject_id", "image_path", "label",
        "age_at_visit", "GENDER", "EDUC", "APOE_e4",
    ]
    splits = {
        "oasis3_train": train_df,
        "oasis3_val": val_df,
        "oasis3_test": test_df,
    }
    for name, df in splits.items():
        out_path = splits_dir / f"{name}.csv"
        df[out_cols].to_csv(out_path, index=False)

    print(f"\n  Splits guardados en: {splits_dir}")
    print_split_summary("train", train_df)
    print_split_summary("val", val_df)
    print_split_summary("test", test_df)

    print("\n[OK] CSVs NIfTI generados (rutas Linux):")
    for name in splits:
        print(f"  {splits_dir / (name + '.csv')}")
    print("\nSiguiente paso: python preprocess_to_pt.py --dataset oasis3")


if __name__ == "__main__":
    main()
