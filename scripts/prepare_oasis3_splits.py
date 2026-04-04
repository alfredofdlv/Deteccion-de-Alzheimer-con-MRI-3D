"""
prepare_oasis3_splits.py — Genera oasis3_{train,val,test}_pt.csv a partir de los
tensores .pt ya existentes en data/preprocessed/oasis3/ y el CSV clínico.

Cuándo usar cada script (pipeline Linux recomendado):
  - **Tienes NIfTI en cfg.OASIS3_RAW_DIR y aún no tienes .pt:** ejecuta primero
    `prepare_oasis3_nifti_splits.py` (genera oasis3_{train,val,test}.csv con rutas
    Linux) y luego `preprocess_to_pt.py --dataset oasis3` (crea los .pt y los *_pt.csv).
  - **Ya tienes .pt y quieres regenerar solo los *_pt.csv** (p. ej. nuevo split
    o re-matching clínico sin volver a pasar por NIfTI): usa este script.

Estrategia de matching .pt <-> etiqueta clínica:
  Para cada escáner (subject_id, scan_day), busca la visita clínica más cercana
  del mismo sujeto en oasis3_master_clinical.csv (ventana máx: ±365 días por defecto).

Split sujeto-nivel estratificado (70/15/15) para evitar data leakage.

Uso:
    python prepare_oasis3_splits.py
    python prepare_oasis3_splits.py --window 180   # ventana más estricta
    python prepare_oasis3_splits.py --verbose       # mostrar filas descartadas
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import cfg

PATTERN = re.compile(r"sub-(OAS3\d+)_sess?-d(\d+)")


def scan_pt_files(pt_dir: Path) -> pd.DataFrame:
    """Escanea directorio .pt y extrae subject_id, scan_day, pt_path."""
    records = []
    for pt_file in sorted(pt_dir.glob("*.pt")):
        m = PATTERN.match(pt_file.name)
        if m is None:
            continue
        subject_id = m.group(1)
        scan_day = int(m.group(2))
        records.append({
            "subject_id": subject_id,
            "scan_day": scan_day,
            "image_path": str(pt_file),
        })
    return pd.DataFrame(records)


def match_labels(scans: pd.DataFrame, clinical: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Para cada fila de scans, busca la etiqueta clínica más cercana del mismo sujeto.
    Descarta scans cuya visita más cercana esté a más de `window` días.
    """
    clin = clinical[
        ["OASISID", "days_to_visit", "label", "age at visit", "GENDER", "EDUC", "APOE_e4"]
    ].dropna(subset=["label"]).copy()
    clin["days_to_visit"] = clin["days_to_visit"].astype(int)
    clin["label"] = clin["label"].astype(int)

    matched = []
    discarded = []

    for _, row in scans.iterrows():
        subj_rows = clin[clin["OASISID"] == row["subject_id"]]
        if subj_rows.empty:
            discarded.append((row["subject_id"], row["scan_day"], "no clinical data"))
            continue
        diffs = (subj_rows["days_to_visit"] - row["scan_day"]).abs()
        min_diff = diffs.min()
        if min_diff > window:
            discarded.append((row["subject_id"], row["scan_day"],
                               f"closest visit {min_diff}d > {window}d window"))
            continue
        best = subj_rows.loc[diffs.idxmin()]
        matched.append({
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "label": int(best["label"]),
            "age_at_visit": float(best["age at visit"]),
            "GENDER": str(best["GENDER"]),
            "EDUC": float(best["EDUC"]) if pd.notna(best["EDUC"]) else 12.0,
            "APOE_e4": float(best["APOE_e4"]) if pd.notna(best["APOE_e4"]) else 0.0,
        })

    return pd.DataFrame(matched), discarded


def subject_majority_label(df: pd.DataFrame) -> pd.Series:
    """Etiqueta mayoritaria por sujeto (para estratificación del split)."""
    return df.groupby("subject_id")["label"].agg(lambda x: x.value_counts().idxmax())


def stratified_subject_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split sujeto-nivel estratificado.
    Primero separa test (1 - train - val), luego val del resto.
    """
    subjects = subject_majority_label(df).reset_index()
    subjects.columns = ["subject_id", "strat_label"]

    test_ratio = 1.0 - train_ratio - val_ratio
    train_subj, temp_subj = train_test_split(
        subjects,
        test_size=(val_ratio + test_ratio),
        stratify=subjects["strat_label"],
        random_state=seed,
    )
    # val_ratio relativo al subconjunto temp
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
    dist = "  ".join(f"{label_names.get(l, l)}={c}" for l, c in counts.items())
    print(f"  {name:<6}: {len(df):>4} muestras, {n_subj:>3} sujetos  |  {dist}")


def main():
    parser = argparse.ArgumentParser(description="Generar splits OASIS-3 desde .pt + CSV clínico")
    parser.add_argument("--window", type=int, default=365,
                        help="Ventana máxima en días entre escáner y visita clínica (default: 365)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar detalles de scans descartados")
    args = parser.parse_args()

    pt_dir = cfg.PREPROCESSED_DIR / "oasis3"
    clinical_csv = cfg.OASIS3_CLINICAL_CSV
    splits_dir = cfg.DATA_SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Escaneando .pt en: {pt_dir}")
    scans = scan_pt_files(pt_dir)
    print(f"      {len(scans)} ficheros .pt encontrados ({scans['subject_id'].nunique()} sujetos)")

    print(f"[2/4] Cargando datos clínicos: {clinical_csv}")
    clinical = pd.read_csv(clinical_csv)
    print(f"      {len(clinical)} visitas clínicas ({clinical['OASISID'].nunique()} sujetos)")

    print(f"[3/4] Matching .pt <-> etiqueta clínica (ventana ±{args.window}d)...")
    matched, discarded = match_labels(scans, clinical, window=args.window)
    print(f"      Matcheados: {len(matched)}  |  Descartados: {len(discarded)}")

    if args.verbose and discarded:
        print("      Scans descartados:")
        for subj, day, reason in discarded[:20]:
            print(f"        {subj} d{day}: {reason}")
        if len(discarded) > 20:
            print(f"        ... y {len(discarded) - 20} más")

    if matched.empty:
        print("[ERROR] No hay muestras tras el matching. Revisa el CSV clínico y los .pt.")
        return

    print(f"\n      Distribución de clases (total):")
    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    for lbl, cnt in matched["label"].value_counts().sort_index().items():
        print(f"        {label_names.get(lbl, lbl)}: {cnt} ({cnt/len(matched):.1%})")

    print(f"\n[4/4] Generando splits sujeto-nivel estratificados (70/15/15)...")
    train_df, val_df, test_df = stratified_subject_split(
        matched, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.RANDOM_SEED
    )

    splits = {"oasis3_train_pt": train_df, "oasis3_val_pt": val_df, "oasis3_test_pt": test_df}
    for name, df in splits.items():
        out_path = splits_dir / f"{name}.csv"
        df[["subject_id", "image_path", "label",
            "age_at_visit", "GENDER", "EDUC", "APOE_e4"]].to_csv(out_path, index=False)

    print(f"\n  Splits guardados en: {splits_dir}")
    print_split_summary("train", train_df)
    print_split_summary("val", val_df)
    print_split_summary("test", test_df)

    print(f"\n[OK] CSVs generados:")
    for name in splits:
        path = splits_dir / f"{name}.csv"
        print(f"  {path}")


if __name__ == "__main__":
    main()
