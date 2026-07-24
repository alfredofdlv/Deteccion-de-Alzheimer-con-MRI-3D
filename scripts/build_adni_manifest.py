"""
build_adni_manifest.py — Genera manifest ADNI piloto + lista de descarga IDA.

Construye data/splits/adni_{train,val,test}.csv y adni_download_list.csv a partir
de los CSV en Docs/ADNI-Context/ (sin imágenes DICOM).

Flujo:
  1. Catálogo T1w (All_Subjects_Key_MRI) ∩ no borrados (DELMRSCANS)
  2. Filtro visita (--viscode bl por defecto en piloto)
  3. Whitelist listas estandarizadas (baseline) si --use-standardized-lists
  4. 1 T1w por (PTID, VISCODE): MP-RAGE sin REPEAT, prioridad lista estandarizada
  5. Join CDR (CDGLOBAL → CN/MCI/AD)
  6. Split estratificado por PTID (70/15/15)

Uso:
    python scripts/build_adni_manifest.py --viscode bl
    python scripts/build_adni_manifest.py --catalog cohort --label-mode baseline_sc
    python scripts/build_adni_manifest.py --catalog cohort --label-mode baseline
    python scripts/build_adni_manifest.py --catalog cohort --label-mode cdr_longitudinal
    python scripts/build_adni_manifest.py --catalog cohort --label-mode union
    python scripts/build_adni_manifest.py --viscode bl --limit 200  # piloto acotado
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_oasis3_splits import print_split_summary, stratified_subject_split
from src.config import cfg

DOCS = cfg.ADNI_DOCS_DIR
EXCLUDED_VISIT_CODES = {"sc", "f"}


def _latest_csv(folder: Path, prefix: str) -> Path:
    matches = sorted(folder.glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No se encontró {prefix}_*.csv en {folder}")
    return matches[-1]


def cdr_to_label(cdglobal) -> int | None:
    if pd.isna(cdglobal):
        return None
    v = float(cdglobal)
    if v == 0.0:
        return 0
    if v == 0.5:
        return 1
    if v >= 1.0:
        return 2
    return None


def load_standardized_image_ids(lists_dir: Path, viscode: str | None) -> set[str]:
    """Union de Image.ID en listas estandarizadas relevantes."""
    ids: set[str] = set()
    if not lists_dir.is_dir():
        return ids

    baseline_3t = lists_dir / "ADNI_BaselineList_3T_8_28_12.csv"
    if viscode in (None, "bl") and baseline_3t.exists():
        df = pd.read_csv(baseline_3t)
        col = "Image.ID" if "Image.ID" in df.columns else "image_id"
        ids.update(df[col].astype(str))

    for path in lists_dir.glob("ADNI_CompleteVisitList*.csv"):
        df = pd.read_csv(path)
        if "Visit" in df.columns:
            mask = df["Visit"].astype(str).str.contains("Baseline", case=False, na=False)
            df = df[mask]
        col = "Image.ID" if "Image.ID" in df.columns else "image_id"
        if col in df.columns:
            ids.update(df[col].astype(str))

    return ids


def load_deleted_image_ids(study_info_dir: Path) -> set[str]:
    path = _latest_csv(study_info_dir, "DELMRSCANS")
    df = pd.read_csv(path)
    col = "IMAGEID" if "IMAGEID" in df.columns else "image_id"
    return set(df[col].astype(str))


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    df = pd.read_csv(catalog_path, low_memory=False)
    df = df[df["series_type"].astype(str).str.upper() == "T1W"].copy()
    df["image_id"] = df["image_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["image_visit"] = df["image_visit"].astype(str).str.lower()
    return df


def load_cohort_catalog(cohort_path: Path, key_mri_path: Path | None = None) -> pd.DataFrame:
    """
    Carga Clinical_T1w_Imaging_Cohort_Manifest y normaliza columnas al esquema del catálogo Key MRI.

    Enriquece study/series UID desde Key MRI cuando exista match por image_id.
    """
    df = pd.read_csv(cohort_path, low_memory=False)
    df = df.rename(columns={"image_description": "series_description"})
    df["image_id"] = df["image_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["image_visit"] = df["image_visit"].astype(str).str.lower()
    df["series_type"] = "T1W"

    for col in (
        "study_instance_uid", "series_instance_uid",
        "magnetic_field_strength", "mri_protocol_phase",
    ):
        if col not in df.columns:
            df[col] = ""

    if key_mri_path is not None and key_mri_path.exists():
        key = pd.read_csv(key_mri_path, low_memory=False, usecols=[
            "image_id", "study_instance_uid", "series_instance_uid",
            "magnetic_field_strength", "mri_protocol_phase",
        ])
        key["image_id"] = key["image_id"].astype(str)
        key = key.drop_duplicates(subset=["image_id"], keep="last")
        meta_cols = [
            "study_instance_uid", "series_instance_uid",
            "magnetic_field_strength", "mri_protocol_phase",
        ]
        df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
        df = df.merge(key, on="image_id", how="left")

    return df


def pick_one_t1_per_visit(
    df: pd.DataFrame,
    std_ids: set[str],
) -> pd.DataFrame:
    """Una serie T1w por (subject_id, image_visit)."""
    records: list[dict] = []
    group_cols = ["subject_id", "image_visit"]

    for (_, _), grp in df.groupby(group_cols, sort=False):
        g = grp.copy()
        g["_in_std"] = g["image_id"].isin(std_ids).astype(int)
        g["_repeat"] = g["series_description"].astype(str).str.contains(
            "REPEAT", case=False, na=False,
        ).astype(int)
        g["_mprage"] = g["series_description"].astype(str).str.contains(
            "MP-RAGE|MPRAGE|MPR", case=False, na=False,
        ).astype(int)
        g = g.sort_values(
            by=["_in_std", "_mprage", "_repeat", "image_id"],
            ascending=[False, False, True, True],
        )
        records.append(g.iloc[0].to_dict())

    return pd.DataFrame(records)


def dxsum_to_label(diagnosis) -> int | None:
    """ADNI DXSUM: 1=CN, 2=MCI, 3=AD → proyecto 0/1/2."""
    if pd.isna(diagnosis):
        return None
    v = int(float(diagnosis))
    mapping = {1: 0, 2: 1, 3: 2}
    return mapping.get(v)


def attach_labels(
    images: pd.DataFrame,
    cdr_path: Path,
    dxsum_path: Path,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    if images.empty:
        return pd.DataFrame(), []

    cdr = pd.read_csv(cdr_path, low_memory=False)
    cdr["PTID"] = cdr["PTID"].astype(str)
    cdr["VISCODE"] = cdr["VISCODE"].astype(str).str.lower()
    cdr["label_cdr"] = cdr["CDGLOBAL"].apply(cdr_to_label)
    cdr_key = cdr[["PTID", "VISCODE", "label_cdr", "CDGLOBAL"]].drop_duplicates(
        subset=["PTID", "VISCODE"], keep="last",
    )

    dx = pd.read_csv(dxsum_path, low_memory=False)
    dx["PTID"] = dx["PTID"].astype(str)
    dx["VISCODE"] = dx["VISCODE"].astype(str).str.lower()
    dx["label_dx"] = dx["DIAGNOSIS"].apply(dxsum_to_label)
    dx_key = dx[["PTID", "VISCODE", "label_dx", "DIAGNOSIS"]].drop_duplicates(
        subset=["PTID", "VISCODE"], keep="last",
    )

    merged = images.merge(
        cdr_key,
        left_on=["subject_id", "image_visit"],
        right_on=["PTID", "VISCODE"],
        how="left",
    )
    # Tras left join sin match, PTID/VISCODE del lado derecho quedan NaN; usar claves imagen
    merged = merged.merge(
        dx_key.rename(columns={"PTID": "PTID_dx", "VISCODE": "VISCODE_dx"}),
        left_on=["subject_id", "image_visit"],
        right_on=["PTID_dx", "VISCODE_dx"],
        how="left",
    )

    # Prioridad: CDR en la misma visita; si no hay (p. ej. bl), DXSUM
    merged["label"] = merged["label_cdr"]
    merged.loc[merged["label"].isna(), "label"] = merged.loc[
        merged["label"].isna(), "label_dx",
    ]
    merged["label_source"] = "cdr"
    merged.loc[merged["label_cdr"].isna() & merged["label_dx"].notna(), "label_source"] = "dxsum"

    matched = merged[merged["label"].notna()].copy()
    matched["label"] = matched["label"].astype(int)

    discarded: list[tuple[str, str, str]] = []
    for _, row in merged[merged["label"].isna()].iterrows():
        discarded.append((
            row["subject_id"],
            row["image_visit"],
            "sin CDR ni DXSUM para la visita",
        ))
    return matched, discarded


def load_dxsum_baseline_labels(dxsum_path: Path) -> pd.DataFrame:
    """DXSUM en visita clínica bl: una etiqueta por PTID."""
    dx = pd.read_csv(dxsum_path, low_memory=False)
    dx["PTID"] = dx["PTID"].astype(str)
    dx["VISCODE"] = dx["VISCODE"].astype(str).str.lower()
    dx_bl = dx[dx["VISCODE"] == "bl"].copy()
    dx_bl["label"] = dx_bl["DIAGNOSIS"].apply(dxsum_to_label)
    dx_bl = dx_bl.dropna(subset=["label"])
    return (
        dx_bl.drop_duplicates(subset=["PTID"], keep="last")
        [["PTID", "label", "DIAGNOSIS"]]
        .rename(columns={"PTID": "subject_id", "DIAGNOSIS": "dxsum_diagnosis"})
    )


def attach_baseline_sc_labels(
    images: pd.DataFrame,
    dxsum_path: Path,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """
    MRI visita sc + etiqueta DXSUM visita bl (mismo PTID, join cross-visit).
    """
    if images.empty:
        return pd.DataFrame(), []

    dx_bl = load_dxsum_baseline_labels(dxsum_path)
    merged = images.merge(dx_bl, on="subject_id", how="left")
    merged["label_source"] = "dxsum"
    merged["label_visit"] = "bl"

    matched = merged[merged["label"].notna()].copy()
    matched["label"] = matched["label"].astype(int)

    discarded: list[tuple[str, str, str]] = []
    for _, row in merged[merged["label"].isna()].iterrows():
        discarded.append((
            row["subject_id"],
            row["image_visit"],
            "sin DXSUM bl para el sujeto",
        ))
    return matched, discarded


def attach_covariates(df: pd.DataFrame, pt_path: Path) -> pd.DataFrame:
    if not pt_path.exists():
        return df
    demo = pd.read_csv(pt_path, low_memory=False)
    demo["PTID"] = demo["PTID"].astype(str)
    demo["VISCODE"] = demo["VISCODE"].astype(str).str.lower()
    cols = ["PTID", "VISCODE"]
    rename = {}
    if "PTGENDER" in demo.columns:
        cols.append("PTGENDER")
        rename["PTGENDER"] = "GENDER"
    if "PTEDUCAT" in demo.columns:
        cols.append("PTEDUCAT")
        rename["PTEDUCAT"] = "EDUC"
    if "PTMARRY" in demo.columns:
        cols.append("PTMARRY")
    demo = demo[cols].drop_duplicates(subset=["PTID", "VISCODE"], keep="last")
    demo = demo.rename(columns=rename)
    out = df.copy()
    if "subject_id" in out.columns:
        out["PTID"] = out.get("PTID", out["subject_id"])
        out["PTID"] = out["PTID"].fillna(out["subject_id"])
    if "image_visit" in out.columns:
        out["VISCODE"] = out.get("VISCODE", out["image_visit"])
        out["VISCODE"] = out["VISCODE"].fillna(out["image_visit"])
    out = out.merge(demo, on=["PTID", "VISCODE"], how="left")
    if "EDUC" in out.columns:
        out["EDUC"] = out["EDUC"].fillna(12.0)
    return out


def expected_nifti_path(subject_id: str, viscode: str, image_id: str) -> str:
    name = f"{subject_id}_{viscode}_{image_id}.nii.gz"
    return str((cfg.ADNI_NIFTI_DIR / name).resolve())


def _label_mode_filters(label_mode: str) -> tuple[str | None, set[str]]:
    """Devuelve (viscode_filtro, visitas_excluidas) según label-mode."""
    if label_mode == "baseline_sc":
        return "sc", {"f"}
    if label_mode == "cdr_longitudinal":
        return None, set(EXCLUDED_VISIT_CODES)
    raise ValueError(f"label_mode no soportado en build_labeled: {label_mode!r}")


def build_labeled_dataset(
    label_mode: str,
    *,
    catalog: pd.DataFrame,
    cdr_path: Path,
    dxsum_path: Path,
    lists_dir: Path,
    pt_path: Path,
    use_std_filter: bool,
    use_std_rank: bool,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Construye DataFrame etiquetado para baseline_sc o cdr_longitudinal."""
    viscode, excluded_visits = _label_mode_filters(label_mode)
    df = catalog[~catalog["image_visit"].isin(excluded_visits)].copy()
    if viscode:
        df = df[df["image_visit"] == viscode]

    std_ids = load_standardized_image_ids(lists_dir, viscode) if use_std_rank else set()
    if use_std_filter and std_ids:
        df = df[df["image_id"].isin(std_ids)]

    picked = pick_one_t1_per_visit(df, std_ids)
    if label_mode == "baseline_sc":
        labeled, discarded = attach_baseline_sc_labels(picked, dxsum_path)
    else:
        labeled, discarded = attach_labels(picked, cdr_path, dxsum_path)

    if not labeled.empty:
        labeled = attach_covariates(labeled, pt_path)
    return labeled, discarded


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar manifest ADNI piloto")
    parser.add_argument(
        "--viscode", type=str, default="bl",
        help="Filtrar por VISCODE (default: bl = baseline)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Máximo de muestras tras filtros (0 = sin límite, default)",
    )
    parser.add_argument(
        "--catalog", type=str, default="cohort",
        choices=["cohort", "key_mri"],
        help="Fuente de imágenes: cohort (Clinical T1w, default) o key_mri (piloto)",
    )
    parser.add_argument(
        "--label-mode", type=str, default="baseline_sc",
        choices=["baseline", "baseline_sc", "cdr_longitudinal", "union"],
        help=(
            "baseline_sc (default): MRI sc + DXSUM bl; "
            "baseline: MRI bl + DXSUM bl; "
            "cdr_longitudinal: todas las visitas con CDR; "
            "union: baseline_sc + cdr_longitudinal (image_id únicos)"
        ),
    )
    parser.add_argument(
        "--use-standardized-lists", action="store_true",
        help="Exigir Image.ID en listas estandarizadas (filtro duro)",
    )
    parser.add_argument(
        "--no-standardized-lists", action="store_true",
        help="No usar listas estandarizadas ni siquiera para priorizar",
    )
    parser.add_argument(
        "--seed", type=int, default=cfg.RANDOM_SEED,
        help=f"Semilla split (default: {cfg.RANDOM_SEED})",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    use_std_filter = args.use_standardized_lists and not args.no_standardized_lists
    use_std_rank = not args.no_standardized_lists
    viscode = args.viscode.lower() if args.viscode else None
    excluded_visits = set(EXCLUDED_VISIT_CODES)
    if args.label_mode == "union":
        print("[INFO] label-mode=union: baseline_sc + cdr_longitudinal (image_id únicos)")
    elif args.label_mode == "baseline_sc":
        viscode = "sc"
        excluded_visits = {"f"}
        print("[INFO] label-mode=baseline_sc: MRI sc + DXSUM bl (cross-visit, 1 img/sujeto)")
    elif args.label_mode == "cdr_longitudinal":
        viscode = None
        print("[INFO] label-mode=cdr_longitudinal: todas las visitas con CDR (excl. sc/f)")

    key_mri_path = DOCS / "03_imaging" / "catalog" / "All_Subjects_Key_MRI_16Jun2026.csv"
    if args.catalog == "cohort":
        catalog_path = cfg.ADNI_COHORT_MANIFEST
        if not catalog_path.exists():
            catalog_path = DOCS / "03_imaging" / "cohort" / (
                "Clinical_T1w_Imaging_Cohort_Manifest_17Jun2026.csv"
            )
    else:
        catalog_path = key_mri_path

    cdr_path = _latest_csv(DOCS / "02_study_files" / "assessments", "CDR")
    dxsum_path = _latest_csv(DOCS / "02_study_files" / "assessments", "DXSUM")
    del_path_dir = DOCS / "02_study_files" / "study_info"
    lists_dir = DOCS / "03_imaging" / "standardized_lists"
    pt_path = _latest_csv(DOCS / "02_study_files" / "subject_characteristics", "PTDEMOG")

    cfg.DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.ADNI_NIFTI_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Catálogo T1w ({args.catalog}): {catalog_path.name}")
    if args.catalog == "cohort":
        catalog = load_cohort_catalog(catalog_path, key_mri_path)
    else:
        catalog = load_catalog(catalog_path)
    print(f"      {len(catalog)} series T1w en catálogo")

    print("[2/6] Excluyendo escaneos borrados (DELMRSCANS)...")
    deleted = load_deleted_image_ids(del_path_dir)
    catalog = catalog[~catalog["image_id"].isin(deleted)]
    print(f"      {len(catalog)} tras excluir borrados")

    discarded: list[tuple[str, str, str]] = []
    if args.label_mode == "union":
        print("[3/6] Etiquetando baseline_sc...")
        labeled_sc, disc_sc = build_labeled_dataset(
            "baseline_sc",
            catalog=catalog,
            cdr_path=cdr_path,
            dxsum_path=dxsum_path,
            lists_dir=lists_dir,
            pt_path=pt_path,
            use_std_filter=use_std_filter,
            use_std_rank=use_std_rank,
        )
        print(f"      {len(labeled_sc)} baseline_sc")
        print("[4/6] Etiquetando cdr_longitudinal...")
        labeled_long, disc_long = build_labeled_dataset(
            "cdr_longitudinal",
            catalog=catalog,
            cdr_path=cdr_path,
            dxsum_path=dxsum_path,
            lists_dir=lists_dir,
            pt_path=pt_path,
            use_std_filter=use_std_filter,
            use_std_rank=use_std_rank,
        )
        print(f"      {len(labeled_long)} cdr_longitudinal")
        labeled = pd.concat([labeled_sc, labeled_long], ignore_index=True)
        before_dedup = len(labeled)
        labeled = labeled.drop_duplicates(subset=["image_id"], keep="first")
        print(f"      {len(labeled)} únicos tras unión ({before_dedup - len(labeled)} duplicados)")
        discarded = disc_sc + disc_long
    else:
        catalog = catalog[~catalog["image_visit"].isin(excluded_visits)]
        if viscode:
            catalog = catalog[catalog["image_visit"] == viscode]
            print(f"      {len(catalog)} tras filtrar VISCODE imagen={viscode!r}")

        std_ids = load_standardized_image_ids(lists_dir, viscode) if use_std_rank else set()
        if use_std_filter and std_ids:
            before = len(catalog)
            catalog = catalog[catalog["image_id"].isin(std_ids)]
            print(f"      {len(catalog)} tras whitelist listas estandarizadas ({before} antes)")
        elif std_ids:
            in_std = catalog["image_id"].isin(std_ids).sum()
            print(f"      {len(catalog)} visitas (listas std: {in_std} en catálogo, usadas para priorizar)")

        print("[3/6] Seleccionando 1 T1w por (PTID, VISCODE)...")
        picked = pick_one_t1_per_visit(catalog, std_ids)
        print(f"      {len(picked)} visitas únicas")

        if args.label_mode == "baseline_sc":
            print(f"[4/6] Join DXSUM bl (cross-visit): {dxsum_path.name}")
            labeled, discarded = attach_baseline_sc_labels(picked, dxsum_path)
            print(f"      {len(labeled)} con label (DXSUM bl)  |  {len(discarded)} sin label")
        else:
            print(f"[4/6] Join labels CDR + DXSUM: {cdr_path.name}, {dxsum_path.name}")
            labeled, discarded = attach_labels(picked, cdr_path, dxsum_path)
            n_dx = (labeled.get("label_source", pd.Series()) == "dxsum").sum() if not labeled.empty else 0
            print(f"      {len(labeled)} con label ({n_dx} vía DXSUM)  |  {len(discarded)} sin label")
        if not labeled.empty:
            labeled = attach_covariates(labeled, pt_path)

    if args.verbose and discarded:
        for ptid, vc, reason in discarded[:15]:
            print(f"        {ptid} {vc}: {reason}")

    if labeled.empty:
        print("[ERROR] No hay muestras con label. Revisa filtros o CDR.")
        sys.exit(1)

    if args.limit and len(labeled) > args.limit:
        # Muestreo estratificado por label para mantener balance aproximado
        labeled = (
            labeled.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), max(1, args.limit // labeled["label"].nunique())),
                random_state=args.seed,
            ))
            .head(args.limit)
            .reset_index(drop=True)
        )
        print(f"      Limitado a {len(labeled)} muestras (--limit {args.limit})")

    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    print("\n      Distribución de clases:")
    for lbl, cnt in labeled["label"].value_counts().sort_index().items():
        print(f"        {label_names.get(lbl, lbl)}: {cnt} ({cnt/len(labeled):.1%})")

    print(f"\n[5/6] Split sujeto-nivel ({cfg.TRAIN_RATIO:.0%}/{cfg.VAL_RATIO:.0%}/{cfg.TEST_RATIO:.0%})...")
    train_df, val_df, test_df = stratified_subject_split(
        labeled.rename(columns={"subject_id": "subject_id"}),
        cfg.TRAIN_RATIO,
        cfg.VAL_RATIO,
        args.seed,
    )

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            sid = row["subject_id"]
            vc = row["image_visit"]
            iid = row["image_id"]
            entry = {
                "subject_id": sid,
                "viscode": vc,
                "image_id": iid,
                "image_path": expected_nifti_path(sid, vc, iid),
                "label": int(row["label"]),
                "label_visit": row.get("label_visit", vc),
                "label_source": row.get("label_source", ""),
                "series_description": row.get("series_description", ""),
                "study_instance_uid": row.get("study_instance_uid", ""),
                "series_instance_uid": row.get("series_instance_uid", ""),
                "mri_protocol_phase": row.get("mri_protocol_phase", ""),
                "magnetic_field_strength": row.get("magnetic_field_strength", ""),
                "GENDER": row.get("GENDER", ""),
                "EDUC": row.get("EDUC", 12.0),
            }
            rows.append(entry)
        return pd.DataFrame(rows)

    prefix = "adni_baseline_sc" if args.label_mode == "baseline_sc" else "adni"
    splits = {
        f"{prefix}_train": finalize(train_df),
        f"{prefix}_val": finalize(val_df),
        f"{prefix}_test": finalize(test_df),
    }

    out_cols = [
        "subject_id", "viscode", "image_id", "image_path", "label",
        "label_visit", "label_source",
        "series_description", "study_instance_uid", "series_instance_uid",
        "mri_protocol_phase", "magnetic_field_strength", "GENDER", "EDUC",
    ]
    for name, df in splits.items():
        path = cfg.DATA_SPLITS_DIR / f"{name}.csv"
        df[out_cols].to_csv(path, index=False)

    download_df = pd.concat(splits.values(), ignore_index=True).drop_duplicates(
        subset=["image_id"],
    )
    dl_cols = [
        "subject_id", "viscode", "image_id", "label",
        "study_instance_uid", "series_instance_uid", "series_description",
    ]
    if args.label_mode == "baseline_sc":
        download_path = cfg.DATA_SPLITS_DIR / "adni_baseline_sc_download_list.csv"
    else:
        download_path = cfg.ADNI_DICOM_MANIFEST
    download_df[dl_cols].to_csv(download_path, index=False)

    print(f"\n[6/6] Guardado en {cfg.DATA_SPLITS_DIR}")
    print_split_summary("train", splits[f"{prefix}_train"])
    print_split_summary("val", splits[f"{prefix}_val"])
    print_split_summary("test", splits[f"{prefix}_test"])
    print(f"\n  Lista descarga IDA: {download_path} ({len(download_df)} imágenes)")
    print("\n[OK] Siguiente paso:")
    if args.label_mode == "baseline_sc":
        print("  1. python scripts/prepare_adni_splits.py --label-mode baseline_sc")
    else:
        print("  1. Descargar DICOM desde IDA (ver scripts/download_adni_pilot.py)")
        print("  2. python scripts/convert_adni_dicom_to_nifti.py")
        print("  3. python scripts/preprocess_to_pt.py --dataset adni")


if __name__ == "__main__":
    main()
