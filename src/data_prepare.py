"""
data_prepare.py — Pipeline ETL para el dataset OASIS-1.

Tres pasos secuenciales:
  1. extract_and_standardize() — Extrae pares .img/.hdr de T88_111 (masked_gfc)
     desde carpetas ya descomprimidas o directamente desde archivos .tar.gz.
  2. generate_master_csv()     — Genera dataset_master.csv con etiquetas CDR.
  3. stratified_split()        — Divide en train/val/test estratificado.

Uso:
    python -m src.data_prepare

El script es idempotente: puede ejecutarse múltiples veces sin duplicar datos.
"""

import glob
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import cfg

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------
_SUBJECT_ID_RE = re.compile(r"(OAS1_\d{4}_MR\d)")
_MASKED_GFC_PATTERN = re.compile(r".*T88_111[/\\].*_masked_gfc\.(img|hdr)$")
_DISC_TARBALL_RE = re.compile(r"^oasis_cross-sectional_disc\d+\.tar\.gz$")
_DISC_FOLDER_RE = re.compile(r"^oasis_cross-sectional_disc\d+$")
_OASIS1_EXPECTED_SUBJECTS = 416


# ===================================================================
# Step 1 — Extraction & Standardization
# ===================================================================

def _extract_subject_id(filename: str) -> str | None:
    """Extrae el subject_id (ej. 'OAS1_0001_MR1') de un nombre de archivo."""
    match = _SUBJECT_ID_RE.search(filename)
    return match.group(1) if match else None


def _is_target_file(filepath: str) -> bool:
    """Comprueba si un archivo es un _masked_gfc.img/.hdr dentro de T88_111."""
    return bool(_MASKED_GFC_PATTERN.search(filepath.replace("\\", "/")))


def _copy_to_processed(src_path: Path, subject_id: str, ext: str) -> bool:
    """
    Copia un archivo al directorio procesado con nombre estandarizado.

    Returns:
        True si se copió, False si ya existía (idempotente).
    """
    dest = cfg.PROCESSED_IMAGES_DIR / f"{subject_id}{ext}"
    if dest.exists():
        return False
    shutil.copy2(src_path, dest)
    return True


def _write_bytes_to_processed(data: bytes, subject_id: str, ext: str) -> bool:
    """
    Escribe bytes extraídos de un tarball al directorio procesado.

    Returns:
        True si se escribió, False si ya existía (idempotente).
    """
    dest = cfg.PROCESSED_IMAGES_DIR / f"{subject_id}{ext}"
    if dest.exists():
        return False
    dest.write_bytes(data)
    return True


def _discover_discs() -> Tuple[List[Path], List[Path]]:
    """
    Escanea OASIS_RAW_DIR y clasifica cada disco como carpeta extraída
    o archivo .tar.gz. Si ambos existen para un mismo disco, se prioriza
    la carpeta (más rápida de procesar).

    Returns:
        (extracted_dirs, tarball_paths) — listas de Paths.
    """
    raw_dir = cfg.OASIS_RAW_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio OASIS: {raw_dir}\n"
            f"Coloca los archivos descargados de OASIS-1 en esa ruta."
        )

    # Recopilar todos los tarballs y carpetas que coincidan con el patrón
    tarballs: dict[str, Path] = {}  # disc_key -> Path
    folders: dict[str, Path] = {}   # disc_key -> Path

    for entry in sorted(raw_dir.iterdir()):
        name = entry.name
        if entry.is_file() and _DISC_TARBALL_RE.match(name):
            # Extraer clave: "oasis_cross-sectional_disc3"
            key = name.replace(".tar.gz", "")
            tarballs[key] = entry
        elif entry.is_dir() and _DISC_FOLDER_RE.match(name):
            key = name
            folders[key] = entry

    # Para cada disco, decidir si usar carpeta o tarball
    all_keys = sorted(set(list(tarballs.keys()) + list(folders.keys())))
    extracted_dirs: List[Path] = []
    tarball_paths: List[Path] = []

    for key in all_keys:
        if key in folders:
            extracted_dirs.append(folders[key])
        elif key in tarballs:
            tarball_paths.append(tarballs[key])

    total = len(extracted_dirs) + len(tarball_paths)
    if total == 0:
        raise FileNotFoundError(
            f"No se encontraron discos OASIS en {raw_dir}.\n"
            f"Se esperan archivos oasis_cross-sectional_disc*.tar.gz "
            f"o carpetas oasis_cross-sectional_disc*/."
        )

    print(f"[INFO] Discos detectados: {total} total")
    print(f"       - Carpetas extraídas: {len(extracted_dirs)}")
    print(f"       - Archivos .tar.gz:   {len(tarball_paths)}")

    return extracted_dirs, tarball_paths


def _process_extracted_folder(folder: Path) -> int:
    """
    Procesa un disco ya extraído: busca pares _masked_gfc.img/.hdr
    en T88_111 y los copia a data/processed/images/.

    Returns:
        Número de archivos copiados.
    """
    copied = 0
    # Buscar recursivamente en T88_111
    pattern_img = str(folder / "**" / "T88_111" / "*_masked_gfc.img")
    pattern_hdr = str(folder / "**" / "T88_111" / "*_masked_gfc.hdr")

    targets = glob.glob(pattern_img, recursive=True) + glob.glob(
        pattern_hdr, recursive=True
    )

    for filepath in targets:
        fname = os.path.basename(filepath)
        subject_id = _extract_subject_id(fname)
        if subject_id is None:
            print(f"  [WARN] No se pudo extraer subject_id de: {fname}")
            continue
        ext = Path(fname).suffix  # .img o .hdr
        if _copy_to_processed(Path(filepath), subject_id, ext):
            copied += 1

    return copied


def _process_tarball(tarball_path: Path) -> int:
    """
    Procesa un .tar.gz: extrae selectivamente solo los archivos
    _masked_gfc.img/.hdr de T88_111 sin descomprimir el archivo completo.

    Returns:
        Número de archivos extraídos.
    """
    extracted = 0
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            members = tar.getnames()
            target_members = [m for m in members if _is_target_file(m)]

            for member_name in target_members:
                fname = os.path.basename(member_name)
                subject_id = _extract_subject_id(fname)
                if subject_id is None:
                    print(f"  [WARN] No se pudo extraer subject_id de: {fname}")
                    continue
                ext = Path(fname).suffix

                # Comprobar si ya existe (idempotente)
                dest = cfg.PROCESSED_IMAGES_DIR / f"{subject_id}{ext}"
                if dest.exists():
                    continue

                # Extraer solo este miembro en memoria y escribirlo
                member = tar.getmember(member_name)
                f = tar.extractfile(member)
                if f is not None:
                    data = f.read()
                    _write_bytes_to_processed(data, subject_id, ext)
                    extracted += 1
    except tarfile.TarError as e:
        print(f"  [ERROR] Error procesando {tarball_path.name}: {e}")

    return extracted


def extract_and_standardize() -> None:
    """
    Step 1: Extrae y estandariza los archivos skull-stripped (masked_gfc)
    del dataset OASIS-1.

    - Auto-detecta discos extraídos vs .tar.gz.
    - Copia/extrae solo *_masked_gfc.img y *_masked_gfc.hdr de T88_111.
    - Renombra a formato plano: OAS1_XXXX_MR1.img / .hdr
    - Destino: data/processed/images/
    """
    print("=" * 60)
    print("STEP 1: Extraction & Standardization")
    print("=" * 60)

    # Crear directorio destino
    cfg.PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Detectar discos automáticamente
    extracted_dirs, tarball_paths = _discover_discs()

    total_copied = 0

    # Procesar carpetas extraídas
    if extracted_dirs:
        print(f"\n[1/2] Procesando {len(extracted_dirs)} carpeta(s) extraída(s)...")
        for folder in tqdm(extracted_dirs, desc="Carpetas"):
            n = _process_extracted_folder(folder)
            total_copied += n

    # Procesar tarballs
    if tarball_paths:
        print(f"\n[2/2] Procesando {len(tarball_paths)} archivo(s) .tar.gz...")
        for tarball in tqdm(tarball_paths, desc="Tarballs"):
            n = _process_tarball(tarball)
            total_copied += n

    # Validación final
    img_files = list(cfg.PROCESSED_IMAGES_DIR.glob("*.img"))
    hdr_files = list(cfg.PROCESSED_IMAGES_DIR.glob("*.hdr"))

    print(f"\n[RESULTADO]")
    print(f"  Archivos .img: {len(img_files)}")
    print(f"  Archivos .hdr: {len(hdr_files)}")
    print(f"  Pares totales: {min(len(img_files), len(hdr_files))}")
    print(f"  Nuevos copiados en esta ejecución: {total_copied}")

    # Verificar que cada .img tenga su .hdr correspondiente
    img_ids = {f.stem for f in img_files}
    hdr_ids = {f.stem for f in hdr_files}
    orphan_img = img_ids - hdr_ids
    orphan_hdr = hdr_ids - img_ids

    if orphan_img:
        print(f"  [WARN] .img sin .hdr correspondiente: {sorted(orphan_img)}")
    if orphan_hdr:
        print(f"  [WARN] .hdr sin .img correspondiente: {sorted(orphan_hdr)}")

    n_pairs = len(img_ids & hdr_ids)
    assert n_pairs > 0, (
        "No se encontraron pares .img/.hdr válidos. "
        "Verifica que los archivos OASIS estén en la ruta correcta."
    )

    if n_pairs == _OASIS1_EXPECTED_SUBJECTS:
        print(f"\n  [OK] VALIDACION: {n_pairs} pares ({_OASIS1_EXPECTED_SUBJECTS} esperados).")
    else:
        print(
            f"\n  [WARN] Se encontraron {n_pairs} pares ({_OASIS1_EXPECTED_SUBJECTS} esperados)."
            f"\n         Puede que falten discos por descargar/extraer."
        )


# ===================================================================
# Step 2 — Master CSV Generation
# ===================================================================

def _map_cdr_to_label(cdr: float) -> int:
    """
    Mapea CDR (Clinical Dementia Rating) a etiqueta numérica.

    CDR 0.0 -> 0 (CN)
    CDR 0.5 -> 1 (MCI)
    CDR >= 1 -> 2 (AD)
    """
    if cdr == 0.0:
        return 0
    elif cdr == 0.5:
        return 1
    else:  # 1.0, 2.0, 3.0
        return 2


def generate_master_csv() -> pd.DataFrame:
    """
    Step 2: Genera el CSV maestro con etiquetas y rutas a imágenes.

    - Lee el Excel clínico de OASIS-1.
    - Filtra sujetos sin CDR.
    - Mapea CDR a labels (0=CN, 1=MCI, 2=AD).
    - Verifica que las imágenes existen en disco.
    - Guarda data/processed/dataset_master.csv.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Master CSV Generation")
    print("=" * 60)

    # Cargar datos clínicos
    clinical_path = cfg.OASIS_CLINICAL_FILE
    if not clinical_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo clínico: {clinical_path}\n"
            f"Asegúrate de que el Excel de OASIS está en la ruta correcta."
        )

    print(f"  Cargando: {clinical_path.name}")
    df = pd.read_excel(clinical_path, engine="openpyxl")
    print(f"  Filas totales: {len(df)}")

    # Filtrar sujetos sin CDR
    n_before = len(df)
    df = df.dropna(subset=["CDR"]).copy()
    n_after = len(df)
    print(f"  Filas con CDR válido: {n_after} (eliminadas {n_before - n_after} sin CDR)")

    # Mapear CDR a label
    df["label"] = df["CDR"].apply(_map_cdr_to_label)

    # Renombrar columna ID a subject_id
    df = df.rename(columns={"ID": "subject_id"})

    # Construir rutas a imágenes
    df["image_path"] = df["subject_id"].apply(
        lambda sid: str(cfg.PROCESSED_IMAGES_DIR / f"{sid}.img")
    )

    # Sanity check: verificar que las imágenes existen
    exists_mask = df["image_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()

    if n_missing > 0:
        missing_ids = df.loc[~exists_mask, "subject_id"].tolist()
        print(f"  [WARN] {n_missing} imágenes no encontradas en disco.")
        print(f"         Primeros 10: {missing_ids[:10]}")
        print(f"         Estas filas serán eliminadas del CSV maestro.")
        df = df[exists_mask].copy()

    print(f"  Sujetos válidos finales: {len(df)}")

    # Seleccionar y ordenar columnas de salida
    cols_out = ["subject_id", "image_path", "label", "Age", "CDR", "M/F"]
    cols_available = [c for c in cols_out if c in df.columns]
    df_out = df[cols_available].reset_index(drop=True)

    # Guardar
    cfg.PROCESSED_IMAGES_DIR.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg.MASTER_CSV_PATH, index=False)
    print(f"\n  Guardado: {cfg.MASTER_CSV_PATH}")

    # Distribución de clases
    print("\n  Distribución de clases:")
    for label_val, count in df_out["label"].value_counts().sort_index().items():
        label_name = cfg.CLASS_LABELS.get(label_val, "Unknown")
        print(f"    Clase {label_val} ({label_name}): {count}")

    return df_out


# ===================================================================
# Step 3 — Stratified Splitting
# ===================================================================

def stratified_split() -> None:
    """
    Step 3: Divide el dataset en Train/Val/Test con estratificación.

    - Train: 70%, Val: 15%, Test: 15%.
    - Estratificado por 'label' para mantener balance de clases.
    - random_state=42 para reproducibilidad.
    - Verifica que no haya fuga de datos (subject_id compartido).
    """
    print("\n" + "=" * 60)
    print("STEP 3: Stratified Splitting")
    print("=" * 60)

    # Cargar master CSV
    if not cfg.MASTER_CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {cfg.MASTER_CSV_PATH}.\n"
            f"Ejecuta generate_master_csv() primero."
        )

    df = pd.read_csv(cfg.MASTER_CSV_PATH)
    print(f"  Total de sujetos: {len(df)}")

    labels = df["label"]
    seed = cfg.RANDOM_SEED

    # Calcular ratio del segundo split:
    # Queremos val=15%, test=15% del total -> del 30% restante, 50/50
    test_val_ratio = cfg.VAL_RATIO + cfg.TEST_RATIO  # 0.30
    test_of_remaining = cfg.TEST_RATIO / test_val_ratio  # 0.5

    # Primer split: train vs (val+test)
    df_train, df_temp = train_test_split(
        df,
        test_size=test_val_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Segundo split: val vs test
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_of_remaining,
        stratify=df_temp["label"],
        random_state=seed,
    )

    # Guardar splits
    cfg.DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(cfg.DATA_SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(cfg.DATA_SPLITS_DIR / "val.csv", index=False)
    df_test.to_csv(cfg.DATA_SPLITS_DIR / "test.csv", index=False)

    print(f"\n  Guardados en: {cfg.DATA_SPLITS_DIR}")
    print(f"    train.csv: {len(df_train)} sujetos")
    print(f"    val.csv:   {len(df_val)} sujetos")
    print(f"    test.csv:  {len(df_test)} sujetos")

    # Distribución por split
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\n  [{name}] Distribución de clases:")
        for label_val, count in split_df["label"].value_counts().sort_index().items():
            label_name = cfg.CLASS_LABELS.get(label_val, "Unknown")
            pct = 100 * count / len(split_df)
            print(f"    Clase {label_val} ({label_name}): {count} ({pct:.1f}%)")

    # Verificación anti-leakage
    train_ids = set(df_train["subject_id"])
    val_ids = set(df_val["subject_id"])
    test_ids = set(df_test["subject_id"])

    leak_train_val = train_ids & val_ids
    leak_train_test = train_ids & test_ids
    leak_val_test = val_ids & test_ids

    if leak_train_val or leak_train_test or leak_val_test:
        print("\n  [ERROR] DATA LEAKAGE DETECTADO:")
        if leak_train_val:
            print(f"    Train & Val:  {leak_train_val}")
        if leak_train_test:
            print(f"    Train & Test: {leak_train_test}")
        if leak_val_test:
            print(f"    Val & Test:   {leak_val_test}")
        raise RuntimeError("Data leakage detectado entre splits.")
    else:
        print("\n  [OK] Anti-leakage check: 0 sujetos compartidos entre splits.")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    extract_and_standardize()
    generate_master_csv()
    stratified_split()

    print("\n" + "=" * 60)
    print("Pipeline ETL completado exitosamente.")
    print("=" * 60)
