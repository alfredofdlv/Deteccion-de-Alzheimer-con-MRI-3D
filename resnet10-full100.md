# Proyecto TFG — Detección Temprana de Alzheimer con 3D MRI

## Código Fuente (`src/`)

El proyecto contiene 8 módulos Python.

### `src/__init__.py` (3 líneas)

```python
# src/__init__.py
# Módulo principal del proyecto TFG: Detección Temprana de Alzheimer con 3D MRI
```

### `src/config.py` (112 líneas)

```python
"""
config.py — Configuración centralizada del proyecto.

Todas las variables globales (rutas, hiperparámetros, semillas) se definen aquí
para garantizar reproducibilidad y evitar "magic numbers" dispersos por el código.

Uso:
    from src.config import cfg
    print(cfg.IMAGE_SIZE)
    print(cfg.DATA_RAW_DIR)
"""

from pathlib import Path


class ProjectConfig:
    """Configuración centralizada del proyecto TFG."""

    # ========================
    # Semilla de reproducibilidad
    # ========================
    RANDOM_SEED: int = 42

    # ========================
    # Rutas del proyecto
    # ========================
    # Raíz del proyecto (dos niveles arriba de este archivo: src/config.py -> tfg/)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = PROJECT_ROOT / "data"
    DATA_RAW_DIR: Path = DATA_DIR / "raw"
    DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
    DATA_SPLITS_DIR: Path = DATA_DIR / "splits"
    PREPROCESSED_DIR: Path = DATA_DIR / "preprocessed"

    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

    # ========================
    # Rutas específicas OASIS-1
    # ========================
    OASIS_RAW_DIR: Path = DATA_DIR / "OASIS-1" / "raw"
    OASIS_CLINICAL_FILE: Path = OASIS_RAW_DIR / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
    PROCESSED_IMAGES_DIR: Path = DATA_PROCESSED_DIR / "images"
    MASTER_CSV_PATH: Path = DATA_PROCESSED_DIR / "dataset_master.csv"

    # ========================
    # Ratios de partición
    # ========================
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # ========================
    # Parámetros de imagen 3D
    # ========================
    # Tamaño objetivo para las MRI 3D tras preprocesamiento.
    # (96, 96, 96) es un buen compromiso entre resolución y consumo de VRAM
    # para GPUs con ≤8 GB (ej. RTX 3060, Colab T4).
    IMAGE_SIZE: tuple = (96, 96, 96)

    # ========================
    # Hiperparámetros de entrenamiento
    # ========================
    BATCH_SIZE: int = 4          # Conservador para GPUs con poca VRAM
    NUM_WORKERS: int = 4         # DataLoader workers (ajustar según CPU)
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4         # Regularización L2 (penalización sobre norma de pesos)
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 25  # Epochs sin mejora en val clinical F1 antes de parar

    # ========================
    # Prioridad clinica
    # ========================
    # Multiplicadores sobre los class weights de la loss (penalizacion asimetrica).
    # Fuerzan a la red a "sufrir" mas cuando falla en clases criticas.
    CLINICAL_WEIGHT_MULTIPLIERS: dict = {0: 1.0, 1: 1.5, 2: 2.0}

    # Pesos de la metrica clinical F1 para seleccion de modelo.
    # Priorizan deteccion de AD (60%) sobre MCI (30%) y CN (10%).
    CLINICAL_F1_WEIGHTS: dict = {0: 0.10, 1: 0.30, 2: 0.60}

    # ========================
    # Clases del dataset
    # ========================
    # CDR (Clinical Dementia Rating):
    #   0   = Sin demencia          -> Clase 0 (CN)
    #   0.5 = Demencia muy leve     -> Clase 1 (MCI)
    #   1+  = Demencia leve/mod.    -> Clase 2 (AD)
    NUM_CLASSES: int = 3
    CLASS_LABELS: dict = {
        0: "CN (Cognitively Normal)",
        1: "MCI (Mild Cognitive Impairment)",
        2: "AD (Alzheimer's Disease)",
    }

    def __repr__(self) -> str:
        return (
            f"ProjectConfig(\n"
            f"  RANDOM_SEED         = {self.RANDOM_SEED}\n"
            f"  IMAGE_SIZE          = {self.IMAGE_SIZE}\n"
            f"  BATCH_SIZE          = {self.BATCH_SIZE}\n"
            f"  NUM_WORKERS         = {self.NUM_WORKERS}\n"
            f"  NUM_CLASSES         = {self.NUM_CLASSES}\n"
            f"  PREPROCESSED_DIR    = {self.PREPROCESSED_DIR}\n"
            f"  OUTPUTS_DIR         = {self.OUTPUTS_DIR}\n"
            f")"
        )


# Instancia global — importar directamente:  from src.config import cfg
cfg = ProjectConfig()
```

### `src/data_utils.py` (72 líneas)

```python
"""
data_utils.py — Utilidades para carga de datos y gestion de rutas.

Funciones auxiliares para:
- Leer CSVs de particiones (train/val/test), con soporte para
  datos originales (NIfTI/ANALYZE) y preprocesados (.pt).
- Verificar la integridad de los datos procesados.

Uso:
    from src.data_utils import load_split, verify_data_integrity
"""

from pathlib import Path

import pandas as pd

from src.config import cfg


def load_split(split_name: str, dataset: str = "oasis1") -> pd.DataFrame:
    """
    Carga un CSV de particion desde data/splits/.

    Busca primero la version preprocesada (_pt.csv) y, si no existe,
    usa la original. Esto permite usar tensores .pt automaticamente
    cuando estan disponibles.

    Args:
        split_name: Nombre del split sin extension (ej. 'train', 'val', 'test').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
                 'oasis1' busca {split_name}.csv,
                 otros buscan {dataset}_{split_name}.csv.

    Returns:
        DataFrame con columnas esperadas: ['subject_id', 'image_path', 'label'].

    Raises:
        FileNotFoundError: Si el CSV no existe.
    """
    if dataset == "oasis1":
        pt_path = cfg.DATA_SPLITS_DIR / f"{split_name}_pt.csv"
        csv_path = cfg.DATA_SPLITS_DIR / f"{split_name}.csv"
    else:
        pt_path = cfg.DATA_SPLITS_DIR / f"{dataset}_{split_name}_pt.csv"
        csv_path = cfg.DATA_SPLITS_DIR / f"{dataset}_{split_name}.csv"

    if pt_path.exists():
        return pd.read_csv(pt_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de split: {csv_path}\n"
            f"Dataset: {dataset}, split: {split_name}"
        )
    return pd.read_csv(csv_path)


def verify_data_integrity() -> tuple[int, list[str]]:
    """
    Verifica cuantos pares .img/.hdr validos hay en data/processed/images/.

    Returns:
        Tupla con (numero de pares completos, lista de subject_ids con par).
    """
    if not cfg.PROCESSED_IMAGES_DIR.exists():
        return 0, []

    img_files = {p.stem for p in cfg.PROCESSED_IMAGES_DIR.glob("*.img")}
    hdr_files = {p.stem for p in cfg.PROCESSED_IMAGES_DIR.glob("*.hdr")}

    complete_pairs = sorted(img_files & hdr_files)
    return len(complete_pairs), complete_pairs
```

### `src/data_prepare.py` (480 líneas)

```python
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
```

### `src/dataset.py` (283 líneas)

```python
"""
dataset.py — Dataset MONAI y DataLoaders para MRI 3D (OASIS-1 / OASIS-3).

Proporciona el pipeline de transforms y los DataLoaders listos para
alimentar el modelo con tensores de forma (B, 1, 96, 96, 96).

Pipeline de transforms (base):
    LoadImaged -> EnsureChannelFirstd -> Orientationd(RAS)
    -> ScaleIntensityRangePercentilesd -> Resized(96, 96, 96)

Data augmentation (solo train):
    -> RandFlipd (eje 0) -> RandRotated -> RandGaussianNoised -> RandShiftIntensityd

Uso:
    from src.dataset import get_dataloader

    train_loader = get_dataloader("train")
    for batch in train_loader:
        images = batch["image"]  # (B, 1, 96, 96, 96)
        labels = batch["label"]  # (B,)
"""

from __future__ import annotations

import torch
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRangePercentilesd,
)

from src.config import cfg
from src.data_utils import load_split


class LoadPTd(MapTransform):
    """Carga un tensor .pt preprocesado (reemplaza a LoadImaged para archivos .pt)."""

    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # weights_only=False necesario: .pt contienen MetaTensor de MONAI
            d[key] = torch.load(d[key], weights_only=False, map_location="cpu")
        return d


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str = "train") -> Compose:
    """
    Construye el pipeline de MONAI transforms para un split dado.

    Pasos base (todos los splits):
        1. LoadImaged        — Carga el par .img/.hdr (formato ANALYZE).
        2. EnsureChannelFirstd — Añade dimensión de canal: (D,H,W) -> (1,D,H,W).
        3. Orientationd       — Reorienta a RAS (Right-Anterior-Superior).
        4. ScaleIntensityRangePercentilesd — Normaliza intensidad al rango [0, 1]
           usando percentiles 1-99 para robustez ante outliers.
        5. Resized            — Redimensiona a IMAGE_SIZE (96, 96, 96).

    Data augmentation (solo train):
        6. RandFlipd          — Flip aleatorio en eje LR (prob=0.5).
        7. RandRotated        — Rotación aleatoria en 3D (rango 0.2 rad, prob=0.3).
        8. RandGaussianNoised — Ruido gaussiano suave (prob=0.3, std=0.05).
        9. RandShiftIntensityd— Variación de intensidad (offsets=0.1, prob=0.3).

    Args:
        split: Nombre del split ('train', 'val', 'test').

    Returns:
        Compose con el pipeline de transforms.
    """
    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(keys=["image"], spatial_size=cfg.IMAGE_SIZE),
    ]

    if split == "train":
        transforms.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandRotated(keys=["image"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        ])

    return Compose(transforms)


def get_transforms_pt(split: str = "train") -> Compose:
    """
    Pipeline para tensores .pt ya preprocesados.

    Solo carga el tensor y aplica data augmentation si es train.
    Las transforms determinísticas (Orientation, Scale, Resize) ya se
    aplicaron offline por preprocess_to_pt.py.
    """
    transforms: list = [LoadPTd(keys=["image"])]

    if split == "train":
        transforms.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandRotated(keys=["image"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        ])

    return Compose(transforms)


def describe_transforms(split: str = "train", is_pt: bool = False) -> str:
    """
    Devuelve una descripcion legible del pipeline de transforms para un split.

    Args:
        split: Nombre del split ('train', 'val', 'test').
        is_pt: Si True, describe el flujo para tensores .pt preprocesados.
    """
    lines = [
        f"Pipeline de transforms para split: '{split}'",
        "=" * 55,
    ]

    if is_pt:
        lines += [
            "",
            "--- Formato: tensores .pt preprocesados offline ---",
            f"  1. LoadPTd              keys=['image'] (torch.load)",
            "",
            "  Preprocesamiento aplicado offline por preprocess_to_pt.py:",
            f"    LoadImage, EnsureChannelFirst, Orientation(RAS),",
            f"    ScaleIntensityRangePercentiles(1-99), Resize{cfg.IMAGE_SIZE}",
        ]
    else:
        lines += [
            "",
            "--- Preprocesamiento (deterministico, todos los splits) ---",
            f"  1. LoadImaged           keys=['image'], image_only=True",
            f"  2. EnsureChannelFirstd  keys=['image']",
            f"  3. Orientationd         keys=['image'], axcodes='RAS'",
            f"  4. ScaleIntensityRangePercentilesd",
            f"       keys=['image'], lower=1, upper=99",
            f"       b_min=0.0, b_max=1.0, clip=True",
            f"  5. Resized              keys=['image'], spatial_size={cfg.IMAGE_SIZE}",
        ]

    if split == "train":
        lines += [
            "",
            "--- Data Augmentation (estocastico, solo train) ---",
            f"  RandFlipd            keys=['image'], prob=0.5, spatial_axis=0",
            f"  RandRotated          keys=['image'], range_xyz=0.2, prob=0.3",
            f"  RandGaussianNoised   keys=['image'], prob=0.3, mean=0.0, std=0.05",
            f"  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.3",
            "",
            "Nota: las transforms Rand* se aplican on-the-fly en cada epoch.",
        ]
    else:
        lines += [
            "",
            "--- Sin Data Augmentation ---",
            f"  (split='{split}': solo preprocesamiento deterministico)",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data dicts
# ---------------------------------------------------------------------------

def _build_data_dicts(split: str, dataset: str = "oasis1") -> list[dict]:
    """
    Convierte un CSV de split en la lista de dicts que MONAI espera.

    Cada dict tiene la forma:
        {"image": "/ruta/al/imagen.img_o_.nii.gz", "label": 0}

    Args:
        split: Nombre del split ('train', 'val', 'test').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').

    Returns:
        Lista de diccionarios con claves 'image' y 'label'.
    """
    df = load_split(split, dataset=dataset)
    data_dicts = [
        {"image": row["image_path"], "label": int(row["label"])}
        for _, row in df.iterrows()
    ]
    return data_dicts


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def get_dataloader(
    split: str,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    use_cache: bool = False,
    dataset: str = "oasis1",
    subset: int | None = None,
) -> DataLoader:
    """
    Crea un DataLoader MONAI listo para iterar.

    Args:
        split: Nombre del split ('train', 'val', 'test').
        batch_size: Tamaño de batch. Por defecto cfg.BATCH_SIZE (4).
        shuffle: Mezclar datos. Por defecto True para 'train', False para el resto.
        num_workers: Workers del DataLoader. Por defecto cfg.NUM_WORKERS (2).
        use_cache: Si True, usa CacheDataset (precarga todos los volúmenes en RAM).
                   Recomendado solo si tienes >16 GB de RAM disponible.
                   Por defecto False (usa Dataset estándar).
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Si se indica, limita a los primeros N samples (para pruebas rapidas).

    Returns:
        monai.data.DataLoader con batches de:
            batch['image'] -> (B, 1, 96, 96, 96) float32
            batch['label'] -> (B,) int64
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if shuffle is None:
        shuffle = (split == "train")
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS

    data_dicts = _build_data_dicts(split, dataset=dataset)
    if subset is not None and subset < len(data_dicts):
        data_dicts = data_dicts[:subset]

    is_pt = len(data_dicts) > 0 and data_dicts[0]["image"].endswith(".pt")
    transforms = get_transforms_pt(split) if is_pt else get_transforms(split)

    if use_cache:
        ds = CacheDataset(
            data=data_dicts,
            transform=transforms,
            num_workers=num_workers,
        )
    else:
        ds = Dataset(data=data_dicts, transform=transforms)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return loader
```

### `src/model.py` (144 líneas)

```python
"""
model.py — Modelos 3D para clasificacion de MRI cerebral.

Modelos disponibles:
    - resnet10:     ResNet-10 3D de MONAI (~14.3M params)
    - simple3dcnn:  CNN manual de 4 bloques conv (~1.16M params)

Uso:
    from src.model import get_model
    model = get_model("resnet10")
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificacion rapida:
    python -m src.model
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet10

from src.config import cfg


# ---------------------------------------------------------------------------
# Simple3DCNN — CNN manual de 4 bloques
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloque Conv3d(3x3x3, pad=1) -> BatchNorm3d -> ReLU -> MaxPool3d(2)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2),
    )


class Simple3DCNN(nn.Module):
    """
    CNN 3D con 4 bloques convolucionales para volumenes cerebrales.

    Progresion de canales: 1 -> 32 -> 64 -> 128 -> 256.
    Con entrada 96^3: 96 -> 48 -> 24 -> 12 -> 6, luego
    AdaptiveAvgPool3d colapsa a 1^3. ~1.16M parametros.

    Args:
        in_channels: Canales de entrada (1 para MRI).
        num_classes: Numero de clases de salida (3: CN, MCI, AD).
        dropout: Probabilidad de dropout antes de la capa lineal.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = cfg.NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# AlzheimerResNet — ResNet-10 3D de MONAI
# ---------------------------------------------------------------------------

class AlzheimerResNet(nn.Module):
    """
    ResNet-10 3D para clasificacion de volumenes cerebrales MRI.

    Wrapper de monai.networks.nets.resnet10: entrada 3D monocanal,
    3 clases de salida. ~14.3M parametros.

    Args:
        num_classes: Numero de clases de salida.
    """

    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.net = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = ["resnet10", "simple3dcnn"]


def get_model(name: str = "resnet10") -> nn.Module:
    """Instancia un modelo por nombre.

    Args:
        name: 'resnet10' o 'simple3dcnn'.

    Returns:
        nn.Module listo para .to(device).
    """
    if name == "resnet10":
        return AlzheimerResNet()
    elif name == "simple3dcnn":
        return Simple3DCNN()
    else:
        raise ValueError(f"Modelo '{name}' no reconocido. Opciones: {AVAILABLE_MODELS}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(1, 1, 96, 96, 96, device=device)

    for name in AVAILABLE_MODELS:
        print(f"\n{'=' * 50}")
        print(f"Modelo: {name}")
        print(f"{'=' * 50}")
        model = get_model(name).to(device)
        out = model(dummy)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Device:       {device}")
        print(f"  Input shape:  {dummy.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Parametros:   {n_params:,}")
        assert out.shape == (1, cfg.NUM_CLASSES)
        print(f"  CHECK PASSED")
```

### `src/train.py` (618 líneas)

```python
"""
train.py — Training loop para AlzheimerResNet sobre OASIS-1 / OASIS-3.

Metrica de seleccion de modelo: Clinical F1-Score (ponderado: 60% AD, 30% MCI, 10% CN).
Ademas de penalizacion asimetrica en la loss (multiplicadores clinicos sobre class weights).

Genera automaticamente en outputs/<run_name>/:
    - training_log.csv       — metricas por epoch (incluye macro F1 y clinical F1)
    - curves_loss.png        — grafica de loss (train vs val)
    - curves_accuracy.png    — grafica de accuracy (train vs val)
    - curves_f1.png          — grafica de clinical F1 (train vs val)
    - best_model.pth         — pesos del mejor modelo (mayor val clinical F1)
    - training_summary.txt   — resumen legible del entrenamiento

Modos de ejecucion:
    python -m src.train --overfit                   # sanity check
    python -m src.train --epochs 2 --run test_2ep   # prueba rapida
    python -m src.train --epochs 100 --run full     # entrenamiento largo
    python -m src.train --patience 15 --run exp1    # patience custom
"""

from __future__ import annotations

import csv
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.config import cfg
from src.data_utils import load_split
from src.dataset import describe_transforms, get_dataloader
from src.model import AVAILABLE_MODELS, get_model


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Para el entrenamiento si val macro F1 no mejora en `patience` epochs."""

    def __init__(self, patience: int = cfg.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_f1 = 0.0
        self.counter = 0
        self.triggered = False

    def step(self, val_f1: float) -> bool:
        """Retorna True si se debe parar el entrenamiento."""
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: str = "oasis1") -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase,
    con multiplicadores clinicos para penalizar mas los errores en AD/MCI.

    Formula base: weight_i = N_total / (N_classes * N_i)
    Luego: weight_i *= CLINICAL_WEIGHT_MULTIPLIERS[i]
    """
    df = load_split("train", dataset=dataset)
    counts = df["label"].value_counts().sort_index()
    n_total = len(df)
    n_classes = cfg.NUM_CLASSES

    weights = []
    for c in range(n_classes):
        n_c = counts.get(c, 1)
        weights.append(n_total / (n_classes * n_c))

    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights (base):   {w.tolist()}")

    for c, mult in cfg.CLINICAL_WEIGHT_MULTIPLIERS.items():
        w[c] *= mult
    print(f"[INFO] Class weights (clinico): {w.tolist()}")
    print(f"[INFO] Multiplicadores:         {cfg.CLINICAL_WEIGHT_MULTIPLIERS}")
    return w


# ---------------------------------------------------------------------------
# Clinical F1 metric
# ---------------------------------------------------------------------------

def compute_clinical_f1(labels: list[int], preds: list[int]) -> float:
    """F1 ponderado con prioridad clinica: 60% AD, 30% MCI, 10% CN."""
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    weights = cfg.CLINICAL_F1_WEIGHTS
    return sum(weights[c] * f1_per_class[c] for c in range(len(f1_per_class)))


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def _progress_log(step: int, total_steps: int, t_start: float, prefix: str,
                   running_loss: float, correct: int, total_samples: int) -> None:
    """Imprime progreso intra-epoch."""
    pct = step / total_steps
    elapsed = time.time() - t_start
    eta = (elapsed / step) * (total_steps - step) if step > 0 else 0
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_acc = correct / total_samples if total_samples > 0 else 0
    print(
        f"\r  {prefix} [{step}/{total_steps}] "
        f"{pct:>6.1%} | loss: {avg_loss:.4f} | acc: {avg_acc:.2%} | "
        f"{elapsed:.0f}s / ETA {eta:.0f}s",
        end="", flush=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Ejecuta un epoch de entrenamiento. Retorna metricas incluyendo clinical_f1."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    t0 = time.time()

    for i, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if i % log_every == 0 or i == n_batches:
            _progress_log(i, n_batches, t0, "Train", running_loss, correct, total)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    clin_f1 = compute_clinical_f1(all_labels, all_preds)
    print()
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "clinical_f1": clin_f1,
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evalua el modelo sin gradientes. Retorna metricas incluyendo clinical_f1."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if i % log_every == 0 or i == n_batches:
                _progress_log(i, n_batches, t0, "Val  ", running_loss, correct, total)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    clin_f1 = compute_clinical_f1(all_labels, all_preds)
    print()
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "clinical_f1": clin_f1,
    }


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def _save_plots(history: list[dict], run_dir: Path) -> None:
    """Genera y guarda graficas de loss, accuracy y clinical F1."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] * 100 for r in history]
    val_acc = [r["val_acc"] * 100 for r in history]
    train_clin_f1 = [r["train_clinical_f1"] * 100 for r in history]
    val_clin_f1 = [r["val_clinical_f1"] * 100 for r in history]

    best_idx = max(range(len(val_clin_f1)), key=lambda i: val_clin_f1[i])

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "o-", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "s-", label="Val Loss", linewidth=2, markersize=4)
    ax.axvline(x=epochs[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best epoch ({epochs[best_idx]})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "curves_loss.png", dpi=150)
    plt.close(fig)

    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, "o-", label="Train Acc", linewidth=2, markersize=4)
    ax.plot(epochs, val_acc, "s-", label="Val Acc", linewidth=2, markersize=4)
    ax.axvline(x=epochs[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best epoch ({epochs[best_idx]})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(run_dir / "curves_accuracy.png", dpi=150)
    plt.close(fig)

    # --- Clinical F1 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_clin_f1, "o-", label="Train Clinical F1", linewidth=2, markersize=4)
    ax.plot(epochs, val_clin_f1, "s-", label="Val Clinical F1", linewidth=2, markersize=4)
    ax.axvline(x=epochs[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best epoch ({epochs[best_idx]})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Clinical F1 (%)", fontsize=12)
    ax.set_title("Training & Validation Clinical F1-Score (60% AD, 30% MCI, 10% CN)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(run_dir / "curves_f1.png", dpi=150)
    plt.close(fig)


def _save_summary(history: list[dict], run_dir: Path, device: torch.device,
                  n_params: int, elapsed_total: float, class_weights: list,
                  early_stopped: bool = False, patience: int = 0,
                  model_name: str = "resnet10") -> None:
    """Genera un archivo de resumen legible."""
    best = max(history, key=lambda r: r["val_clinical_f1"])
    last = history[-1]

    stop_reason = f"Early stopping (patience={patience})" if early_stopped else "Completado"

    lines = [
        "=" * 60,
        "TRAINING SUMMARY",
        "=" * 60,
        f"Fecha:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device:             {device}",
        f"Modelo:             {model_name}",
        f"Parametros:         {n_params:,}",
        f"Class weights:      {class_weights}",
        f"  Multiplicadores:  {cfg.CLINICAL_WEIGHT_MULTIPLIERS}",
        f"Learning rate:      {cfg.LEARNING_RATE}",
        f"Weight decay (L2):  {cfg.WEIGHT_DECAY}",
        f"Batch size:         {cfg.BATCH_SIZE}",
        f"Image size:         {cfg.IMAGE_SIZE}",
        f"Epochs:             {len(history)}",
        f"Finalizacion:       {stop_reason}",
        f"Metrica seleccion:  Clinical F1 (pesos: {cfg.CLINICAL_F1_WEIGHTS})",
        f"Tiempo total:       {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)",
        f"Tiempo por epoch:   {elapsed_total/len(history):.1f}s",
        "",
        "--- Mejor Epoch (por clinical F1) ---",
        f"  Epoch:               {best['epoch']}",
        f"  Train Loss:          {best['train_loss']:.4f}",
        f"  Train Acc:           {best['train_acc']:.2%}",
        f"  Train F1 (macro):    {best['train_f1']:.4f}",
        f"  Train F1 (clinical): {best['train_clinical_f1']:.4f}",
        f"  Val Loss:            {best['val_loss']:.4f}",
        f"  Val Acc:             {best['val_acc']:.2%}",
        f"  Val F1 (macro):      {best['val_f1']:.4f}",
        f"  Val F1 (clinical):   {best['val_clinical_f1']:.4f}",
        "",
        "--- Ultimo Epoch ---",
        f"  Epoch:               {last['epoch']}",
        f"  Train Loss:          {last['train_loss']:.4f}",
        f"  Train Acc:           {last['train_acc']:.2%}",
        f"  Train F1 (macro):    {last['train_f1']:.4f}",
        f"  Train F1 (clinical): {last['train_clinical_f1']:.4f}",
        f"  Val Loss:            {last['val_loss']:.4f}",
        f"  Val Acc:             {last['val_acc']:.2%}",
        f"  Val F1 (macro):      {last['val_f1']:.4f}",
        f"  Val F1 (clinical):   {last['val_clinical_f1']:.4f}",
        "",
        f"Archivos generados en: {run_dir}",
        "=" * 60,
    ]
    text = "\n".join(lines)
    (run_dir / "training_summary.txt").write_text(text, encoding="utf-8")
    print(f"\n{text}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    num_epochs: int = cfg.NUM_EPOCHS,
    overfit_one_batch: bool = False,
    run_name: str | None = None,
    patience: int = cfg.EARLY_STOPPING_PATIENCE,
    dataset: str = "oasis1",
    subset: int | None = None,
    model_name: str = "resnet10",
) -> None:
    """
    Funcion principal de entrenamiento.

    Args:
        num_epochs: Numero maximo de epochs.
        overfit_one_batch: Si True, entrena solo con 4 imagenes durante
                          100 epochs (sanity check de convergencia).
        run_name: Nombre de la carpeta dentro de outputs/ para esta ejecucion.
                  Si None, se genera uno automatico con timestamp.
        patience: Epochs sin mejora en val clinical F1 antes de early stopping.
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Limitar cada split a N samples (para pruebas rapidas).
        model_name: Nombre del modelo ('resnet10' o 'simple3dcnn').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    torch.manual_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    model = get_model(model_name).to(device)
    print(f"[INFO] Modelo: {model_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parametros entrenables: {n_params:,}")

    class_weights = compute_class_weights(dataset=dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # -- Overfit-one-batch mode -----------------------------------------------
    if overfit_one_batch:
        print("\n" + "=" * 60)
        print("MODO OVERFIT-ONE-BATCH (sanity check)")
        print("=" * 60)

        loader = get_dataloader("train", batch_size=4, num_workers=0, shuffle=False, dataset=dataset, subset=subset)
        single_batch = next(iter(loader))
        images = single_batch["image"].to(device)
        labels = single_batch["label"].to(device)
        print(f"Batch labels: {labels.tolist()}")

        num_epochs = 100
        model.train()
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{num_epochs} — "
                    f"Loss: {loss.item():.4f}  Acc: {acc:.2%}"
                )

        print(f"\n  Final — Loss: {loss.item():.6f}  Acc: {acc:.2%}")
        if loss.item() < 0.05 and acc == 1.0:
            print("  === CHECKPOINT PASSED ===")
        else:
            print("  [WARN] No convergio completamente. Revisar el modelo.")
        return

    # -- Entrenamiento completo -----------------------------------------------
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience)

    transforms_desc = describe_transforms("train")
    (run_dir / "transforms_config.txt").write_text(
        transforms_desc + "\n", encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO — max {num_epochs} epochs (early stopping: patience={patience})")
    print(f"Metrica de seleccion: Clinical F1 (pesos: {cfg.CLINICAL_F1_WEIGHTS})")
    print(f"Resultados en: {run_dir}")
    print("=" * 60)

    t_load = time.time()
    if subset:
        print(f"[INFO] Modo subset: limitando a {subset} samples por split")
    print("[INFO] Cargando datos de entrenamiento...")
    train_loader = get_dataloader("train", dataset=dataset, subset=subset)
    print(f"[INFO] Cargando datos de validacion...")
    val_loader = get_dataloader("val", dataset=dataset, subset=subset)
    print(
        f"[INFO] Datos listos en {time.time() - t_load:.1f}s — "
        f"Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples), "
        f"Val: {len(val_loader)} batches ({len(val_loader.dataset)} samples)"
    )

    csv_path = run_dir / "training_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch", "train_loss", "train_acc", "train_f1", "train_clinical_f1",
            "val_loss", "val_acc", "val_f1", "val_clinical_f1",
            "epoch_time_s", "is_best",
        ],
    )
    csv_writer.writeheader()

    history: list[dict] = []
    best_val_clinical_f1 = 0.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        is_best = val_metrics["clinical_f1"] > best_val_clinical_f1

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "train_f1": round(train_metrics["macro_f1"], 6),
            "train_clinical_f1": round(train_metrics["clinical_f1"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "val_f1": round(val_metrics["macro_f1"], 6),
            "val_clinical_f1": round(val_metrics["clinical_f1"], 6),
            "epoch_time_s": round(elapsed, 1),
            "is_best": is_best,
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        if is_best:
            best_val_clinical_f1 = val_metrics["clinical_f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_clinical_f1": best_val_clinical_f1,
            }, run_dir / "best_model.pth")

        should_stop = early_stopper.step(val_metrics["clinical_f1"])
        scheduler.step(val_metrics["loss"])

        # -- Logging informativo --
        elapsed_total_so_far = time.time() - t_start
        avg_epoch_time = elapsed_total_so_far / epoch
        remaining_epochs = num_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        elapsed_str = str(timedelta(seconds=int(elapsed_total_so_far)))

        best_marker = " << BEST" if is_best else ""
        es_counter = early_stopper.counter
        es_bar = f"[{'#' * es_counter}{'.' * (patience - es_counter)}]"
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n--- Epoch {epoch}/{num_epochs} "
            f"({elapsed:.0f}s | total: {elapsed_str} | ETA: {eta_str}) ---\n"
            f"  Train  ->  Loss: {train_metrics['loss']:.4f}  |  Acc: {train_metrics['accuracy']:.2%}  "
            f"|  F1m: {train_metrics['macro_f1']:.4f}  |  F1c: {train_metrics['clinical_f1']:.4f}\n"
            f"  Val    ->  Loss: {val_metrics['loss']:.4f}  |  Acc: {val_metrics['accuracy']:.2%}  "
            f"|  F1m: {val_metrics['macro_f1']:.4f}  |  F1c: {val_metrics['clinical_f1']:.4f}{best_marker}\n"
            f"  LR: {current_lr:.2e}  |  Best: epoch {best_epoch} (clin_f1={best_val_clinical_f1:.4f})  "
            f"| Early stop: {es_bar} {es_counter}/{patience}"
        )

        if should_stop:
            print(
                f"\n{'=' * 60}\n"
                f"[EARLY STOPPING] Val clinical F1 no mejoro en {patience} epochs.\n"
                f"Mejor epoch: {best_epoch} (clin_f1={best_val_clinical_f1:.4f})\n"
                f"{'=' * 60}"
            )
            break

    csv_file.close()
    elapsed_total = time.time() - t_start

    _save_plots(history, run_dir)
    _save_summary(
        history, run_dir, device, n_params, elapsed_total,
        class_weights.cpu().tolist(),
        early_stopped=early_stopper.triggered,
        patience=patience,
        model_name=model_name,
    )
    print(f"\nArchivos generados:")
    for f in sorted(run_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar modelo 3D sobre OASIS-1 / OASIS-3")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help=f"Numero de epochs (default: {cfg.NUM_EPOCHS})")
    parser.add_argument("--overfit", action="store_true",
                        help="Modo overfit-one-batch (sanity check)")
    parser.add_argument("--run", type=str, default=None,
                        help="Nombre de la carpeta de resultados (default: run_TIMESTAMP)")
    parser.add_argument("--patience", type=int, default=cfg.EARLY_STOPPING_PATIENCE,
                        help=f"Early stopping patience (default: {cfg.EARLY_STOPPING_PATIENCE})")
    parser.add_argument("--dataset", type=str, default="oasis1",
                        choices=["oasis1", "oasis3"],
                        help="Dataset a utilizar (default: oasis1)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limitar a N samples por split (para pruebas rapidas)")
    parser.add_argument("--model", type=str, default="resnet10",
                        choices=AVAILABLE_MODELS,
                        help="Modelo a usar (default: resnet10)")
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        overfit_one_batch=args.overfit,
        run_name=args.run,
        patience=args.patience,
        dataset=args.dataset,
        subset=args.subset,
        model_name=args.model,
    )
```

### `src/evaluate.py` (241 líneas)

```python
"""
evaluate.py — Evaluación del modelo sobre el test set con métricas detalladas (OASIS-1 / OASIS-3).

Genera en outputs/<run_name>/:
    - classification_report.txt  — precision, recall, F1 por clase
    - confusion_matrix.png       — heatmap de la matriz de confusión

Uso:
    python -m src.evaluate --run full_100ep
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import cfg
from src.dataset import get_dataloader
from src.model import AVAILABLE_MODELS, get_model


CLASS_NAMES = ["CN", "MCI", "AD"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Ejecuta inferencia y recopila todas las predicciones y labels."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

            if i % log_every == 0 or i == n_batches:
                elapsed = time.time() - t0
                eta = (elapsed / i) * (n_batches - i) if i > 0 else 0
                print(
                    f"\r  Eval [{i}/{n_batches}] {i/n_batches:>6.1%} | "
                    f"{elapsed:.0f}s / ETA {eta:.0f}s",
                    end="", flush=True,
                )

    print()
    return all_labels, all_preds


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    accuracy: float,
    save_path: Path,
) -> None:
    """Genera y guarda un heatmap de la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_xlabel("Predicción", fontsize=13)
    ax.set_ylabel("Real", fontsize=13)
    ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.2%})", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_model(run_name: str, split: str = "test", dataset: str = "oasis1",
                   subset: int | None = None, model_name: str | None = None) -> None:
    """
    Evalua el mejor modelo de un run sobre un split y genera reportes.

    Args:
        run_name: Nombre de la carpeta en outputs/ que contiene best_model.pth.
        split: Split a evaluar ('test' por defecto, tambien acepta 'val').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Limitar a N samples (para pruebas rapidas).
        model_name: Nombre del modelo. Si None, se lee del checkpoint.
    """
    run_dir = cfg.OUTPUTS_DIR / run_name
    model_path = run_dir / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo: {model_path}\n"
            f"Asegurate de haber entrenado con --run {run_name}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    resolved_model = model_name or checkpoint.get("model_name", "resnet10")
    print(f"[INFO] Modelo: {resolved_model}")
    model = get_model(resolved_model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_f1 = checkpoint.get("val_f1")
    ckpt_info = (
        f"[INFO] Modelo cargado desde epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_accuracy']:.2%}"
    )
    if val_f1 is not None:
        ckpt_info += f", val_f1={val_f1:.4f}"
    ckpt_info += ")"
    print(ckpt_info)

    # Inferencia
    loader = get_dataloader(split, shuffle=False, num_workers=0, dataset=dataset, subset=subset)
    print(f"[INFO] Evaluando sobre '{split}' ({len(loader.dataset)} muestras)...")

    all_labels, all_preds = collect_predictions(model, loader, device)

    # Métricas
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.NUM_CLASSES)))

    # Mostrar resultados
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN — {split.upper()} SET ({run_name})")
    print(f"{'=' * 60}")
    print(f"\nAccuracy global: {acc:.2%}")
    print(f"\n{report}")
    print(f"Confusion Matrix:")
    print(cm)

    # Guardar classification report
    report_path = run_dir / f"classification_report_{split}.txt"
    report_text = (
        f"{'=' * 60}\n"
        f"EVALUACIÓN — {split.upper()} SET\n"
        f"Run: {run_name}\n"
        f"Modelo: epoch {checkpoint['epoch']}\n"
        f"{'=' * 60}\n\n"
        f"Accuracy global: {acc:.2%}\n\n"
        f"{report}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Reporte guardado en: {report_path}")

    # Guardar confusion matrix plot
    cm_path = run_dir / f"confusion_matrix_{split}.png"
    _plot_confusion_matrix(cm, CLASS_NAMES, acc, cm_path)
    print(f"[OK] Gráfica guardada en: {cm_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluar modelo 3D sobre test/val set"
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Nombre de la carpeta en outputs/ (ej. full_100ep)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"],
        help="Split a evaluar (default: test)",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis1",
        choices=["oasis1", "oasis3"],
        help="Dataset a utilizar (default: oasis1)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limitar a N samples (para pruebas rapidas)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=AVAILABLE_MODELS,
        help="Modelo a usar (default: auto-detectar del checkpoint)",
    )
    args = parser.parse_args()

    evaluate_model(run_name=args.run, split=args.split, dataset=args.dataset,
                   subset=args.subset, model_name=args.model)
```

---

## Resultados: `resnet10-oasis3-100ep`

Directorio: `outputs/resnet10-oasis3-100ep/`

### `pipeline.log`

```log
Inicio: 2026-03-23 12:07:48  |  Run: resnet10-oasis3-100ep
============================================================

============================================================
  PASO 1/3 — Entrenamiento
  > /home/aflorez/venv/bin/python -m src.train --run resnet10-oasis3-100ep --dataset oasis3 --model resnet10 --epochs 100
============================================================

[INFO] Device: cuda
[INFO] Modelo: resnet10
[INFO] Parametros entrenables: 14,357,955
[INFO] Class weights (base):   [0.4254224896430969, 2.105454444885254, 5.732673168182373]
[INFO] Class weights (clinico): [0.4254224896430969, 3.158181667327881, 11.465346336364746]
[INFO] Multiplicadores:         {0: 1.0, 1: 1.5, 2: 2.0}

============================================================
ENTRENAMIENTO — max 100 epochs (early stopping: patience=25)
Metrica de seleccion: Clinical F1 (pesos: {0: 0.1, 1: 0.3, 2: 0.6})
Resultados en: /media/nas/aflorez/Deteccion-de-Alzheimer-con-MRI-3D/outputs/resnet10-oasis3-100ep
============================================================
[INFO] Cargando datos de entrenamiento...
[INFO] Cargando datos de validacion...
[INFO] Datos listos en 0.1s — Train: 435 batches (1737 samples), Val: 90 batches (359 samples)
  Train [435/435] 100.0% | loss: 1.6945 | acc: 12.90% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7347 | acc: 6.13% | 11s / ETA 0s

--- Epoch 1/100 (161s | total: 0:02:43 | ETA: 4:29:01) ---
  Train  ->  Loss: 1.6945  |  Acc: 12.90%  |  F1m: 0.1356  |  F1c: 0.1221
  Val    ->  Loss: 1.7347  |  Acc: 6.13%  |  F1m: 0.0409  |  F1c: 0.0629 << BEST
  LR: 1.00e-04  |  Best: epoch 1 (clin_f1=0.0629)  | Early stop: [.........................] 0/25
  Train [435/435] 100.0% | loss: 1.6535 | acc: 12.90% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7167 | acc: 6.13% | 11s / ETA 0s

--- Epoch 2/100 (161s | total: 0:05:25 | ETA: 4:25:49) ---
  Train  ->  Loss: 1.6535  |  Acc: 12.90%  |  F1m: 0.1305  |  F1c: 0.1254
  Val    ->  Loss: 1.7167  |  Acc: 6.13%  |  F1m: 0.0496  |  F1c: 0.0698 << BEST
  LR: 1.00e-04  |  Best: epoch 2 (clin_f1=0.0698)  | Early stop: [.........................] 0/25
  Train [435/435] 100.0% | loss: 1.6677 | acc: 10.42% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7093 | acc: 11.98% | 11s / ETA 0s

--- Epoch 3/100 (162s | total: 0:08:09 | ETA: 4:23:41) ---
  Train  ->  Loss: 1.6677  |  Acc: 10.42%  |  F1m: 0.1086  |  F1c: 0.1130
  Val    ->  Loss: 1.7093  |  Acc: 11.98%  |  F1m: 0.1421  |  F1c: 0.1378 << BEST
  LR: 1.00e-04  |  Best: epoch 3 (clin_f1=0.1378)  | Early stop: [.........................] 0/25
  Train [435/435] 100.0% | loss: 1.6619 | acc: 12.78% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7090 | acc: 13.09% | 11s / ETA 0s

--- Epoch 4/100 (161s | total: 0:10:50 | ETA: 4:20:05) ---
  Train  ->  Loss: 1.6619  |  Acc: 12.78%  |  F1m: 0.1203  |  F1c: 0.1117
  Val    ->  Loss: 1.7090  |  Acc: 13.09%  |  F1m: 0.1132  |  F1c: 0.0987
  LR: 1.00e-04  |  Best: epoch 3 (clin_f1=0.1378)  | Early stop: [#........................] 1/25
  Train [435/435] 100.0% | loss: 1.6592 | acc: 15.37% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7910 | acc: 5.29% | 12s / ETA 0s

--- Epoch 5/100 (162s | total: 0:13:32 | ETA: 4:17:09) ---
  Train  ->  Loss: 1.6592  |  Acc: 15.37%  |  F1m: 0.1552  |  F1c: 0.1343
  Val    ->  Loss: 1.7910  |  Acc: 5.29%  |  F1m: 0.0335  |  F1c: 0.0603
  LR: 1.00e-04  |  Best: epoch 3 (clin_f1=0.1378)  | Early stop: [##.......................] 2/25
  Train [435/435] 100.0% | loss: 1.6441 | acc: 20.26% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7266 | acc: 48.47% | 11s / ETA 0s

--- Epoch 6/100 (162s | total: 0:16:15 | ETA: 4:14:49) ---
  Train  ->  Loss: 1.6441  |  Acc: 20.26%  |  F1m: 0.1824  |  F1c: 0.1460
  Val    ->  Loss: 1.7266  |  Acc: 48.47%  |  F1m: 0.3214  |  F1c: 0.1972 << BEST
  LR: 1.00e-04  |  Best: epoch 6 (clin_f1=0.1972)  | Early stop: [.........................] 0/25
  Train [435/435] 100.0% | loss: 1.6394 | acc: 18.54% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7668 | acc: 11.70% | 11s / ETA 0s

--- Epoch 7/100 (161s | total: 0:18:57 | ETA: 4:11:47) ---
  Train  ->  Loss: 1.6394  |  Acc: 18.54%  |  F1m: 0.1687  |  F1c: 0.1351
  Val    ->  Loss: 1.7668  |  Acc: 11.70%  |  F1m: 0.1489  |  F1c: 0.1589
  LR: 1.00e-04  |  Best: epoch 6 (clin_f1=0.1972)  | Early stop: [#........................] 1/25
  Train [435/435] 100.0% | loss: 1.6363 | acc: 20.78% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7755 | acc: 17.27% | 11s / ETA 0s

--- Epoch 8/100 (162s | total: 0:21:39 | ETA: 4:08:59) ---
  Train  ->  Loss: 1.6363  |  Acc: 20.78%  |  F1m: 0.2131  |  F1c: 0.1722
  Val    ->  Loss: 1.7755  |  Acc: 17.27%  |  F1m: 0.1597  |  F1c: 0.1403
  LR: 1.00e-04  |  Best: epoch 6 (clin_f1=0.1972)  | Early stop: [##.......................] 2/25
  Train [435/435] 100.0% | loss: 1.6397 | acc: 21.19% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7092 | acc: 20.89% | 11s / ETA 0s

--- Epoch 9/100 (161s | total: 0:24:20 | ETA: 4:06:08) ---
  Train  ->  Loss: 1.6397  |  Acc: 21.19%  |  F1m: 0.2076  |  F1c: 0.1630
  Val    ->  Loss: 1.7092  |  Acc: 20.89%  |  F1m: 0.2055  |  F1c: 0.1570
  LR: 1.00e-04  |  Best: epoch 6 (clin_f1=0.1972)  | Early stop: [###......................] 3/25
  Train [435/435] 100.0% | loss: 1.6406 | acc: 22.57% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7228 | acc: 57.66% | 11s / ETA 0s

--- Epoch 10/100 (161s | total: 0:27:03 | ETA: 4:03:27) ---
  Train  ->  Loss: 1.6406  |  Acc: 22.57%  |  F1m: 0.2164  |  F1c: 0.1680
  Val    ->  Loss: 1.7228  |  Acc: 57.66%  |  F1m: 0.4181  |  F1c: 0.2736 << BEST
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [.........................] 0/25
  Train [435/435] 100.0% | loss: 1.6195 | acc: 26.42% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7070 | acc: 15.60% | 12s / ETA 0s

--- Epoch 11/100 (162s | total: 0:29:45 | ETA: 4:00:45) ---
  Train  ->  Loss: 1.6195  |  Acc: 26.42%  |  F1m: 0.2373  |  F1c: 0.1740
  Val    ->  Loss: 1.7070  |  Acc: 15.60%  |  F1m: 0.1394  |  F1c: 0.1136
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [#........................] 1/25
  Train [435/435] 100.0% | loss: 1.6204 | acc: 26.66% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7120 | acc: 25.91% | 11s / ETA 0s

--- Epoch 12/100 (161s | total: 0:32:26 | ETA: 3:57:51) ---
  Train  ->  Loss: 1.6204  |  Acc: 26.66%  |  F1m: 0.2636  |  F1c: 0.2000
  Val    ->  Loss: 1.7120  |  Acc: 25.91%  |  F1m: 0.2647  |  F1c: 0.1992
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [##.......................] 2/25
  Train [435/435] 100.0% | loss: 1.6171 | acc: 25.50% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7115 | acc: 38.16% | 11s / ETA 0s

--- Epoch 13/100 (161s | total: 0:35:07 | ETA: 3:55:01) ---
  Train  ->  Loss: 1.6171  |  Acc: 25.50%  |  F1m: 0.2588  |  F1c: 0.1993
  Val    ->  Loss: 1.7115  |  Acc: 38.16%  |  F1m: 0.3064  |  F1c: 0.2053
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [###......................] 3/25
  Train [435/435] 100.0% | loss: 1.6167 | acc: 30.45% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7718 | acc: 8.91% | 12s / ETA 0s

--- Epoch 14/100 (162s | total: 0:37:49 | ETA: 3:52:18) ---
  Train  ->  Loss: 1.6167  |  Acc: 30.45%  |  F1m: 0.2943  |  F1c: 0.2229
  Val    ->  Loss: 1.7718  |  Acc: 8.91%  |  F1m: 0.1025  |  F1c: 0.1127
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [####.....................] 4/25
  Train [435/435] 100.0% | loss: 1.5982 | acc: 25.96% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7247 | acc: 14.48% | 11s / ETA 0s

--- Epoch 15/100 (161s | total: 0:40:29 | ETA: 3:49:28) ---
  Train  ->  Loss: 1.5982  |  Acc: 25.96%  |  F1m: 0.2571  |  F1c: 0.1982
  Val    ->  Loss: 1.7247  |  Acc: 14.48%  |  F1m: 0.1752  |  F1c: 0.1532
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [#####....................] 5/25
  Train [435/435] 100.0% | loss: 1.6170 | acc: 31.38% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7266 | acc: 39.00% | 11s / ETA 0s

--- Epoch 16/100 (159s | total: 0:43:08 | ETA: 3:46:30) ---
  Train  ->  Loss: 1.6170  |  Acc: 31.38%  |  F1m: 0.2967  |  F1c: 0.2221
  Val    ->  Loss: 1.7266  |  Acc: 39.00%  |  F1m: 0.2692  |  F1c: 0.1662
  LR: 5.00e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [######...................] 6/25
  Train [435/435] 100.0% | loss: 1.5945 | acc: 32.64% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7768 | acc: 29.25% | 11s / ETA 0s

--- Epoch 17/100 (159s | total: 0:45:48 | ETA: 3:43:37) ---
  Train  ->  Loss: 1.5945  |  Acc: 32.64%  |  F1m: 0.3207  |  F1c: 0.2465
  Val    ->  Loss: 1.7768  |  Acc: 29.25%  |  F1m: 0.2636  |  F1c: 0.1859
  LR: 2.50e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [#######..................] 7/25
  Train [435/435] 100.0% | loss: 1.5681 | acc: 34.95% | 152s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7553 | acc: 33.15% | 11s / ETA 0s

--- Epoch 18/100 (163s | total: 0:48:30 | ETA: 3:41:00) ---
  Train  ->  Loss: 1.5681  |  Acc: 34.95%  |  F1m: 0.3461  |  F1c: 0.2661
  Val    ->  Loss: 1.7553  |  Acc: 33.15%  |  F1m: 0.2975  |  F1c: 0.2097
  LR: 2.50e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [########.................] 8/25
  Train [435/435] 100.0% | loss: 1.5402 | acc: 36.10% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7614 | acc: 30.92% | 11s / ETA 0s

--- Epoch 19/100 (161s | total: 0:51:11 | ETA: 3:38:15) ---
  Train  ->  Loss: 1.5402  |  Acc: 36.10%  |  F1m: 0.3559  |  F1c: 0.2761
  Val    ->  Loss: 1.7614  |  Acc: 30.92%  |  F1m: 0.2787  |  F1c: 0.1955
  LR: 2.50e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [#########................] 9/25
  Train [435/435] 100.0% | loss: 1.5474 | acc: 38.28% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7842 | acc: 57.66% | 11s / ETA 0s

--- Epoch 20/100 (160s | total: 0:53:52 | ETA: 3:35:28) ---
  Train  ->  Loss: 1.5474  |  Acc: 38.28%  |  F1m: 0.3801  |  F1c: 0.2973
  Val    ->  Loss: 1.7842  |  Acc: 57.66%  |  F1m: 0.3683  |  F1c: 0.2206
  LR: 2.50e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [##########...............] 10/25
  Train [435/435] 100.0% | loss: 1.5056 | acc: 40.93% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.7755 | acc: 13.65% | 11s / ETA 0s

--- Epoch 21/100 (163s | total: 0:56:34 | ETA: 3:32:50) ---
  Train  ->  Loss: 1.5056  |  Acc: 40.93%  |  F1m: 0.3925  |  F1c: 0.3027
  Val    ->  Loss: 1.7755  |  Acc: 13.65%  |  F1m: 0.1608  |  F1c: 0.1464
  LR: 2.50e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [###########..............] 11/25
  Train [435/435] 100.0% | loss: 1.4591 | acc: 43.64% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8213 | acc: 52.65% | 11s / ETA 0s

--- Epoch 22/100 (161s | total: 0:59:15 | ETA: 3:30:07) ---
  Train  ->  Loss: 1.4591  |  Acc: 43.64%  |  F1m: 0.4293  |  F1c: 0.3387
  Val    ->  Loss: 1.8213  |  Acc: 52.65%  |  F1m: 0.3874  |  F1c: 0.2615
  LR: 2.50e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [############.............] 12/25
  Train [435/435] 100.0% | loss: 1.4977 | acc: 43.64% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8033 | acc: 18.94% | 12s / ETA 0s

--- Epoch 23/100 (162s | total: 1:01:58 | ETA: 3:27:28) ---
  Train  ->  Loss: 1.4977  |  Acc: 43.64%  |  F1m: 0.4301  |  F1c: 0.3359
  Val    ->  Loss: 1.8033  |  Acc: 18.94%  |  F1m: 0.1769  |  F1c: 0.1367
  LR: 1.25e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [#############............] 13/25
  Train [435/435] 100.0% | loss: 1.4392 | acc: 46.86% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8506 | acc: 44.29% | 11s / ETA 0s

--- Epoch 24/100 (162s | total: 1:04:40 | ETA: 3:24:47) ---
  Train  ->  Loss: 1.4392  |  Acc: 46.86%  |  F1m: 0.4598  |  F1c: 0.3590
  Val    ->  Loss: 1.8506  |  Acc: 44.29%  |  F1m: 0.3274  |  F1c: 0.2123
  LR: 1.25e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [##############...........] 14/25
  Train [435/435] 100.0% | loss: 1.4440 | acc: 48.36% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8408 | acc: 37.05% | 11s / ETA 0s

--- Epoch 25/100 (161s | total: 1:07:21 | ETA: 3:22:05) ---
  Train  ->  Loss: 1.4440  |  Acc: 48.36%  |  F1m: 0.4687  |  F1c: 0.3715
  Val    ->  Loss: 1.8408  |  Acc: 37.05%  |  F1m: 0.2994  |  F1c: 0.2046
  LR: 1.25e-05  |  Best: epoch 10 (clin_f1=0.2736)  | Early stop: [###############..........] 15/25
  Train [435/435] 100.0% | loss: 1.4469 | acc: 49.22% | 149s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8415 | acc: 60.17% | 11s / ETA 0s

--- Epoch 26/100 (159s | total: 1:10:02 | ETA: 3:19:21) ---
  Train  ->  Loss: 1.4469  |  Acc: 49.22%  |  F1m: 0.4716  |  F1c: 0.3669
  Val    ->  Loss: 1.8415  |  Acc: 60.17%  |  F1m: 0.4328  |  F1c: 0.2988 << BEST
  LR: 1.25e-05  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [.........................] 0/25
  Train [435/435] 100.0% | loss: 1.4296 | acc: 50.03% | 149s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8762 | acc: 36.21% | 11s / ETA 0s

--- Epoch 27/100 (160s | total: 1:12:42 | ETA: 3:16:35) ---
  Train  ->  Loss: 1.4296  |  Acc: 50.03%  |  F1m: 0.4900  |  F1c: 0.3849
  Val    ->  Loss: 1.8762  |  Acc: 36.21%  |  F1m: 0.2910  |  F1c: 0.2028
  LR: 1.25e-05  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#........................] 1/25
  Train [435/435] 100.0% | loss: 1.4188 | acc: 51.81% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.9471 | acc: 16.16% | 11s / ETA 0s

--- Epoch 28/100 (161s | total: 1:15:23 | ETA: 3:13:52) ---
  Train  ->  Loss: 1.4188  |  Acc: 51.81%  |  F1m: 0.5019  |  F1c: 0.3941
  Val    ->  Loss: 1.9471  |  Acc: 16.16%  |  F1m: 0.1492  |  F1c: 0.1221
  LR: 1.25e-05  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [##.......................] 2/25
  Train [435/435] 100.0% | loss: 1.3985 | acc: 50.72% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.9007 | acc: 18.66% | 11s / ETA 0s

--- Epoch 29/100 (159s | total: 1:18:02 | ETA: 3:11:03) ---
  Train  ->  Loss: 1.3985  |  Acc: 50.72%  |  F1m: 0.5003  |  F1c: 0.3965
  Val    ->  Loss: 1.9007  |  Acc: 18.66%  |  F1m: 0.1773  |  F1c: 0.1362
  LR: 6.25e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [###......................] 3/25
  Train [435/435] 100.0% | loss: 1.4120 | acc: 52.85% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8614 | acc: 43.45% | 11s / ETA 0s

--- Epoch 30/100 (161s | total: 1:20:43 | ETA: 3:08:21) ---
  Train  ->  Loss: 1.4120  |  Acc: 52.85%  |  F1m: 0.5212  |  F1c: 0.4131
  Val    ->  Loss: 1.8614  |  Acc: 43.45%  |  F1m: 0.3219  |  F1c: 0.2098
  LR: 6.25e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [####.....................] 4/25
  Train [435/435] 100.0% | loss: 1.4054 | acc: 50.60% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.9462 | acc: 50.97% | 11s / ETA 0s

--- Epoch 31/100 (162s | total: 1:23:25 | ETA: 3:05:40) ---
  Train  ->  Loss: 1.4054  |  Acc: 50.60%  |  F1m: 0.4949  |  F1c: 0.3880
  Val    ->  Loss: 1.9462  |  Acc: 50.97%  |  F1m: 0.3415  |  F1c: 0.2186
  LR: 6.25e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#####....................] 5/25
  Train [435/435] 100.0% | loss: 1.3666 | acc: 51.81% | 147s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8937 | acc: 50.14% | 11s / ETA 0s

--- Epoch 32/100 (158s | total: 1:26:02 | ETA: 3:02:51) ---
  Train  ->  Loss: 1.3666  |  Acc: 51.81%  |  F1m: 0.5207  |  F1c: 0.4133
  Val    ->  Loss: 1.8937  |  Acc: 50.14%  |  F1m: 0.3517  |  F1c: 0.2297
  LR: 6.25e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [######...................] 6/25
  Train [435/435] 100.0% | loss: 1.3736 | acc: 53.37% | 149s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8625 | acc: 43.45% | 11s / ETA 0s

--- Epoch 33/100 (160s | total: 1:28:42 | ETA: 3:00:06) ---
  Train  ->  Loss: 1.3736  |  Acc: 53.37%  |  F1m: 0.5226  |  F1c: 0.4096
  Val    ->  Loss: 1.8625  |  Acc: 43.45%  |  F1m: 0.3417  |  F1c: 0.2313
  LR: 6.25e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#######..................] 7/25
  Train [435/435] 100.0% | loss: 1.3733 | acc: 53.66% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8782 | acc: 57.66% | 11s / ETA 0s

--- Epoch 34/100 (159s | total: 1:31:21 | ETA: 2:57:21) ---
  Train  ->  Loss: 1.3733  |  Acc: 53.66%  |  F1m: 0.5285  |  F1c: 0.4161
  Val    ->  Loss: 1.8782  |  Acc: 57.66%  |  F1m: 0.3934  |  F1c: 0.2683
  LR: 6.25e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [########.................] 8/25
  Train [435/435] 100.0% | loss: 1.3659 | acc: 53.83% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8815 | acc: 32.87% | 11s / ETA 0s

--- Epoch 35/100 (159s | total: 1:34:01 | ETA: 2:54:36) ---
  Train  ->  Loss: 1.3659  |  Acc: 53.83%  |  F1m: 0.5263  |  F1c: 0.4135
  Val    ->  Loss: 1.8815  |  Acc: 32.87%  |  F1m: 0.2597  |  F1c: 0.1778
  LR: 3.13e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#########................] 9/25
  Train [435/435] 100.0% | loss: 1.3473 | acc: 54.00% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8674 | acc: 46.52% | 11s / ETA 0s

--- Epoch 36/100 (161s | total: 1:36:42 | ETA: 2:51:55) ---
  Train  ->  Loss: 1.3473  |  Acc: 54.00%  |  F1m: 0.5280  |  F1c: 0.4176
  Val    ->  Loss: 1.8674  |  Acc: 46.52%  |  F1m: 0.3532  |  F1c: 0.2500
  LR: 3.13e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [##########...............] 10/25
  Train [435/435] 100.0% | loss: 1.3692 | acc: 55.09% | 149s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8600 | acc: 43.45% | 11s / ETA 0s

--- Epoch 37/100 (160s | total: 1:39:22 | ETA: 2:49:12) ---
  Train  ->  Loss: 1.3692  |  Acc: 55.09%  |  F1m: 0.5405  |  F1c: 0.4287
  Val    ->  Loss: 1.8600  |  Acc: 43.45%  |  F1m: 0.3371  |  F1c: 0.2362
  LR: 3.13e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [###########..............] 11/25
  Train [435/435] 100.0% | loss: 1.3110 | acc: 56.71% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8702 | acc: 44.57% | 11s / ETA 0s

--- Epoch 38/100 (159s | total: 1:42:01 | ETA: 2:46:28) ---
  Train  ->  Loss: 1.3110  |  Acc: 56.71%  |  F1m: 0.5572  |  F1c: 0.4428
  Val    ->  Loss: 1.8702  |  Acc: 44.57%  |  F1m: 0.3349  |  F1c: 0.2246
  LR: 3.13e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [############.............] 12/25
  Train [435/435] 100.0% | loss: 1.3429 | acc: 55.04% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8724 | acc: 42.34% | 11s / ETA 0s

--- Epoch 39/100 (160s | total: 1:44:42 | ETA: 2:43:45) ---
  Train  ->  Loss: 1.3429  |  Acc: 55.04%  |  F1m: 0.5437  |  F1c: 0.4325
  Val    ->  Loss: 1.8724  |  Acc: 42.34%  |  F1m: 0.3273  |  F1c: 0.2294
  LR: 3.13e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#############............] 13/25
  Train [435/435] 100.0% | loss: 1.3345 | acc: 55.38% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8969 | acc: 40.67% | 11s / ETA 0s

--- Epoch 40/100 (159s | total: 1:47:21 | ETA: 2:41:01) ---
  Train  ->  Loss: 1.3345  |  Acc: 55.38%  |  F1m: 0.5403  |  F1c: 0.4256
  Val    ->  Loss: 1.8969  |  Acc: 40.67%  |  F1m: 0.3131  |  F1c: 0.2111
  LR: 3.13e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [##############...........] 14/25
  Train [435/435] 100.0% | loss: 1.3259 | acc: 57.92% | 148s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8547 | acc: 35.10% | 11s / ETA 0s

--- Epoch 41/100 (159s | total: 1:49:59 | ETA: 2:38:17) ---
  Train  ->  Loss: 1.3259  |  Acc: 57.92%  |  F1m: 0.5637  |  F1c: 0.4460
  Val    ->  Loss: 1.8547  |  Acc: 35.10%  |  F1m: 0.2880  |  F1c: 0.1974
  LR: 1.56e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [###############..........] 15/25
  Train [435/435] 100.0% | loss: 1.3195 | acc: 55.04% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8677 | acc: 40.95% | 11s / ETA 0s

--- Epoch 42/100 (161s | total: 1:52:41 | ETA: 2:35:37) ---
  Train  ->  Loss: 1.3195  |  Acc: 55.04%  |  F1m: 0.5425  |  F1c: 0.4288
  Val    ->  Loss: 1.8677  |  Acc: 40.95%  |  F1m: 0.3268  |  F1c: 0.2284
  LR: 1.56e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [################.........] 16/25
  Train [435/435] 100.0% | loss: 1.3485 | acc: 56.25% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8789 | acc: 37.05% | 11s / ETA 0s

--- Epoch 43/100 (161s | total: 1:55:22 | ETA: 2:32:56) ---
  Train  ->  Loss: 1.3485  |  Acc: 56.25%  |  F1m: 0.5558  |  F1c: 0.4396
  Val    ->  Loss: 1.8789  |  Acc: 37.05%  |  F1m: 0.2978  |  F1c: 0.2065
  LR: 1.56e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#################........] 17/25
  Train [435/435] 100.0% | loss: 1.3403 | acc: 56.13% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8727 | acc: 35.93% | 11s / ETA 0s

--- Epoch 44/100 (162s | total: 1:58:04 | ETA: 2:30:16) ---
  Train  ->  Loss: 1.3403  |  Acc: 56.13%  |  F1m: 0.5498  |  F1c: 0.4349
  Val    ->  Loss: 1.8727  |  Acc: 35.93%  |  F1m: 0.2943  |  F1c: 0.2057
  LR: 1.56e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [##################.......] 18/25
  Train [435/435] 100.0% | loss: 1.3432 | acc: 55.27% | 151s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8763 | acc: 38.16% | 11s / ETA 0s

--- Epoch 45/100 (161s | total: 2:00:45 | ETA: 2:27:35) ---
  Train  ->  Loss: 1.3432  |  Acc: 55.27%  |  F1m: 0.5402  |  F1c: 0.4239
  Val    ->  Loss: 1.8763  |  Acc: 38.16%  |  F1m: 0.3065  |  F1c: 0.2137
  LR: 1.56e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [###################......] 19/25
  Train [435/435] 100.0% | loss: 1.3287 | acc: 55.61% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8614 | acc: 38.16% | 11s / ETA 0s

--- Epoch 46/100 (160s | total: 2:03:25 | ETA: 2:24:53) ---
  Train  ->  Loss: 1.3287  |  Acc: 55.61%  |  F1m: 0.5461  |  F1c: 0.4332
  Val    ->  Loss: 1.8614  |  Acc: 38.16%  |  F1m: 0.3036  |  F1c: 0.2098
  LR: 1.56e-06  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [####################.....] 20/25
  Train [435/435] 100.0% | loss: 1.2975 | acc: 56.76% | 150s / ETA 0s
  Val   [90/90] 100.0% | loss: 1.8705 | acc: 38.16% | 11s / ETA 0s

--- Epoch 47/100 (161s | total: 2:06:06 | ETA: 2:22:12) ---
  Train  ->  Loss: 1.2975  |  Acc: 56.76%  |  F1m: 0.5540  |  F1c: 0.4401
  Val    ->  Loss: 1.8705  |  Acc: 38.16%  |  F1m: 0.3073  |  F1c: 0.2190
  LR: 7.81e-07  |  Best: epoch 26 (clin_f1=0.2988)  | Early stop: [#####################....] 21/25
  Train [435/435] 100.0% | loss: 1.2973 | acc: 57.28% | 150s / ETA 0s
```

### `training_log.csv`

```csv
epoch,train_loss,train_acc,train_f1,train_clinical_f1,val_loss,val_acc,val_f1,val_clinical_f1,epoch_time_s,is_best
1,1.694499,0.128958,0.135578,0.122097,1.734746,0.061281,0.04087,0.062928,161.5,True
2,1.653486,0.128958,0.130499,0.125375,1.716662,0.061281,0.049626,0.069771,160.9,True
3,1.667659,0.104203,0.10856,0.113046,1.709275,0.119777,0.142124,0.137795,162.2,True
4,1.661903,0.127807,0.120306,0.111709,1.70899,0.130919,0.113156,0.098673,160.9,False
5,1.659167,0.153713,0.15524,0.134316,1.790992,0.052925,0.03351,0.060317,161.9,False
6,1.644141,0.202648,0.182401,0.146036,1.726611,0.48468,0.321434,0.197171,162.2,True
7,1.639425,0.185377,0.16866,0.135081,1.766765,0.116992,0.148857,0.158916,161.2,False
8,1.636305,0.20783,0.213089,0.172241,1.775521,0.172702,0.15968,0.140296,162.0,False
9,1.639737,0.21186,0.207575,0.163048,1.709241,0.208914,0.205542,0.157035,161.5,False
10,1.640558,0.225676,0.216401,0.167972,1.722833,0.576602,0.418087,0.273597,160.9,True
11,1.619521,0.264249,0.23732,0.174003,1.70696,0.155989,0.139386,0.113618,162.3,False
12,1.62044,0.266552,0.263582,0.200036,1.711952,0.259053,0.264729,0.199242,160.8,False
13,1.61714,0.255037,0.258771,0.199344,1.711495,0.381616,0.306424,0.205293,160.9,False
14,1.616696,0.304548,0.294335,0.222866,1.771834,0.089136,0.102529,0.112657,162.0,False
15,1.598217,0.259643,0.25712,0.198194,1.724656,0.144847,0.175153,0.153207,160.7,False
16,1.616971,0.313759,0.296719,0.222115,1.726578,0.389972,0.269223,0.166243,159.0,False
17,1.594474,0.326425,0.320664,0.246543,1.776835,0.292479,0.26356,0.185946,159.4,False
18,1.568099,0.349453,0.346113,0.266111,1.755303,0.331476,0.297501,0.209671,162.8,False
19,1.540223,0.360967,0.355895,0.276144,1.76139,0.309192,0.278704,0.195454,161.0,False
20,1.547448,0.382844,0.38006,0.2973,1.784153,0.576602,0.368323,0.220592,160.3,False
21,1.505566,0.409326,0.392479,0.302671,1.775479,0.13649,0.160835,0.146355,162.5,False
22,1.459138,0.436385,0.429324,0.33866,1.821344,0.526462,0.387353,0.261475,161.3,False
23,1.497697,0.436385,0.430113,0.335855,1.803291,0.189415,0.176889,0.136734,162.3,False
24,1.439244,0.468624,0.459769,0.359015,1.850612,0.442897,0.32738,0.212266,162.1,False
25,1.444022,0.483592,0.468744,0.371543,1.840829,0.370474,0.299368,0.204635,161.4,False
26,1.446875,0.492228,0.471565,0.366938,1.841542,0.601671,0.432817,0.298811,159.4,True
27,1.429566,0.500288,0.489989,0.384883,1.876184,0.362117,0.29095,0.202836,160.1,False
28,1.418847,0.518135,0.501928,0.394067,1.947094,0.16156,0.149247,0.122085,160.8,False
29,1.398455,0.507196,0.500311,0.39653,1.900737,0.18663,0.177308,0.136246,158.8,False
30,1.412042,0.528497,0.521167,0.413053,1.861431,0.43454,0.321924,0.209811,160.9,False
31,1.405381,0.506045,0.494906,0.387974,1.946226,0.509749,0.341454,0.218571,161.8,False
32,1.366558,0.518135,0.5207,0.413338,1.893688,0.501393,0.351709,0.229651,157.8,False
33,1.373601,0.533679,0.522565,0.409563,1.862541,0.43454,0.341678,0.23128,159.9,False
34,1.373266,0.536557,0.528469,0.416079,1.878212,0.576602,0.393387,0.268279,159.1,False
35,1.365902,0.538284,0.526256,0.413458,1.881514,0.328691,0.259731,0.177807,159.4,False
36,1.347317,0.540012,0.527954,0.417615,1.867401,0.465181,0.353164,0.250004,161.3,False
37,1.369217,0.55095,0.540464,0.428748,1.860026,0.43454,0.337117,0.236184,160.2,False
38,1.310962,0.56707,0.557166,0.442801,1.870234,0.445682,0.334934,0.224618,159.2,False
39,1.342944,0.550374,0.54366,0.43251,1.872405,0.423398,0.32731,0.229358,160.2,False
40,1.334507,0.553828,0.540334,0.42558,1.896916,0.406685,0.313072,0.211059,159.0,False
41,1.325856,0.579159,0.563706,0.446003,1.854674,0.350975,0.288044,0.19743,158.8,False
42,1.319504,0.550374,0.542466,0.428842,1.867691,0.409471,0.326826,0.228418,161.4,False
43,1.348541,0.562464,0.555832,0.439602,1.878868,0.370474,0.29778,0.206506,161.2,False
44,1.340316,0.561313,0.549772,0.434903,1.872685,0.359331,0.294299,0.205664,161.5,False
45,1.343191,0.552677,0.54024,0.423881,1.876322,0.381616,0.306502,0.213671,161.3,False
46,1.328704,0.556131,0.546102,0.433186,1.861389,0.381616,0.303563,0.209751,160.5,False
47,1.297511,0.567645,0.554025,0.440113,1.870522,0.381616,0.307322,0.219043,160.9,False
```

### `transforms_config.txt`

```txt
Pipeline de transforms para split: 'train'
=======================================================

--- Preprocesamiento (deterministico, todos los splits) ---
  1. LoadImaged           keys=['image'], image_only=True
  2. EnsureChannelFirstd  keys=['image']
  3. Orientationd         keys=['image'], axcodes='RAS'
  4. ScaleIntensityRangePercentilesd
       keys=['image'], lower=1, upper=99
       b_min=0.0, b_max=1.0, clip=True
  5. Resized              keys=['image'], spatial_size=(96, 96, 96)

--- Data Augmentation (estocastico, solo train) ---
  RandFlipd            keys=['image'], prob=0.5, spatial_axis=0
  RandRotated          keys=['image'], range_xyz=0.2, prob=0.3
  RandGaussianNoised   keys=['image'], prob=0.3, mean=0.0, std=0.05
  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.3

Nota: las transforms Rand* se aplican on-the-fly en cada epoch.
```

### Otros archivos (no incluidos en texto)

- `best_model.pth` (168340.9 KB)
- `nohup.out` (115.2 KB)
