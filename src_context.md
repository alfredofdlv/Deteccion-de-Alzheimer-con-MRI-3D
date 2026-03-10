# Contexto de la carpeta `src`

Este archivo contiene todo el código fuente de los scripts ubicados en la carpeta `src/` del proyecto.

## `src/__init__.py`
```python
# src/__init__.py
# Módulo principal del proyecto TFG: Detección Temprana de Alzheimer con 3D MRI
```

## `src/config.py`
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
    NUM_WORKERS: int = 2         # DataLoader workers (ajustar según CPU)
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4         # Regularización L2 (penalización sobre norma de pesos)
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 30  # Epochs sin mejora en val_loss antes de parar

    # ========================
    # Clases del dataset OASIS-1
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
            f"  NUM_CLASSES         = {self.NUM_CLASSES}\n"
            f"  OASIS_RAW_DIR       = {self.OASIS_RAW_DIR}\n"
            f"  PROCESSED_IMAGES_DIR= {self.PROCESSED_IMAGES_DIR}\n"
            f"  OUTPUTS_DIR         = {self.OUTPUTS_DIR}\n"
            f")"
        )


# Instancia global — importar directamente:  from src.config import cfg
cfg = ProjectConfig()
```

## `src/data_prepare.py`
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

    # Nota: la aserción de exactamente 416 es orientativa.
    # Si faltan discos, el script informa pero no falla fatalmente.
    if n_pairs == 416:
        print(f"\n  ✅ VALIDACIÓN OK: {n_pairs} pares (416 esperados).")
    else:
        print(
            f"\n  ⚠️  Se encontraron {n_pairs} pares (416 esperados)."
            f"\n      Puede que falten discos por descargar/extraer."
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
        print("\n  ❌ DATA LEAKAGE DETECTADO:")
        if leak_train_val:
            print(f"    Train ∩ Val:  {leak_train_val}")
        if leak_train_test:
            print(f"    Train ∩ Test: {leak_train_test}")
        if leak_val_test:
            print(f"    Val ∩ Test:   {leak_val_test}")
        raise RuntimeError("Data leakage detectado entre splits.")
    else:
        print("\n  ✅ Anti-leakage check: OK (0 sujetos compartidos entre splits).")


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

## `src/data_utils.py`
```python
"""
data_utils.py — Utilidades para carga de datos y gestión de rutas.

Funciones auxiliares para:
- Leer CSVs de particiones (train/val/test).
- Construir rutas absolutas a los archivos ANALYZE (.img/.hdr).
- Verificar la integridad de los datos procesados.

Uso:
    from src.data_utils import load_split, verify_data_integrity
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config import cfg


def load_split(split_name: str) -> pd.DataFrame:
    """
    Carga un CSV de partición desde data/splits/.

    Args:
        split_name: Nombre del split sin extensión (ej. 'train', 'val', 'test').

    Returns:
        DataFrame con columnas esperadas: ['subject_id', 'image_path', 'label'].

    Raises:
        FileNotFoundError: Si el CSV no existe.
    """
    csv_path = cfg.DATA_SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de split: {csv_path}\n"
            f"Asegúrate de haber ejecutado: python -m src.data_prepare"
        )
    df = pd.read_csv(csv_path)
    return df


def build_image_path(subject_id: str) -> Path:
    """
    Construye la ruta absoluta al archivo .img procesado de un sujeto OASIS.

    Args:
        subject_id: Identificador del sujeto OASIS (ej. 'OAS1_0001_MR1').

    Returns:
        Path al archivo ANALYZE (.img) en data/processed/images/.
    """
    return cfg.PROCESSED_IMAGES_DIR / f"{subject_id}.img"


def list_available_subjects() -> List[str]:
    """
    Lista todos los sujetos disponibles en data/processed/images/.

    Returns:
        Lista de subject_ids (sin extensión) de los .img procesados.
    """
    if not cfg.PROCESSED_IMAGES_DIR.exists():
        return []
    return sorted([p.stem for p in cfg.PROCESSED_IMAGES_DIR.glob("*.img")])


def verify_data_integrity() -> Tuple[int, List[str]]:
    """
    Verifica cuántos pares .img/.hdr válidos hay en data/processed/images/.

    Returns:
        Tupla con (número de pares completos, lista de subject_ids con par).
    """
    if not cfg.PROCESSED_IMAGES_DIR.exists():
        return 0, []

    img_files = {p.stem for p in cfg.PROCESSED_IMAGES_DIR.glob("*.img")}
    hdr_files = {p.stem for p in cfg.PROCESSED_IMAGES_DIR.glob("*.hdr")}

    # Solo contar pares completos (.img + .hdr)
    complete_pairs = sorted(img_files & hdr_files)
    return len(complete_pairs), complete_pairs

```

## `src/dataset.py`
```python
"""
dataset.py — Dataset MONAI y DataLoaders para MRI 3D (OASIS-1).

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

from typing import List

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
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


def describe_transforms(split: str = "train") -> str:
    """
    Devuelve una descripción legible del pipeline de transforms para un split.

    Útil para guardar en cada run qué preprocesamiento y augmentation se aplicó.
    """
    lines = [
        f"Pipeline de transforms para split: '{split}'",
        "=" * 55,
        "",
        "--- Preprocesamiento (determinístico, todos los splits) ---",
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
            "--- Data Augmentation (estocástico, solo train) ---",
            f"  6. RandFlipd            keys=['image'], prob=0.5, spatial_axis=0",
            f"  7. RandRotated          keys=['image'], range_xyz=0.2, prob=0.3",
            f"  8. RandGaussianNoised   keys=['image'], prob=0.3, mean=0.0, std=0.05",
            f"  9. RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.3",
            "",
            "Nota: las transforms Rand* se aplican on-the-fly en cada epoch.",
            "Cada imagen puede recibir combinaciones distintas cada vez.",
        ]
    else:
        lines += [
            "",
            "--- Sin Data Augmentation ---",
            f"  (split='{split}': solo preprocesamiento determinístico)",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data dicts
# ---------------------------------------------------------------------------

def _build_data_dicts(split: str) -> List[dict]:
    """
    Convierte un CSV de split en la lista de dicts que MONAI espera.

    Cada dict tiene la forma:
        {"image": "/ruta/al/OAS1_XXXX_MR1.img", "label": 0}

    Args:
        split: Nombre del split ('train', 'val', 'test').

    Returns:
        Lista de diccionarios con claves 'image' y 'label'.
    """
    df = load_split(split)
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

    data_dicts = _build_data_dicts(split)
    transforms = get_transforms(split)

    if use_cache:
        dataset = CacheDataset(
            data=data_dicts,
            transform=transforms,
            num_workers=num_workers,
        )
    else:
        dataset = Dataset(data=data_dicts, transform=transforms)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
```

## `src/evaluate.py`
```python
"""
evaluate.py — Evaluación del modelo sobre el test set con métricas detalladas.

Genera en outputs/<run_name>/:
    - classification_report.txt  — precision, recall, F1 por clase
    - confusion_matrix.png       — heatmap de la matriz de confusión

Uso:
    python -m src.evaluate --run full_100ep
"""

from __future__ import annotations

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
from src.model import Simple3DCNN


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

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

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

def evaluate_model(run_name: str, split: str = "test") -> None:
    """
    Evalúa el mejor modelo de un run sobre un split y genera reportes.

    Args:
        run_name: Nombre de la carpeta en outputs/ que contiene best_model.pth.
        split: Split a evaluar ('test' por defecto, también acepta 'val').
    """
    run_dir = cfg.OUTPUTS_DIR / run_name
    model_path = run_dir / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}\n"
            f"Asegúrate de haber entrenado con --run {run_name}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Cargar modelo
    model = Simple3DCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"[INFO] Modelo cargado desde epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_accuracy']:.2%})"
    )

    # Inferencia
    loader = get_dataloader(split, shuffle=False, num_workers=0)
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
        description="Evaluar Simple3DCNN sobre test/val set"
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Nombre de la carpeta en outputs/ (ej. full_100ep)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"],
        help="Split a evaluar (default: test)",
    )
    args = parser.parse_args()

    evaluate_model(run_name=args.run, split=args.split)
```

## `src/model.py`
```python
"""
model.py — Arquitectura Simple3DCNN para clasificación de MRI 3D.

4 bloques convolucionales seguidos de un clasificador global.

Arquitectura:
    Input (B, 1, 96, 96, 96)
    -> [Conv3d -> BN3d -> ReLU -> MaxPool3d] x4
    -> AdaptiveAvgPool3d(1) -> Dropout -> Linear(256, 3)

Uso:
    from src.model import Simple3DCNN
    model = Simple3DCNN()
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificación rápida:
    python -m src.model
"""

import torch
import torch.nn as nn

from src.config import cfg


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
    CNN 3D vanilla para clasificación de volúmenes cerebrales.

    Progresión de canales: 1 -> 32 -> 64 -> 128 -> 256.
    Cada bloque reduce las dimensiones espaciales a la mitad con MaxPool3d(2).
    AdaptiveAvgPool3d colapsa las dimensiones espaciales a 1x1x1,
    haciendo la red agnóstica al tamaño de entrada.

    Args:
        in_channels: Canales de entrada (1 para MRI).
        num_classes: Número de clases de salida (3: CN, MCI, AD).
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
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DCNN().to(device)

    dummy = torch.randn(1, 1, 96, 96, 96, device=device)
    out = model(dummy)

    print(f"Device:       {device}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output:       {out.detach().cpu()}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales:      {n_params:,}")
    print(f"Parámetros entrenables:  {n_train:,}")

    assert out.shape == (1, cfg.NUM_CLASSES), (
        f"Shape incorrecto: {out.shape}, esperado (1, {cfg.NUM_CLASSES})"
    )
    print("\n=== CHECKPOINT 3.1 PASSED ===")
```

## `src/train.py`
```python
"""
train.py — Training loop para Simple3DCNN sobre OASIS-1.

Genera automáticamente en outputs/<run_name>/:
    - training_log.csv       — métricas por epoch
    - curves_loss.png        — gráfica de loss (train vs val)
    - curves_accuracy.png    — gráfica de accuracy (train vs val)
    - best_model.pth         — pesos del mejor modelo (menor val_loss)
    - training_summary.txt   — resumen legible del entrenamiento

Modos de ejecución:
    python -m src.train --overfit                   # sanity check
    python -m src.train --epochs 2 --run test_2ep   # prueba rápida
    python -m src.train --epochs 100 --run full     # entrenamiento largo
    python -m src.train --patience 15 --run exp1    # patience custom
"""

from __future__ import annotations

import csv
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Silenciar warnings repetitivos de ITK/C++ sobre archivos Analyze
import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.config import cfg
from src.data_utils import load_split
from src.dataset import describe_transforms, get_dataloader
from src.model import Simple3DCNN


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Para el entrenamiento si val_loss no mejora en `patience` epochs consecutivos."""

    def __init__(self, patience: int = cfg.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Retorna True si se debe parar el entrenamiento."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights() -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Formula: weight_i = N_total / (N_classes * N_i)
    """
    df = load_split("train")
    counts = df["label"].value_counts().sort_index()
    n_total = len(df)
    n_classes = cfg.NUM_CLASSES

    weights = []
    for c in range(n_classes):
        n_c = counts.get(c, 1)
        weights.append(n_total / (n_classes * n_c))

    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights: {w.tolist()}")
    return w


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Ejecuta un epoch de entrenamiento. Retorna dict con loss y accuracy medias."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
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

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evalúa el modelo sin gradientes. Retorna dict con loss y accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def _save_plots(history: list[dict], run_dir: Path) -> None:
    """Genera y guarda gráficas de loss y accuracy."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] * 100 for r in history]
    val_acc = [r["val_acc"] * 100 for r in history]

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "o-", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "s-", label="Val Loss", linewidth=2, markersize=4)
    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
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


def _save_summary(history: list[dict], run_dir: Path, device: torch.device,
                  n_params: int, elapsed_total: float, class_weights: list,
                  early_stopped: bool = False, patience: int = 0) -> None:
    """Genera un archivo de resumen legible."""
    best = min(history, key=lambda r: r["val_loss"])
    last = history[-1]

    stop_reason = f"Early stopping (patience={patience})" if early_stopped else "Completado"

    lines = [
        "=" * 60,
        "TRAINING SUMMARY",
        "=" * 60,
        f"Fecha:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device:             {device}",
        f"Parametros:         {n_params:,}",
        f"Class weights:      {class_weights}",
        f"Learning rate:      {cfg.LEARNING_RATE}",
        f"Weight decay (L2):  {cfg.WEIGHT_DECAY}",
        f"Batch size:         {cfg.BATCH_SIZE}",
        f"Image size:         {cfg.IMAGE_SIZE}",
        f"Epochs:             {len(history)}",
        f"Finalizacion:       {stop_reason}",
        f"Tiempo total:       {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)",
        f"Tiempo por epoch:   {elapsed_total/len(history):.1f}s",
        "",
        "--- Mejor Epoch ---",
        f"  Epoch:            {best['epoch']}",
        f"  Train Loss:       {best['train_loss']:.4f}",
        f"  Train Acc:        {best['train_acc']:.2%}",
        f"  Val Loss:         {best['val_loss']:.4f}",
        f"  Val Acc:          {best['val_acc']:.2%}",
        "",
        "--- Ultimo Epoch ---",
        f"  Epoch:            {last['epoch']}",
        f"  Train Loss:       {last['train_loss']:.4f}",
        f"  Train Acc:        {last['train_acc']:.2%}",
        f"  Val Loss:         {last['val_loss']:.4f}",
        f"  Val Acc:          {last['val_acc']:.2%}",
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
) -> None:
    """
    Función principal de entrenamiento.

    Args:
        num_epochs: Número máximo de epochs.
        overfit_one_batch: Si True, entrena solo con 4 imágenes durante
                          100 epochs (sanity check de convergencia).
        run_name: Nombre de la carpeta dentro de outputs/ para esta ejecución.
        Si None, se genera uno automático con timestamp.
        patience: Epochs sin mejora en val_loss antes de early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    torch.manual_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    model = Simple3DCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parametros entrenables: {n_params:,}")

    class_weights = compute_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )

    # ── Overfit-one-batch mode ───────────────────────────────────────────
    if overfit_one_batch:
        print("\n" + "=" * 60)
        print("MODO OVERFIT-ONE-BATCH (sanity check)")
        print("=" * 60)

        loader = get_dataloader("train", batch_size=4, num_workers=0, shuffle=False)
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
            print("  === CHECKPOINT 3.2 PASSED ===")
        else:
            print("  [WARN] No convergio completamente. Revisar el modelo.")
        return

    # ── Entrenamiento completo ───────────────────────────────────────────
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience)

    # Guardar ficha de transforms usadas en este run
    transforms_desc = describe_transforms("train")
    (run_dir / "transforms_config.txt").write_text(
        transforms_desc + "\n", encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO — max {num_epochs} epochs (early stopping: patience={patience})")
    print(f"Resultados en: {run_dir}")
    print("=" * 60)

    train_loader = get_dataloader("train", num_workers=0)
    val_loader = get_dataloader("val", num_workers=0)
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # CSV log
    csv_path = run_dir / "training_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s", "is_best"],
    )
    csv_writer.writeheader()

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        is_best = val_metrics["loss"] < best_val_loss

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "epoch_time_s": round(elapsed, 1),
            "is_best": is_best,
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_accuracy": val_metrics["accuracy"],
            }, run_dir / "best_model.pth")

        should_stop = early_stopper.step(val_metrics["loss"])

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

        print(
            f"\n--- Epoch {epoch}/{num_epochs} "
            f"({elapsed:.0f}s | total: {elapsed_str} | ETA: {eta_str}) ---\n"
            f"  Train  ->  Loss: {train_metrics['loss']:.4f}  |  Acc: {train_metrics['accuracy']:.2%}\n"
            f"  Val    ->  Loss: {val_metrics['loss']:.4f}  |  Acc: {val_metrics['accuracy']:.2%}{best_marker}\n"
            f"  Best: epoch {best_epoch} (val_loss={best_val_loss:.4f})  "
            f"| Early stop: {es_bar} {es_counter}/{patience}"
        )

        if should_stop:
            print(
                f"\n{'=' * 60}\n"
                f"[EARLY STOPPING] Val loss no mejoro en {patience} epochs.\n"
                f"Mejor epoch: {best_epoch} (val_loss={best_val_loss:.4f})\n"
                f"{'=' * 60}"
            )
            break

    csv_file.close()
    elapsed_total = time.time() - t_start

    # Generar graficas y resumen
    _save_plots(history, run_dir)
    _save_summary(
        history, run_dir, device, n_params, elapsed_total,
        class_weights.cpu().tolist(),
        early_stopped=early_stopper.triggered,
        patience=patience,
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

    parser = argparse.ArgumentParser(description="Entrenar Simple3DCNN sobre OASIS-1")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help=f"Numero de epochs (default: {cfg.NUM_EPOCHS})")
    parser.add_argument("--overfit", action="store_true",
                        help="Modo overfit-one-batch (sanity check)")
    parser.add_argument("--run", type=str, default=None,
                        help="Nombre de la carpeta de resultados (default: run_TIMESTAMP)")
    parser.add_argument("--patience", type=int, default=cfg.EARLY_STOPPING_PATIENCE,
                        help=f"Early stopping patience (default: {cfg.EARLY_STOPPING_PATIENCE})")
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        overfit_one_batch=args.overfit,
        run_name=args.run,
        patience=args.patience,
    )
```
