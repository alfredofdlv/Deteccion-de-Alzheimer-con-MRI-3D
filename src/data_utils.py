"""
data_utils.py — Utilidades para carga de datos y gestión de rutas.

Funciones auxiliares para:
- Leer CSVs de particiones (train/val/test).
- Construir rutas absolutas a los archivos .nii.gz.
- Verificar la integridad de los datos descargados.

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
        DataFrame con columnas esperadas: ['subject_id', 'filepath', 'label'].

    Raises:
        FileNotFoundError: Si el CSV no existe.
    """
    csv_path = cfg.DATA_SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de split: {csv_path}\n"
            f"Asegúrate de haber generado las particiones primero."
        )
    df = pd.read_csv(csv_path)
    return df


def build_nifti_path(subject_id: str) -> Path:
    """
    Construye la ruta absoluta a un archivo .nii.gz dado un subject_id de OASIS.

    Args:
        subject_id: Identificador del sujeto OASIS (ej. 'OAS1_0001_MR1').

    Returns:
        Path al archivo NIfTI.
    """
    # Patrón típico de OASIS-1: data/raw/{subject_id}/{subject_id}_mpr-1_anon.nii.gz
    # Ajustar según la estructura real del dataset descargado.
    nifti_path = cfg.DATA_RAW_DIR / subject_id
    return nifti_path


def list_available_subjects() -> List[str]:
    """
    Lista todos los sujetos disponibles en data/raw/.

    Returns:
        Lista de nombres de carpetas/archivos en data/raw/.
    """
    if not cfg.DATA_RAW_DIR.exists():
        return []
    return sorted([p.name for p in cfg.DATA_RAW_DIR.iterdir()])


def verify_data_integrity() -> Tuple[int, List[str]]:
    """
    Verifica cuántos archivos .nii.gz hay disponibles en data/raw/.

    Returns:
        Tupla con (número de archivos encontrados, lista de rutas).
    """
    nifti_files = list(cfg.DATA_RAW_DIR.rglob("*.nii.gz"))
    file_paths = [str(f) for f in sorted(nifti_files)]
    return len(nifti_files), file_paths

