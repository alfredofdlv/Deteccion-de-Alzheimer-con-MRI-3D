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

