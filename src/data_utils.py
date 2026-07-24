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


def load_split(
    split_name: str,
    dataset: str = "oasis1",
    variant: str | None = None,
) -> pd.DataFrame:
    """
    Carga un CSV de particion desde data/splits/.

    Busca primero la version preprocesada (_pt.csv o _pt_mni.csv) y, si no existe,
    usa la original. Esto permite usar tensores .pt automaticamente
    cuando estan disponibles.

    Args:
        split_name: Nombre del split sin extension (ej. 'train', 'val', 'test').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        variant: Variante de preprocesado ('cropped' o 'mni'). None = cropped.

    Returns:
        DataFrame con columnas esperadas: ['subject_id', 'image_path', 'label'].

    Raises:
        FileNotFoundError: Si el CSV no existe.
    """
    variant = variant or cfg.PREPROCESS_VARIANT_DEFAULT

    if split_name == "all":
        frames: list[pd.DataFrame] = []
        for part in ("train", "val", "test"):
            try:
                frames.append(
                    load_split(part, dataset=dataset, variant=variant),
                )
            except FileNotFoundError:
                continue
        if not frames:
            raise FileNotFoundError(
                f"No se encontraron splits train/val/test para dataset={dataset!r}"
            )
        combined = pd.concat(frames, ignore_index=True)
        if "subject_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["subject_id"], keep="first")
        elif "image_path" in combined.columns:
            combined = combined.drop_duplicates(subset=["image_path"], keep="first")
        return combined

    pt_path = cfg.DATA_SPLITS_DIR / cfg.split_csv_name(dataset, split_name, variant)

    if dataset == "oasis1":
        csv_path = cfg.DATA_SPLITS_DIR / f"{split_name}.csv"
    else:
        csv_path = cfg.DATA_SPLITS_DIR / f"{dataset}_{split_name}.csv"

    if pt_path.exists():
        return pd.read_csv(pt_path)

    if variant != "cropped":
        raise FileNotFoundError(
            f"No se encontro el split preprocesado para variant={variant!r}: {pt_path}\n"
            f"Genera los .pt con: python scripts/preprocess_to_pt.py "
            f"--dataset {dataset} --variant {variant}"
        )

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
