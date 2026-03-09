"""
dataset.py — Dataset MONAI y DataLoaders para MRI 3D (OASIS-1).

Proporciona el pipeline de transforms y los DataLoaders listos para
alimentar el modelo con tensores de forma (B, 1, 96, 96, 96).

Pipeline de transforms:
    LoadImaged -> EnsureChannelFirstd -> Orientationd(RAS)
    -> ScaleIntensityRangePercentilesd -> Resized(96, 96, 96)

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

    Pasos:
        1. LoadImaged        — Carga el par .img/.hdr (formato ANALYZE).
        2. EnsureChannelFirstd — Añade dimensión de canal: (D,H,W) -> (1,D,H,W).
        3. Orientationd       — Reorienta a RAS (Right-Anterior-Superior).
        4. ScaleIntensityRangePercentilesd — Normaliza intensidad al rango [0, 1]
           usando percentiles 1-99 para robustez ante outliers.
        5. Resized            — Redimensiona a IMAGE_SIZE (96, 96, 96).

    Args:
        split: Nombre del split ('train', 'val', 'test').
              En Sprint 2 todas las transforms son iguales.
              La data augmentation para 'train' se añadirá en Sprint 4.

    Returns:
        Compose con el pipeline de transforms.
    """
    return Compose([
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
    ])


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
