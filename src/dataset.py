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
    Compose, EnsureChannelFirstd, LoadImaged, MapTransform, Orientationd,
    RandFlipd, RandGaussianNoised, RandShiftIntensityd, RandAdjustContrastd,
    RandAffined, RandCoarseDropoutd, Resized, ScaleIntensityRangePercentilesd,
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
        8. RandGaussianNoised — Ruido gaussiano suave (prob=0.1, std=0.05).
        9. RandShiftIntensityd— Variación de intensidad (offsets=0.1, prob=0.1).

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
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
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
         # 1. Anatomical spatial variations (Flip LR only, no up/down)
         RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),

         # 2. Scanner positioning & head size variations (Combines Rotate, Zoom, Translate)
         RandAffined(
             keys=["image"], 
             prob=0.6, 
             rotate_range=(0.2, 0.2, 0.2), 
             scale_range=(0.1, 0.1, 0.1), # +/- 10% Zoom
             translate_range=(4, 4, 4),   # Slight shifting
             padding_mode="zeros"
         ),

         # 3. Scanner noise and magnetic field variations
         RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
         RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
         RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),

         # 4. Regional occlusion (Forces network to look at multiple brain areas)
         RandCoarseDropoutd(
             keys=["image"], holes=1, spatial_size=(16, 16, 16), fill_value=0, prob=0.3
         ),
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
            f"  RandGaussianNoised   keys=['image'], prob=0.1, mean=0.0, std=0.05",
            f"  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.1",
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
