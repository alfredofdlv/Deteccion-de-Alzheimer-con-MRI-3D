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

import pandas as pd
import torch
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, MapTransform, Orientationd,
    RandFlipd, RandGaussianNoised, RandShiftIntensityd, RandAdjustContrastd,
    RandAffined, RandCoarseDropoutd, Resized, ScaleIntensityRangePercentilesd,
    CropForegroundd, SpatialCropd,
)

from src.config import cfg
from src.data_utils import load_split
from src.label_scheme import get_scheme, remap_label


def default_variant_for_dataset(dataset: str) -> str:
    """Variante de preprocesado por defecto según el identificador de dataset."""
    if dataset in ("oasis3_adni", "oasis3_adni_baseline"):
        return "mni"
    return cfg.PREPROCESS_VARIANT_DEFAULT


def _clinical_vector_from_row(row, use_etiv: bool = False) -> torch.Tensor:
    """Covariables normalizadas [age, sex, educ, apoe] (+ eTIV opcional)."""
    age = float(row.get("age_at_visit", 75.0)) / 100.0
    sex = 1.0 if str(row.get("GENDER", "F")).strip().upper() == "F" else 0.0
    educ_val = row.get("EDUC", 12.0)
    educ = float(educ_val) / 20.0 if pd.notna(educ_val) else 12.0 / 20.0
    apoe_val = row.get("APOE_e4", 0.0)
    apoe = float(apoe_val) if pd.notna(apoe_val) else 0.0
    feats = [age, sex, educ, apoe]
    if use_etiv:
        etiv_val = row.get("eTIV_norm")
        if pd.isna(etiv_val):
            raise ValueError(
                "use_etiv=True pero falta eTIV_norm en el split CSV. "
                "Ejecuta: python scripts/enrich_splits_etiv.py"
            )
        feats.append(float(etiv_val))
    return torch.tensor(feats, dtype=torch.float32)

AUG_MODES = ("full", "light", "none")


def _ensure_single_channel(tensor: torch.Tensor) -> torch.Tensor:
    """NIfTI 4D (p. ej. multi-volumen ADNI) puede dejar C>1; el modelo espera (1, D, H, W)."""
    if tensor.shape[0] > 1:
        tensor = tensor[0:1]
    return tensor


class LoadPTd(MapTransform):
    """Carga un tensor .pt preprocesado (reemplaza a LoadImaged para archivos .pt)."""

    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # weights_only=False necesario: .pt contienen MetaTensor de MONAI
            tensor = torch.load(d[key], weights_only=False, map_location="cpu")
            d[key] = _ensure_single_channel(tensor)
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
        Resized(keys=["image"], spatial_size=cfg.IMAGE_SIZE)
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
    Pipeline para tensores .pt ya preprocesados (variante cropped).

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


def get_transforms_pt_light(split: str = "train") -> Compose:
    """
    Augmentación suave (opción B): flip LR + variaciones de intensidad leves.

    Sin RandAffined ni RandCoarseDropout — adecuada para cohortes pequeñas (ADNI).
    """
    transforms: list = [LoadPTd(keys=["image"])]

    if split == "train":
        transforms.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.8, 1.2)),
        ])

    return Compose(transforms)


def get_transforms_pt_mni(split: str = "train") -> Compose:
    """
    Pipeline para tensores .pt preprocesados con registro MNI152.

    Sin RandAffined agresivo: el registro afín ya alineó pose/escala global.
    Mantiene augmentations de intensidad y dropout regional.
    """
    transforms: list = [LoadPTd(keys=["image"])]

    if split == "train":
        transforms.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),
            RandCoarseDropoutd(
                keys=["image"], holes=1, spatial_size=(16, 16, 16), fill_value=0, prob=0.3,
            ),
        ])

    return Compose(transforms)


def _spatial_roi_transforms(
    roi_mode: str | None = None,
    spatial_size: tuple[int, int, int] | None = None,
) -> list:
    """Crop MTL o resize online (p. ej. 96³→128³ upsample) tras LoadPTd."""
    if roi_mode == "mtl":
        return [
            SpatialCropd(
                keys=["image"],
                roi_start=cfg.MTL_ROI_START,
                roi_end=cfg.MTL_ROI_END,
            ),
            Resized(keys=["image"], spatial_size=cfg.MTL_OUTPUT_SIZE),
        ]
    if spatial_size is not None and spatial_size != cfg.IMAGE_SIZE:
        return [Resized(keys=["image"], spatial_size=spatial_size)]
    return []


def _resolve_pt_transforms(
    split: str,
    variant: str,
    aug_mode: str = "full",
    roi_mode: str | None = None,
    spatial_size: tuple[int, int, int] | None = None,
) -> Compose:
    """Selecciona pipeline .pt según variante de preprocesado y modo de augmentation."""
    if aug_mode not in AUG_MODES:
        raise ValueError(f"aug_mode debe ser uno de {AUG_MODES}, recibido: {aug_mode!r}")

    spatial = _spatial_roi_transforms(roi_mode, spatial_size)

    if aug_mode == "none":
        return Compose([LoadPTd(keys=["image"]), *spatial])

    if variant in cfg.SPATIAL_PREPROCESS_VARIANTS:
        if aug_mode == "light":
            transforms: list = [LoadPTd(keys=["image"]), *spatial]
            if split == "train":
                transforms.extend([
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
                    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
                    RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.8, 1.2)),
                ])
            return Compose(transforms)
        base = get_transforms_pt_mni(split)
        if not spatial:
            return base
        return Compose([LoadPTd(keys=["image"]), *spatial, *base.transforms[1:]])

    if aug_mode == "light":
        base = get_transforms_pt_light(split)
        if not spatial:
            return base
        return Compose([LoadPTd(keys=["image"]), *spatial, *base.transforms[1:]])
    base = get_transforms_pt(split)
    if not spatial:
        return base
    return Compose([LoadPTd(keys=["image"]), *spatial, *base.transforms[1:]])


def describe_transforms(
    split: str = "train",
    is_pt: bool = False,
    variant: str | None = None,
    aug_mode: str = "full",
    roi_mode: str | None = None,
    spatial_size: tuple[int, int, int] | None = None,
) -> str:
    """
    Devuelve una descripcion legible del pipeline de transforms para un split.

    Args:
        split: Nombre del split ('train', 'val', 'test').
        is_pt: Si True, describe el flujo para tensores .pt preprocesados.
        variant: Variante de preprocesado ('cropped' o 'mni').
    """
    variant = variant or cfg.PREPROCESS_VARIANT_DEFAULT
    lines = [
        f"Pipeline de transforms para split: '{split}'",
        f"Preprocess variant: {variant}",
        f"Augmentation mode: {aug_mode}",
    ]
    if roi_mode:
        lines.append(f"ROI mode: {roi_mode}")
    if spatial_size and spatial_size != cfg.IMAGE_SIZE:
        lines.append(f"Spatial size (online): {spatial_size}")
    lines.append("=" * 55)

    if is_pt:
        offline_desc = (
            "    LoadImage, EnsureChannelFirst, Orientation(RAS),"
            f" ScaleIntensityRangePercentiles(1-99), Resize{cfg.IMAGE_SIZE}"
        )
        if variant in cfg.SPATIAL_PREPROCESS_VARIANTS:
            offline_desc = (
                "    SkullStrip, Affine registration to MNI152,"
                f" ScaleIntensityRangePercentiles(1-99), Resize{cfg.spatial_size_for_variant(variant)}"
            )
        lines += [
            "",
            "--- Formato: tensores .pt preprocesados offline ---",
            f"  1. LoadPTd              keys=['image'] (torch.load)",
        ]
        if roi_mode == "mtl":
            lines += [
                f"  2. SpatialCropd         MTL bbox {cfg.MTL_ROI_START}→{cfg.MTL_ROI_END}",
                f"  3. Resized              spatial_size={cfg.MTL_OUTPUT_SIZE}",
            ]
        elif spatial_size and spatial_size != cfg.IMAGE_SIZE:
            lines += [f"  2. Resized              spatial_size={spatial_size}"]
        lines += [
            "",
            "  Preprocesamiento aplicado offline por preprocess_to_pt.py:",
            offline_desc,
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
        ]
        if aug_mode == "none":
            lines += ["  (sin augmentation — solo LoadPTd / preprocesado offline)"]
        elif aug_mode == "light":
            lines += [
                f"  RandFlipd            keys=['image'], prob=0.5, spatial_axis=0",
                f"  RandGaussianNoised   keys=['image'], prob=0.1, std=0.05",
                f"  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.1",
                f"  RandAdjustContrastd  keys=['image'], gamma=(0.8, 1.2), prob=0.1",
                "  (sin RandAffined ni RandCoarseDropout)",
            ]
        elif is_pt and variant in cfg.SPATIAL_PREPROCESS_VARIANTS:
            lines += [
                f"  RandFlipd            keys=['image'], prob=0.5, spatial_axis=0",
                f"  RandGaussianNoised   keys=['image'], prob=0.2, std=0.05",
                f"  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.2",
                f"  RandAdjustContrastd  keys=['image'], gamma=(0.7, 1.3), prob=0.2",
                f"  RandCoarseDropoutd   spatial_size=(16,16,16), prob=0.3",
                "  (sin RandAffined: registro MNI ya alineó pose/escala)",
            ]
        elif is_pt:
            lines += [
                f"  RandFlipd            keys=['image'], prob=0.5, spatial_axis=0",
                f"  RandAffined          prob=0.6, rotate/scale/translate",
                f"  RandGaussianNoised   keys=['image'], prob=0.2, std=0.05",
                f"  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.2",
                f"  RandAdjustContrastd  keys=['image'], gamma=(0.7, 1.3), prob=0.2",
                f"  RandCoarseDropoutd   spatial_size=(16,16,16), prob=0.3",
            ]
        else:
            lines += [
                f"  RandFlipd            keys=['image'], prob=0.5, spatial_axis=0",
                f"  RandRotated          keys=['image'], range_xyz=0.2, prob=0.3",
                f"  RandGaussianNoised   keys=['image'], prob=0.1, mean=0.0, std=0.05",
                f"  RandShiftIntensityd  keys=['image'], offsets=0.1, prob=0.1",
            ]
        lines += [
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

def _build_data_dicts(
    split: str,
    dataset: str = "oasis1",
    use_clinical: bool = False,
    use_etiv: bool = False,
    variant: str | None = None,
    source_filter: str | None = None,
    label_scheme: str = "multiclass",
    include_raw_labels: tuple[int, ...] | None = None,
) -> list[dict]:
    """
    Convierte un CSV de split en la lista de dicts que MONAI espera.

    Cada dict tiene la forma:
        {"image": "/ruta/al/imagen.img_o_.nii.gz", "label": 0}
    Con use_clinical=True, añade adicionalmente:
        {"clinical": torch.Tensor([age, sex, educ, apoe])}

    Args:
        split: Nombre del split ('train', 'val', 'test').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        use_clinical: Si True, extrae y normaliza covariables clínicas.

    Returns:
        Lista de diccionarios con claves 'image' y 'label' (y 'clinical' si use_clinical).
    """
    df = load_split(split, dataset=dataset, variant=variant)
    if source_filter:
        if "source" not in df.columns:
            raise ValueError(
                f"source_filter={source_filter!r} pero el split no tiene columna 'source' "
                f"(dataset={dataset!r}, split={split!r})"
            )
        df = df[df["source"].astype(str).str.lower() == source_filter.lower()]
    if include_raw_labels is None:
        include_raw_labels = get_scheme(label_scheme).include_raw_labels

    data_dicts = []
    for _, row in df.iterrows():
        raw_label = int(row["label"])
        if include_raw_labels is not None and raw_label not in include_raw_labels:
            continue
        label = remap_label(raw_label, label_scheme)
        d = {"image": row["image_path"], "label": label}
        if use_clinical:
            d["clinical"] = _clinical_vector_from_row(row, use_etiv=use_etiv)
        data_dicts.append(d)
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
    use_clinical: bool = False,
    use_etiv: bool = False,
    variant: str | None = None,
    aug_mode: str = "full",
    source_filter: str | None = None,
    label_scheme: str = "multiclass",
    include_raw_labels: tuple[int, ...] | None = None,
    roi_mode: str | None = None,
    spatial_size: tuple[int, int, int] | None = None,
    balanced_sampler: bool = False,
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
        use_clinical: Si True, cada batch incluye batch['clinical'] de shape (B, 4) o (B, 5)
                      con covariables normalizadas [age/100, sex, educ/20, apoe, (eTIV)].
        use_etiv:     Si True (requiere use_clinical), añade eTIV_norm del CSV de split.
        variant: Variante de preprocesado ('cropped' o 'mni'). None = auto por dataset.
        aug_mode: Modo de augmentation en train ('full', 'light', 'none').
        roi_mode: Si 'mtl', recorta bbox hipocampo/MTL en 96³ MNI y resize a 64³.
        spatial_size: Tamaño online tras carga (p. ej. 128³ upsample desde .pt 96³).
        balanced_sampler: Si True en train, WeightedRandomSampler equilibrado por clase (cRT).

    Returns:
        monai.data.DataLoader con batches de:
            batch['image']    -> (B, 1, 96, 96, 96) float32
            batch['label']    -> (B,) int64
            batch['clinical'] -> (B, 4) float32  (solo si use_clinical=True)
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if shuffle is None:
        shuffle = (split == "train")
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS

    variant = variant or default_variant_for_dataset(dataset)
    data_dicts = _build_data_dicts(
        split,
        dataset=dataset,
        use_clinical=use_clinical,
        use_etiv=use_etiv,
        variant=variant,
        source_filter=source_filter,
        label_scheme=label_scheme,
        include_raw_labels=include_raw_labels,
    )
    if subset is not None and subset < len(data_dicts):
        data_dicts = data_dicts[:subset]

    is_pt = len(data_dicts) > 0 and data_dicts[0]["image"].endswith(".pt")
    if is_pt:
        transforms = _resolve_pt_transforms(
            split, variant, aug_mode, roi_mode=roi_mode, spatial_size=spatial_size,
        )
    else:
        transforms = get_transforms(split)

    if use_cache:
        ds = CacheDataset(
            data=data_dicts,
            transform=transforms,
            num_workers=num_workers,
        )
    else:
        ds = Dataset(data=data_dicts, transform=transforms)

    sampler = None
    if balanced_sampler and split == "train":
        labels = [int(d["label"]) for d in data_dicts]
        import numpy as np
        counts = np.bincount(labels, minlength=cfg.NUM_CLASSES).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        sample_weights = [1.0 / counts[l] for l in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(split == "train"),
    )

    return loader
