"""
yaware_densenet.py — Carga de pesos y-Aware (BHB-10K) en DenseNet121 MONAI.

El checkpoint oficial (Duplums/yAwareContrastiveLearning) usa una DenseNet 3D
custom con claves distintas a MONAI (sin submódulo `.layers.` en cada denselayer).
Este módulo remapea claves del encoder y omite cabeza contrastiva / proyección.

Referencia: https://github.com/Duplums/yAwareContrastiveLearning
Peso: DenseNet121_BHB-10K_yAwareContrastive.pth (Google Drive en README del repo).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

from src.config import cfg
from src.model import _torch_load_checkpoint, _unwrap_state_dict_from_checkpoint

# Google Drive file id del README oficial (Duplums/yAwareContrastiveLearning).
YAWARE_GDRIVE_FILE_ID = "1BmDC4USdZmX0ZSi-jQyUVmKHaDFIJFo2"
YAWARE_CHECKPOINT_FILENAME = "DenseNet121_BHB-10K_yAwareContrastive.pth"

_SKIP_PREFIXES = (
    "hidden_representation.",
    "head_projection.",
    "classifier.",
    "class_layers.",
)

_DENSELAYER_KEY = re.compile(
    r"^(features\.denseblock\d+\.denselayer\d+)\.(norm1|conv1|norm2|conv2)(\..+)$"
)


@dataclass
class YAwareLoadReport:
    """Resumen de la carga de pesos y-Aware en un DenseNet MONAI."""

    weights_path: Path
    source_tensor_keys: int
    remapped_keys: int
    loaded_keys: int
    missing_backbone_keys: int
    unexpected_keys: int
    shape_mismatches: int
    backbone_coverage_pct: float

    def __str__(self) -> str:
        return (
            f"y-Aware load: {self.loaded_keys} tensors cargados "
            f"({self.backbone_coverage_pct:.1f}% del backbone features+norm5), "
            f"missing={self.missing_backbone_keys}, unexpected={self.unexpected_keys}, "
            f"shape_mismatch={self.shape_mismatches}"
        )


def default_yaware_weights_path() -> Path:
    return cfg.YAWARE_DENSENET_CHECKPOINT


def strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def strip_features_prefix(key: str) -> str:
    """Claves de model.features.state_dict() no llevan prefijo 'features.'."""
    if key.startswith("features."):
        return key[len("features.") :]
    return key


def remap_yaware_key_to_monai(key: str) -> str | None:
    """
    Convierte una clave del checkpoint y-Aware a nomenclatura MONAI DenseNet121.

    Returns:
        Clave MONAI, o None si la clave pertenece a cabeza SSL/clasificador y-Aware.
    """
    key = strip_module_prefix(key)
    if any(key.startswith(p) for p in _SKIP_PREFIXES):
        return None
    m = _DENSELAYER_KEY.match(key)
    if m and ".layers." not in key:
        key = f"{m.group(1)}.layers.{m.group(2)}{m.group(3)}"
    return strip_features_prefix(key)


def extract_yaware_backbone_state_dict(raw: object) -> dict[str, torch.Tensor]:
    """Extrae y remapea tensores del encoder desde un checkpoint y-Aware."""
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        sd = raw["model"]
    else:
        sd = _unwrap_state_dict_from_checkpoint(raw)

    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        mk = remap_yaware_key_to_monai(k)
        if mk is not None:
            out[mk] = v
    return out


def _backbone_key_set(model: DenseNet121) -> set[str]:
    return set(model.features.state_dict().keys())


def load_yaware_into_monai_densenet(
    model: DenseNet121,
    weights_path: str | Path,
) -> YAwareLoadReport:
    """
    Carga pesos y-Aware en un DenseNet121 MONAI (solo trunk; cabeza class_layers intacta).

    Usa strict=False; solo asigna claves con forma compatible.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"No se encontró checkpoint y-Aware: {weights_path}\n"
            f"Descarga con: python scripts/download_yaware_densenet.py"
        )

    raw = _torch_load_checkpoint(weights_path)
    yaware_sd = extract_yaware_backbone_state_dict(raw)

    model_sd = model.state_dict()
    backbone_sd = model.features.state_dict()

    to_load: dict[str, torch.Tensor] = {}
    shape_mismatches = 0
    for mk, tensor in yaware_sd.items():
        if mk in backbone_sd:
            if backbone_sd[mk].shape == tensor.shape:
                to_load[mk] = tensor
            else:
                shape_mismatches += 1
        elif mk in model_sd and mk.startswith("class_layers."):
            continue

    missing, unexpected = model.features.load_state_dict(to_load, strict=False)
    loaded = len(to_load)
    total_backbone = len(backbone_sd)
    coverage = 100.0 * loaded / total_backbone if total_backbone else 0.0

    return YAwareLoadReport(
        weights_path=weights_path,
        source_tensor_keys=len(yaware_sd),
        remapped_keys=len(yaware_sd),
        loaded_keys=loaded,
        missing_backbone_keys=len(missing),
        unexpected_keys=len(unexpected),
        shape_mismatches=shape_mismatches,
        backbone_coverage_pct=coverage,
    )


def diagnose_yaware_checkpoint(
    weights_path: str | Path,
    dropout_prob: float | None = None,
) -> dict:
    """
    Compara claves del checkpoint y-Aware con DenseNet121 MONAI.

    Returns:
        dict con estadísticas para scripts/inspect_yaware_checkpoint.py
    """
    weights_path = Path(weights_path)
    p = cfg.DENSENET_DROPOUT if dropout_prob is None else dropout_prob
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=cfg.NUM_CLASSES,
        dropout_prob=p,
    )
    raw = _torch_load_checkpoint(weights_path)
    yaware_sd = extract_yaware_backbone_state_dict(raw)

    monai_backbone = set(model.features.state_dict().keys())
    remapped_keys = set(yaware_sd.keys())

    matched = {k for k in remapped_keys if k in monai_backbone and yaware_sd[k].shape == model.features.state_dict()[k].shape}
    unmatched_yaware = sorted(remapped_keys - monai_backbone)
    missing_monai = sorted(monai_backbone - remapped_keys)

    raw_sd = _unwrap_state_dict_from_checkpoint(raw)
    if isinstance(raw_sd, dict) and "model" in raw_sd:
        raw_sd = raw_sd["model"]
    sample_raw = sorted(strip_module_prefix(k) for k in list(raw_sd.keys())[:20])

    return {
        "weights_path": str(weights_path),
        "raw_key_count": len(raw_sd),
        "remapped_key_count": len(remapped_keys),
        "monai_backbone_key_count": len(monai_backbone),
        "matched_shape_ok": len(matched),
        "coverage_pct": 100.0 * len(matched) / len(monai_backbone) if monai_backbone else 0.0,
        "unmatched_yaware_sample": unmatched_yaware[:15],
        "missing_monai_sample": missing_monai[:15],
        "sample_raw_keys": sample_raw,
    }


def download_yaware_checkpoint(
    dest: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Descarga el .pth oficial desde Google Drive (requiere gdown)."""
    dest = Path(dest or cfg.YAWARE_DENSENET_CHECKPOINT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        return dest

    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "Instala gdown: pip install gdown\n"
            f"O descarga manualmente a {dest}"
        ) from e

    url = f"https://drive.google.com/uc?id={YAWARE_GDRIVE_FILE_ID}"
    gdown.download(url, str(dest), quiet=False)
    if not dest.is_file():
        raise FileNotFoundError(f"Descarga fallida: {dest}")
    return dest
