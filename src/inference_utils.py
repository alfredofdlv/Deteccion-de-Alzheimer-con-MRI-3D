"""
inference_utils.py — Utilidades compartidas de inferencia post-entrenamiento.

Probabilidades (softmax / ordinal), TTA, umbrales, carga de checkpoints y métricas.
Usado por evaluate.py, threshold_tuning.py, ensemble.py y hierarchical.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import fbeta_score

from src.config import cfg
from src.label_scheme import compute_clinical_f2_for_scheme, get_scheme
from src.model import get_model, adapt_efficientnet25d_state_dict


CLASS_NAMES = ["CN", "MCI", "AD"]


def get_class_names(label_scheme: str = "multiclass") -> list[str]:
    return list(get_scheme(label_scheme).class_names)


def ordinal_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convierte logits CORAL/ordinales (B, K-1) en probabilidades 3-clase (B, 3).

    P(CN)=1-P(Y>=MCI), P(MCI)=P(Y>=MCI)-P(Y>=AD), P(AD)=P(Y>=AD).
    """
    sigmoids = torch.sigmoid(logits)
    p0 = sigmoids[:, 0]  # P(Y >= MCI)
    p1 = sigmoids[:, 1]  # P(Y >= AD)
    p_cn = (1.0 - p0).clamp(min=0.0)
    p_mci = (p0 - p1).clamp(min=0.0)
    p_ad = p1.clamp(min=0.0)
    return torch.stack([p_cn, p_mci, p_ad], dim=1)


def ordinal_sigmoids_to_class_probs(sigmoids: torch.Tensor) -> torch.Tensor:
    """
    Convierte 2 sigmoides ordinales P(Y>=MCI), P(Y>=AD) en probs 3-clase.

    P(CN)=1-p0, P(MCI)=p0-p1, P(AD)=p1 (recortado a >=0 y renormalizado).
    """
    p0 = sigmoids[:, 0].clamp(0.0, 1.0)
    p1 = sigmoids[:, 1].clamp(0.0, 1.0)
    p_cn = (1.0 - p0).clamp(min=0.0)
    p_mci = (p0 - p1).clamp(min=0.0)
    p_ad = p1.clamp(min=0.0)
    probs = torch.stack([p_cn, p_mci, p_ad], dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs


def decode_preds(outputs: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Decodifica logits a clases {0, 1, 2}.

    Estándar: argmax sobre 3 logits.
    Ordinal/CORAL: descomposición probabilística + argmax.
    """
    if getattr(model, "uses_ordinal", False):
        return ordinal_logits_to_class_probs(outputs).argmax(dim=1)
    return outputs.argmax(dim=1)


def outputs_to_class_probs(outputs: torch.Tensor, uses_ordinal: bool) -> torch.Tensor:
    """Logits del modelo -> (B, 3) probabilidades de clase."""
    if uses_ordinal:
        probs = ordinal_logits_to_class_probs(outputs)
        return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return F.softmax(outputs, dim=1)


def apply_logit_biases(probs: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """
    Aplica sesgos por clase en espacio logit y renormaliza.

    biases: shape (3,) sumados al logit de cada clase.
    """
    eps = 1e-8
    logits = np.log(np.clip(probs, eps, 1.0))
    logits = logits + biases.reshape(1, 3)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def probs_to_preds(probs: np.ndarray) -> list[int]:
    return probs.argmax(axis=1).tolist()


def compute_clinical_f2_np(labels: list[int], preds: list[int]) -> float:
    f2_per_class = fbeta_score(
        labels, preds, beta=2, labels=list(range(cfg.NUM_CLASSES)),
        average=None, zero_division=0,
    )
    return float(sum(cfg.CLINICAL_F2_WEIGHTS[c] * f2_per_class[c] for c in range(cfg.NUM_CLASSES)))


def load_model_from_run(
    run_name: str,
    device: torch.device | None = None,
    model_name: str | None = None,
    weights_path: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    """Carga best_model.pth desde outputs/<run_name>/."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = cfg.OUTPUTS_DIR / run_name
    model_path = run_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    resolved_model = model_name or checkpoint.get("model_name", "resnet10")
    uses_ordinal = checkpoint.get("uses_ordinal", False)
    num_classes = checkpoint.get("num_classes", cfg.NUM_CLASSES)
    resolved_weights = weights_path or checkpoint.get(
        "weights_path", str(cfg.RESENC_SSL_CHECKPOINT),
    )
    lora_stages = checkpoint.get("ultimate_fm_lora_stages")
    freeze_backbone = checkpoint.get("ultimate_fm_freeze_backbone")

    if resolved_model == "densenet121_coral":
        model = get_model("densenet121_coral", num_classes=num_classes).to(device)
    else:
        model = get_model(
            resolved_model,
            ordinal=uses_ordinal,
            num_classes=num_classes,
            weights_path=resolved_weights,
            lora_stages=lora_stages,
            freeze_backbone=freeze_backbone,
            pretrained=checkpoint.get("pretrained", False),
            backbone_name=checkpoint.get("backbone_name", "resnet10"),
            dropout_prob=checkpoint.get("dropout_prob"),
            yaware_pretrained=checkpoint.get("yaware_pretrained", False),
            yaware_weights_path=checkpoint.get("yaware_weights_path"),
        ).to(device)
    state_dict = checkpoint["model_state_dict"]
    if resolved_model == "efficientnet25d":
        state_dict = adapt_efficientnet25d_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint, run_dir


def _forward_batch_probs(
    model: torch.nn.Module,
    images: torch.Tensor,
    clinical: torch.Tensor | None,
) -> torch.Tensor:
    uses_clin = getattr(model, "uses_clinical", False)
    uses_ordinal = getattr(model, "uses_ordinal", False)
    outputs = model(images, clinical) if uses_clin else model(images)
    return outputs_to_class_probs(outputs, uses_ordinal)


def _tta_flip_lr(images: torch.Tensor) -> torch.Tensor:
    """Flip espejo en eje espacial 0 (LR en volumen MONAI)."""
    return torch.flip(images, dims=[2])


def _set_dropout_layers_train(model: torch.nn.Module) -> None:
    """Activa solo Dropout en modo train (MC Dropout)."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def _predict_batch_mc_dropout(
    model: torch.nn.Module,
    images: torch.Tensor,
    clinical: torch.Tensor | None,
    uses_clin: bool,
    uses_ordinal: bool,
    mc_samples: int,
) -> torch.Tensor:
    """Promedia probabilidades sobre mc_samples pases forward."""
    probs_stack: list[torch.Tensor] = []
    for _ in range(mc_samples):
        outputs = model(images, clinical) if uses_clin else model(images)
        probs_stack.append(outputs_to_class_probs(outputs, uses_ordinal))
    avg_probs = torch.stack(probs_stack, dim=0).mean(dim=0)
    return avg_probs.argmax(dim=1)


def collect_probabilities(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    tta: bool = False,
    mc_dropout: bool = False,
    mc_samples: int = 30,
) -> tuple[list[int], np.ndarray]:
    """
    Inferencia que devuelve labels y matriz de probabilidades (N, 3).

    Con tta=True promedia probs de la imagen original y flip LR.
    """
    model.eval()
    if mc_dropout:
        _set_dropout_layers_train(model)

    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    uses_clin = getattr(model, "uses_clinical", False)
    uses_ordinal = getattr(model, "uses_ordinal", False)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)

            if mc_dropout:
                batch_preds = _predict_batch_mc_dropout(
                    model, images, clinical, uses_clin, uses_ordinal, mc_samples,
                )
                probs = F.one_hot(
                    batch_preds, num_classes=cfg.NUM_CLASSES,
                ).float()
            elif tta:
                p0 = _forward_batch_probs(model, images, clinical)
                p1 = _forward_batch_probs(model, _tta_flip_lr(images), clinical)
                probs = (p0 + p1) / 2.0
            else:
                probs = _forward_batch_probs(model, images, clinical)

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(batch["label"].tolist())

    if mc_dropout:
        model.eval()

    return all_labels, np.vstack(all_probs)


def save_thresholds(
    run_dir: Path,
    biases: np.ndarray,
    val_clinical_f2: float,
    extra: dict | None = None,
) -> Path:
    payload = {
        "logit_biases": biases.tolist(),
        "val_clinical_f2": val_clinical_f2,
        **(extra or {}),
    }
    path = run_dir / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_thresholds(run_dir: Path) -> np.ndarray | None:
    path = run_dir / "thresholds.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["logit_biases"], dtype=np.float64)


def grid_search_logit_biases(
    labels: list[int],
    probs: np.ndarray,
    bias_range: tuple[float, float, float] = (-1.0, 1.0, 0.25),
) -> tuple[np.ndarray, float]:
    """
    Búsqueda en rejilla de sesgos por clase que maximizan Clinical F2 en val.
    """
    lo, hi, step = bias_range
    grid = np.arange(lo, hi + step * 0.5, step)
    best_f2 = -1.0
    best_biases = np.zeros(3, dtype=np.float64)

    for b0 in grid:
        for b1 in grid:
            for b2 in grid:
                biases = np.array([b0, b1, b2], dtype=np.float64)
                adj = apply_logit_biases(probs, biases)
                preds = probs_to_preds(adj)
                f2 = compute_clinical_f2_np(labels, preds)
                if f2 > best_f2:
                    best_f2 = f2
                    best_biases = biases.copy()

    return best_biases, best_f2


def class_priors_from_labels(labels: list[int], num_classes: int | None = None) -> np.ndarray:
    """Frecuencias empíricas por clase (suavizado mínimo para evitar log(0))."""
    k = num_classes or cfg.NUM_CLASSES
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def analytical_prior_biases(
    pi_source: np.ndarray,
    pi_target: np.ndarray,
) -> np.ndarray:
    """
    Sesgos de logit para corregir label shift (Saerens et al., fórmula directa).

    biases_c = log(pi_target_c) - log(pi_source_c)
    """
    eps = 1e-8
    return np.log(np.clip(pi_target, eps, 1.0)) - np.log(np.clip(pi_source, eps, 1.0))


def saerens_estimate_target_prior(
    probs: np.ndarray,
    pi_source: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_prior: float = 0.05,
) -> np.ndarray:
    """
    EM de Saerens (2002) para estimar priors en dominio destino sin etiquetas.

    Ajusta P_t(c|x) ∝ P_s(c|x) * pi_t(c) / pi_s(c) iterando sobre pi_t.
    Inicialización uniforme y suelo min_prior para evitar colapso degenerado.
    """
    eps = 1e-8
    k = probs.shape[1]
    pi_s = np.clip(pi_source, eps, 1.0)
    pi_s = pi_s / pi_s.sum()
    n, _ = probs.shape
    pi_t = np.ones(k, dtype=np.float64) / k

    for _ in range(max_iter):
        ratio = pi_t / pi_s
        adjusted = probs * ratio.reshape(1, k)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True).clip(min=eps)
        pi_new = adjusted.sum(axis=0) / n
        pi_new = np.maximum(pi_new, min_prior / k)
        pi_new = pi_new / pi_new.sum()
        if np.max(np.abs(pi_new - pi_t)) < tol:
            pi_t = pi_new
            break
        pi_t = pi_new

    return pi_t / pi_t.sum()
