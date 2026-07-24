"""
train.py — Training loop para AlzheimerResNet sobre OASIS-1 / OASIS-3.

Metrica de seleccion de modelo: Clinical F2-Score (β=2, ponderado: 60% AD, 30% MCI, 10% CN).
Ademas de penalizacion asimetrica en la loss (multiplicadores clinicos sobre class weights).

Genera automaticamente en outputs/<run_name>/:
    - training_log.csv       — metricas por epoch (incluye macro F2 y clinical F2)
    - curves_loss.png        — grafica de loss (train vs val)
    - curves_accuracy.png    — grafica de accuracy (train vs val)
    - curves_f1.png          — grafica de clinical F2 (train vs val)
    - best_model.pth         — pesos del mejor modelo (mayor val clinical F2)
    - training_summary.txt   — resumen legible del entrenamiento

Modos de ejecucion:
    python -m src.train --overfit                   # sanity check
    python -m src.train --epochs 2 --run test_2ep   # prueba rapida
    python -m src.train --epochs 100 --run full     # entrenamiento largo
    python -m src.train --patience 15 --run exp1    # patience custom
    python -m src.train --model ultimate_fm --run fm_run  # UltimateNeuroFM
"""

from __future__ import annotations

import csv
import random
import signal
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import fbeta_score

from src.config import cfg
from src.data_utils import load_split
from src.dataset import describe_transforms, get_dataloader
from src.label_scheme import LABEL_SCHEME_CHOICES, compute_clinical_f2_for_scheme, get_scheme
from src.losses import ExpectedCostLoss, FocalLoss, OrdinalClinicalF2Loss
from src.metrics import metrics_from_labels
from src.inference_utils import decode_preds
from src.model import AVAILABLE_MODELS, get_model

# Sobrevivir a SIGHUP si el terminal SSH cierra (el job sigue en setsid/nohup).
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Métricas de selección de modelo (early stopping + checkpoint)
# name (config)          -> (clave en dict de métricas de val, modo, clave en fila CSV)
# ---------------------------------------------------------------------------
SELECTION_METRICS: dict[str, tuple[str, str, str]] = {
    "val_loss":                     ("loss",                     "min", "val_loss"),
    "val_accuracy":                 ("accuracy",                 "max", "val_acc"),
    "val_macro_f2":                 ("macro_f2",                 "max", "val_f2"),
    "val_clinical_f2":              ("clinical_f2",              "max", "val_clinical_f2"),
    "val_balanced_accuracy":        ("balanced_accuracy",        "max", "val_balanced_accuracy"),
    "val_qwk":                      ("quadratic_weighted_kappa", "max", "val_qwk"),
    "val_quadratic_weighted_kappa": ("quadratic_weighted_kappa", "max", "val_qwk"),
    "val_expected_cost":            ("expected_cost",            "min", "val_expected_cost"),
    "val_macro_mae":                ("macro_mae",                "min", "val_macro_mae"),
}


def resolve_selection_metric(name: str) -> tuple[str, str, str]:
    """Traduce el nombre de config (p.ej. 'val_expected_cost') a (clave, modo, clave_csv)."""
    if name not in SELECTION_METRICS:
        raise ValueError(
            f"Métrica de selección desconocida: {name!r}. "
            f"Opciones: {list(SELECTION_METRICS)}"
        )
    return SELECTION_METRICS[name]


class EarlyStopping:
    """Para el entrenamiento si la métrica monitorizada no mejora en `patience` epochs.

    mode='max' (F2, accuracy, kappa…) o 'min' (loss, expected_cost, MAE).
    """

    def __init__(self, patience: int = cfg.EARLY_STOPPING_PATIENCE, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.triggered = False

    def _is_better(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, value: float) -> bool:
        """Retorna True si se debe parar el entrenamiento."""
        if self._is_better(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset: str = "oasis1",
    use_clinical_multipliers: bool = False,
    variant: str | None = None,
    label_scheme: str = "multiclass",
) -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Formula: weight_i = N_total / (N_classes * N_i)
    Si use_clinical_multipliers, multiplica por multiplicadores del esquema activo.
    """
    scheme = get_scheme(label_scheme)
    df = load_split("train", dataset=dataset, variant=variant)
    if scheme.include_raw_labels is not None:
        df = df[df["label"].isin(scheme.include_raw_labels)]
    remapped = df["label"].map(lambda x: scheme.remap(int(x)))
    counts = remapped.value_counts().sort_index()
    n_total = len(df)
    n_classes = scheme.num_classes

    weights = []
    for c in range(n_classes):
        n_c = counts.get(c, 1)
        w_c = n_total / (n_classes * n_c)
        if use_clinical_multipliers:
            w_c *= scheme.clinical_weight_multipliers[c]
        weights.append(w_c)

    w = torch.tensor(weights, dtype=torch.float32)
    mult_tag = " (x clinical multipliers)" if use_clinical_multipliers else ""
    print(f"[INFO] Class weights{mult_tag} [{label_scheme}]: {w.tolist()}")
    return w


def compute_pos_weight(
    dataset: str = "oasis1",
    variant: str | None = None,
) -> torch.Tensor:
    """
    Calcula pos_weight para BCEWithLogitsLoss en modo ordinal (2 umbrales).

    Umbral 0 — P(Y >= MCI): positivos = MCI + AD
    Umbral 1 — P(Y >= AD) : positivos = AD

    pos_weight[k] = N_neg_k / N_pos_k  (usado por BCEWithLogitsLoss para compensar imbalanza).
    """
    df = load_split("train", dataset=dataset, variant=variant)
    counts = df["label"].value_counts().sort_index()
    n_cn  = int(counts.get(0, 1))
    n_mci = int(counts.get(1, 1))
    n_ad  = int(counts.get(2, 1))

    pos0 = n_mci + n_ad
    neg0 = n_cn
    pos1 = n_ad
    neg1 = n_cn + n_mci

    pw = torch.tensor([neg0 / max(pos0, 1), neg1 / max(pos1, 1)], dtype=torch.float32)
    print(f"[INFO] Ordinal pos_weight: {pw.tolist()}  "
          f"(N_CN={n_cn}, N_MCI={n_mci}, N_AD={n_ad})")
    return pw


# ---------------------------------------------------------------------------
# Clinical F2 metric (β=2: recall pesa el doble que precision)
# ---------------------------------------------------------------------------

def compute_clinical_f2(
    labels: list[int],
    preds: list[int],
    label_scheme: str = "multiclass",
) -> float:
    """F2 (β=2) ponderado según el esquema de labels activo."""
    return compute_clinical_f2_for_scheme(labels, preds, label_scheme)


def _epoch_metrics(
    all_labels: list[int],
    all_preds: list[int],
    loss_sum: float,
    total: int,
    correct: int,
    label_scheme: str,
) -> dict:
    """Métricas comunes de un epoch (loss/acc/F2 + BA/QWK/MMAE/Expected Cost)."""
    scheme = get_scheme(label_scheme)
    macro_f2 = fbeta_score(all_labels, all_preds, beta=2, average="macro", zero_division=0)
    clin_f2 = compute_clinical_f2(all_labels, all_preds, label_scheme=label_scheme)
    extra = metrics_from_labels(all_labels, all_preds, num_classes=scheme.num_classes)
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "macro_f2": float(macro_f2),
        "clinical_f2": clin_f2,
        "balanced_accuracy": extra["balanced_accuracy"],
        "quadratic_weighted_kappa": extra["quadratic_weighted_kappa"],
        "macro_mae": extra["macro_mae"],
        "expected_cost": extra["expected_cost"],
    }


# ---------------------------------------------------------------------------
# Optimizer helpers (soporte dual Euclidean + Riemannian)
# ---------------------------------------------------------------------------

# Full fine-tune UltimateNeuroFM (ResEnc-L + cabeza)
_FULL_FT_BACKBONE_LR = 1e-5
_FULL_FT_HEAD_LR = 1e-4
_FULL_FT_WEIGHT_DECAY = 1e-3

# MedicalNet / y-Aware pretrained (backbone + cabeza nueva)
_PRETRAINED_BACKBONE_LR = 1e-5
_PRETRAINED_HEAD_LR = 1e-3


def _build_optimizers(
    model: nn.Module,
    full_finetune: bool = False,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """
    Construye uno o dos optimizadores segun si el modelo es hiperbolico.

    Para modelos con is_hyperbolic=True:
        - optimizer_eucl: AdamW para parametros euclidianos (LoRA, MLP, etc.)
        - optimizer_riem: RiemannianAdam para ManifoldParameters (prototipos de Lorentz)

    Para modelos estandar:
        - optimizer_eucl: AdamW para todos los parametros entrenables
        - optimizer_riem: None

    Returns:
        (optimizer_eucl, optimizer_riem)  — optimizer_riem puede ser None
    """
    if getattr(model, "is_hyperbolic", False):
        try:
            import geoopt
            from geoopt.optim import RiemannianAdam

            manifold_params = [
                p for p in model.parameters()
                if isinstance(p, geoopt.ManifoldParameter) and p.requires_grad
            ]
            euclidean_params = [
                p for p in model.parameters()
                if not isinstance(p, geoopt.ManifoldParameter) and p.requires_grad
            ]

            print(f"[INFO] Modo hiperbolico (geoopt): {len(euclidean_params)} params euclidianos, "
                  f"{len(manifold_params)} ManifoldParameters")

            optimizer_eucl = torch.optim.AdamW(
                euclidean_params,
                lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY,
            )
            optimizer_riem = RiemannianAdam(
                manifold_params,
                lr=cfg.LEARNING_RATE * 5,
                stabilize=10,
            )
            return optimizer_eucl, optimizer_riem

        except ImportError:
            print(
                "[WARN] geoopt no disponible: usando AdamW euclidiano para todos los parametros. "
                "Instalar con: pip install geoopt para optimizacion Riemanniana completa."
            )

    if full_finetune and hasattr(model, "feature_extractor"):
        backbone_params = [p for p in model.feature_extractor.parameters() if p.requires_grad]
        head_params = [
            p for name, p in model.named_parameters()
            if not name.startswith("feature_extractor.") and p.requires_grad
        ]
        print(
            f"[INFO] Full fine-tune: backbone lr={_FULL_FT_BACKBONE_LR}, "
            f"head lr={_FULL_FT_HEAD_LR}, wd={_FULL_FT_WEIGHT_DECAY} "
            f"({len(backbone_params)} + {len(head_params)} param tensors)"
        )
        optimizer_eucl = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": _FULL_FT_BACKBONE_LR},
                {"params": head_params, "lr": _FULL_FT_HEAD_LR},
            ],
            weight_decay=_FULL_FT_WEIGHT_DECAY,
        )
        return optimizer_eucl, None

    if getattr(model, "yaware_pretrained", False) and hasattr(model, "net"):
        backbone_params = [p for p in model.net.features.parameters() if p.requires_grad]
        head_params = [p for p in model.net.class_layers.parameters() if p.requires_grad]
        if getattr(model, "uses_ordinal", False):
            head_params += [p for p in model.coral_linear.parameters() if p.requires_grad]
            if model.coral_biases.requires_grad:
                head_params.append(model.coral_biases)
        print(
            f"[INFO] y-Aware DenseNet fine-tune: backbone lr={_PRETRAINED_BACKBONE_LR}, "
            f"head lr={_PRETRAINED_HEAD_LR} "
            f"({len(backbone_params)} + {len(head_params)} param tensors)"
        )
        optimizer_eucl = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": _PRETRAINED_BACKBONE_LR},
                {"params": head_params, "lr": _PRETRAINED_HEAD_LR},
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )
        return optimizer_eucl, None

    if getattr(model, "pretrained", False) and hasattr(model, "backbone") and model.backbone is not None:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        print(
            f"[INFO] MedicalNet fine-tune: backbone lr={_PRETRAINED_BACKBONE_LR}, "
            f"head lr={_PRETRAINED_HEAD_LR} "
            f"({len(backbone_params)} + {len(head_params)} param tensors)"
        )
        optimizer_eucl = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": _PRETRAINED_BACKBONE_LR},
                {"params": head_params, "lr": _PRETRAINED_HEAD_LR},
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )
        return optimizer_eucl, None

    # Fallback estandar: AdamW para todos los parametros entrenables (sin geoopt o modelo no hiperbolico)
    optimizer_eucl = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    return optimizer_eucl, None


def _zero_grad_all(
    optimizer_eucl: torch.optim.Optimizer,
    optimizer_riem: torch.optim.Optimizer | None,
) -> None:
    optimizer_eucl.zero_grad()
    if optimizer_riem is not None:
        optimizer_riem.zero_grad()


def _step_all(
    optimizer_eucl: torch.optim.Optimizer,
    optimizer_riem: torch.optim.Optimizer | None,
) -> None:
    optimizer_eucl.step()
    if optimizer_riem is not None:
        optimizer_riem.step()


def _get_lr(optimizer_eucl: torch.optim.Optimizer) -> float:
    return optimizer_eucl.param_groups[0]["lr"]


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
    optimizer_eucl: torch.optim.Optimizer,
    optimizer_riem: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool = False,
    label_scheme: str = "multiclass",
) -> dict:
    """
    Ejecuta un epoch de entrenamiento. Retorna metricas incluyendo clinical_f2.

    Soporta modelos estandar (optimizer_riem=None) y modelos hiperbolicos
    (optimizer_eucl + optimizer_riem en paralelo).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 100)
    t0 = time.time()

    uses_clin = getattr(model, "uses_clinical", False)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    for i, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        clinical = batch.get("clinical")
        if clinical is not None:
            clinical = clinical.to(device)
            # Modality Dropout (solo train, p=0.5): fuerza al backbone a aprender
            # representaciones visuales robustas sin depender de los clinicos
            if torch.rand(1).item() < 0.5:
                clinical = torch.zeros_like(clinical)

        _zero_grad_all(optimizer_eucl, optimizer_riem)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images, clinical) if uses_clin else model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer_eucl)
            if optimizer_riem is not None:
                scaler.step(optimizer_riem)
            scaler.update()
        else:
            loss.backward()
            _step_all(optimizer_eucl, optimizer_riem)

        running_loss += loss.item() * images.size(0)
        preds = decode_preds(outputs, model)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if i % log_every == 0 or i == n_batches:
            _progress_log(i, n_batches, t0, "Train", running_loss, correct, total)

    print()
    return _epoch_metrics(
        all_labels, all_preds, running_loss, total, correct, label_scheme,
    )


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    label_scheme: str = "multiclass",
) -> dict:
    """Evalua el modelo sin gradientes. Retorna metricas incluyendo clinical_f2."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    t0 = time.time()

    uses_clin = getattr(model, "uses_clinical", False)
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)

            outputs = model(images, clinical) if uses_clin else model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = decode_preds(outputs, model)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if i % log_every == 0 or i == n_batches:
                _progress_log(i, n_batches, t0, "Val  ", running_loss, correct, total)

    print()
    return _epoch_metrics(
        all_labels, all_preds, running_loss, total, correct, label_scheme,
    )


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def _save_plots(history: list[dict], run_dir: Path) -> None:
    """Genera y guarda graficas de loss, accuracy y clinical F2."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] * 100 for r in history]
    val_acc = [r["val_acc"] * 100 for r in history]
    train_clin_f2 = [r["train_clinical_f2"] * 100 for r in history]
    val_clin_f2 = [r["val_clinical_f2"] * 100 for r in history]

    best_idx = max(range(len(val_clin_f2)), key=lambda i: val_clin_f2[i])

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

    # --- Clinical F2 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_clin_f2, "o-", label="Train Clinical F2", linewidth=2, markersize=4)
    ax.plot(epochs, val_clin_f2, "s-", label="Val Clinical F2", linewidth=2, markersize=4)
    ax.axvline(x=epochs[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best epoch ({epochs[best_idx]})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Clinical F2 β=2 (%)", fontsize=12)
    ax.set_title("Training & Validation Clinical F2-Score β=2 (60% AD, 30% MCI, 10% CN)",
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
                  model_name: str = "resnet10",
                  label_scheme: str = "multiclass",
                  selection_metric: str = "val_clinical_f2") -> None:
    """Genera un archivo de resumen legible."""
    scheme = get_scheme(label_scheme)
    _, sel_mode, sel_row = resolve_selection_metric(selection_metric)
    best = (min if sel_mode == "min" else max)(history, key=lambda r: r[sel_row])
    last = history[-1]

    stop_reason = f"Early stopping (patience={patience})" if early_stopped else "Completado"

    lines = [
        "=" * 60,
        "TRAINING SUMMARY",
        "=" * 60,
        f"Fecha:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device:             {device}",
        f"Modelo:             {model_name}",
        f"Label scheme:       {label_scheme} ({scheme.num_classes} clases)",
        f"Parametros:         {n_params:,}",
        f"Class weights:      {class_weights}",
        f"Learning rate:      {cfg.LEARNING_RATE}",
        f"Weight decay (L2):  {cfg.WEIGHT_DECAY}",
        f"Batch size:         {cfg.BATCH_SIZE}",
        f"Image size:         {cfg.IMAGE_SIZE}",
        f"Epochs:             {len(history)}",
        f"Finalizacion:       {stop_reason}",
        f"Metrica seleccion:  {selection_metric} (modo {sel_mode})",
        f"Tiempo total:       {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)",
        f"Tiempo por epoch:   {elapsed_total/len(history):.1f}s",
        "",
        f"--- Mejor Epoch (por {selection_metric}) ---",
        f"  Epoch:               {best['epoch']}",
        f"  Train Loss:          {best['train_loss']:.4f}",
        f"  Train Acc:           {best['train_acc']:.2%}",
        f"  Train F2 (macro):    {best['train_f2']:.4f}",
        f"  Train F2 (clinical): {best['train_clinical_f2']:.4f}",
        f"  Val Loss:            {best['val_loss']:.4f}",
        f"  Val Acc:             {best['val_acc']:.2%}",
        f"  Val F2 (macro):      {best['val_f2']:.4f}",
        f"  Val F2 (clinical):   {best['val_clinical_f2']:.4f}",
        f"  Val Balanced Acc:    {best.get('val_balanced_accuracy', float('nan')):.4f}",
        f"  Val QWK (ordinal):   {best.get('val_qwk', float('nan')):.4f}",
        f"  Val Macro MAE:       {best.get('val_macro_mae', float('nan')):.4f}",
        f"  Val Expected Cost:   {best.get('val_expected_cost', float('nan')):.4f}",
        "",
        "--- Ultimo Epoch ---",
        f"  Epoch:               {last['epoch']}",
        f"  Train Loss:          {last['train_loss']:.4f}",
        f"  Train Acc:           {last['train_acc']:.2%}",
        f"  Train F2 (macro):    {last['train_f2']:.4f}",
        f"  Train F2 (clinical): {last['train_clinical_f2']:.4f}",
        f"  Val Loss:            {last['val_loss']:.4f}",
        f"  Val Acc:             {last['val_acc']:.2%}",
        f"  Val F2 (macro):      {last['val_f2']:.4f}",
        f"  Val F2 (clinical):   {last['val_clinical_f2']:.4f}",
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
    use_clinical: bool | None = None,
    ordinal: bool = False,
    focal: bool = False,
    expected_cost_loss: bool = False,
    cost_lambda: float = 0.5,
    weights_path: str | None = None,
    no_lora: bool = False,
    full_finetune: bool = False,
    clinical_weights: bool = False,
    pretrained: bool = False,
    backbone_name: str = "resnet10",
    preprocess_variant: str | None = None,
    aug_mode: str = "full",
    dropout_prob: float | None = None,
    yaware_pretrained: bool = False,
    yaware_weights_path: str | None = None,
    label_scheme: str = "multiclass",
    roi_mode: str | None = None,
    spatial_size: tuple[int, int, int] | None = None,
    batch_size_override: int | None = None,
    checkpoint_metric: str | None = None,
    early_stop_metric: str | None = None,
    seed: int | None = None,
) -> None:
    """
    Funcion principal de entrenamiento.

    Args:
        num_epochs:    Numero maximo de epochs.
        overfit_one_batch: Si True, entrena solo con 4 imagenes durante
                           100 epochs (sanity check de convergencia).
        run_name:      Nombre de la carpeta dentro de outputs/. Si None,
                       se genera uno automatico con timestamp.
        patience:      Epochs sin mejora en val clinical F2 antes de early stopping.
        dataset:       Identificador del dataset ('oasis1' o 'oasis3').
        subset:        Limitar cada split a N samples (para pruebas rapidas).
        model_name:    Nombre del modelo.
        use_clinical:  Si True, pasa covariables clinicas al modelo.
                       Por defecto True para 'multimodal_densenet' y 'ultimate_fm'.
        ordinal:       Si True, usa OrdinalClinicalF2Loss (solo compatible con densenet121
                       y multimodal_densenet). El modelo emitira 2 logits en lugar de 3.
        weights_path:  Ruta al checkpoint pre-entrenado para 'ultimate_fm'.
        no_lora:       Linear probe: backbone congelado sin LoRA (solo cabeza).
        full_finetune: Descongela ResEnc-L; LR diferencial backbone/head + AMP.
        clinical_weights: Multiplica class weights por cfg.CLINICAL_WEIGHT_MULTIPLIERS.
        pretrained:      Carga MedicalNet 3D (solo resnet10 con --pretrained).
        backbone_name:   'resnet10' o 'resnet18' si pretrained.
        yaware_pretrained: Encoder DenseNet121 desde checkpoint y-Aware BHB-10K.
        yaware_weights_path: Ruta al .pth y-Aware (default cfg.YAWARE_DENSENET_CHECKPOINT).
        preprocess_variant: Variante de preprocesado ('cropped' o 'mni').
    """
    preprocess_variant = preprocess_variant or cfg.PREPROCESS_VARIANT_DEFAULT
    scheme = get_scheme(label_scheme)
    if label_scheme in ("binary_cn_imp", "binary_mci_ad"):
        if ordinal or model_name == "densenet121_coral":
            raise ValueError(
                f"{label_scheme} no es compatible con --ordinal ni densenet121_coral"
            )
    if full_finetune and no_lora:
        print("[WARN] --full-finetune tiene prioridad sobre --no-lora")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    active_seed = seed if seed is not None else cfg.RANDOM_SEED
    random.seed(active_seed)
    np.random.seed(active_seed)
    torch.manual_seed(active_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(active_seed)
    if active_seed != cfg.RANDOM_SEED:
        print(f"[INFO] Semilla override: {active_seed} (cfg.RANDOM_SEED={cfg.RANDOM_SEED})")

    if use_clinical is None:
        use_clinical = model_name in ("multimodal_densenet", "ultimate_fm")

    wp = weights_path if weights_path is not None else str(cfg.RESENC_SSL_CHECKPOINT)
    fm_lora_stages: list[str] | None = None
    fm_freeze_backbone: bool | None = None
    if model_name == "ultimate_fm":
        if full_finetune:
            fm_lora_stages = []
            fm_freeze_backbone = False
        elif no_lora:
            fm_lora_stages = []
            fm_freeze_backbone = True
    if pretrained and yaware_pretrained:
        raise ValueError("No usar --pretrained y --yaware-pretrained a la vez")
    if pretrained and model_name != "resnet10":
        raise ValueError("--pretrained solo es compatible con --model resnet10")
    if yaware_pretrained and model_name not in ("densenet121", "densenet121_coral"):
        raise ValueError("--yaware-pretrained solo es compatible con densenet121")
    if model_name == "densenet121_coral":
        ordinal = True
    resolved_dropout = dropout_prob
    if resolved_dropout is None and model_name in ("densenet121", "densenet121_coral"):
        resolved_dropout = cfg.DENSENET_DROPOUT
    elif resolved_dropout is None and model_name == "efficientnet25d":
        resolved_dropout = cfg.EFFICIENTNET25D_DROPOUT
    yaware_wp = yaware_weights_path or str(cfg.YAWARE_DENSENET_CHECKPOINT)
    model = get_model(
        model_name,
        ordinal=ordinal,
        num_classes=scheme.num_classes,
        weights_path=wp,
        lora_stages=fm_lora_stages,
        freeze_backbone=fm_freeze_backbone,
        pretrained=pretrained,
        backbone_name=backbone_name,
        dropout_prob=dropout_prob,
        yaware_pretrained=yaware_pretrained,
        yaware_weights_path=yaware_wp if yaware_pretrained else None,
    ).to(device)
    uses_ordinal = getattr(model, "uses_ordinal", False)
    uses_coral = getattr(model, "uses_coral", False)
    is_hyperbolic = getattr(model, "is_hyperbolic", False)
    mode_tag = ""
    if model_name == "ultimate_fm":
        if full_finetune:
            mode_tag = " | full_finetune"
        elif no_lora:
            mode_tag = " | linear_probe (no LoRA)"
    if yaware_pretrained:
        mode_tag += " | y-Aware pretrained"
    if pretrained:
        mode_tag += f" | MedicalNet ({backbone_name})"
    coral_tag = " | coral=True" if uses_coral else ""
    print(f"[INFO] Modelo: {model_name} | ordinal={uses_ordinal}{coral_tag} | hiperbolico={is_hyperbolic}{mode_tag}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parametros entrenables: {n_params:,}")

    if focal and ordinal:
        raise ValueError("No se puede usar --focal y --ordinal a la vez.")
    if expected_cost_loss and (focal or ordinal):
        raise ValueError("--expected-cost-loss no es compatible con --focal ni --ordinal.")

    if uses_ordinal:
        pos_weight = compute_pos_weight(
            dataset=dataset, variant=preprocess_variant,
        ).to(device)
        criterion = OrdinalClinicalF2Loss(
            weights=cfg.CLINICAL_F2_WEIGHTS,
            alpha=0.5,
            pos_weight=pos_weight,
        )
        print(f"[INFO] Loss: OrdinalClinicalF2Loss (alpha=0.5)")
        class_weights = None
    else:
        class_weights = compute_class_weights(
            dataset=dataset,
            use_clinical_multipliers=clinical_weights,
            variant=preprocess_variant,
            label_scheme=label_scheme,
        ).to(device)
        if focal:
            criterion = FocalLoss(
                weight=class_weights,
                gamma=2.0,
                label_smoothing=0.1,
            )
            print("[INFO] Loss: FocalLoss (gamma=2.0, label_smoothing=0.1)")
        elif expected_cost_loss:
            n_classes = scheme.num_classes
            if n_classes != len(cfg.COST_MATRIX):
                raise ValueError(
                    f"--expected-cost-loss requiere COST_MATRIX {n_classes}x{n_classes}, "
                    f"pero cfg.COST_MATRIX es {len(cfg.COST_MATRIX)}x{len(cfg.COST_MATRIX)}."
                )
            criterion = ExpectedCostLoss(
                cost_matrix=cfg.COST_MATRIX,
                weight=class_weights,
                label_smoothing=0.1,
                lam=cost_lambda,
            )
            print(
                f"[INFO] Loss: ExpectedCostLoss (CE + lambda={cost_lambda}*coste, "
                f"label_smoothing=0.1, C={cfg.COST_MATRIX})"
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            print("[INFO] Loss: CrossEntropyLoss (label_smoothing=0.1)")

    optimizer_eucl, optimizer_riem = _build_optimizers(model, full_finetune=full_finetune)
    if model_name == "efficientnet25d":
        optimizer_eucl = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.EFFICIENTNET25D_WEIGHT_DECAY,
        )
        print(f"[INFO] EfficientNet25D: weight_decay={cfg.EFFICIENTNET25D_WEIGHT_DECAY}")
    use_amp = full_finetune and model_name == "ultimate_fm"
    if use_amp:
        print("[INFO] AMP (mixed precision) activo en entrenamiento")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_eucl, mode="min", factor=0.5, patience=5,
    )

    # -- Overfit-one-batch mode -----------------------------------------------
    if overfit_one_batch:
        print("\n" + "=" * 60)
        print("MODO OVERFIT-ONE-BATCH (sanity check)")
        print("=" * 60)

        loader = get_dataloader(
            "train", batch_size=4, num_workers=0, shuffle=False,
            dataset=dataset, subset=subset, use_clinical=use_clinical,
            variant=preprocess_variant, aug_mode=aug_mode,
            label_scheme=label_scheme,
            roi_mode=roi_mode, spatial_size=spatial_size,
        )
        single_batch = next(iter(loader))
        images = single_batch["image"].to(device)
        labels = single_batch["label"].to(device)
        ob_clinical = single_batch.get("clinical")
        if ob_clinical is not None:
            ob_clinical = ob_clinical.to(device)
        print(f"Batch labels: {labels.tolist()}")

        num_epochs = 100
        _uses_clin = getattr(model, "uses_clinical", False)
        model.train()
        for epoch in range(1, num_epochs + 1):
            _zero_grad_all(optimizer_eucl, optimizer_riem)
            outputs = model(images, ob_clinical) if _uses_clin else model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            _step_all(optimizer_eucl, optimizer_riem)

            preds = decode_preds(outputs, model)
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

    # Métricas de selección (checkpoint) y de early stopping — configurables por
    # parámetro (CLI/pipeline) con fallback a cfg.
    checkpoint_metric = checkpoint_metric or cfg.CHECKPOINT_METRIC
    early_stop_metric = early_stop_metric or cfg.EARLY_STOP_METRIC
    ckpt_key, ckpt_mode, _ = resolve_selection_metric(checkpoint_metric)
    es_key, es_mode, _ = resolve_selection_metric(early_stop_metric)
    if "expected_cost" in (ckpt_key, es_key) and scheme.num_classes != len(cfg.COST_MATRIX):
        raise ValueError(
            f"Selección por expected_cost requiere COST_MATRIX {len(cfg.COST_MATRIX)}x"
            f"{len(cfg.COST_MATRIX)} pero el esquema '{label_scheme}' tiene "
            f"{scheme.num_classes} clases."
        )

    early_stopper = EarlyStopping(patience=patience, mode=es_mode)

    transforms_desc = describe_transforms(
        "train", is_pt=True, variant=preprocess_variant, aug_mode=aug_mode,
        roi_mode=roi_mode, spatial_size=spatial_size,
    )
    (run_dir / "transforms_config.txt").write_text(
        transforms_desc + "\n", encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO — max {num_epochs} epochs (early stopping: patience={patience})")
    print(f"Preprocess variant: {preprocess_variant}")
    print(f"Augmentation mode:  {aug_mode}")
    if roi_mode:
        print(f"ROI mode:           {roi_mode}")
    if spatial_size and spatial_size != cfg.IMAGE_SIZE:
        print(f"Spatial size:       {spatial_size}")
    if resolved_dropout is not None:
        print(f"DenseNet dropout:   {resolved_dropout}")
    print(f"Label scheme:       {label_scheme} ({scheme.num_classes} clases)")
    print(f"Metrica checkpoint:  {checkpoint_metric} (modo {ckpt_mode})")
    print(f"Metrica early stop:  {early_stop_metric} (modo {es_mode})")
    if checkpoint_metric == "val_clinical_f2":
        print(f"  (Clinical F2 β=2, pesos: {scheme.clinical_f2_weights})")
    print(f"Resultados en: {run_dir}")
    print("=" * 60)

    t_load = time.time()
    if subset:
        print(f"[INFO] Modo subset: limitando a {subset} samples por split")
    batch_size = batch_size_override
    if batch_size is None:
        if model_name == "efficientnet25d":
            batch_size = cfg.EFFICIENTNET25D_BATCH_SIZE
        elif roi_mode == "mtl":
            batch_size = cfg.MTL_BATCH_SIZE
        elif spatial_size == cfg.IMAGE_SIZE_HIGH or preprocess_variant == "mni_128":
            batch_size = cfg.HIGH_RES_BATCH_SIZE
        else:
            batch_size = cfg.BATCH_SIZE
    if model_name == "efficientnet25d":
        print(f"[INFO] Batch size EfficientNet25D: {batch_size}")
        if resolved_dropout is not None:
            print(f"[INFO] EfficientNet25D dropout: {resolved_dropout}")
    elif batch_size != cfg.BATCH_SIZE:
        print(f"[INFO] Batch size: {batch_size}")
    print("[INFO] Cargando datos de entrenamiento...")
    train_loader = get_dataloader(
        "train", dataset=dataset, subset=subset, use_clinical=use_clinical,
        variant=preprocess_variant, aug_mode=aug_mode,
        label_scheme=label_scheme, batch_size=batch_size,
        roi_mode=roi_mode, spatial_size=spatial_size,
    )
    print(f"[INFO] Cargando datos de validacion...")
    val_loader = get_dataloader(
        "val", dataset=dataset, subset=subset, use_clinical=use_clinical,
        variant=preprocess_variant, aug_mode=aug_mode,
        label_scheme=label_scheme, batch_size=batch_size,
        roi_mode=roi_mode, spatial_size=spatial_size,
    )
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
            "epoch", "train_loss", "train_acc", "train_f2", "train_clinical_f2",
            "val_loss", "val_acc", "val_f2", "val_clinical_f2",
            "val_balanced_accuracy", "val_qwk", "val_macro_mae", "val_expected_cost",
            "epoch_time_s", "is_best",
        ],
    )
    csv_writer.writeheader()

    history: list[dict] = []
    best_ckpt_value = float("-inf") if ckpt_mode == "max" else float("inf")
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_eucl, optimizer_riem, device,
            use_amp=use_amp,
            label_scheme=label_scheme,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, label_scheme=label_scheme,
        )

        elapsed = time.time() - t0
        val_ckpt_value = val_metrics[ckpt_key]
        is_best = (
            val_ckpt_value > best_ckpt_value if ckpt_mode == "max"
            else val_ckpt_value < best_ckpt_value
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "train_f2": round(train_metrics["macro_f2"], 6),
            "train_clinical_f2": round(train_metrics["clinical_f2"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "val_f2": round(val_metrics["macro_f2"], 6),
            "val_clinical_f2": round(val_metrics["clinical_f2"], 6),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
            "val_qwk": round(val_metrics["quadratic_weighted_kappa"], 6),
            "val_macro_mae": round(val_metrics["macro_mae"], 6),
            "val_expected_cost": round(val_metrics["expected_cost"], 6),
            "epoch_time_s": round(elapsed, 1),
            "is_best": is_best,
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        if is_best:
            best_ckpt_value = val_ckpt_value
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_name": model_name,
                "label_scheme": label_scheme,
                "num_classes": scheme.num_classes,
                "use_clinical": use_clinical,
                "uses_ordinal": uses_ordinal,
                "uses_coral": uses_coral,
                "is_hyperbolic": is_hyperbolic,
                "pretrained": pretrained,
                "backbone_name": backbone_name,
                "yaware_pretrained": yaware_pretrained,
                "yaware_weights_path": yaware_wp if yaware_pretrained else None,
                "clinical_weights": clinical_weights,
                "preprocess_variant": preprocess_variant,
                "roi_mode": roi_mode,
                "spatial_size": spatial_size,
                "dropout_prob": resolved_dropout,
                "weights_path": wp,
                "ultimate_fm_lora_stages": fm_lora_stages,
                "ultimate_fm_freeze_backbone": fm_freeze_backbone,
                "model_state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f2": val_metrics["macro_f2"],
                "val_clinical_f2": val_metrics["clinical_f2"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_qwk": val_metrics["quadratic_weighted_kappa"],
                "val_macro_mae": val_metrics["macro_mae"],
                "val_expected_cost": val_metrics["expected_cost"],
                "selection_metric": checkpoint_metric,
                "selection_value": best_ckpt_value,
            }
            if is_hyperbolic:
                checkpoint["optimizer_eucl_state_dict"] = optimizer_eucl.state_dict()
                if optimizer_riem is not None:
                    checkpoint["optimizer_riem_state_dict"] = optimizer_riem.state_dict()
            else:
                checkpoint["optimizer_state_dict"] = optimizer_eucl.state_dict()

            torch.save(checkpoint, run_dir / "best_model.pth")

        should_stop = early_stopper.step(val_metrics[es_key])
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
        current_lr = _get_lr(optimizer_eucl)

        print(
            f"\n--- Epoch {epoch}/{num_epochs} "
            f"({elapsed:.0f}s | total: {elapsed_str} | ETA: {eta_str}) ---\n"
            f"  Train  ->  Loss: {train_metrics['loss']:.4f}  |  Acc: {train_metrics['accuracy']:.2%}  "
            f"|  F2m: {train_metrics['macro_f2']:.4f}  |  F2c: {train_metrics['clinical_f2']:.4f}\n"
            f"  Val    ->  Loss: {val_metrics['loss']:.4f}  |  Acc: {val_metrics['accuracy']:.2%}  "
            f"|  F2m: {val_metrics['macro_f2']:.4f}  |  F2c: {val_metrics['clinical_f2']:.4f}{best_marker}\n"
            f"  LR: {current_lr:.2e}  |  Best: epoch {best_epoch} "
            f"({checkpoint_metric}={best_ckpt_value:.4f})  "
            f"| Early stop: {es_bar} {es_counter}/{patience}"
        )

        if should_stop:
            print(
                f"\n{'=' * 60}\n"
                f"[EARLY STOPPING] {early_stop_metric} no mejoro en {patience} epochs.\n"
                f"Mejor epoch: {best_epoch} ({checkpoint_metric}={best_ckpt_value:.4f})\n"
                f"{'=' * 60}"
            )
            break

    csv_file.close()
    elapsed_total = time.time() - t_start

    _save_plots(history, run_dir)
    cw_list = class_weights.cpu().tolist() if class_weights is not None else "N/A (ordinal)"
    _save_summary(
        history, run_dir, device, n_params, elapsed_total,
        cw_list,
        early_stopped=early_stopper.triggered,
        patience=patience,
        model_name=model_name,
        label_scheme=label_scheme,
        selection_metric=checkpoint_metric,
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
                        choices=["oasis1", "oasis3", "adni", "adni_baseline_sc", "oasis3_adni",
                                 "oasis3_adni_baseline"],
                        help="Dataset a utilizar (default: oasis1)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limitar a N samples por split (para pruebas rapidas)")
    parser.add_argument("--model", type=str, default="resnet10",
                        choices=AVAILABLE_MODELS,
                        help="Modelo a usar (default: resnet10)")
    parser.add_argument("--ordinal", action="store_true",
                        help="CORAL + OrdinalClinicalF2Loss (densenet121 / densenet121_coral)")
    parser.add_argument(
        "--focal",
        action="store_true",
        help="Usar Focal Loss multiclase (gamma=2) en lugar de CrossEntropyLoss",
    )
    parser.add_argument(
        "--expected-cost-loss",
        action="store_true",
        help="Usar ExpectedCostLoss (CE + lambda*coste, cfg.COST_MATRIX) en lugar de CE",
    )
    parser.add_argument(
        "--cost-lambda",
        type=float,
        default=0.5,
        help="Peso lambda del termino de coste esperado en ExpectedCostLoss (default 0.5)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=str(cfg.RESENC_SSL_CHECKPOINT),
        help="Ruta al checkpoint pre-entrenado del backbone (solo ultimate_fm); default: cfg.RESENC_SSL_CHECKPOINT",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="ultimate_fm: backbone congelado sin LoRA (linear probe, solo cabeza)",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="ultimate_fm: full fine-tune ResEnc-L con LR diferencial y AMP",
    )
    parser.add_argument(
        "--clinical-weights",
        action="store_true",
        help="Aplicar cfg.CLINICAL_WEIGHT_MULTIPLIERS sobre los class weights de la loss",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="MedicalNet 3D preentrenado (solo --model resnet10)",
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="resnet10",
        choices=["resnet10", "resnet18"],
        help="Backbone MONAI si --pretrained (default: resnet10)",
    )
    parser.add_argument(
        "--yaware-pretrained",
        action="store_true",
        help="Encoder DenseNet121 desde checkpoint y-Aware BHB-10K (solo densenet121)",
    )
    parser.add_argument(
        "--yaware-weights",
        type=str,
        default=None,
        help="Ruta al .pth y-Aware (default: cfg.YAWARE_DENSENET_CHECKPOINT)",
    )
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=cfg.PREPROCESS_VARIANT_DEFAULT,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado offline: cropped (baseline) o mni",
    )
    parser.add_argument(
        "--aug",
        type=str,
        default="full",
        choices=["full", "light", "none"],
        help="Modo de augmentation en train (default: full)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        metavar="P",
        help="Dropout DenseNet (default: cfg.DENSENET_DROPOUT; usar 0 para desactivar)",
    )
    parser.add_argument(
        "--label-scheme",
        type=str,
        default="multiclass",
        choices=list(LABEL_SCHEME_CHOICES),
        help="Esquema de labels: multiclass (CN/MCI/AD) o binary_cn_imp (CN vs MCI+AD)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        choices=["mtl"],
        help="Recorte ROI online: mtl = hipocampo/MTL desde .pt MNI 96³",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        metavar="N",
        help="Resize online a N³ tras carga (p. ej. 128 upsample desde .pt 96³)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="B",
        help="Batch size (default: auto según modelo/ROI/resolución)",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        choices=list(SELECTION_METRICS.keys()),
        help="Métrica para checkpoint y early stopping (default: cfg.CHECKPOINT_METRIC). "
             "Aplica a ambos salvo que se use --early-stop-metric.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default=None,
        choices=list(SELECTION_METRICS.keys()),
        help="Métrica solo para early stopping (default: igual que --selection-metric o cfg).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Semilla aleatoria (default: cfg.RANDOM_SEED={cfg.RANDOM_SEED}). "
             "Para replicación multi-semilla del mismo experimento.",
    )
    args = parser.parse_args()

    spatial_size = None
    if args.spatial_size is not None:
        if args.roi:
            parser.error("No combinar --roi con --spatial-size")
        spatial_size = (args.spatial_size,) * 3

    if args.label_scheme in ("binary_cn_imp", "binary_mci_ad") and (
        args.ordinal or args.model == "densenet121_coral"
    ):
        parser.error(
            f"{args.label_scheme} no es compatible con --ordinal ni densenet121_coral"
        )

    train(
        num_epochs=args.epochs,
        overfit_one_batch=args.overfit,
        run_name=args.run,
        patience=args.patience,
        dataset=args.dataset,
        subset=args.subset,
        model_name=args.model,
        ordinal=args.ordinal,
        focal=args.focal,
        expected_cost_loss=args.expected_cost_loss,
        cost_lambda=args.cost_lambda,
        weights_path=args.weights_path,
        no_lora=args.no_lora,
        full_finetune=args.full_finetune,
        clinical_weights=args.clinical_weights,
        pretrained=args.pretrained,
        backbone_name=args.backbone_name,
        preprocess_variant=args.preprocess_variant,
        aug_mode=args.aug,
        dropout_prob=args.dropout,
        yaware_pretrained=args.yaware_pretrained,
        yaware_weights_path=args.yaware_weights,
        label_scheme=args.label_scheme,
        roi_mode=args.roi,
        spatial_size=spatial_size,
        batch_size_override=args.batch_size,
        checkpoint_metric=args.selection_metric,
        early_stop_metric=args.early_stop_metric or args.selection_metric,
        seed=args.seed,
    )
