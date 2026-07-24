"""
evaluate.py — Evaluación del modelo sobre el test set con métricas detalladas (OASIS-1 / OASIS-3).

Genera en outputs/<run_name>/:
    - classification_report_<split>.txt  — precision, recall, F1 por clase
    - confusion_matrix_<split>.png       — heatmap de la matriz de confusión
    Con --mc-dropout: *_mcdropout.txt / *_mcdropout.png (inferencia bayesiana aproximada).

Uso:
    python -m src.evaluate --run full_100ep
    python -m src.evaluate --run densenet-multimodal-ordinal --dataset oasis3 --mc-dropout --mc-samples 30
"""

from __future__ import annotations

import time
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
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
)

from src.config import cfg
from src.dataset import get_dataloader
from src.inference_utils import (
    CLASS_NAMES,
    apply_logit_biases,
    collect_probabilities,
    decode_preds,
    get_class_names,
    load_model_from_run,
    load_thresholds,
    probs_to_preds,
)
from src.label_scheme import LABEL_SCHEME_CHOICES, compute_clinical_f2_for_scheme, get_scheme
from src.metrics import metrics_from_confusion
from src.model import AVAILABLE_MODELS


def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mc_dropout: bool = False,
    mc_samples: int = 30,
) -> tuple[list[int], list[int]]:
    """Ejecuta inferencia y recopila todas las predicciones y labels."""
    model.eval()
    from src.inference_utils import _predict_batch_mc_dropout, _set_dropout_layers_train

    if mc_dropout:
        _set_dropout_layers_train(model)

    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)  # ~5% de los batches
    t0 = time.time()

    uses_clin = getattr(model, "uses_clinical", False)
    uses_ordinal = getattr(model, "uses_ordinal", False)

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)

            if not mc_dropout:
                outputs = model(images, clinical) if uses_clin else model(images)
                batch_preds = decode_preds(outputs, model)
            else:
                batch_preds = _predict_batch_mc_dropout(
                    model, images, clinical, uses_clin, uses_ordinal, mc_samples
                )

            all_preds.extend(batch_preds.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

            if i % log_every == 0 or i == n_batches:
                elapsed = time.time() - t0
                eta = (elapsed / i) * (n_batches - i) if i > 0 else 0
                mode = f"MCx{mc_samples}" if mc_dropout else "1-pass"
                print(
                    f"\r  Eval [{mode}] [{i}/{n_batches}] {i/n_batches:>6.1%} | "
                    f"{elapsed:.0f}s / ETA {eta:.0f}s",
                    end="", flush=True,
                )

    print()
    if mc_dropout:
        model.eval()
    return all_labels, all_preds


def collect_predictions_from_probs(
    labels: list[int],
    probs: np.ndarray,
    run_dir: Path | None = None,
) -> list[int]:
    """Aplica umbrales guardados (logit biases) si existen en run_dir."""
    biases = load_thresholds(run_dir) if run_dir is not None else None
    if biases is not None:
        probs = apply_logit_biases(probs, biases)
    return probs_to_preds(probs)


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

def evaluate_model(
    run_name: str,
    split: str = "test",
    dataset: str = "oasis1",
    subset: int | None = None,
    model_name: str | None = None,
    weights_path: str | None = None,
    mc_dropout: bool = False,
    mc_samples: int = 30,
    tta: bool = False,
    use_thresholds: bool = True,
    preprocess_variant: str | None = None,
    source_filter: str | None = None,
    label_scheme: str | None = None,
) -> None:
    """
    Evalua el mejor modelo de un run sobre un split y genera reportes.

    Args:
        run_name: Nombre de la carpeta en outputs/ que contiene best_model.pth.
        split: Split a evaluar ('test' por defecto, tambien acepta 'val').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Limitar a N samples (para pruebas rapidas).
        model_name: Nombre del modelo. Si None, se lee del checkpoint.
        weights_path: Ruta al checkpoint del backbone para ultimate_fm. Si None, se usa la
                      guardada en el checkpoint de entrenamiento o el default de get_model.
        mc_dropout: Si True, inferencia con MC Dropout (Dropout activo, promedio de probabilidades).
        mc_samples: Numero de pases forward por batch cuando mc_dropout es True.
        tta: Test-Time Augmentation (promedio original + flip LR).
        use_thresholds: Si True y existe thresholds.json, aplica sesgos de logit.
        preprocess_variant: Variante de preprocesado; si None, se lee del checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model, checkpoint, run_dir = load_model_from_run(
        run_name, device=device, model_name=model_name, weights_path=weights_path,
    )
    resolved_model = model_name or checkpoint.get("model_name", "resnet10")
    use_clinical = checkpoint.get("use_clinical", False)
    uses_ordinal = checkpoint.get("uses_ordinal", False)
    variant = preprocess_variant or checkpoint.get(
        "preprocess_variant", cfg.PREPROCESS_VARIANT_DEFAULT,
    )
    roi_mode = checkpoint.get("roi_mode")
    spatial_size = checkpoint.get("spatial_size")
    if spatial_size is not None and not isinstance(spatial_size, tuple):
        spatial_size = tuple(spatial_size)
    resolved_scheme = label_scheme or checkpoint.get("label_scheme", "multiclass")
    scheme = get_scheme(resolved_scheme)
    class_names = get_class_names(resolved_scheme)
    num_classes = scheme.num_classes
    print(f"[INFO] Modelo: {resolved_model} | ordinal={uses_ordinal}")
    print(f"[INFO] Label scheme: {resolved_scheme} ({num_classes} clases)")
    print(f"[INFO] Preprocess variant: {variant}")
    if roi_mode:
        print(f"[INFO] ROI mode: {roi_mode}")
    if spatial_size and spatial_size != cfg.IMAGE_SIZE:
        print(f"[INFO] Spatial size: {spatial_size}")
    val_clinical_f2 = checkpoint.get("val_clinical_f2")
    ckpt_info = (
        f"[INFO] Modelo cargado desde epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_accuracy']:.2%}"
    )
    if val_clinical_f2 is not None:
        ckpt_info += f", val_clinical_f2={val_clinical_f2:.4f}"
    ckpt_info += ")"
    print(ckpt_info)

    loader = get_dataloader(
        split, shuffle=False, num_workers=0, dataset=dataset,
        subset=subset, use_clinical=use_clinical, variant=variant,
        source_filter=source_filter,
        label_scheme=resolved_scheme,
        roi_mode=roi_mode, spatial_size=spatial_size,
    )
    if mc_dropout:
        print(
            f"[INFO] MC Dropout activo: {mc_samples} muestras por batch "
            "(Dropout en train, resto en eval)"
        )
    if tta:
        print("[INFO] TTA activo: promedio imagen original + flip LR")
    if use_thresholds and load_thresholds(run_dir) is not None:
        print(f"[INFO] Umbrales cargados desde {run_dir / 'thresholds.json'}")
    if source_filter:
        print(f"[INFO] Filtro cohorte: source={source_filter}")
    print(f"[INFO] Evaluando sobre '{split}' ({len(loader.dataset)} muestras)...")

    all_labels, probs = collect_probabilities(
        model, loader, device,
        tta=tta, mc_dropout=mc_dropout, mc_samples=mc_samples,
    )
    all_preds = collect_predictions_from_probs(
        all_labels, probs, run_dir if use_thresholds else None,
    )

    # Métricas
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # F2 (β=2) por clase, macro y clinical
    f2_per_class = fbeta_score(
        all_labels, all_preds, beta=2,
        labels=list(range(num_classes)),
        average=None, zero_division=0,
    )
    macro_f2 = fbeta_score(
        all_labels, all_preds, beta=2,
        labels=list(range(num_classes)),
        average="macro", zero_division=0,
    )
    clinical_f2 = compute_clinical_f2_for_scheme(
        all_labels, all_preds, resolved_scheme,
    )

    f2_lines = [
        f"  {'Clase':<6}  {'F2 (β=2)':>10}",
        f"  {'-'*20}",
    ]
    for i, name in enumerate(class_names):
        f2_lines.append(f"  {name:<6}  {f2_per_class[i]:>10.4f}")
    f2_lines += [
        f"  {'-'*20}",
        f"  {'Macro':<6}  {macro_f2:>10.4f}",
        f"  {'Clinical':<6}  {clinical_f2:>10.4f}  (pesos: {scheme.clinical_f2_weights})",
    ]
    f2_report = "\n".join(f2_lines)

    # Métricas robustas al desbalance / ordinales / de coste (src/metrics.py)
    extra = metrics_from_confusion(cm)
    can_cost = num_classes == len(cfg.COST_MATRIX)
    extra_lines = [
        f"  {'Balanced Accuracy':<20} {extra['balanced_accuracy']:>8.4f}",
        f"  {'Quadratic W. Kappa':<20} {extra['quadratic_weighted_kappa']:>8.4f}   (acuerdo ordinal)",
        f"  {'Macro MAE':<20} {extra['macro_mae']:>8.4f}   (distancia ordinal, menor mejor)",
    ]
    if can_cost:
        extra_lines.append(
            f"  {'Expected Cost':<20} {extra['expected_cost']:>8.4f}   "
            f"(coste clínico, menor mejor; matriz {cfg.COST_MATRIX})"
        )
    else:
        extra_lines.append(f"  {'Expected Cost':<20} {'N/A':>8}   (COST_MATRIX no aplica a este esquema)")
    extra_report = "\n".join(extra_lines)

    # Mostrar resultados
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN — {split.upper()} SET ({run_name})")
    if mc_dropout:
        print(f"MC Dropout: {mc_samples} pases forward / batch (promedio de probabilidades)")
    print(f"{'=' * 60}")
    print(f"\nAccuracy global: {acc:.2%}")
    print(f"\n{report}")
    print(f"--- F2-Score (β=2) ---")
    print(f2_report)
    print(f"\n--- Métricas robustas / ordinales / de coste ---")
    print(extra_report)
    print(f"\nConfusion Matrix:")
    print(cm)

    suffix_parts = []
    if mc_dropout:
        suffix_parts.append("mcdropout")
    if tta:
        suffix_parts.append("tta")
    if use_thresholds and load_thresholds(run_dir) is not None:
        suffix_parts.append("thresh")
    if source_filter:
        suffix_parts.append(source_filter)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    report_path = run_dir / f"classification_report_{split}{suffix}.txt"
    mc_header = (
        f"MC Dropout: activo | mc_samples={mc_samples} (promedio de probabilidades por batch)\n"
        if mc_dropout
        else ""
    )
    report_text = (
        f"{'=' * 60}\n"
        f"EVALUACIÓN — {split.upper()} SET\n"
        f"Run: {run_name}\n"
        f"Modelo: epoch {checkpoint['epoch']}\n"
        f"{mc_header}"
        f"{'=' * 60}\n\n"
        f"Accuracy global: {acc:.2%}\n\n"
        f"{report}\n"
        f"--- F2-Score (β=2) ---\n"
        f"{f2_report}\n\n"
        f"--- Métricas robustas / ordinales / de coste ---\n"
        f"{extra_report}\n\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Reporte guardado en: {report_path}")

    # Guardar confusion matrix plot
    cm_path = run_dir / f"confusion_matrix_{split}{suffix}.png"
    _plot_confusion_matrix(cm, class_names, acc, cm_path)
    print(f"[OK] Gráfica guardada en: {cm_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluar modelo 3D sobre test/val set"
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Nombre de la carpeta en outputs/ (ej. full_100ep)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val"],
        help="Split a evaluar (default: test)",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis1",
        choices=["oasis1", "oasis3", "adni", "adni_baseline_sc", "oasis3_adni", "oasis3_adni_baseline"],
        help="Dataset a utilizar (default: oasis1)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limitar a N samples (para pruebas rapidas)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=AVAILABLE_MODELS,
        help="Modelo a usar (default: auto-detectar del checkpoint)",
    )
    parser.add_argument(
        "--weights-path", type=str, default=None,
        help="Ruta al checkpoint del backbone para ultimate_fm (default: del checkpoint o resenc_l_ssl3d.pth)",
    )
    parser.add_argument(
        "--mc-dropout", action="store_true",
        help="Inferencia con MC Dropout: promedia probabilidades sobre N pases (no reentrena)",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=30,
        metavar="N",
        help="Pases forward por batch con --mc-dropout (default: 30)",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Test-Time Augmentation: promedia probs original + flip LR",
    )
    parser.add_argument(
        "--no-thresholds", action="store_true",
        help="No aplicar thresholds.json aunque exista",
    )
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado (default: leer del checkpoint)",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        choices=["oasis3", "adni"],
        help="Filtrar filas del split fusionado por columna source",
    )
    parser.add_argument(
        "--label-scheme",
        type=str,
        default=None,
        choices=list(LABEL_SCHEME_CHOICES),
        help="Esquema de labels (default: del checkpoint o multiclass)",
    )
    args = parser.parse_args()

    evaluate_model(
        run_name=args.run,
        split=args.split,
        dataset=args.dataset,
        subset=args.subset,
        model_name=args.model,
        weights_path=args.weights_path,
        mc_dropout=args.mc_dropout,
        mc_samples=args.mc_samples,
        tta=args.tta,
        use_thresholds=not args.no_thresholds,
        preprocess_variant=args.preprocess_variant,
        source_filter=args.source_filter,
        label_scheme=args.label_scheme,
    )
