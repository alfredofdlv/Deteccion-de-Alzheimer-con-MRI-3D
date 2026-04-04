"""
evaluate.py — Evaluación del modelo sobre el test set con métricas detalladas (OASIS-1 / OASIS-3).

Genera en outputs/<run_name>/:
    - classification_report.txt  — precision, recall, F1 por clase
    - confusion_matrix.png       — heatmap de la matriz de confusión

Uso:
    python -m src.evaluate --run full_100ep
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
)

from src.config import cfg
from src.dataset import get_dataloader
from src.model import AVAILABLE_MODELS, get_model
from src.train import decode_preds


CLASS_NAMES = ["CN", "MCI", "AD"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Ejecuta inferencia y recopila todas las predicciones y labels."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)  # ~5% de los batches
    t0 = time.time()

    uses_clin = getattr(model, "uses_clinical", False)
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            clinical = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)
            outputs = model(images, clinical) if uses_clin else model(images)
            all_preds.extend(decode_preds(outputs, model).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

            if i % log_every == 0 or i == n_batches:
                elapsed = time.time() - t0
                eta = (elapsed / i) * (n_batches - i) if i > 0 else 0
                print(
                    f"\r  Eval [{i}/{n_batches}] {i/n_batches:>6.1%} | "
                    f"{elapsed:.0f}s / ETA {eta:.0f}s",
                    end="", flush=True,
                )

    print()
    return all_labels, all_preds


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

def evaluate_model(run_name: str, split: str = "test", dataset: str = "oasis1",
                   subset: int | None = None, model_name: str | None = None) -> None:
    """
    Evalua el mejor modelo de un run sobre un split y genera reportes.

    Args:
        run_name: Nombre de la carpeta en outputs/ que contiene best_model.pth.
        split: Split a evaluar ('test' por defecto, tambien acepta 'val').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Limitar a N samples (para pruebas rapidas).
        model_name: Nombre del modelo. Si None, se lee del checkpoint.
    """
    run_dir = cfg.OUTPUTS_DIR / run_name
    model_path = run_dir / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo: {model_path}\n"
            f"Asegurate de haber entrenado con --run {run_name}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    resolved_model = model_name or checkpoint.get("model_name", "resnet10")
    use_clinical = checkpoint.get("use_clinical", False)
    uses_ordinal = checkpoint.get("uses_ordinal", False)
    print(f"[INFO] Modelo: {resolved_model} | ordinal={uses_ordinal}")
    model = get_model(resolved_model, ordinal=uses_ordinal).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_clinical_f2 = checkpoint.get("val_clinical_f2")
    ckpt_info = (
        f"[INFO] Modelo cargado desde epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_accuracy']:.2%}"
    )
    if val_clinical_f2 is not None:
        ckpt_info += f", val_clinical_f2={val_clinical_f2:.4f}"
    ckpt_info += ")"
    print(ckpt_info)

    # Inferencia
    loader = get_dataloader(split, shuffle=False, num_workers=0, dataset=dataset,
                            subset=subset, use_clinical=use_clinical)
    print(f"[INFO] Evaluando sobre '{split}' ({len(loader.dataset)} muestras)...")

    all_labels, all_preds = collect_predictions(model, loader, device)

    # Métricas
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(cfg.NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.NUM_CLASSES)))

    # F2 (β=2) por clase, macro y clinical
    f2_per_class = fbeta_score(
        all_labels, all_preds, beta=2,
        labels=list(range(cfg.NUM_CLASSES)),
        average=None, zero_division=0,
    )
    macro_f2 = fbeta_score(
        all_labels, all_preds, beta=2,
        labels=list(range(cfg.NUM_CLASSES)),
        average="macro", zero_division=0,
    )
    clinical_f2 = sum(cfg.CLINICAL_F2_WEIGHTS[c] * f2_per_class[c] for c in range(cfg.NUM_CLASSES))

    f2_lines = [
        f"  {'Clase':<6}  {'F2 (β=2)':>10}",
        f"  {'-'*20}",
    ]
    for i, name in enumerate(CLASS_NAMES):
        f2_lines.append(f"  {name:<6}  {f2_per_class[i]:>10.4f}")
    f2_lines += [
        f"  {'-'*20}",
        f"  {'Macro':<6}  {macro_f2:>10.4f}",
        f"  {'Clinical':<6}  {clinical_f2:>10.4f}  (pesos: {cfg.CLINICAL_F2_WEIGHTS})",
    ]
    f2_report = "\n".join(f2_lines)

    # Mostrar resultados
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN — {split.upper()} SET ({run_name})")
    print(f"{'=' * 60}")
    print(f"\nAccuracy global: {acc:.2%}")
    print(f"\n{report}")
    print(f"--- F2-Score (β=2) ---")
    print(f2_report)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Guardar classification report
    report_path = run_dir / f"classification_report_{split}.txt"
    report_text = (
        f"{'=' * 60}\n"
        f"EVALUACIÓN — {split.upper()} SET\n"
        f"Run: {run_name}\n"
        f"Modelo: epoch {checkpoint['epoch']}\n"
        f"{'=' * 60}\n\n"
        f"Accuracy global: {acc:.2%}\n\n"
        f"{report}\n"
        f"--- F2-Score (β=2) ---\n"
        f"{f2_report}\n\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Reporte guardado en: {report_path}")

    # Guardar confusion matrix plot
    cm_path = run_dir / f"confusion_matrix_{split}.png"
    _plot_confusion_matrix(cm, CLASS_NAMES, acc, cm_path)
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
        choices=["oasis1", "oasis3"],
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
    args = parser.parse_args()

    evaluate_model(run_name=args.run, split=args.split, dataset=args.dataset,
                   subset=args.subset, model_name=args.model)
