"""
evaluate.py — Evaluación del modelo sobre el test set con métricas detalladas (OASIS-1 / OASIS-3).

Genera en outputs/<run_name>/:
    - classification_report.txt  — precision, recall, F1 por clase
    - confusion_matrix.png       — heatmap de la matriz de confusión

Uso:
    python -m src.evaluate --run full_100ep
"""

from __future__ import annotations

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
)

from src.config import cfg
from src.dataset import get_dataloader
from src.model import Simple3DCNN


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
    import time
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
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
                   subset: int | None = None) -> None:
    """
    Evalúa el mejor modelo de un run sobre un split y genera reportes.

    Args:
        run_name: Nombre de la carpeta en outputs/ que contiene best_model.pth.
        split: Split a evaluar ('test' por defecto, también acepta 'val').
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Limitar a N samples (para pruebas rapidas).
    """
    run_dir = cfg.OUTPUTS_DIR / run_name
    model_path = run_dir / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}\n"
            f"Asegúrate de haber entrenado con --run {run_name}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Cargar modelo
    model = Simple3DCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"[INFO] Modelo cargado desde epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_accuracy']:.2%})"
    )

    # Inferencia
    loader = get_dataloader(split, shuffle=False, num_workers=0, dataset=dataset, subset=subset)
    print(f"[INFO] Evaluando sobre '{split}' ({len(loader.dataset)} muestras)...")

    all_labels, all_preds = collect_predictions(model, loader, device)

    # Métricas
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.NUM_CLASSES)))

    # Mostrar resultados
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN — {split.upper()} SET ({run_name})")
    print(f"{'=' * 60}")
    print(f"\nAccuracy global: {acc:.2%}")
    print(f"\n{report}")
    print(f"Confusion Matrix:")
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
        description="Evaluar Simple3DCNN sobre test/val set"
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
    args = parser.parse_args()

    evaluate_model(run_name=args.run, split=args.split, dataset=args.dataset,
                   subset=args.subset)
