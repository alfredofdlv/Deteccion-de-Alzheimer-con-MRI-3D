"""
train.py — Training loop para Simple3DCNN sobre OASIS-1.

Genera automáticamente en outputs/<run_name>/:
    - training_log.csv       — métricas por epoch
    - curves_loss.png        — gráfica de loss (train vs val)
    - curves_accuracy.png    — gráfica de accuracy (train vs val)
    - best_model.pth         — pesos del mejor modelo (menor val_loss)
    - training_summary.txt   — resumen legible del entrenamiento

Modos de ejecución:
    python -m src.train --overfit                   # sanity check
    python -m src.train --epochs 2 --run test_2ep   # prueba rápida
    python -m src.train --epochs 100 --run full     # entrenamiento largo
    python -m src.train --patience 15 --run exp1    # patience custom
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.config import cfg
from src.data_utils import load_split
from src.dataset import get_dataloader
from src.model import Simple3DCNN


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Para el entrenamiento si val_loss no mejora en `patience` epochs consecutivos."""

    def __init__(self, patience: int = cfg.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Retorna True si se debe parar el entrenamiento."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights() -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Formula: weight_i = N_total / (N_classes * N_i)
    """
    df = load_split("train")
    counts = df["label"].value_counts().sort_index()
    n_total = len(df)
    n_classes = cfg.NUM_CLASSES

    weights = []
    for c in range(n_classes):
        n_c = counts.get(c, 1)
        weights.append(n_total / (n_classes * n_c))

    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights: {w.tolist()}")
    return w


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Ejecuta un epoch de entrenamiento. Retorna dict con loss y accuracy medias."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evalúa el modelo sin gradientes. Retorna dict con loss y accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def _save_plots(history: list[dict], run_dir: Path) -> None:
    """Genera y guarda gráficas de loss y accuracy."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] * 100 for r in history]
    val_acc = [r["val_acc"] * 100 for r in history]

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "o-", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "s-", label="Val Loss", linewidth=2, markersize=4)
    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
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


def _save_summary(history: list[dict], run_dir: Path, device: torch.device,
                  n_params: int, elapsed_total: float, class_weights: list,
                  early_stopped: bool = False, patience: int = 0) -> None:
    """Genera un archivo de resumen legible."""
    best = min(history, key=lambda r: r["val_loss"])
    last = history[-1]

    stop_reason = f"Early stopping (patience={patience})" if early_stopped else "Completado"

    lines = [
        "=" * 60,
        "TRAINING SUMMARY",
        "=" * 60,
        f"Fecha:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device:             {device}",
        f"Parametros:         {n_params:,}",
        f"Class weights:      {class_weights}",
        f"Learning rate:      {cfg.LEARNING_RATE}",
        f"Batch size:         {cfg.BATCH_SIZE}",
        f"Image size:         {cfg.IMAGE_SIZE}",
        f"Epochs:             {len(history)}",
        f"Finalizacion:       {stop_reason}",
        f"Tiempo total:       {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)",
        f"Tiempo por epoch:   {elapsed_total/len(history):.1f}s",
        "",
        "--- Mejor Epoch ---",
        f"  Epoch:            {best['epoch']}",
        f"  Train Loss:       {best['train_loss']:.4f}",
        f"  Train Acc:        {best['train_acc']:.2%}",
        f"  Val Loss:         {best['val_loss']:.4f}",
        f"  Val Acc:          {best['val_acc']:.2%}",
        "",
        "--- Ultimo Epoch ---",
        f"  Epoch:            {last['epoch']}",
        f"  Train Loss:       {last['train_loss']:.4f}",
        f"  Train Acc:        {last['train_acc']:.2%}",
        f"  Val Loss:         {last['val_loss']:.4f}",
        f"  Val Acc:          {last['val_acc']:.2%}",
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
) -> None:
    """
    Función principal de entrenamiento.

    Args:
        num_epochs: Número máximo de epochs.
        overfit_one_batch: Si True, entrena solo con 4 imágenes durante
                          100 epochs (sanity check de convergencia).
        run_name: Nombre de la carpeta dentro de outputs/ para esta ejecución.
                  Si None, se genera uno automático con timestamp.
        patience: Epochs sin mejora en val_loss antes de early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    torch.manual_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    model = Simple3DCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parametros entrenables: {n_params:,}")

    class_weights = compute_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # ── Overfit-one-batch mode ───────────────────────────────────────────
    if overfit_one_batch:
        print("\n" + "=" * 60)
        print("MODO OVERFIT-ONE-BATCH (sanity check)")
        print("=" * 60)

        loader = get_dataloader("train", batch_size=4, num_workers=0, shuffle=False)
        single_batch = next(iter(loader))
        images = single_batch["image"].to(device)
        labels = single_batch["label"].to(device)
        print(f"Batch labels: {labels.tolist()}")

        num_epochs = 100
        model.train()
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{num_epochs} — "
                    f"Loss: {loss.item():.4f}  Acc: {acc:.2%}"
                )

        print(f"\n  Final — Loss: {loss.item():.6f}  Acc: {acc:.2%}")
        if loss.item() < 0.05 and acc == 1.0:
            print("  === CHECKPOINT 3.2 PASSED ===")
        else:
            print("  [WARN] No convergio completamente. Revisar el modelo.")
        return

    # ── Entrenamiento completo ───────────────────────────────────────────
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience)

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO — max {num_epochs} epochs (early stopping: patience={patience})")
    print(f"Resultados en: {run_dir}")
    print("=" * 60)

    train_loader = get_dataloader("train", num_workers=0)
    val_loader = get_dataloader("val", num_workers=0)
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # CSV log
    csv_path = run_dir / "training_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s", "is_best"],
    )
    csv_writer.writeheader()

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        is_best = val_metrics["loss"] < best_val_loss

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "epoch_time_s": round(elapsed, 1),
            "is_best": is_best,
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        best_tag = ""
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_accuracy": val_metrics["accuracy"],
            }, run_dir / "best_model.pth")
            best_tag = "  *best*"

        print(
            f"Epoch {epoch:3d}/{num_epochs} ({elapsed:.0f}s) — "
            f"Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.2%}  |  "
            f"Val Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.2%}"
            f"{best_tag}"
        )

        if early_stopper.step(val_metrics["loss"]):
            print(
                f"\n[EARLY STOPPING] Val loss no mejoro en {patience} epochs. "
                f"Mejor epoch: {best_epoch} (val_loss={best_val_loss:.4f})"
            )
            break

    csv_file.close()
    elapsed_total = time.time() - t_start

    # Generar graficas y resumen
    _save_plots(history, run_dir)
    _save_summary(
        history, run_dir, device, n_params, elapsed_total,
        class_weights.cpu().tolist(),
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

    parser = argparse.ArgumentParser(description="Entrenar Simple3DCNN sobre OASIS-1")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help=f"Numero de epochs (default: {cfg.NUM_EPOCHS})")
    parser.add_argument("--overfit", action="store_true",
                        help="Modo overfit-one-batch (sanity check)")
    parser.add_argument("--run", type=str, default=None,
                        help="Nombre de la carpeta de resultados (default: run_TIMESTAMP)")
    args = parser.parse_args()

    train(num_epochs=args.epochs, overfit_one_batch=args.overfit, run_name=args.run)
