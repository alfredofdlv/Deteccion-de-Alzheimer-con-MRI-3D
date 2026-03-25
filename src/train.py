"""
train.py — Training loop para AlzheimerResNet sobre OASIS-1 / OASIS-3.

Metrica de seleccion de modelo: Clinical F1-Score (ponderado: 60% AD, 30% MCI, 10% CN).
Ademas de penalizacion asimetrica en la loss (multiplicadores clinicos sobre class weights).

Genera automaticamente en outputs/<run_name>/:
    - training_log.csv       — metricas por epoch (incluye macro F1 y clinical F1)
    - curves_loss.png        — grafica de loss (train vs val)
    - curves_accuracy.png    — grafica de accuracy (train vs val)
    - curves_f1.png          — grafica de clinical F1 (train vs val)
    - best_model.pth         — pesos del mejor modelo (mayor val clinical F1)
    - training_summary.txt   — resumen legible del entrenamiento

Modos de ejecucion:
    python -m src.train --overfit                   # sanity check
    python -m src.train --epochs 2 --run test_2ep   # prueba rapida
    python -m src.train --epochs 100 --run full     # entrenamiento largo
    python -m src.train --patience 15 --run exp1    # patience custom
"""

from __future__ import annotations

import csv
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

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
from src.model import AVAILABLE_MODELS, get_model
from monai.losses import FocalLoss


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Para el entrenamiento si val macro F1 no mejora en `patience` epochs."""

    def __init__(self, patience: int = cfg.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_f1 = 0.0
        self.counter = 0
        self.triggered = False

    def step(self, val_f1: float) -> bool:
        """Retorna True si se debe parar el entrenamiento."""
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: str = "oasis1") -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    Formula: weight_i = N_total / (N_classes * N_i)
    """
    df = load_split("train", dataset=dataset)
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
# Clinical F2 metric (β=2: recall pesa el doble que precision)
# ---------------------------------------------------------------------------

def compute_clinical_f2(labels: list[int], preds: list[int]) -> float:
    """F2 (β=2) ponderado con prioridad clinica: 60% AD, 30% MCI, 10% CN."""
    f2_per_class = fbeta_score(labels, preds, beta=2, average=None, zero_division=0)
    weights = cfg.CLINICAL_F2_WEIGHTS
    return sum(weights[c] * f2_per_class[c] for c in range(len(f2_per_class)))


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
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Ejecuta un epoch de entrenamiento. Retorna metricas incluyendo clinical_f2."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 100)  # ~1% de los batches
    t0 = time.time()

    uses_clin = getattr(model, "uses_clinical", False)
    for i, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        clinical = batch.get("clinical")
        if clinical is not None:
            clinical = clinical.to(device)

        optimizer.zero_grad()
        outputs = model(images, clinical) if uses_clin else model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if i % log_every == 0 or i == n_batches:
            _progress_log(i, n_batches, t0, "Train", running_loss, correct, total)

    macro_f2 = fbeta_score(all_labels, all_preds, beta=2, average="macro", zero_division=0)
    clin_f2 = compute_clinical_f2(all_labels, all_preds)
    print()
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "macro_f2": macro_f2,
        "clinical_f2": clin_f2,
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evalua el modelo sin gradientes. Retorna metricas incluyendo clinical_f2."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)  # ~5% de los batches
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
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if i % log_every == 0 or i == n_batches:
                _progress_log(i, n_batches, t0, "Val  ", running_loss, correct, total)

    macro_f2 = fbeta_score(all_labels, all_preds, beta=2, average="macro", zero_division=0)
    clin_f2 = compute_clinical_f2(all_labels, all_preds)
    print()
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "macro_f2": macro_f2,
        "clinical_f2": clin_f2,
    }


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
                  model_name: str = "resnet10") -> None:
    """Genera un archivo de resumen legible."""
    best = max(history, key=lambda r: r["val_clinical_f2"])
    last = history[-1]

    stop_reason = f"Early stopping (patience={patience})" if early_stopped else "Completado"

    lines = [
        "=" * 60,
        "TRAINING SUMMARY",
        "=" * 60,
        f"Fecha:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device:             {device}",
        f"Modelo:             {model_name}",
        f"Parametros:         {n_params:,}",
        f"Class weights:      {class_weights}",
        f"Learning rate:      {cfg.LEARNING_RATE}",
        f"Weight decay (L2):  {cfg.WEIGHT_DECAY}",
        f"Batch size:         {cfg.BATCH_SIZE}",
        f"Image size:         {cfg.IMAGE_SIZE}",
        f"Epochs:             {len(history)}",
        f"Finalizacion:       {stop_reason}",
        f"Metrica seleccion:  Clinical F2 β=2 (pesos: {cfg.CLINICAL_F2_WEIGHTS})",
        f"Tiempo total:       {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)",
        f"Tiempo por epoch:   {elapsed_total/len(history):.1f}s",
        "",
        "--- Mejor Epoch (por clinical F2) ---",
        f"  Epoch:               {best['epoch']}",
        f"  Train Loss:          {best['train_loss']:.4f}",
        f"  Train Acc:           {best['train_acc']:.2%}",
        f"  Train F2 (macro):    {best['train_f2']:.4f}",
        f"  Train F2 (clinical): {best['train_clinical_f2']:.4f}",
        f"  Val Loss:            {best['val_loss']:.4f}",
        f"  Val Acc:             {best['val_acc']:.2%}",
        f"  Val F2 (macro):      {best['val_f2']:.4f}",
        f"  Val F2 (clinical):   {best['val_clinical_f2']:.4f}",
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
) -> None:
    """
    Funcion principal de entrenamiento.

    Args:
        num_epochs: Numero maximo de epochs.
        overfit_one_batch: Si True, entrena solo con 4 imagenes durante
                          100 epochs (sanity check de convergencia).
        run_name: Nombre de la carpeta dentro de outputs/ para esta ejecucion.
                  Si None, se genera uno automatico con timestamp.
        patience: Epochs sin mejora en val clinical F1 antes de early stopping.
        dataset: Identificador del dataset ('oasis1' o 'oasis3').
        subset: Limitar cada split a N samples (para pruebas rapidas).
        model_name: Nombre del modelo.
        use_clinical: Si True, pasa covariables clínicas al modelo.
                      Por defecto True para 'multimodal_densenet', False para el resto.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    torch.manual_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    if use_clinical is None:
        use_clinical = (model_name == "multimodal_densenet")

    model = get_model(model_name).to(device)
    print(f"[INFO] Modelo: {model_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parametros entrenables: {n_params:,}")

    class_weights = compute_class_weights(dataset=dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # criterion = FocalLoss(weight=class_weights, gamma=2.0, to_onehot_y=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # -- Overfit-one-batch mode -----------------------------------------------
    if overfit_one_batch:
        print("\n" + "=" * 60)
        print("MODO OVERFIT-ONE-BATCH (sanity check)")
        print("=" * 60)

        loader = get_dataloader("train", batch_size=4, num_workers=0, shuffle=False,
                                dataset=dataset, subset=subset, use_clinical=use_clinical)
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
            optimizer.zero_grad()
            outputs = model(images, ob_clinical) if _uses_clin else model(images)
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
            print("  === CHECKPOINT PASSED ===")
        else:
            print("  [WARN] No convergio completamente. Revisar el modelo.")
        return

    # -- Entrenamiento completo -----------------------------------------------
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience)

    transforms_desc = describe_transforms("train")
    (run_dir / "transforms_config.txt").write_text(
        transforms_desc + "\n", encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO — max {num_epochs} epochs (early stopping: patience={patience})")
    print(f"Metrica de seleccion: Clinical F2 β=2 (pesos: {cfg.CLINICAL_F2_WEIGHTS})")
    print(f"Resultados en: {run_dir}")
    print("=" * 60)

    t_load = time.time()
    if subset:
        print(f"[INFO] Modo subset: limitando a {subset} samples por split")
    print("[INFO] Cargando datos de entrenamiento...")
    train_loader = get_dataloader("train", dataset=dataset, subset=subset, use_clinical=use_clinical)
    print(f"[INFO] Cargando datos de validacion...")
    val_loader = get_dataloader("val", dataset=dataset, subset=subset, use_clinical=use_clinical)
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
            "epoch_time_s", "is_best",
        ],
    )
    csv_writer.writeheader()

    history: list[dict] = []
    best_val_clinical_f2 = 0.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        is_best = val_metrics["clinical_f2"] > best_val_clinical_f2

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
            "epoch_time_s": round(elapsed, 1),
            "is_best": is_best,
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        if is_best:
            best_val_clinical_f2 = val_metrics["clinical_f2"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "use_clinical": use_clinical,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f2": val_metrics["macro_f2"],
                "val_clinical_f2": best_val_clinical_f2,
            }, run_dir / "best_model.pth")

        should_stop = early_stopper.step(val_metrics["clinical_f2"])
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
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n--- Epoch {epoch}/{num_epochs} "
            f"({elapsed:.0f}s | total: {elapsed_str} | ETA: {eta_str}) ---\n"
            f"  Train  ->  Loss: {train_metrics['loss']:.4f}  |  Acc: {train_metrics['accuracy']:.2%}  "
            f"|  F2m: {train_metrics['macro_f2']:.4f}  |  F2c: {train_metrics['clinical_f2']:.4f}\n"
            f"  Val    ->  Loss: {val_metrics['loss']:.4f}  |  Acc: {val_metrics['accuracy']:.2%}  "
            f"|  F2m: {val_metrics['macro_f2']:.4f}  |  F2c: {val_metrics['clinical_f2']:.4f}{best_marker}\n"
            f"  LR: {current_lr:.2e}  |  Best: epoch {best_epoch} (clin_f2={best_val_clinical_f2:.4f})  "
            f"| Early stop: {es_bar} {es_counter}/{patience}"
        )

        if should_stop:
            print(
                f"\n{'=' * 60}\n"
                f"[EARLY STOPPING] Val clinical F2 no mejoro en {patience} epochs.\n"
                f"Mejor epoch: {best_epoch} (clin_f2={best_val_clinical_f2:.4f})\n"
                f"{'=' * 60}"
            )
            break

    csv_file.close()
    elapsed_total = time.time() - t_start

    _save_plots(history, run_dir)
    _save_summary(
        history, run_dir, device, n_params, elapsed_total,
        class_weights.cpu().tolist(),
        early_stopped=early_stopper.triggered,
        patience=patience,
        model_name=model_name,
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
                        choices=["oasis1", "oasis3"],
                        help="Dataset a utilizar (default: oasis1)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limitar a N samples por split (para pruebas rapidas)")
    parser.add_argument("--model", type=str, default="resnet10",
                        choices=AVAILABLE_MODELS,
                        help="Modelo a usar (default: resnet10)")
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        overfit_one_batch=args.overfit,
        run_name=args.run,
        patience=args.patience,
        dataset=args.dataset,
        subset=args.subset,
        model_name=args.model,
    )
