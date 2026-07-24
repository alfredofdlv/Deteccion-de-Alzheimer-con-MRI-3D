"""
crt_train.py — Classifier Re-Training (cRT / decoupling Kang et al.).

Fase 2: congela el encoder, reentrena solo la cabeza con batches balanceados
(WeightedRandomSampler) y CE sin pesos de clase.

Uso:
    python -m src.crt_train \\
        --source-run densenet-oasis3-adni-mni \\
        --run densenet-oasis3-adni-mni-crt \\
        --dataset oasis3_adni --preprocess-variant mni
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src.config import cfg
from src.dataset import get_dataloader
from src.inference_utils import load_model_from_run
from src.label_scheme import get_scheme
from src.train import EarlyStopping, evaluate, train_one_epoch


def _freeze_encoder(model: nn.Module, model_name: str) -> int:
    """Congela representación; deja entrenable solo cabeza (+ atención en 2.5D)."""
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_name == "efficientnet25d":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.avgpool.parameters():
            p.requires_grad = False
    elif model_name in ("densenet121", "densenet121_coral"):
        net = getattr(model, "net", model)
        for p in net.features.parameters():
            p.requires_grad = False
        if hasattr(net, "class_layers"):
            for name, p in net.class_layers.named_parameters():
                if name.startswith("out"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        if getattr(model, "uses_coral", False):
            for p in model.coral_linear.parameters():
                p.requires_grad = True
            model.coral_biases.requires_grad = True
    elif model_name == "resnet10":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc") or name.startswith("classifier")
    else:
        raise ValueError(
            f"cRT no implementado para {model_name!r}. "
            "Soportado: densenet121, densenet121_coral, efficientnet25d, resnet10"
        )

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] cRT freeze: {trainable_before:,} → {trainable_after:,} params entrenables"
    )
    return trainable_after


def _head_parameters(model: nn.Module, model_name: str) -> list[nn.Parameter]:
    if model_name == "efficientnet25d":
        return (
            list(model.attention.parameters())
            + list(model.classifier.parameters())
        )
    if model_name in ("densenet121", "densenet121_coral"):
        params = []
        net = getattr(model, "net", model)
        if hasattr(net, "class_layers"):
            params += list(net.class_layers.out.parameters())
        if getattr(model, "uses_coral", False):
            params += list(model.coral_linear.parameters())
            if model.coral_biases.requires_grad:
                params.append(model.coral_biases)
        return [p for p in params if p.requires_grad]
    return [p for p in model.parameters() if p.requires_grad]


def train_crt(
    source_run: str,
    run_name: str,
    dataset: str = "oasis3_adni",
    preprocess_variant: str = "mni",
    num_epochs: int | None = None,
    patience: int | None = None,
    roi_mode: str | None = None,
    aug_mode: str = "none",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = num_epochs or cfg.CRT_NUM_EPOCHS
    patience = patience or cfg.CRT_PATIENCE
    scheme = get_scheme("multiclass")

    model, checkpoint, _ = load_model_from_run(source_run, device=device)
    model_name = checkpoint.get("model_name", "densenet121")
    label_scheme = checkpoint.get("label_scheme", "multiclass")
    variant = preprocess_variant or checkpoint.get("preprocess_variant", "mni")
    roi_mode = roi_mode or checkpoint.get("roi_mode")
    spatial_size = checkpoint.get("spatial_size")
    if spatial_size is not None and not isinstance(spatial_size, tuple):
        spatial_size = tuple(spatial_size)

    print(f"[INFO] cRT source: {source_run} | modelo: {model_name}")
    print(f"[INFO] Dataset: {dataset} | variant: {variant} | aug: {aug_mode}")

    _freeze_encoder(model, model_name)
    head_params = _head_parameters(model, model_name)
    if not head_params:
        raise RuntimeError("No hay parámetros de cabeza entrenables tras freeze")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        head_params, lr=cfg.CRT_HEAD_LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    batch_size = (
        cfg.EFFICIENTNET25D_BATCH_SIZE
        if model_name == "efficientnet25d"
        else cfg.BATCH_SIZE
    )
    train_loader = get_dataloader(
        "train", dataset=dataset, variant=variant, aug_mode=aug_mode,
        balanced_sampler=True, batch_size=batch_size,
        roi_mode=roi_mode, spatial_size=spatial_size,
        label_scheme=label_scheme,
    )
    val_loader = get_dataloader(
        "val", dataset=dataset, variant=variant, aug_mode="none",
        batch_size=batch_size, roi_mode=roi_mode, spatial_size=spatial_size,
        label_scheme=label_scheme,
    )

    run_dir = cfg.OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    early_stopper = EarlyStopping(patience=patience)
    csv_path = run_dir / "crt_training_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch", "train_loss", "train_acc", "train_f2", "train_clinical_f2",
            "val_loss", "val_acc", "val_f2", "val_clinical_f2", "is_best",
        ],
    )
    writer.writeheader()

    best_val_clinical_f2 = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, None, device,
            label_scheme=label_scheme,
        )
        val_m = evaluate(
            model, val_loader, criterion, device, label_scheme=label_scheme,
        )
        is_best = val_m["clinical_f2"] > best_val_clinical_f2
        if is_best:
            best_val_clinical_f2 = val_m["clinical_f2"]
            best_epoch = epoch
            ckpt = {
                **checkpoint,
                "epoch": epoch,
                "crt_source_run": source_run,
                "crt_stage": 2,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_m["loss"],
                "val_accuracy": val_m["accuracy"],
                "val_macro_f2": val_m["macro_f2"],
                "val_clinical_f2": best_val_clinical_f2,
            }
            torch.save(ckpt, run_dir / "best_model.pth")

        writer.writerow({
            "epoch": epoch,
            "train_loss": round(train_m["loss"], 6),
            "train_acc": round(train_m["accuracy"], 6),
            "train_f2": round(train_m["macro_f2"], 6),
            "train_clinical_f2": round(train_m["clinical_f2"], 6),
            "val_loss": round(val_m["loss"], 6),
            "val_acc": round(val_m["accuracy"], 6),
            "val_f2": round(val_m["macro_f2"], 6),
            "val_clinical_f2": round(val_m["clinical_f2"], 6),
            "is_best": is_best,
        })
        csv_file.flush()

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train acc {train_m['accuracy']:.2%} F2c {train_m['clinical_f2']:.4f} | "
            f"val acc {val_m['accuracy']:.2%} F2c {val_m['clinical_f2']:.4f}"
            + (" << BEST" if is_best else "")
        )
        scheduler.step(val_m["loss"])
        if early_stopper.step(val_m["clinical_f2"]):
            print(f"[INFO] Early stopping en epoch {epoch}")
            break

    csv_file.close()
    summary = run_dir / "crt_summary.txt"
    summary.write_text(
        f"cRT stage 2\n"
        f"source_run: {source_run}\n"
        f"model: {model_name}\n"
        f"dataset: {dataset}\n"
        f"best_epoch: {best_epoch}\n"
        f"best_val_clinical_f2: {best_val_clinical_f2:.4f}\n"
        f"finished: {datetime.now().isoformat()}\n",
        encoding="utf-8",
    )
    print(f"[OK] cRT guardado en {run_dir} (best ep {best_epoch}, val F2c {best_val_clinical_f2:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifier Re-Training (cRT fase 2)")
    parser.add_argument("--source-run", required=True, help="Run con encoder preentrenado")
    parser.add_argument("--run", required=True, help="Nombre salida cRT")
    parser.add_argument("--dataset", default="oasis3_adni")
    parser.add_argument("--preprocess-variant", default="mni", choices=list(cfg.PREPROCESS_VARIANTS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--aug", default="none", choices=["full", "light", "none"])
    parser.add_argument("--roi", default=None, choices=["mtl"])
    args = parser.parse_args()

    train_crt(
        source_run=args.source_run,
        run_name=args.run,
        dataset=args.dataset,
        preprocess_variant=args.preprocess_variant,
        num_epochs=args.epochs,
        patience=args.patience,
        aug_mode=args.aug,
        roi_mode=args.roi,
    )


if __name__ == "__main__":
    main()
