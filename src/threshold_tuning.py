"""
threshold_tuning.py — Optimiza sesgos de logit en validación para maximizar Clinical F2.

Uso:
    python -m src.threshold_tuning --run densenet-cropped --dataset oasis3
    python -m src.threshold_tuning --run densenet-cropped --dataset oasis3 --tta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

from src.config import cfg
from src.dataset import get_dataloader
from src.inference_utils import (
    CLASS_NAMES,
    apply_logit_biases,
    collect_probabilities,
    compute_clinical_f2_np,
    grid_search_logit_biases,
    load_model_from_run,
    probs_to_preds,
    save_thresholds,
)


def tune_and_evaluate(
    run_name: str,
    dataset: str = "oasis3",
    subset: int | None = None,
    tta: bool = False,
    bias_step: float = 0.25,
    output_run: str | None = None,
    preprocess_variant: str | None = None,
) -> None:
    """
    Busca umbrales en val, guarda thresholds.json y evalúa en test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, run_dir = load_model_from_run(run_name, device=device)
    use_clinical = checkpoint.get("use_clinical", False)
    variant = preprocess_variant or checkpoint.get("preprocess_variant", "cropped")
    out_dir = cfg.OUTPUTS_DIR / (output_run or run_name)

    if subset is not None:
        print(f"[WARN] --subset={subset}: resultados no representativos; "
              "no usar thresholds.json para produccion sin re-ejecutar en val completo.")

    for split in ("val", "test"):
        loader = get_dataloader(
            split, shuffle=False, num_workers=0, dataset=dataset,
            subset=subset, use_clinical=use_clinical, variant=variant,
        )
        labels, probs = collect_probabilities(model, loader, device, tta=tta)
        if split == "val":
            biases, val_f2 = grid_search_logit_biases(
                labels, probs, bias_range=(-1.0, 1.0, bias_step),
            )
            save_thresholds(
                out_dir, biases, val_f2,
                extra={"run": run_name, "dataset": dataset, "tta": tta},
            )
            print(f"[OK] Umbrales guardados en {out_dir / 'thresholds.json'}")
            print(f"     Val Clinical F2 (con umbrales): {val_f2:.4f}")
            print(f"     logit_biases: {biases.tolist()}")
            val_labels, val_probs = labels, probs
        else:
            adj = apply_logit_biases(probs, biases)
            preds = probs_to_preds(adj)
            test_f2 = compute_clinical_f2_np(labels, preds)
            argmax_preds = probs.argmax(axis=1).tolist()
            baseline_f2 = compute_clinical_f2_np(labels, argmax_preds)

            report = classification_report(
                labels, preds,
                labels=list(range(cfg.NUM_CLASSES)),
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
            cm = confusion_matrix(
                labels, preds, labels=list(range(cfg.NUM_CLASSES)),
            )
            f2_per = fbeta_score(
                labels, preds, beta=2,
                labels=list(range(cfg.NUM_CLASSES)),
                average=None, zero_division=0,
            )

            report_path = out_dir / f"classification_report_{split}_tuned.txt"
            text = (
                f"Threshold tuning — test\n"
                f"Run: {run_name}\n"
                f"TTA: {tta}\n"
                f"logit_biases: {biases.tolist()}\n"
                f"Test Clinical F2 (argmax): {baseline_f2:.4f}\n"
                f"Test Clinical F2 (tuned):  {test_f2:.4f}\n\n"
                f"{report}\n\nConfusion Matrix:\n{cm}\n"
            )
            report_path.write_text(text, encoding="utf-8")
            print(f"\n{'=' * 60}")
            print(f"TEST — Clinical F2 argmax: {baseline_f2:.4f} -> tuned: {test_f2:.4f}")
            print(f"Per-class F2: CN={f2_per[0]:.4f} MCI={f2_per[1]:.4f} AD={f2_per[2]:.4f}")
            print(f"[OK] Reporte: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning de umbrales (logit biases) en val")
    parser.add_argument("--run", type=str, required=True, help="Run con best_model.pth")
    parser.add_argument("--dataset", type=str, default="oasis3",
                        choices=["oasis1", "oasis3", "adni", "oasis3_adni", "oasis3_adni_baseline"])
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--tta", action="store_true", help="TTA al extraer probabilidades")
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado (default: leer del checkpoint)",
    )
    parser.add_argument("--bias-step", type=float, default=0.25, help="Paso del grid de sesgos")
    parser.add_argument(
        "--output-run", type=str, default=None,
        help="Carpeta outputs donde guardar thresholds.json (default: mismo --run)",
    )
    args = parser.parse_args()
    tune_and_evaluate(
        run_name=args.run,
        dataset=args.dataset,
        subset=args.subset,
        tta=args.tta,
        bias_step=args.bias_step,
        output_run=args.output_run,
        preprocess_variant=args.preprocess_variant,
    )
