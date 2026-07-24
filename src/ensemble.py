"""
ensemble.py — Ensemble por promedio de probabilidades de varios checkpoints.

Uso:
    python -m src.ensemble --runs densenet-cropped densenet-ordinal resnet10-recrop --dataset oasis3
    python -m src.ensemble --runs densenet-cropped densenet-ordinal --dataset oasis3 --tune-thresholds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

from src.config import cfg
from src.dataset import default_variant_for_dataset, get_dataloader
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


DEFAULT_ENSEMBLE_RUNS = ["densenet-cropped", "densenet-ordinal", "resnet10-recrop"]


def collect_ensemble_probs(
    run_names: list[str],
    split: str,
    dataset: str,
    device: torch.device,
    subset: int | None = None,
    tta: bool = False,
    weights: list[float] | None = None,
    preprocess_variant: str | None = None,
) -> tuple[list[int], np.ndarray]:
    """Promedia probabilidades 3-clase de varios runs (mismo orden de muestras)."""
    if weights is None:
        weights = [1.0 / len(run_names)] * len(run_names)
    else:
        w = np.array(weights, dtype=np.float64)
        weights = (w / w.sum()).tolist()

    labels: list[int] | None = None
    avg_probs: np.ndarray | None = None
    variant = preprocess_variant or default_variant_for_dataset(dataset)

    for run_name, w in zip(run_names, weights):
        model, checkpoint, _ = load_model_from_run(run_name, device=device)
        use_clinical = checkpoint.get("use_clinical", False)
        roi_mode = checkpoint.get("roi_mode")
        spatial_size = checkpoint.get("spatial_size")
        if spatial_size is not None and not isinstance(spatial_size, tuple):
            spatial_size = tuple(spatial_size)
        run_variant = checkpoint.get("preprocess_variant", variant)
        loader = get_dataloader(
            split, shuffle=False, num_workers=0, dataset=dataset,
            subset=subset, use_clinical=use_clinical, variant=run_variant,
            roi_mode=roi_mode, spatial_size=spatial_size,
        )
        batch_labels, probs = collect_probabilities(model, loader, device, tta=tta)
        if labels is None:
            labels = batch_labels
        elif labels != batch_labels:
            raise ValueError(f"Labels distintos entre runs; revisar split {split}")
        if avg_probs is None:
            avg_probs = w * probs
        else:
            avg_probs += w * probs

    assert labels is not None and avg_probs is not None
    return labels, avg_probs


def evaluate_ensemble(
    run_names: list[str],
    dataset: str = "oasis3",
    subset: int | None = None,
    tta: bool = False,
    tune_thresholds: bool = True,
    output_name: str = "ensemble-baseline",
    weights: list[float] | None = None,
    preprocess_variant: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.OUTPUTS_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "runs": run_names,
        "dataset": dataset,
        "tta": tta,
        "weights": weights,
    }
    (out_dir / "ensemble_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    # Val: umbrales
    val_labels, val_probs = collect_ensemble_probs(
        run_names, "val", dataset, device, subset=subset, tta=tta, weights=weights,
        preprocess_variant=preprocess_variant,
    )
    biases = np.zeros(3, dtype=np.float64)
    val_f2_tuned = compute_clinical_f2_np(val_labels, probs_to_preds(val_probs))
    if tune_thresholds:
        biases, val_f2_tuned = grid_search_logit_biases(val_labels, val_probs)
        save_thresholds(out_dir, biases, val_f2_tuned, extra=meta)
        print(f"[OK] Ensemble val Clinical F2 (tuned): {val_f2_tuned:.4f}")

    val_preds_argmax = probs_to_preds(val_probs)
    val_f2_argmax = compute_clinical_f2_np(val_labels, val_preds_argmax)
    print(f"[INFO] Ensemble val Clinical F2 (argmax): {val_f2_argmax:.4f}")

    # Test
    test_labels, test_probs = collect_ensemble_probs(
        run_names, "test", dataset, device, subset=subset, tta=tta, weights=weights,
        preprocess_variant=preprocess_variant,
    )
    test_adj = apply_logit_biases(test_probs, biases) if tune_thresholds else test_probs
    test_preds = probs_to_preds(test_adj)
    test_f2 = compute_clinical_f2_np(test_labels, test_preds)
    test_f2_argmax = compute_clinical_f2_np(test_labels, probs_to_preds(test_probs))

    report = classification_report(
        test_labels, test_preds,
        labels=list(range(cfg.NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(cfg.NUM_CLASSES)))
    f2_per = fbeta_score(
        test_labels, test_preds, beta=2,
        labels=list(range(cfg.NUM_CLASSES)),
        average=None, zero_division=0,
    )

    report_path = out_dir / "classification_report_test_ensemble.txt"
    report_path.write_text(
        f"Ensemble test\nRuns: {run_names}\nTTA: {tta}\n"
        f"Test F2c argmax: {test_f2_argmax:.4f}\n"
        f"Test F2c tuned:  {test_f2:.4f}\n"
        f"logit_biases: {biases.tolist()}\n\n{report}\n\n{cm}\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE TEST — F2c argmax: {test_f2_argmax:.4f} | tuned: {test_f2:.4f}")
    print(f"  CN={f2_per[0]:.4f}  MCI={f2_per[1]:.4f}  AD={f2_per[2]:.4f}")
    print(f"[OK] {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble de checkpoints CNN")
    parser.add_argument(
        "--runs", nargs="+", default=DEFAULT_ENSEMBLE_RUNS,
        help="Runs a promediar (default: densenet-cropped densenet-ordinal resnet10-recrop)",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=[
            "oasis1", "oasis3", "adni", "adni_baseline_sc",
            "oasis3_adni", "oasis3_adni_baseline",
        ],
    )
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-tune-thresholds", action="store_true")
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado (default: auto segun dataset)",
    )
    parser.add_argument(
        "--output-name", type=str, default="ensemble-baseline",
        help="Carpeta en outputs/ para artefactos del ensemble",
    )
    parser.add_argument(
        "--model-weights", nargs="+", type=float, default=None,
        help="Pesos por run (misma longitud que --runs)",
    )
    args = parser.parse_args()
    evaluate_ensemble(
        run_names=args.runs,
        dataset=args.dataset,
        subset=args.subset,
        tta=args.tta,
        tune_thresholds=not args.no_tune_thresholds,
        output_name=args.output_name,
        weights=args.model_weights,
        preprocess_variant=args.preprocess_variant,
    )
