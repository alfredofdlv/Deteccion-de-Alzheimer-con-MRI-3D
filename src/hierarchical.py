"""
hierarchical.py — Clasificación jerárquica en inferencia (CN vs deterioro → MCI vs AD).

Etapa 1: si P(CN) >= t_cn → CN; si no, etapa 2.
Etapa 2: entre MCI/AD, predice AD si P(AD)/(P(MCI)+P(AD)) >= t_ad.

Las probabilidades pueden venir de un solo run o del ensemble.

Uso:
    python -m src.hierarchical --dataset oasis3
    python -m src.hierarchical --prob-source ensemble --dataset oasis3
    python -m src.hierarchical --run densenet-cropped --dataset oasis3
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

from src.config import cfg
from src.ensemble import DEFAULT_ENSEMBLE_RUNS, collect_ensemble_probs
from src.inference_utils import (
    CLASS_NAMES,
    collect_probabilities,
    compute_clinical_f2_np,
    load_model_from_run,
    probs_to_preds,
)
from src.dataset import default_variant_for_dataset, get_dataloader


def hierarchical_decode(
    probs: np.ndarray,
    t_cn: float,
    t_ad: float,
) -> list[int]:
    """
    Decodifica (N,3) probabilidades a etiquetas {0,1,2}.

    t_cn: umbral sobre P(CN); por encima → CN.
    t_ad: umbral sobre P(AD | no-CN); por encima → AD, si no MCI.
    """
    preds: list[int] = []
    for row in probs:
        p_cn, p_mci, p_ad = row
        if p_cn >= t_cn:
            preds.append(0)
            continue
        denom = p_mci + p_ad
        if denom < 1e-8:
            preds.append(1)
        elif p_ad / denom >= t_ad:
            preds.append(2)
        else:
            preds.append(1)
    return preds


def grid_search_hierarchical(
    labels: list[int],
    probs: np.ndarray,
    t_cn_range: tuple[float, float, float] = (0.3, 0.9, 0.05),
    t_ad_range: tuple[float, float, float] = (0.3, 0.7, 0.05),
) -> tuple[float, float, float]:
    lo_cn, hi_cn, step_cn = t_cn_range
    lo_ad, hi_ad, step_ad = t_ad_range
    best_f2 = -1.0
    best_t_cn, best_t_ad = 0.5, 0.5

    for t_cn in np.arange(lo_cn, hi_cn + step_cn * 0.5, step_cn):
        for t_ad in np.arange(lo_ad, hi_ad + step_ad * 0.5, step_ad):
            preds = hierarchical_decode(probs, float(t_cn), float(t_ad))
            f2 = compute_clinical_f2_np(labels, preds)
            if f2 > best_f2:
                best_f2 = f2
                best_t_cn, best_t_ad = float(t_cn), float(t_ad)

    return best_t_cn, best_t_ad, best_f2


def evaluate_hierarchical(
    dataset: str = "oasis3",
    run_name: str | None = None,
    ensemble_runs: list[str] | None = None,
    prob_source: str = "ensemble",
    subset: int | None = None,
    tta: bool = False,
    output_name: str = "hierarchical-decode",
    preprocess_variant: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.OUTPUTS_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = ensemble_runs or DEFAULT_ENSEMBLE_RUNS

    variant = preprocess_variant or default_variant_for_dataset(dataset)

    def get_probs(split: str) -> tuple[list[int], np.ndarray]:
        if prob_source == "ensemble":
            return collect_ensemble_probs(
                runs, split, dataset, device, subset=subset, tta=tta,
                preprocess_variant=variant,
            )
        if run_name is None:
            raise ValueError("--run requerido si prob_source=single")
        model, checkpoint, _ = load_model_from_run(run_name, device=device)
        use_clinical = checkpoint.get("use_clinical", False)
        loader = get_dataloader(
            split, shuffle=False, num_workers=0, dataset=dataset,
            subset=subset, use_clinical=use_clinical, variant=variant,
        )
        return collect_probabilities(model, loader, device, tta=tta)

    val_labels, val_probs = get_probs("val")
    t_cn, t_ad, val_f2 = grid_search_hierarchical(val_labels, val_probs)
    flat_f2 = compute_clinical_f2_np(val_labels, probs_to_preds(val_probs))

    test_labels, test_probs = get_probs("test")
    test_preds = hierarchical_decode(test_probs, t_cn, t_ad)
    test_f2 = compute_clinical_f2_np(test_labels, test_preds)
    test_flat = compute_clinical_f2_np(test_labels, probs_to_preds(test_probs))

    config = {
        "prob_source": prob_source,
        "runs": runs if prob_source == "ensemble" else [run_name],
        "t_cn": t_cn,
        "t_ad": t_ad,
        "val_f2_flat": flat_f2,
        "val_f2_hierarchical": val_f2,
        "test_f2_flat": test_flat,
        "test_f2_hierarchical": test_f2,
        "tta": tta,
    }
    (out_dir / "hierarchical_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    report = classification_report(
        test_labels, test_preds,
        labels=list(range(cfg.NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(cfg.NUM_CLASSES)))
    (out_dir / "classification_report_test_hierarchical.txt").write_text(
        f"Hierarchical decode\n{json.dumps(config, indent=2)}\n\n{report}\n\n{cm}\n",
        encoding="utf-8",
    )

    print(f"[OK] Umbrales: t_cn={t_cn:.3f} t_ad={t_ad:.3f}")
    print(f"     Val F2c flat={flat_f2:.4f} -> hierarchical={val_f2:.4f}")
    print(f"     Test F2c flat={test_flat:.4f} -> hierarchical={test_f2:.4f}")
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificación jerárquica (inferencia)")
    parser.add_argument(
        "--prob-source", choices=["ensemble", "single"], default="ensemble",
    )
    parser.add_argument("--run", type=str, default=None, help="Run si prob-source=single")
    parser.add_argument("--runs", nargs="+", default=None, help="Runs para ensemble")
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=[
            "oasis1", "oasis3", "adni", "adni_baseline_sc",
            "oasis3_adni", "oasis3_adni_baseline",
        ],
    )
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output-name", type=str, default="hierarchical-decode")
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
    )
    args = parser.parse_args()
    evaluate_hierarchical(
        dataset=args.dataset,
        run_name=args.run,
        ensemble_runs=args.runs,
        prob_source=args.prob_source,
        subset=args.subset,
        tta=args.tta,
        output_name=args.output_name,
        preprocess_variant=args.preprocess_variant,
    )
