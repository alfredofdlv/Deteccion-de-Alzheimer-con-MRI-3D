"""
inference_pipeline.py — Pipeline consolidado: ensemble + TTA + umbrales (bloque A5).

Uso:
    python -m src.inference_pipeline --dataset oasis3
    python -m src.inference_pipeline --dataset oasis3 --runs densenet-cropped densenet-ordinal
"""

from __future__ import annotations

import argparse
import json

import torch

from src.config import cfg
from src.ensemble import DEFAULT_ENSEMBLE_RUNS, evaluate_ensemble


def run_consolidated_pipeline(
    run_names: list[str] | None = None,
    dataset: str = "oasis3",
    subset: int | None = None,
    output_name: str = "ensemble-tta-tuned",
    preprocess_variant: str | None = None,
) -> None:
    """
    Ejecuta ensemble con TTA y tuning de umbrales en val (configuración A5).
    """
    runs = run_names or DEFAULT_ENSEMBLE_RUNS
    print(f"[A5] Ensemble + TTA + umbrales | runs={runs}")
    evaluate_ensemble(
        run_names=runs,
        dataset=dataset,
        subset=subset,
        tta=True,
        tune_thresholds=True,
        output_name=output_name,
        preprocess_variant=preprocess_variant,
    )
    summary_path = cfg.OUTPUTS_DIR / output_name / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps({
            "pipeline": "A5",
            "runs": runs,
            "tta": True,
            "tune_thresholds": True,
            "dataset": dataset,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Resumen en {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline A5: ensemble + TTA + umbrales")
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=[
            "oasis1", "oasis3", "adni", "adni_baseline_sc",
            "oasis3_adni", "oasis3_adni_baseline",
        ],
    )
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="ensemble-tta-tuned")
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
    )
    args = parser.parse_args()
    run_consolidated_pipeline(
        run_names=args.runs,
        dataset=args.dataset,
        subset=args.subset,
        output_name=args.output_name,
        preprocess_variant=args.preprocess_variant,
    )
