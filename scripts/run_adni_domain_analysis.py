"""
run_adni_domain_analysis.py — Análisis de brecha de dominio OASIS-3 → ADNI.

Fase 1 del plan de validación externa:
  - Baseline argmax (sin umbrales)
  - 1a: corrección analítica de label-shift (oráculo + EM Saerens)
  - 1b: recalibración de umbrales en ADNI (grid-search en val, eval en test)

Uso:
    python scripts/run_adni_domain_analysis.py --run densenet-cropped
    python scripts/run_adni_domain_analysis.py --run densenet-mni --preprocess-variant mni
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.data_utils import load_split
from src.dataset import get_dataloader
from src.evaluate import _plot_confusion_matrix
from src.inference_utils import (
    CLASS_NAMES,
    analytical_prior_biases,
    apply_logit_biases,
    class_priors_from_labels,
    collect_probabilities,
    compute_clinical_f2_np,
    grid_search_logit_biases,
    load_model_from_run,
    probs_to_preds,
    saerens_estimate_target_prior,
)


def _metrics(labels: list[int], preds: list[int]) -> dict:
    acc = accuracy_score(labels, preds)
    f2_per = fbeta_score(
        labels, preds, beta=2,
        labels=list(range(cfg.NUM_CLASSES)),
        average=None, zero_division=0,
    )
    macro_f2 = fbeta_score(
        labels, preds, beta=2,
        labels=list(range(cfg.NUM_CLASSES)),
        average="macro", zero_division=0,
    )
    clinical_f2 = compute_clinical_f2_np(labels, preds)
    cm = confusion_matrix(labels, preds, labels=list(range(cfg.NUM_CLASSES)))
    report = classification_report(
        labels, preds,
        labels=list(range(cfg.NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    recalls = {}
    recall_per_class = []
    for i in range(cfg.NUM_CLASSES):
        row_sum = cm[i].sum()
        recall_per_class.append(float(cm[i, i] / row_sum) if row_sum > 0 else 0.0)

    return {
        "accuracy": float(acc),
        "macro_f2": float(macro_f2),
        "clinical_f2": float(clinical_f2),
        "f2_per_class": {CLASS_NAMES[i]: float(f2_per[i]) for i in range(cfg.NUM_CLASSES)},
        "recall_per_class": {CLASS_NAMES[i]: recall_per_class[i] for i in range(cfg.NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _priors_from_csv(split_csv: str) -> np.ndarray:
    df = pd.read_csv(cfg.DATA_SPLITS_DIR / split_csv)
    return class_priors_from_labels(df["label"].tolist())


def _format_method_block(name: str, m: dict, biases: list[float] | None = None) -> str:
    lines = [
        f"### {name}",
        f"Accuracy: {m['accuracy']:.2%}",
        f"Macro F2: {m['macro_f2']:.4f}",
        f"Clinical F2: {m['clinical_f2']:.4f}",
    ]
    if biases is not None:
        lines.append(f"logit_biases: {biases}")
    lines.append("Recall por clase:")
    for cls in CLASS_NAMES:
        lines.append(f"  {cls}: {m['recall_per_class'][cls]:.4f}")
    lines.append(f"\n{m['classification_report']}")
    lines.append(f"Confusion Matrix:\n{np.array(m['confusion_matrix'])}")
    return "\n".join(lines)


def run_analysis(
    run_name: str,
    model_name: str | None = None,
    preprocess_variant: str | None = None,
    bias_step: float = 0.25,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.OUTPUTS_DIR / "adni-domain-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, run_dir = load_model_from_run(
        run_name, device=device, model_name=model_name,
    )
    variant = preprocess_variant or checkpoint.get(
        "preprocess_variant", cfg.PREPROCESS_VARIANT_DEFAULT,
    )

    pi_source = _priors_from_csv("oasis3_train_pt.csv")
    pi_adni_all = _priors_from_csv("adni_train_pt.csv")
    # Combinar train+val+test para oráculo ADNI
    adni_all_df = load_split("all", dataset="adni", variant=variant)
    pi_adni_oracle = class_priors_from_labels(adni_all_df["label"].tolist())

    print(f"[INFO] pi_source (OASIS-3 train): {pi_source.round(4).tolist()}")
    print(f"[INFO] pi_target oráculo (ADNI all): {pi_adni_oracle.round(4).tolist()}")

    results: dict = {
        "run": run_name,
        "preprocess_variant": variant,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pi_source": pi_source.tolist(),
        "pi_adni_oracle": pi_adni_oracle.tolist(),
        "methods": {},
    }

    split_data: dict[str, tuple[list[int], np.ndarray]] = {}
    for split in ("train", "val", "test", "all"):
        loader = get_dataloader(
            split, shuffle=False, num_workers=0,
            dataset="adni", variant=variant,
        )
        labels, probs = collect_probabilities(model, loader, device)
        split_data[split] = (labels, probs)
        print(f"[INFO] ADNI {split}: n={len(labels)}")

    # --- Baseline argmax (sin umbrales) ---
    for eval_split in ("all", "test"):
        labels, probs = split_data[eval_split]
        preds = probs.argmax(axis=1).tolist()
        key = f"baseline_argmax_{eval_split}"
        results["methods"][key] = _metrics(labels, preds)

    # --- 1a oráculo: corrección analítica con priors ADNI conocidos ---
    biases_oracle = analytical_prior_biases(pi_source, pi_adni_oracle)
    for eval_split in ("all", "test"):
        labels, probs = split_data[eval_split]
        adj = apply_logit_biases(probs, biases_oracle)
        preds = probs_to_preds(adj)
        key = f"prior_oracle_{eval_split}"
        results["methods"][key] = _metrics(labels, preds)
    results["biases_oracle"] = biases_oracle.tolist()

    # --- 1a Saerens EM: estimar prior en ADNI all (sin etiquetas en inferencia) ---
    _, probs_all = split_data["all"]
    pi_saerens = saerens_estimate_target_prior(probs_all, pi_source)
    biases_saerens = analytical_prior_biases(pi_source, pi_saerens)
    print(f"[INFO] pi_target Saerens EM: {pi_saerens.round(4).tolist()}")
    results["pi_saerens"] = pi_saerens.tolist()
    results["biases_saerens"] = biases_saerens.tolist()

    for eval_split in ("all", "test"):
        labels, probs = split_data[eval_split]
        adj = apply_logit_biases(probs, biases_saerens)
        preds = probs_to_preds(adj)
        key = f"prior_saerens_{eval_split}"
        results["methods"][key] = _metrics(labels, preds)

    # --- 1b: grid-search en ADNI val, aplicar a test ---
    val_labels, val_probs = split_data["val"]
    train_labels, train_probs = split_data["train"]
    cal_labels = val_labels + train_labels
    cal_probs = np.vstack([val_probs, train_probs])

    biases_grid, val_f2_grid = grid_search_logit_biases(
        cal_labels, cal_probs, bias_range=(-2.0, 2.0, bias_step),
    )
    # También buscar solo en val (más conservador, reportable en test)
    biases_grid_valonly, val_f2_valonly = grid_search_logit_biases(
        val_labels, val_probs, bias_range=(-2.0, 2.0, bias_step),
    )

    adni_thresh_path = out_dir / f"thresholds_adni_{run_name}.json"
    thresh_payload = {
        "logit_biases": biases_grid_valonly.tolist(),
        "val_clinical_f2": float(val_f2_valonly),
        "run": run_name,
        "dataset": "adni",
        "calibration_split": "val_only",
        "note": "Grid-search clinical F2 en adni_val; aplicar a adni_test",
    }
    adni_thresh_path.write_text(json.dumps(thresh_payload, indent=2), encoding="utf-8")

    results["biases_grid_trainval"] = biases_grid.tolist()
    results["biases_grid_valonly"] = biases_grid_valonly.tolist()
    results["val_clinical_f2_grid_trainval"] = float(val_f2_grid)
    results["val_clinical_f2_grid_valonly"] = float(val_f2_valonly)

    for label, biases in (
        ("grid_trainval", biases_grid),
        ("grid_valonly", biases_grid_valonly),
    ):
        test_labels, test_probs = split_data["test"]
        adj = apply_logit_biases(test_probs, biases)
        preds = probs_to_preds(adj)
        results["methods"][f"threshold_{label}_test"] = _metrics(test_labels, preds)

        all_labels, all_probs = split_data["all"]
        adj_all = apply_logit_biases(all_probs, biases)
        preds_all = probs_to_preds(adj_all)
        results["methods"][f"threshold_{label}_all"] = _metrics(all_labels, preds_all)

    # Brecha: referencia OASIS-3 test (mismo run)
    try:
        oasis_loader = get_dataloader(
            "test", shuffle=False, num_workers=0,
            dataset="oasis3", variant=variant,
        )
        oasis_labels, oasis_probs = collect_probabilities(model, oasis_loader, device)
        oasis_preds = oasis_probs.argmax(axis=1).tolist()
        results["oasis3_test_reference"] = _metrics(oasis_labels, oasis_preds)
    except FileNotFoundError as e:
        print(f"[WARN] No se pudo evaluar referencia OASIS-3: {e}")

    # Guardar JSON
    json_path = out_dir / f"domain_analysis_{run_name}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Reporte legible
    baseline_f2c = results["methods"]["baseline_argmax_all"]["clinical_f2"]
    oracle_f2c = results["methods"]["prior_oracle_all"]["clinical_f2"]
    saerens_f2c = results["methods"]["prior_saerens_all"]["clinical_f2"]
    grid_test_f2c = results["methods"]["threshold_grid_valonly_test"]["clinical_f2"]
    oasis_f2c = results.get("oasis3_test_reference", {}).get("clinical_f2")

    gap_prior_oracle = oracle_f2c - baseline_f2c
    gap_remaining_oracle = (oasis_f2c - oracle_f2c) if oasis_f2c else None

    report_lines = [
        "=" * 70,
        f"ANÁLISIS DE DOMINIO ADNI — {run_name}",
        f"Fecha: {results['timestamp']}",
        f"Preprocess variant: {variant}",
        "=" * 70,
        "",
        "## Priors",
        f"  pi_source (OASIS-3 train): {pi_source.round(4).tolist()}",
        f"  pi_target oráculo (ADNI):  {pi_adni_oracle.round(4).tolist()}",
        f"  pi_target Saerens EM:      {pi_saerens.round(4).tolist()}",
        "",
        "## Brecha cuantificada (clinical F2)",
        f"  OASIS-3 test (referencia interna): {oasis_f2c:.4f}" if oasis_f2c else "  OASIS-3 test: N/A",
        f"  ADNI all baseline (argmax):          {baseline_f2c:.4f}",
        f"  ADNI all + prior oráculo (1a):       {oracle_f2c:.4f}  (Δ prior: {gap_prior_oracle:+.4f})",
        f"  ADNI all + prior Saerens (1a):       {saerens_f2c:.4f}",
        f"  ADNI test + grid val-only (1b):      {grid_test_f2c:.4f}",
    ]
    if gap_remaining_oracle is not None:
        report_lines.append(
            f"  Brecha restante (OASIS test - ADNI oráculo): {gap_remaining_oracle:.4f}"
        )

    oracle_preds = probs_to_preds(
        apply_logit_biases(split_data["all"][1], biases_oracle),
    )
    oracle_ad_frac = float(np.mean(np.array(oracle_preds) == 2))
    if oracle_ad_frac > 0.9:
        report_lines.append(
            f"\nADVERTENCIA: corrección oráculo predice AD en {oracle_ad_frac:.0%} "
            "(sobre-corrección del prior; preferir grid-search val-only para despliegue)."
        )
    report_lines.append("")
    report_lines.append("NOTA: thresholds.json de OASIS tiene biases uniformes [-1,-1,-1] (no-op).")
    report_lines.append("")

    for method_key in sorted(results["methods"].keys()):
        m = results["methods"][method_key]
        biases = None
        if "oracle" in method_key:
            biases = results["biases_oracle"]
        elif "saerens" in method_key:
            biases = results["biases_saerens"]
        elif "grid_valonly" in method_key:
            biases = results["biases_grid_valonly"]
        elif "grid_trainval" in method_key:
            biases = results["biases_grid_trainval"]
        report_lines.append(_format_method_block(method_key, m, biases))
        report_lines.append("")

    report_path = out_dir / f"domain_analysis_{run_name}.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Confusion matrices para métodos clave
    for method_key in (
        "baseline_argmax_all",
        "prior_oracle_all",
        "prior_saerens_all",
        "threshold_grid_valonly_test",
    ):
        if method_key not in results["methods"]:
            continue
        m = results["methods"][method_key]
        cm = np.array(m["confusion_matrix"])
        cm_path = out_dir / f"cm_{run_name}_{method_key}.png"
        _plot_confusion_matrix(cm, CLASS_NAMES, m["accuracy"], cm_path)

    print(f"\n[OK] JSON: {json_path}")
    print(f"[OK] Reporte: {report_path}")
    print(f"[OK] Umbrales ADNI: {adni_thresh_path}")
    print(f"\nResumen clinical F2:")
    print(f"  Baseline ADNI all:     {baseline_f2c:.4f}")
    print(f"  + Prior oráculo:       {oracle_f2c:.4f}  (Δ {gap_prior_oracle:+.4f})")
    print(f"  + Prior Saerens:       {saerens_f2c:.4f}")
    print(f"  Grid val→test:         {grid_test_f2c:.4f}")
    if oasis_f2c:
        print(f"  OASIS-3 test ref:      {oasis_f2c:.4f}")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis brecha dominio OASIS→ADNI")
    parser.add_argument("--run", type=str, default="densenet-cropped")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
    )
    parser.add_argument("--bias-step", type=float, default=0.25)
    args = parser.parse_args()

    run_analysis(
        run_name=args.run,
        model_name=args.model,
        preprocess_variant=args.preprocess_variant,
        bias_step=args.bias_step,
    )


if __name__ == "__main__":
    main()
