"""
cascade_two_stage.py — Cascada de dos modelos entrenados por separado.

Etapa 1 (binary_cn_imp): CN vs deterioro (MCI+AD).
Etapa 2 (binary_mci_ad): MCI vs AD, entrenado solo sobre deterioro.

Decodificación en test (3 clases):
  - Si P(CN) >= t_cn → CN
  - Si no → etapa 2: P(AD) >= t_ad → AD, si no MCI

Los umbrales t_cn y t_ad se ajustan en val maximizando clinical F2 (3 clases).

Uso:
    python -m src.cascade_two_stage \\
        --stage1-run densenet-oasis3-adni-mni-n4-bin-cn-imp \\
        --stage2-run densenet-oasis3-adni-mni-n4-bin-mci-ad \\
        --dataset oasis3_adni --preprocess-variant mni_n4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import cfg
from src.inference_utils import (
    CLASS_NAMES,
    collect_probabilities,
    compute_clinical_f2_np,
    load_model_from_run,
)
from src.label_scheme import compute_clinical_f2_for_scheme
from src.dataset import get_dataloader


def cascade_decode(
    probs_stage1: np.ndarray,
    probs_stage2: np.ndarray,
    t_cn: float,
    t_ad: float,
) -> list[int]:
    """
    Decodifica probabilidades binarias a etiquetas multiclase {0=CN, 1=MCI, 2=AD}.

    probs_stage1: (N, 2) con [P(CN), P(Impaired)]
    probs_stage2: (N, 2) con [P(MCI), P(AD)]
    """
    preds: list[int] = []
    for i, (p_cn, _p_imp) in enumerate(probs_stage1):
        if p_cn >= t_cn:
            preds.append(0)
            continue
        p_mci, p_ad = probs_stage2[i]
        preds.append(2 if p_ad >= t_ad else 1)
    return preds


def grid_search_cascade(
    labels: list[int],
    probs_stage1: np.ndarray,
    probs_stage2: np.ndarray,
    t_cn_range: tuple[float, float, float] = (0.3, 0.9, 0.05),
    t_ad_range: tuple[float, float, float] = (0.3, 0.7, 0.05),
) -> tuple[float, float, float]:
    lo_cn, hi_cn, step_cn = t_cn_range
    lo_ad, hi_ad, step_ad = t_ad_range
    best_f2 = -1.0
    best_t_cn, best_t_ad = 0.5, 0.5

    for t_cn in np.arange(lo_cn, hi_cn + step_cn * 0.5, step_cn):
        for t_ad in np.arange(lo_ad, hi_ad + step_ad * 0.5, step_ad):
            preds = cascade_decode(
                probs_stage1, probs_stage2, float(t_cn), float(t_ad),
            )
            f2 = compute_clinical_f2_np(labels, preds)
            if f2 > best_f2:
                best_f2 = f2
                best_t_cn, best_t_ad = float(t_cn), float(t_ad)

    return best_t_cn, best_t_ad, best_f2


def _collect_stage_probs(
    run_name: str,
    split: str,
    dataset: str,
    device: torch.device,
    variant: str,
    subset: int | None = None,
    tta: bool = False,
    *,
    for_cascade: bool = False,
) -> np.ndarray:
    """Inferencia binaria. Con for_cascade=True usa el split completo (sin filtrar CN)."""
    model, checkpoint, _ = load_model_from_run(run_name, device=device)
    use_clinical = checkpoint.get("use_clinical", False)
    label_scheme = checkpoint.get("label_scheme", "multiclass")
    loader_scheme = "multiclass" if for_cascade else label_scheme
    loader = get_dataloader(
        split,
        shuffle=False,
        num_workers=0,
        dataset=dataset,
        subset=subset,
        use_clinical=use_clinical,
        variant=variant,
        label_scheme=loader_scheme,
    )
    _, probs = collect_probabilities(model, loader, device, tta=tta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def _multiclass_labels(
    split: str,
    dataset: str,
    variant: str,
    subset: int | None = None,
) -> list[int]:
    loader = get_dataloader(
        split,
        shuffle=False,
        num_workers=0,
        dataset=dataset,
        subset=subset,
        variant=variant,
        label_scheme="multiclass",
    )
    labels: list[int] = []
    for batch in loader:
        labels.extend(batch["label"].tolist())
    return labels


def evaluate_cascade(
    stage1_run: str,
    stage2_run: str,
    dataset: str = "oasis3_adni",
    preprocess_variant: str = "mni_n4",
    subset: int | None = None,
    tta: bool = False,
    output_name: str = "cascade-two-stage-mni-n4",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.OUTPUTS_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Etapa 1: {stage1_run}")
    print(f"[INFO] Etapa 2: {stage2_run}")
    print(f"[INFO] Dataset: {dataset} | variant: {preprocess_variant}")

    val_labels = _multiclass_labels("val", dataset, preprocess_variant, subset=subset)
    val_p1 = _collect_stage_probs(
        stage1_run, "val", dataset, device, preprocess_variant,
        subset=subset, tta=tta, for_cascade=True,
    )
    val_p2 = _collect_stage_probs(
        stage2_run, "val", dataset, device, preprocess_variant,
        subset=subset, tta=tta, for_cascade=True,
    )
    if len(val_labels) != len(val_p1) or len(val_labels) != len(val_p2):
        raise RuntimeError(
            f"Tamaños inconsistentes en val: labels={len(val_labels)}, "
            f"stage1={len(val_p1)}, stage2={len(val_p2)}"
        )

    t_cn, t_ad, val_f2 = grid_search_cascade(val_labels, val_p1, val_p2)
    print(f"[INFO] Val — t_cn={t_cn:.2f}, t_ad={t_ad:.2f}, F2c={val_f2:.4f}")

    test_labels = _multiclass_labels("test", dataset, preprocess_variant, subset=subset)
    test_p1 = _collect_stage_probs(
        stage1_run, "test", dataset, device, preprocess_variant,
        subset=subset, tta=tta, for_cascade=True,
    )
    test_p2 = _collect_stage_probs(
        stage2_run, "test", dataset, device, preprocess_variant,
        subset=subset, tta=tta, for_cascade=True,
    )
    test_preds = cascade_decode(test_p1, test_p2, t_cn, t_ad)
    test_f2 = compute_clinical_f2_np(test_labels, test_preds)

    stage1_bin_labels = [0 if y == 0 else 1 for y in test_labels]
    stage1_bin_preds = [0 if row[0] >= t_cn else 1 for row in test_p1]
    stage1_f2_binary = compute_clinical_f2_for_scheme(
        stage1_bin_labels, stage1_bin_preds, "binary_cn_imp",
    )

    config = {
        "stage1_run": stage1_run,
        "stage2_run": stage2_run,
        "dataset": dataset,
        "preprocess_variant": preprocess_variant,
        "t_cn": t_cn,
        "t_ad": t_ad,
        "val_f2_cascade": val_f2,
        "test_f2_cascade": test_f2,
        "test_f2_stage1_binary": stage1_f2_binary,
        "tta": tta,
        "baselines": {
            "flat_multiclass_f2c": 0.5889,
            "hierarchical_decode_f2c": 0.596,
            "best_cnn_peak_f2c": 0.6121,
        },
    }
    config_path = out_dir / "cascade_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(cfg.NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report_test.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(
        test_labels, test_preds, labels=list(range(cfg.NUM_CLASSES)),
    )
    cm_lines = [
        "Confusion matrix (rows=true, cols=pred):",
        f"       {' '.join(f'{n:>6}' for n in CLASS_NAMES)}",
    ]
    for i, name in enumerate(CLASS_NAMES):
        cm_lines.append(f"{name:>6} " + " ".join(f"{cm[i, j]:>6}" for j in range(3)))
    (out_dir / "confusion_matrix_test.txt").write_text(
        "\n".join(cm_lines) + "\n", encoding="utf-8",
    )

    summary = (
        f"Cascada dos modelos — test F2c={test_f2:.4f}\n"
        f"  t_cn={t_cn:.2f}, t_ad={t_ad:.2f}\n"
        f"  val F2c={val_f2:.4f}\n"
        f"  stage1 binario F2c (CN vs imp): {stage1_f2_binary:.4f}\n"
    )
    (out_dir / "cascade_summary.txt").write_text(summary, encoding="utf-8")

    print("\n" + "=" * 60)
    print(summary.strip())
    print(f"Resultados en: {out_dir}")
    print("=" * 60)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación cascada de dos modelos binarios (CN vs imp → MCI vs AD)",
    )
    parser.add_argument(
        "--stage1-run",
        default="densenet-oasis3-adni-mni-n4-bin-cn-imp",
        help="Run etapa 1 (binary_cn_imp)",
    )
    parser.add_argument(
        "--stage2-run",
        default="densenet-oasis3-adni-mni-n4-bin-mci-ad",
        help="Run etapa 2 (binary_mci_ad)",
    )
    parser.add_argument("--dataset", default="oasis3_adni")
    parser.add_argument("--preprocess-variant", default="mni_n4")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--output-name",
        default="cascade-two-stage-mni-n4",
    )
    args = parser.parse_args()

    evaluate_cascade(
        stage1_run=args.stage1_run,
        stage2_run=args.stage2_run,
        dataset=args.dataset,
        preprocess_variant=args.preprocess_variant,
        subset=args.subset,
        tta=args.tta,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
