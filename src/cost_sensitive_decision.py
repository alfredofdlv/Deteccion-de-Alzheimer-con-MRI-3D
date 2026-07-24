"""
cost_sensitive_decision.py — Decisión sensible al coste en inferencia (Fase 0b).

En lugar de decidir con argmax(P), aplica la REGLA DE DECISIÓN DE BAYES que minimiza
el riesgo esperado dado la matriz de coste C:

    pred(x) = argmin_j  sum_i  P(i|x) * C[i, j]

Como las CNN suelen estar sobreconfiadas, antes se CALIBRAN las probabilidades con
Temperature Scaling (Guo et al., 2017): se ajusta un único escalar T en el conjunto
de validación minimizando la NLL, y se aplica en test. Todo post-hoc, sin reentrenar.

Compara tres estrategias en test:
    1. argmax                 (baseline actual)
    2. Bayes-risk sin calibrar
    3. Bayes-risk calibrado (T ajustada en val)

Reporta por estrategia: matriz de confusión, Expected Cost, Balanced Accuracy,
Clinical F2 y recall por clase, además del ECE (calibración) antes/después.

Uso (CPU por defecto, para respetar la GPU 1 reservada):
    python -m src.cost_sensitive_decision --run densenet-cropped --dataset oasis3
    python -m src.cost_sensitive_decision --run densenet-cropped --dataset oasis3 --device cpu

Familias de coste (para Bayes-risk):
    --cost-families deepresearch|all|sym-linear,sym-quad,asym-5:1,...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.config import cfg
from src.dataset import get_dataloader
from src.inference_utils import (
    CLASS_NAMES,
    collect_probabilities,
    compute_clinical_f2_np,
    load_model_from_run,
)
from src.metrics import balanced_accuracy, expected_cost, quadratic_weighted_kappa


def make_cost_matrix(under: float, over: float, mode: str = "linear", n: int = 3) -> np.ndarray:
    """Construye una matriz de coste 3x3 con asimetría under/over y forma linear/quad."""
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            w = d if mode == "linear" else d * d
            if j < i:  # predicho menos severo -> infra-diagnóstico
                C[i, j] = under * w
            elif j > i:  # predicho más severo -> sobre-diagnóstico
                C[i, j] = over * w
    return C


def cost_families() -> dict[str, np.ndarray]:
    n = cfg.NUM_CLASSES
    return {
        "sym-linear": make_cost_matrix(1, 1, "linear", n=n),
        "sym-quad": make_cost_matrix(1, 1, "quad", n=n),
        "asym-2:1": make_cost_matrix(2, 1, "linear", n=n),
        "asym-5:1": make_cost_matrix(5, 1, "linear", n=n),
        "asym-10:1": make_cost_matrix(10, 1, "linear", n=n),
        "deepresearch": np.asarray(cfg.COST_MATRIX, dtype=np.float64),
    }


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + eps)))


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Re-escala en espacio logit: softmax(log(p)/T). T>1 suaviza (menos confianza)."""
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0)) / T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_probs: np.ndarray, val_labels: np.ndarray) -> float:
    """Ajuste 1D de T minimizando NLL en validación (búsqueda en rejilla).

    Rango amplio: con label smoothing el modelo queda infra-confiado y el óptimo
    puede estar en T<1 (afilar); con sobreconfianza clásica, en T>1 (suavizar).
    """
    grid = np.arange(0.05, 8.01, 0.05)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = _nll(_apply_temperature(val_probs, T), val_labels)
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE estándar con confianza = max prob."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def bayes_decision(probs: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """argmin_j sum_i P_i C_ij. probs (N,K), C (K,K) -> preds (N,)."""
    risk = probs @ cost_matrix          # (N, K): riesgo de cada predicción j
    return risk.argmin(axis=1)


def _metrics_block(name: str, labels: np.ndarray, preds: np.ndarray, C: np.ndarray) -> str:
    cm = confusion_matrix(labels, preds, labels=list(range(cfg.NUM_CLASSES)))
    ec = expected_cost(cm, C)
    ba = balanced_accuracy(cm)
    qwk = quadratic_weighted_kappa(cm)
    f2c = compute_clinical_f2_np(labels.tolist(), preds.tolist())
    recalls = []
    for i in range(cfg.NUM_CLASSES):
        s = cm[i].sum()
        recalls.append(cm[i, i] / s if s > 0 else 0.0)
    lines = [
        f"--- {name} ---",
        f"  Expected Cost : {ec:.4f}   (menor mejor)",
        f"  Balanced Acc  : {ba:.4f}",
        f"  QWK           : {qwk:.4f}",
        f"  Clinical F2   : {f2c:.4f}",
        f"  Recall CN/MCI/AD: {recalls[0]:.3f} / {recalls[1]:.3f} / {recalls[2]:.3f}",
        f"  Confusion:\n{cm}",
    ]
    return "\n".join(lines)


def run(
    run_name: str,
    dataset: str = "oasis3",
    device_str: str = "cpu",
    preprocess_variant: str | None = None,
    subset: int | None = None,
    cost_families_arg: str = "deepresearch",
) -> None:
    device = torch.device(device_str)
    model, checkpoint, run_dir = load_model_from_run(run_name, device=device)
    use_clinical = checkpoint.get("use_clinical", False)
    variant = preprocess_variant or checkpoint.get("preprocess_variant", "cropped")
    num_classes = checkpoint.get("num_classes", cfg.NUM_CLASSES)

    if num_classes != len(cfg.COST_MATRIX):
        raise ValueError(
            f"El run '{run_name}' tiene {num_classes} clases pero COST_MATRIX es "
            f"{len(cfg.COST_MATRIX)}x{len(cfg.COST_MATRIX)}. Solo multiclase (3) soportado."
        )

    all_families = cost_families()
    if cost_families_arg == "all":
        families = list(all_families.keys())
    else:
        families = [f.strip() for f in cost_families_arg.split(",") if f.strip()]
        unknown = [f for f in families if f not in all_families]
        if unknown:
            raise ValueError(f"Familias de coste desconocidas: {unknown}. Opciones: {list(all_families)}")

    print(f"[INFO] Run: {run_name} | dataset: {dataset} | variante: {variant} | device: {device}")
    print("[INFO] Extrayendo probabilidades en val (para calibrar) y test...")

    val_loader = get_dataloader("val", shuffle=False, num_workers=0, dataset=dataset,
                                subset=subset, use_clinical=use_clinical, variant=variant)
    test_loader = get_dataloader("test", shuffle=False, num_workers=0, dataset=dataset,
                                 subset=subset, use_clinical=use_clinical, variant=variant)

    val_labels, val_probs = collect_probabilities(model, val_loader, device)
    test_labels, test_probs = collect_probabilities(model, test_loader, device)
    val_labels = np.asarray(val_labels)
    test_labels = np.asarray(test_labels)

    T = fit_temperature(val_probs, val_labels)
    test_probs_cal = _apply_temperature(test_probs, T)

    ece_before = expected_calibration_error(test_probs, test_labels)
    ece_after = expected_calibration_error(test_probs_cal, test_labels)

    argmax_preds = test_probs.argmax(axis=1)
    header_lines = [
        "=" * 60,
        f"DECISIÓN SENSIBLE AL COSTE — TEST ({run_name})",
        "=" * 60,
        f"Temperature (val): T = {T:.3f}   (>1 = red sobreconfiada, se suaviza)",
        f"ECE test: {ece_before:.4f} (sin calibrar) -> {ece_after:.4f} (calibrado)",
        "",
    ]

    blocks: list[str] = []
    # Baseline: predicción por argmax (misma confusión para cualquier C)
    blocks.append(_metrics_block("1) argmax (baseline)", test_labels, argmax_preds, all_families["deepresearch"]))
    blocks.append("")

    for fam in families:
        C = all_families[fam]
        bayes_raw = bayes_decision(test_probs, C)
        bayes_cal = bayes_decision(test_probs_cal, C)
        blocks.append(_metrics_block(f"2) Bayes-risk sin calibrar ({fam})", test_labels, bayes_raw, C))
        blocks.append("")
        blocks.append(_metrics_block(f"3) Bayes-risk CALIBRADO (T) ({fam})", test_labels, bayes_cal, C))
        blocks.append("")

    report = "\n".join(header_lines + blocks).rstrip() + "\n"

    fam_slug = cost_families_arg.replace(":", "-").replace(",", "_")
    out_path = run_dir / f"cost_sensitive_decision_test_cost-families_{fam_slug}.txt"
    out_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"[OK] Reporte guardado en: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regla de Bayes + temperature scaling (post-hoc)")
    parser.add_argument("--run", type=str, required=True, help="Run con best_model.pth")
    parser.add_argument("--dataset", type=str, default="oasis3",
                        choices=["oasis1", "oasis3", "adni", "adni_baseline_sc",
                                 "oasis3_adni", "oasis3_adni_baseline"])
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu (recomendado; GPU 1 reservada) o cuda")
    parser.add_argument("--preprocess-variant", type=str, default=None,
                        choices=list(cfg.PREPROCESS_VARIANTS.keys()))
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument(
        "--cost-families",
        type=str,
        default="deepresearch",
        help="deepresearch|all|sym-linear,sym-quad,asym-5:1,... (se evalúa tras calibración de probabilidades)",
    )
    args = parser.parse_args()
    run(args.run, dataset=args.dataset, device_str=args.device,
        preprocess_variant=args.preprocess_variant, subset=args.subset, cost_families_arg=args.cost_families)
