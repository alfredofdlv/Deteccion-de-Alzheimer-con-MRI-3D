"""
metrics.py — Métricas robustas al desbalance, ordinales y sensibles al coste.

Todas las funciones principales operan sobre la matriz de confusión (numpy array
NxN con filas = clase real, columnas = clase predicha), de modo que pueden
calcularse *post-hoc* a partir de las matrices ya guardadas en outputs/<run>/
sin necesidad de reentrenar ni de disponer de las probabilidades.

Métricas implementadas:
    - balanced_accuracy(cm)          : media de recalls por clase (robusta al desbalance).
    - quadratic_weighted_kappa(cm)   : acuerdo ordinal (penaliza cuadráticamente la distancia).
    - macro_mae(cm)                  : error absoluto medio ordinal, promediado por clase real.
    - expected_cost(cm, cost_matrix) : coste clínico esperado (menor es mejor).
    - metrics_from_confusion(cm, ..) : dict con todas las anteriores.

Referencias metodológicas (ver Docs/Info-Other_papers/):
    - Balanced Accuracy: estándar de facto en neuroimagen desbalanceada (Wen et al. 2020, ClinicaDL).
    - QWK: métrica ordinal para estadios ordenados (CN < MCI < AD).
    - Expected Cost: aprendizaje sensible al coste para errores diagnósticos asimétricos.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Sequence

import numpy as np

from src.config import cfg


def parse_confusion_from_text(text: str) -> np.ndarray | None:
    """
    Extrae la matriz de confusión de un classification_report_*.txt del proyecto.

    Busca el bloque tras 'Confusion Matrix:'; si no existe (reportes de ensemble/LF),
    toma el último bloque con forma [[...]]. Devuelve un array NxN o None.
    """
    marker = "Confusion Matrix:"
    idx = text.find(marker)
    if idx != -1:
        tail = text[idx + len(marker):]
    else:
        blocks = re.findall(r"\[\[.*?\]\]", text, flags=re.DOTALL)
        if not blocks:
            return None
        tail = blocks[-1]
    ints = re.findall(r"-?\d+", tail)
    if not ints:
        return None
    n = int(math.isqrt(len(ints)))
    if n < 2:
        return None
    vals = np.array([int(x) for x in ints[: n * n]], dtype=np.float64)
    return vals.reshape(n, n)


def _as_cm(cm) -> np.ndarray:
    """Convierte a matriz de confusión float, validando que sea cuadrada."""
    arr = np.asarray(cm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"La matriz de confusión debe ser cuadrada NxN, recibido {arr.shape}")
    return arr


def confusion_from_labels(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Construye la matriz de confusión (filas=real, columnas=predicho) sin sklearn."""
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def balanced_accuracy(cm) -> float:
    """Media aritmética de la sensibilidad (recall) de cada clase con soporte > 0."""
    cm = _as_cm(cm)
    row_sums = cm.sum(axis=1)
    recalls = []
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            recalls.append(cm[i, i] / row_sums[i])
    return float(np.mean(recalls)) if recalls else 0.0


def _ordinal_weight_matrix(n: int) -> np.ndarray:
    """Matriz de pesos cuadráticos w_ij = (i-j)^2 / (n-1)^2."""
    idx = np.arange(n)
    diff = idx[:, None] - idx[None, :]
    denom = (n - 1) ** 2 if n > 1 else 1
    return (diff ** 2) / denom


def quadratic_weighted_kappa(cm) -> float:
    """
    Quadratic Weighted Kappa a partir de la matriz de confusión.

    Penaliza cuadráticamente la distancia ordinal entre clase real y predicha:
    confundir AD con CN (distancia 2) penaliza mucho más que AD con MCI (distancia 1).
    Rango típico: 0 (azar) a 1 (acuerdo perfecto); puede ser negativo.
    """
    cm = _as_cm(cm)
    n = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return 0.0

    w = _ordinal_weight_matrix(n)
    observed = cm / total

    row_marg = cm.sum(axis=1) / total
    col_marg = cm.sum(axis=0) / total
    expected = np.outer(row_marg, col_marg)

    denom = float((w * expected).sum())
    if denom == 0:
        return 0.0
    return float(1.0 - (w * observed).sum() / denom)


def macro_mae(cm) -> float:
    """
    Macro-Averaged Mean Absolute Error ordinal.

    Para cada clase real i calcula el error absoluto medio |i - j| ponderado por
    la distribución de predicciones de esa clase, y promedia sobre las clases con
    soporte. Cuantifica la magnitud (distancia) de los fallos, no solo si acierta.
    """
    cm = _as_cm(cm)
    n = cm.shape[0]
    idx = np.arange(n)
    row_sums = cm.sum(axis=1)
    maes = []
    for i in range(n):
        if row_sums[i] > 0:
            maes.append(float((cm[i, :] * np.abs(idx - i)).sum() / row_sums[i]))
    return float(np.mean(maes)) if maes else 0.0


def expected_cost(cm, cost_matrix=None) -> float:
    """
    Coste clínico esperado por muestra: sum(cm * cost) / sum(cm). Menor es mejor.

    cost_matrix[i, j] = coste de predecir j cuando la clase real es i.
    Por defecto usa cfg.COST_MATRIX (ilustrativa; consensuar con tutor).
    """
    cm = _as_cm(cm)
    if cost_matrix is None:
        cost_matrix = cfg.COST_MATRIX
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.shape != cm.shape:
        raise ValueError(
            f"cost_matrix {cost.shape} no coincide con la matriz de confusión {cm.shape}"
        )
    total = cm.sum()
    if total == 0:
        return 0.0
    return float((cm * cost).sum() / total)


def metrics_from_confusion(cm, cost_matrix=None) -> Dict[str, float]:
    """
    Devuelve todas las métricas derivables de la matriz de confusión.

    Si la matriz de coste no coincide en tamaño con la de confusión (p.ej. esquema
    binario con COST_MATRIX 3x3), expected_cost se devuelve como NaN en lugar de fallar.
    """
    cm = _as_cm(cm)
    diag = float(np.trace(cm))
    total = float(cm.sum())
    ref_cost = cfg.COST_MATRIX if cost_matrix is None else cost_matrix
    if np.asarray(ref_cost).shape == cm.shape:
        exp_cost = expected_cost(cm, ref_cost)
    else:
        exp_cost = float("nan")
    return {
        "accuracy": diag / total if total > 0 else 0.0,
        "balanced_accuracy": balanced_accuracy(cm),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(cm),
        "macro_mae": macro_mae(cm),
        "expected_cost": exp_cost,
    }


def metrics_from_labels(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int | None = None,
    cost_matrix=None,
) -> Dict[str, float]:
    """Como metrics_from_confusion pero a partir de listas de etiquetas."""
    if num_classes is None:
        num_classes = int(max(max(y_true, default=0), max(y_pred, default=0)) + 1)
    cm = confusion_from_labels(y_true, y_pred, num_classes)
    return metrics_from_confusion(cm, cost_matrix)
