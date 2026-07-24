"""
compute_extra_metrics.py — Métricas post-hoc (Balanced Accuracy, QWK, MMAE, Expected Cost)
a partir de las matrices de confusión ya guardadas en outputs/<run>/classification_report_test*.txt.

No requiere reentrenar ni disponer de las probabilidades: todas las métricas se derivan
de la matriz de confusión. Objetivo: comprobar si el ranking de modelos por Expected Cost /
QWK coincide con el ranking por Clinical F2 (métrica de selección actual).

Uso:
    python scripts/compute_extra_metrics.py                 # todos los reports test
    python scripts/compute_extra_metrics.py --runs densenet-cropped densenet-oasis3-adni-mni
    python scripts/compute_extra_metrics.py --sort expected_cost

Salidas:
    outputs/<run>/extra_metrics_<report>.txt   — ficha por report
    outputs/_metrics/extra_metrics_summary.csv — tabla comparativa ordenable
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg
from src.metrics import metrics_from_confusion, parse_confusion_from_text

REPORT_GLOB = "classification_report_test*.txt"


def parse_confusion_matrix(text: str) -> np.ndarray | None:
    """Wrapper del parser compartido en src.metrics."""
    return parse_confusion_from_text(text)


def parse_reported(text: str) -> dict:
    """Extrae accuracy global y Clinical F2 ya reportados (para comparar)."""
    out: dict = {}
    m = re.search(r"Accuracy global:\s*([\d.]+)%", text)
    if m:
        out["reported_accuracy"] = float(m.group(1)) / 100.0
    m = re.search(r"Clinical\s+([\d.]+)", text)
    if not m:
        m = re.search(r"Test F2c tuned:\s+([\d.]+)", text)
    if m:
        out["reported_clinical_f2"] = float(m.group(1))
    return out


def format_report(run: str, report_name: str, cm: np.ndarray, metrics: dict, reported: dict) -> str:
    n = cm.shape[0]
    can_cost = n == np.asarray(cfg.COST_MATRIX).shape[0]
    lines = [
        "=" * 60,
        f"MÉTRICAS EXTRA (post-hoc) — {run}",
        f"Report: {report_name}",
        "=" * 60,
        "",
        f"Matriz de confusión ({n}x{n}):",
        str(cm.astype(int)),
        "",
        f"  Accuracy            : {metrics['accuracy']:.4f}",
        f"  Balanced Accuracy   : {metrics['balanced_accuracy']:.4f}",
        f"  Quadratic W. Kappa  : {metrics['quadratic_weighted_kappa']:.4f}",
        f"  Macro MAE (ordinal) : {metrics['macro_mae']:.4f}   (menor es mejor)",
    ]
    if can_cost:
        lines.append(f"  Expected Cost       : {metrics['expected_cost']:.4f}   (menor es mejor)")
    else:
        lines.append("  Expected Cost       : N/A (matriz de coste 3x3; report no multiclase)")
    if reported:
        lines += ["", "Ya reportado en el fichero original:"]
        if "reported_accuracy" in reported:
            lines.append(f"  accuracy    : {reported['reported_accuracy']:.4f}")
        if "reported_clinical_f2" in reported:
            lines.append(f"  clinical_f2 : {reported['reported_clinical_f2']:.4f}")
    if can_cost:
        lines += ["", f"Matriz de coste usada (cfg.COST_MATRIX): {cfg.COST_MATRIX}"]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Nombres de runs a procesar (por defecto: todos)")
    parser.add_argument("--sort", default="expected_cost",
                        help="Columna para ordenar el CSV (default: expected_cost)")
    parser.add_argument("--outputs-dir", default=str(cfg.OUTPUTS_DIR))
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    reports = sorted(outputs_dir.glob(f"**/{REPORT_GLOB}"))
    if args.runs:
        wanted = set(args.runs)
        reports = [r for r in reports if r.parent.name in wanted]

    rows: list[dict] = []
    n_ok = 0
    for report_path in reports:
        run = report_path.parent.name
        try:
            text = report_path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] No se pudo leer {report_path}: {e}")
            continue
        cm = parse_confusion_matrix(text)
        if cm is None:
            print(f"[WARN] Sin matriz de confusión en {report_path}")
            continue

        can_cost = cm.shape[0] == np.asarray(cfg.COST_MATRIX).shape[0]
        cost = cfg.COST_MATRIX if can_cost else None
        metrics = metrics_from_confusion(cm, cost_matrix=cost)
        if not can_cost:
            metrics["expected_cost"] = float("nan")
        reported = parse_reported(text)

        ficha = format_report(run, report_path.name, cm, metrics, reported)
        ficha_path = report_path.parent / report_path.name.replace(
            "classification_report_", "extra_metrics_"
        )
        ficha_path.write_text(ficha, encoding="utf-8")
        n_ok += 1

        rows.append({
            "run": run,
            "report": report_path.name,
            "n_classes": cm.shape[0],
            "accuracy": round(metrics["accuracy"], 4),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "quadratic_weighted_kappa": round(metrics["quadratic_weighted_kappa"], 4),
            "macro_mae": round(metrics["macro_mae"], 4),
            "expected_cost": round(metrics["expected_cost"], 4) if can_cost else "",
            "reported_clinical_f2": round(reported["reported_clinical_f2"], 4)
            if "reported_clinical_f2" in reported else "",
        })

    if not rows:
        print("[ERROR] No se procesó ningún report.")
        return

    def sort_key(r: dict):
        v = r.get(args.sort, "")
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, 0.0)

    rows.sort(key=sort_key)

    metrics_dir = outputs_dir / "_metrics"
    metrics_dir.mkdir(exist_ok=True)
    csv_path = metrics_dir / "extra_metrics_summary.csv"
    fieldnames = [
        "run", "report", "n_classes", "accuracy", "balanced_accuracy",
        "quadratic_weighted_kappa", "macro_mae", "expected_cost", "reported_clinical_f2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {n_ok} reports procesados. CSV: {csv_path}")
    print(f"\nTop por menor {args.sort} (o el orden elegido):")
    header = f"{'run':<48} {'BA':>7} {'QWK':>7} {'MMAE':>7} {'ExpCost':>8} {'F2c':>7}"
    print(header)
    print("-" * len(header))
    for r in rows[:25]:
        print(f"{r['run'][:47]:<48} {r['balanced_accuracy']:>7} "
              f"{r['quadratic_weighted_kappa']:>7} {r['macro_mae']:>7} "
              f"{str(r['expected_cost']):>8} {str(r['reported_clinical_f2']):>7}")


if __name__ == "__main__":
    main()
