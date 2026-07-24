"""
cost_sensitivity_analysis.py — Análisis de sensibilidad de la matriz de coste (Fase 0a).

Responde a la crítica de que los pesos de la matriz de coste son arbitrarios: en vez
de defender un único conjunto de números, se comprueba si el *ranking* de modelos es
ROBUSTO ante distintas familias de matrices de coste. Si el mejor modelo apenas cambia,
la conclusión no depende de los pesos exactos (resultado defendible ante un tribunal).

Todo es post-hoc desde las matrices de confusión ya guardadas (sin GPU, sin reentrenar).

Familias de coste evaluadas (real i, predicho j; infra-diagnóstico j<i = "under",
sobre-diagnóstico j>i = "over"):
    - sym-linear     : coste = |i-j|                     (simétrico, distancia)
    - sym-quad       : coste = (i-j)^2                    (simétrico, cuadrático)
    - asym-2:1       : under = 2*|i-j|, over = 1*|i-j|
    - asym-5:1       : under = 5*|i-j|, over = 1*|i-j|
    - asym-10:1      : under = 10*|i-j|, over = 1*|i-j|
    - deepresearch   : cfg.COST_MATRIX (la ilustrativa actual)

Uso:
    python scripts/cost_sensitivity_analysis.py
    python scripts/cost_sensitivity_analysis.py --runs densenet-cropped densenet-oasis3-adni-mni
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import cfg
from src.metrics import expected_cost, parse_confusion_from_text

# Solo el report principal de test (evita duplicados _adni/_oasis3/_tta/_thresh...).
PRIMARY_REPORTS = (
    "classification_report_test.txt",
    "classification_report_test_ensemble.txt",
    "classification_report_test_hierarchical.txt",
)


def make_cost_matrix(under: float, over: float, mode: str = "linear", n: int = 3) -> np.ndarray:
    """Matriz de coste parametrizada por asimetría under/over y forma linear/quad."""
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            w = d if mode == "linear" else d * d
            if j < i:      # predicho menos severo que real -> infra-diagnóstico
                C[i, j] = under * w
            elif j > i:    # sobre-diagnóstico
                C[i, j] = over * w
    return C


def cost_families() -> dict[str, np.ndarray]:
    fams = {
        "sym-linear": make_cost_matrix(1, 1, "linear"),
        "sym-quad":   make_cost_matrix(1, 1, "quad"),
        "asym-2:1":   make_cost_matrix(2, 1, "linear"),
        "asym-5:1":   make_cost_matrix(5, 1, "linear"),
        "asym-10:1":  make_cost_matrix(10, 1, "linear"),
        "deepresearch": np.asarray(cfg.COST_MATRIX, dtype=np.float64),
    }
    return fams


def spearman(a: list[float], b: list[float]) -> float:
    """Correlación de Spearman sin scipy (rangos + Pearson)."""
    def ranks(x):
        order = np.argsort(x)
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x))
        return r
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Runs a incluir (default: todos los que tengan report principal 3x3)")
    parser.add_argument("--min-support", type=int, default=100,
                        help="Excluir runs con menos de N muestras en test (pilotos/subsets)")
    parser.add_argument("--outputs-dir", default=str(cfg.OUTPUTS_DIR))
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    fams = cost_families()

    rows: list[dict] = []
    for report_name in PRIMARY_REPORTS:
        for report_path in sorted(outputs_dir.glob(f"**/{report_name}")):
            run = report_path.parent.name
            if args.runs and run not in set(args.runs):
                continue
            if not args.runs and ("smoke" in run.lower() or run.startswith("_")):
                continue  # descartar pruebas de humo salvo petición explícita
            text = report_path.read_text(encoding="utf-8")
            cm = parse_confusion_from_text(text)
            if cm is None or cm.shape != (3, 3):
                continue
            if cm.sum() < args.min_support:
                continue  # test demasiado pequeño (piloto/subset), no comparable
            entry = {"run": run}
            for fam_name, C in fams.items():
                entry[fam_name] = round(expected_cost(cm, C), 4)
            rows.append(entry)

    if not rows:
        print("[ERROR] No se encontraron matrices de confusión 3x3.")
        return

    # Deduplicar por run (quedarse con el primero)
    seen = set()
    uniq = []
    for r in rows:
        if r["run"] in seen:
            continue
        seen.add(r["run"])
        uniq.append(r)
    rows = uniq

    fam_names = list(fams.keys())

    # --- CSV ---
    metrics_dir = outputs_dir / "_metrics"
    metrics_dir.mkdir(exist_ok=True)
    csv_path = metrics_dir / "cost_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + fam_names)
        writer.writeheader()
        writer.writerows(rows)

    # --- Mejor modelo por familia (menor Expected Cost) ---
    print(f"[OK] {len(rows)} runs. CSV: {csv_path}\n")
    print("Mejor modelo (menor Expected Cost) por familia de coste:")
    print("-" * 60)
    best_per_fam = {}
    for fam in fam_names:
        best = min(rows, key=lambda r: r[fam])
        best_per_fam[fam] = best["run"]
        print(f"  {fam:<14} -> {best['run']:<40} ({best[fam]})")

    n_distinct = len(set(best_per_fam.values()))
    print("-" * 60)
    print(f"Modelos ganadores distintos entre {len(fam_names)} familias: {n_distinct}")
    if n_distinct == 1:
        print("=> RANKING ROBUSTO: el mejor modelo NO depende de los pesos de coste.")
    else:
        print("=> El mejor modelo DEPENDE de la matriz de coste (justifica el análisis).")

    # --- Estabilidad global: Spearman entre rankings de familias vs deepresearch ---
    print("\nCorrelación de Spearman del ranking de runs (referencia: deepresearch):")
    ref = [r["deepresearch"] for r in rows]
    for fam in fam_names:
        if fam == "deepresearch":
            continue
        rho = spearman([r[fam] for r in rows], ref)
        print(f"  {fam:<14} rho = {rho:+.3f}")


if __name__ == "__main__":
    main()
