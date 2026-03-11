"""
run_pipeline.py — Ejecuta la pipeline completa: entrenar, evaluar y exportar.

Uso:
    python run_pipeline.py mi-experimento
    python run_pipeline.py mi-experimento --epochs 80 --patience 30
    python run_pipeline.py mi-experimento --no-export
    python run_pipeline.py mi-experimento --dataset oasis3
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(description: str, cmd: list[str]) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {description}")
    print(f"  > {' '.join(cmd)}")
    print(f"{sep}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Falló: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completa: train → evaluate → export"
    )
    parser.add_argument(
        "name", type=str,
        help="Nombre del run (carpeta en outputs/)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Número máximo de epochs (default: el de config.py)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early stopping patience (default: el de config.py)",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Omitir la exportación del Markdown de contexto",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis1",
        choices=["oasis1", "oasis3"],
        help="Dataset a utilizar (default: oasis1)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limitar a N samples por split (para pruebas rapidas)",
    )
    args = parser.parse_args()

    py = sys.executable

    # 1. Entrenar
    train_cmd = [py, "-m", "src.train", "--run", args.name, "--dataset", args.dataset]
    if args.epochs is not None:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.patience is not None:
        train_cmd += ["--patience", str(args.patience)]
    if args.subset is not None:
        train_cmd += ["--subset", str(args.subset)]
    run_step("PASO 1/3 — Entrenamiento", train_cmd)

    # 2. Evaluar
    eval_cmd = [py, "-m", "src.evaluate", "--run", args.name, "--dataset", args.dataset]
    if args.subset is not None:
        eval_cmd += ["--subset", str(args.subset)]
    run_step("PASO 2/3 — Evaluación (test set)", eval_cmd)

    # 3. Exportar contexto (opcional)
    md_path = f"outputs/{args.name}/{args.name}.md"
    if not args.no_export:
        export_cmd = [
            py, "export_context.py",
            "--run", args.name,
            "-o", md_path,
        ]
        run_step("PASO 3/3 — Exportar Markdown", export_cmd)
    else:
        print(f"\n{'=' * 60}")
        print(f"  PASO 3/3 — Exportar Markdown (omitido: --no-export)")
        print(f"{'=' * 60}\n")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline completada para '{args.name}'")
    print(f"  Resultados en: outputs/{args.name}/")
    if not args.no_export:
        print(f"  Markdown en:   {md_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
