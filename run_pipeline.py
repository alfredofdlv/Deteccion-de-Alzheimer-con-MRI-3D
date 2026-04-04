"""
run_pipeline.py — Ejecuta la pipeline completa: entrenar, evaluar y exportar.

Toda la salida se escribe en tiempo real tanto en stdout como en
outputs/<run_name>/pipeline.log, lo que permite monitorizar via:
    tail -f outputs/<run_name>/pipeline.log

Uso:
    python run_pipeline.py mi-experimento
    python run_pipeline.py mi-experimento --epochs 80 --patience 30
    python run_pipeline.py mi-experimento --no-export
    python run_pipeline.py mi-experimento --dataset oasis3
    python run_pipeline.py mi-run --dataset oasis3 --model densenet121 --ordinal
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step_with_log(description: str, cmd: list[str], log_file) -> None:
    sep = "=" * 60
    header = f"\n{sep}\n  {description}\n  > {' '.join(cmd)}\n{sep}\n\n"
    _tee(header, log_file)

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # mergea stderr para capturar warnings de PyTorch/MONAI
    )

    # Buffer de bytes: \r (0x0D) y \n (0x0A) son ASCII de 1 byte y nunca aparecen
    # como bytes de continuación UTF-8 (0x80-0xBF), por lo que es seguro buscarlos
    # en el stream crudo sin romper secuencias multibyte.
    line_buf = b""
    while True:
        raw = proc.stdout.read(1)
        if not raw:
            break
        sys.stdout.buffer.write(raw)   # terminal: byte crudo (animación \r funciona)
        sys.stdout.buffer.flush()
        if raw == b"\r":
            line_buf = b""             # descartar estado intermedio del progress bar
        elif raw == b"\n":
            log_file.write(line_buf.decode("utf-8", errors="replace") + "\n")
            log_file.flush()
            line_buf = b""
        else:
            line_buf += raw

    # Línea final sin \n (e.g. si el proceso termina sin salto de línea)
    if line_buf:
        log_file.write(line_buf.decode("utf-8", errors="replace") + "\n")
        log_file.flush()

    proc.wait()
    if proc.returncode != 0:
        msg = f"\n[ERROR] Falló: {description} (exit code {proc.returncode})\n"
        _tee(msg, log_file)
        sys.exit(proc.returncode)


def _tee(text: str, log_file) -> None:
    """Escribe text en stdout y en log_file simultáneamente."""
    raw = text.encode("utf-8")
    sys.stdout.buffer.write(raw)   # mismo buffer que el subprocess, sin mezcla de capas
    sys.stdout.buffer.flush()
    log_file.write(text)
    log_file.flush()


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
    parser.add_argument(
        "--model", type=str, default="resnet10",
        choices=["resnet10", "simple3dcnn", "densenet121", "multimodal_densenet"],
        help="Modelo a usar (default: resnet10)",
    )
    parser.add_argument(
        "--ordinal", action="store_true",
        help="Pérdida ordinal BCE + soft F2 (solo densenet121 / multimodal_densenet); se pasa a src.train",
    )
    args = parser.parse_args()

    # Crear directorio de salida y abrir fichero de log
    out_dir = PROJECT_ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _tee(f"Inicio: {timestamp}  |  Run: {args.name}\n{'=' * 60}\n", log_file)

    py = sys.executable

    # 1. Entrenar
    train_cmd = [py, "-m", "src.train", "--run", args.name, "--dataset", args.dataset,
                 "--model", args.model]
    if args.epochs is not None:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.patience is not None:
        train_cmd += ["--patience", str(args.patience)]
    if args.subset is not None:
        train_cmd += ["--subset", str(args.subset)]
    if args.ordinal:
        train_cmd += ["--ordinal"]
    run_step_with_log("PASO 1/3 — Entrenamiento", train_cmd, log_file)

    # 2. Evaluar
    eval_cmd = [py, "-m", "src.evaluate", "--run", args.name, "--dataset", args.dataset,
                "--model", args.model]
    if args.subset is not None:
        eval_cmd += ["--subset", str(args.subset)]
    run_step_with_log("PASO 2/3 — Evaluacion (test set)", eval_cmd, log_file)

    # 3. Exportar contexto (opcional)
    md_path = f"outputs/{args.name}/{args.name}.md"
    if not args.no_export:
        export_cmd = [
            py, "export_context.py",
            "--run", args.name,
            "-o", md_path,
        ]
        run_step_with_log("PASO 3/3 — Exportar Markdown", export_cmd, log_file)
    else:
        _tee(f"\n{'=' * 60}\n  PASO 3/3 — Exportar Markdown (omitido: --no-export)\n{'=' * 60}\n", log_file)

    sep = "=" * 60
    summary = f"\n{sep}\n  Pipeline completada para '{args.name}'\n  Resultados en: outputs/{args.name}/\n"
    if not args.no_export:
        summary += f"  Markdown en:   {md_path}\n"
    summary += f"{sep}\n"
    _tee(summary, log_file)

    log_file.close()


if __name__ == "__main__":
    main()
