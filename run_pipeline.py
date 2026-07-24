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
    python run_pipeline.py densenet-late-fusion --dataset oasis3 --model densenet121 --no-export --late-fusion --run-source densenet-cropped
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# Import diferido no necesario; default de --weights-path alineado con cfg
try:
    from src.config import cfg as _cfg
    from src.label_scheme import LABEL_SCHEME_CHOICES
    from src.model import AVAILABLE_MODELS as _AVAILABLE_MODELS

    _DEFAULT_SSL_WEIGHTS = str(_cfg.RESENC_SSL_CHECKPOINT)
except Exception:  # ej. ejecutar sin PYTHONPATH
    _DEFAULT_SSL_WEIGHTS = "weights/resenc_l_ssl3d.pth"
    LABEL_SCHEME_CHOICES = ("multiclass", "binary_cn_imp")
    _AVAILABLE_MODELS = [
        "resnet10", "simple3dcnn", "densenet121", "densenet121_coral",
        "multimodal_densenet", "ultimate_fm", "efficientnet25d",
    ]


def run_step_with_log(description: str, cmd: list[str], log_file, *, detach: bool = False) -> None:
    sep = "=" * 60
    header = f"\n{sep}\n  {description}\n  > {' '.join(cmd)}\n{sep}\n\n"
    _tee(header, log_file)

    # Nueva sesión: train/eval sobreviven si muere el padre (cola SSH, run_pipeline, etc.)
    popen_kw: dict = {
        "cwd": PROJECT_ROOT,
        "stderr": subprocess.STDOUT,
        "start_new_session": True,
    }

    non_interactive = detach or os.environ.get("RUN_PIPELINE_NON_INTERACTIVE") == "1"
    if non_interactive or not sys.stdout.isatty():
        # Log directo al fichero — evita bloqueo de pipe si el padre desaparece
        with open(log_file.name, "a", encoding="utf-8") as step_log:
            proc = subprocess.Popen(cmd, stdout=step_log, **popen_kw)
        proc.wait()
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, **popen_kw)
        line_buf = b""
        while True:
            raw = proc.stdout.read(1)
            if not raw:
                break
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            if raw == b"\r":
                line_buf = b""
            elif raw == b"\n":
                log_file.write(line_buf.decode("utf-8", errors="replace") + "\n")
                log_file.flush()
                line_buf = b""
            else:
                line_buf += raw
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
        "--detach",
        action="store_true",
        help="Modo resistente a SSH: subprocesos en nueva sesión, log sin tee interactivo",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis1",
        choices=["oasis1", "oasis3", "adni", "adni_baseline_sc", "oasis3_adni", "oasis3_adni_baseline"],
        help="Dataset a utilizar (default: oasis1)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limitar a N samples por split (para pruebas rapidas)",
    )
    parser.add_argument(
        "--model", type=str, default="resnet10",
        choices=_AVAILABLE_MODELS,
        help="Modelo a usar (default: resnet10)",
    )
    parser.add_argument(
        "--ordinal", action="store_true",
        help="Pérdida ordinal CORAL + soft F2 (densenet121 / densenet121_coral); se pasa a src.train",
    )
    parser.add_argument(
        "--focal", action="store_true",
        help="Focal Loss multiclase (gamma=2) en lugar de CrossEntropyLoss; se pasa a src.train",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=_DEFAULT_SSL_WEIGHTS,
        help="Ruta al checkpoint SSL del backbone para ultimate_fm (default: cfg.RESENC_SSL_CHECKPOINT)",
    )
    parser.add_argument(
        "--late-fusion", action="store_true",
        help="Ejecutar Late Fusion (XGBoost) tras la evaluacion, usando las features del CNN entrenado",
    )
    parser.add_argument(
        "--run-source", type=str, default=None,
        help="Run del que se cargan features para Late Fusion (default: el run actual)",
    )
    parser.add_argument(
        "--smote", action="store_true",
        help="Aplicar SMOTE en Late Fusion para balancear clases minoritarias (requiere --late-fusion)",
    )
    parser.add_argument(
        "--grid-verbose", type=int, default=5,
        metavar="N",
        help="Verbose GridSearch en Late Fusion (0-50; default 5). Solo con --late-fusion",
    )
    parser.add_argument(
        "--no-lora", action="store_true",
        help="ultimate_fm: linear probe sin LoRA (solo cabeza)",
    )
    parser.add_argument(
        "--full-finetune", action="store_true",
        help="ultimate_fm: full fine-tune con LR diferencial y AMP",
    )
    parser.add_argument(
        "--late-fusion-only", action="store_true",
        help="Omitir train/eval; solo Late Fusion (--late-fusion requerido)",
    )
    parser.add_argument(
        "--clinical-weights",
        action="store_true",
        help="Aplicar cfg.CLINICAL_WEIGHT_MULTIPLIERS en la loss (class weights)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="MedicalNet 3D preentrenado (solo --model resnet10)",
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="resnet10",
        choices=["resnet10", "resnet18"],
        help="Backbone MONAI con --pretrained",
    )
    parser.add_argument(
        "--yaware-pretrained",
        action="store_true",
        help="Encoder DenseNet121 desde checkpoint y-Aware BHB-10K (solo densenet121)",
    )
    parser.add_argument(
        "--yaware-weights",
        type=str,
        default=None,
        help="Ruta al .pth y-Aware (default: cfg.YAWARE_DENSENET_CHECKPOINT)",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Evaluacion con Test-Time Augmentation (flip LR)",
    )
    parser.add_argument(
        "--tune-thresholds", action="store_true",
        help="Tras evaluar, optimizar umbrales en val (threshold_tuning)",
    )
    parser.add_argument(
        "--inference-pipeline", action="store_true",
        help="Tras entrenar/evaluar, ejecutar ensemble+TTA+umbrales (A5)",
    )
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default="cropped",
        choices=list(_cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado offline (cropped, mni, mni_n4)",
    )
    parser.add_argument(
        "--aug",
        type=str,
        default="full",
        choices=["full", "light", "none"],
        help="Modo de augmentation en train (default: full)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        metavar="P",
        help="Dropout DenseNet (default: cfg.DENSENET_DROPOUT)",
    )
    parser.add_argument(
        "--label-scheme",
        type=str,
        default=None,
        choices=list(LABEL_SCHEME_CHOICES),
        help="Esquema de labels (default: del checkpoint o multiclass)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        choices=["mtl"],
        help="Recorte ROI MTL online (hipocampo desde .pt MNI 96³)",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        metavar="N",
        help="Resize online a N³ (p. ej. 128 desde .pt 96³)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="B",
        help="Batch size de entrenamiento",
    )
    args = parser.parse_args()

    # Crear directorio de salida y abrir fichero de log
    out_dir = PROJECT_ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"
    log_mode = "a" if args.detach and log_path.is_file() else "w"
    log_file = open(log_path, log_mode, encoding="utf-8", buffering=1)
    detach = args.detach

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _tee(f"Inicio: {timestamp}  |  Run: {args.name}\n{'=' * 60}\n", log_file)

    py = sys.executable

    if args.late_fusion_only and not args.late_fusion:
        parser.error("--late-fusion-only requiere --late-fusion")
    if args.focal and args.ordinal:
        parser.error("No se puede usar --focal y --ordinal a la vez.")
    if args.pretrained and args.yaware_pretrained:
        parser.error("No usar --pretrained y --yaware-pretrained a la vez")
    if args.yaware_pretrained and args.model not in ("densenet121", "densenet121_coral"):
        parser.error("--yaware-pretrained solo es compatible con --model densenet121")
    if args.label_scheme in ("binary_cn_imp", "binary_mci_ad") and (
        args.ordinal or args.model == "densenet121_coral"
    ):
        parser.error(
            f"{args.label_scheme} no es compatible con --ordinal ni densenet121_coral"
        )

    total_steps = 4 if args.late_fusion else 3
    if args.late_fusion_only:
        total_steps = 2 if not args.no_export else 1

    step = 0

    if not args.late_fusion_only:
        step += 1
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
        if args.focal:
            train_cmd += ["--focal"]
        if args.model == "ultimate_fm":
            train_cmd += ["--weights-path", args.weights_path]
        if args.no_lora:
            train_cmd += ["--no-lora"]
        if args.full_finetune:
            train_cmd += ["--full-finetune"]
        if args.clinical_weights:
            train_cmd += ["--clinical-weights"]
        if args.pretrained:
            train_cmd += ["--pretrained", "--backbone-name", args.backbone_name]
        if args.yaware_pretrained:
            train_cmd += ["--yaware-pretrained"]
            if args.yaware_weights:
                train_cmd += ["--yaware-weights", args.yaware_weights]
        if args.preprocess_variant != "cropped":
            train_cmd += ["--preprocess-variant", args.preprocess_variant]
        if args.aug != "full":
            train_cmd += ["--aug", args.aug]
        if args.dropout is not None:
            train_cmd += ["--dropout", str(args.dropout)]
        if args.label_scheme is not None:
            train_cmd += ["--label-scheme", args.label_scheme]
        if getattr(args, "roi", None):
            train_cmd += ["--roi", args.roi]
        if getattr(args, "spatial_size", None) is not None:
            train_cmd += ["--spatial-size", str(args.spatial_size)]
        if getattr(args, "batch_size", None) is not None:
            train_cmd += ["--batch-size", str(args.batch_size)]
        run_step_with_log(f"PASO {step}/{total_steps} — Entrenamiento", train_cmd, log_file, detach=detach)

        step += 1
        # 2. Evaluar
        eval_cmd = [py, "-m", "src.evaluate", "--run", args.name, "--dataset", args.dataset,
                    "--model", args.model]
        if args.subset is not None:
            eval_cmd += ["--subset", str(args.subset)]
        if args.model == "ultimate_fm":
            eval_cmd += ["--weights-path", args.weights_path]
        if args.tta:
            eval_cmd += ["--tta"]
        if args.preprocess_variant != "cropped":
            eval_cmd += ["--preprocess-variant", args.preprocess_variant]
        if args.label_scheme is not None:
            eval_cmd += ["--label-scheme", args.label_scheme]
        run_step_with_log(f"PASO {step}/{total_steps} — Evaluacion (test set)", eval_cmd, log_file, detach=detach)

        if args.tune_thresholds:
            tune_cmd = [
                py, "-m", "src.threshold_tuning",
                "--run", args.name, "--dataset", args.dataset,
            ]
            if args.subset is not None:
                tune_cmd += ["--subset", str(args.subset)]
            if args.tta:
                tune_cmd += ["--tta"]
            run_step_with_log("Post-eval — Tuning de umbrales (val)", tune_cmd, log_file, detach=detach)

    # Late Fusion (opcional)
    if args.late_fusion:
        step += 1
        lf_source = args.run_source if args.run_source else args.name
        lf_cmd = [
            py, "-m", "src.late_fusion",
            "--run", args.name,
            "--run-source", lf_source,
            "--dataset", args.dataset,
        ]
        if args.smote:
            lf_cmd += ["--smote"]
        lf_cmd += ["--grid-verbose", str(args.grid_verbose)]
        run_step_with_log(
            f"PASO {step}/{total_steps} — Late Fusion XGBoost (extractor: {lf_source})",
            lf_cmd,
            log_file,
            detach=detach,
        )

    if args.inference_pipeline:
        ip_cmd = [
            py, "-m", "src.inference_pipeline",
            "--dataset", args.dataset,
            "--output-name", f"{args.name}-ensemble-tta",
        ]
        if args.subset is not None:
            ip_cmd += ["--subset", str(args.subset)]
        run_step_with_log("Post-pipeline — Ensemble + TTA + umbrales (A5)", ip_cmd, log_file, detach=detach)

    # Exportar contexto (opcional)
    step_n = total_steps
    md_path = f"outputs/{args.name}/{args.name}.md"
    if not args.no_export:
        export_cmd = [
            py, "export_context.py",
            "--run", args.name,
            "-o", md_path,
        ]
        run_step_with_log(f"PASO {step_n}/{step_n} — Exportar Markdown", export_cmd, log_file, detach=detach)
    else:
        _tee(f"\n{'=' * 60}\n  PASO {step_n}/{step_n} — Exportar Markdown (omitido: --no-export)\n{'=' * 60}\n", log_file)

    sep = "=" * 60
    summary = f"\n{sep}\n  Pipeline completada para '{args.name}'\n  Resultados en: outputs/{args.name}/\n"
    if not args.no_export:
        summary += f"  Markdown en:   {md_path}\n"
    summary += f"{sep}\n"
    _tee(summary, log_file)

    log_file.close()


if __name__ == "__main__":
    main()
