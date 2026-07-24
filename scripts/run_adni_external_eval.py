"""
run_adni_external_eval.py — Validación externa OASIS-3 → ADNI y comparación de dominio.

Evalúa el checkpoint OASIS-3 (p. ej. densenet-cropped) sobre el split ADNI test
y escribe un resumen en outputs/adni-external-eval/.

Uso:
    python scripts/run_adni_external_eval.py --run densenet-cropped
    python scripts/run_adni_external_eval.py --run densenet-cropped --dataset oasis3 --split test --tag oasis3_reference
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación externa OASIS→ADNI")
    parser.add_argument("--run", type=str, default="densenet-cropped")
    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--dataset", type=str, default="adni", choices=["adni", "oasis3"])
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "all"])
    parser.add_argument("--tag", type=str, default=None, help="Sufijo del reporte (default: dataset)")
    parser.add_argument(
        "--no-thresholds", action="store_true",
        help="No aplicar thresholds.json (argmax puro sobre softmax)",
    )
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado (default: leer del checkpoint)",
    )
    args = parser.parse_args()

    tag = args.tag or args.dataset
    if args.no_thresholds:
        tag = f"{tag}_no_thresh"
    out_dir = cfg.OUTPUTS_DIR / "adni-external-eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--run", args.run,
        "--dataset", args.dataset,
        "--split", args.split,
        "--model", args.model,
    ]
    if args.no_thresholds:
        cmd.append("--no-thresholds")
    if args.preprocess_variant:
        cmd.extend(["--preprocess-variant", args.preprocess_variant])
    print(f"[INFO] Ejecutando: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        sys.exit(rc)

    run_dir = cfg.OUTPUTS_DIR / args.run
    report_src = run_dir / f"classification_report_{args.split}_thresh.txt"
    if not report_src.exists():
        report_src = run_dir / f"classification_report_{args.split}.txt"

    report_dst = out_dir / f"classification_report_{tag}_{args.split}.txt"
    if report_src.exists():
        report_dst.write_text(report_src.read_text(encoding="utf-8"), encoding="utf-8")

    cm_src = run_dir / f"confusion_matrix_{args.split}_thresh.png"
    if not cm_src.exists():
        cm_src = run_dir / f"confusion_matrix_{args.split}.png"
    if cm_src.exists():
        import shutil
        shutil.copy2(cm_src, out_dir / f"confusion_matrix_{tag}_{args.split}.png")

    summary_path = out_dir / "external_eval_summary.txt"
    smoke_note = ""
    if args.dataset == "adni" and args.split != "all":
        smoke_note = (
            "\nNOTA: Si usaste adni_smoke_fill_from_oasis.py, los volúmenes NO son ADNI reales.\n"
            "      Repite esta evaluación tras descargar DICOM del IDA y preprocesar.\n"
        )

    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"\n--- {datetime.now().isoformat(timespec='seconds')} ---\n")
        f.write(f"Run: {args.run} | Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset} | Split: {args.split}\n")
        if report_dst.exists():
            f.write(report_dst.read_text(encoding="utf-8"))
        f.write(smoke_note)

    print(f"\n[OK] Reportes en: {out_dir}")
    if smoke_note:
        print(smoke_note.strip())


if __name__ == "__main__":
    main()
