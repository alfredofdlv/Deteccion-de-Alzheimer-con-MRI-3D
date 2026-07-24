#!/usr/bin/env python3
"""Diagnóstico: claves checkpoint y-Aware vs DenseNet121 MONAI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.yaware_densenet import diagnose_yaware_checkpoint, load_yaware_into_monai_densenet
from monai.networks.nets import DenseNet121


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecciona compatibilidad y-Aware → MONAI DenseNet121")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(cfg.YAWARE_DENSENET_CHECKPOINT),
        help="Ruta al checkpoint .pth",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout MONAI (default cfg)")
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"ERROR: no existe {weights}", file=sys.stderr)
        print("Descarga: python scripts/download_yaware_densenet.py", file=sys.stderr)
        sys.exit(1)

    report = diagnose_yaware_checkpoint(weights, dropout_prob=args.dropout)

    p = cfg.DENSENET_DROPOUT if args.dropout is None else args.dropout
    model = DenseNet121(
        spatial_dims=3, in_channels=1, out_channels=cfg.NUM_CLASSES, dropout_prob=p,
    )
    load_report = load_yaware_into_monai_densenet(model, weights)
    report["load_report"] = {
        "loaded_keys": load_report.loaded_keys,
        "missing_backbone_keys": load_report.missing_backbone_keys,
        "unexpected_keys": load_report.unexpected_keys,
        "shape_mismatches": load_report.shape_mismatches,
        "backbone_coverage_pct": load_report.backbone_coverage_pct,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Checkpoint: {report['weights_path']}")
    print(f"Claves raw: {report['raw_key_count']}")
    print(f"Claves remapeadas (encoder): {report['remapped_key_count']}")
    print(f"Claves backbone MONAI: {report['monai_backbone_key_count']}")
    print(f"Match forma OK: {report['matched_shape_ok']} ({report['coverage_pct']:.1f}%)")
    print(f"Carga real: {load_report}")
    print("\nMuestra claves raw y-Aware:")
    for k in report["sample_raw_keys"]:
        print(f"  {k}")
    if report["unmatched_yaware_sample"]:
        print("\nRemapeadas sin match MONAI (muestra):")
        for k in report["unmatched_yaware_sample"]:
            print(f"  {k}")
    if report["missing_monai_sample"]:
        print("\nBackbone MONAI sin peso y-Aware (muestra):")
        for k in report["missing_monai_sample"]:
            print(f"  {k}")


if __name__ == "__main__":
    main()
