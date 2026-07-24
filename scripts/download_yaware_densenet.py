#!/usr/bin/env python3
"""Descarga DenseNet121_BHB-10K_yAwareContrastive.pth (repo yAwareContrastiveLearning)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.yaware_densenet import download_yaware_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga checkpoint y-Aware DenseNet121 BHB-10K")
    parser.add_argument(
        "--dest",
        type=str,
        default=str(cfg.YAWARE_DENSENET_CHECKPOINT),
        help="Ruta destino del .pth",
    )
    parser.add_argument("--force", action="store_true", help="Re-descargar aunque exista")
    args = parser.parse_args()

    path = download_yaware_checkpoint(dest=args.dest, force=args.force)
    print(f"OK: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
