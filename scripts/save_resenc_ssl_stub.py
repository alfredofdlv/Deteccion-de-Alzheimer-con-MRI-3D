#!/usr/bin/env python3
"""
Genera weights/resenc_l_ssl3d.pth compatible con ResEncL_UNet_Encoder (stub local).

Sirve para probar que UltimateNeuroFM carga el checkpoint sin aviso de “backbone aleatorio”.
Sustituye este archivo por el checkpoint SSL real cuando lo tengas.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import cfg


def main() -> None:
    cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    from BrainFM4Challenges.models import ResEncL_UNet_Encoder

    backbone = ResEncL_UNet_Encoder(out_channels=320)
    out_path = cfg.RESENC_SSL_CHECKPOINT
    payload = {
        "state_dict": backbone.state_dict(),
        "note": "stub generado por scripts/save_resenc_ssl_stub.py — reemplazar por SSL real",
    }
    torch.save(payload, out_path)
    print(f"OK: guardado {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
