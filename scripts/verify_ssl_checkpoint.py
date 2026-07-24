#!/usr/bin/env python3
"""Verifica carga del checkpoint SSL ResEnc-L (nnssl network_weights o stub state_dict)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import cfg
from src.model import UltimateNeuroFM, _load_backbone_state_dict


def main() -> None:
    p = Path(cfg.RESENC_SSL_CHECKPOINT)
    if not p.is_file():
        print(f"ERROR: no existe {p}")
        sys.exit(1)
    print(f"Checkpoint: {p.resolve()}")
    print(f"Tamaño (MB): {p.stat().st_size / 1e6:.1f}")

    from BrainFM4Challenges.models import ResEncL_UNet_Encoder

    bb = ResEncL_UNet_Encoder(out_channels=320)
    matched, miss, unexp = _load_backbone_state_dict(bb, p)
    total = len(bb.state_dict())
    print(f"Backbone: cargados={matched}/{total} missing={miss} unexpected={unexp}")

    if miss > total // 4:
        print("WARN: muchas capas sin peso — revisa arquitectura vs checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltimateNeuroFM(weights_path=str(p)).to(device).eval()
    x = torch.randn(2, 1, 96, 96, 96, device=device)
    c = torch.randn(2, 4, device=device)
    with torch.no_grad():
        y = model(x, c)
    n_train = sum(par.numel() for par in model.parameters() if par.requires_grad)
    print(f"Forward OK: logits {tuple(y.shape)} | trainable params: {n_train:,}")


if __name__ == "__main__":
    main()
