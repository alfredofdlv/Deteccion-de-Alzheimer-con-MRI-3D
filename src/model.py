"""
model.py — Arquitectura 3D ResNet para clasificacion de MRI 3D.

Usa resnet10 de MONAI como backbone, adaptado para volumenes cerebrales
monocanal (1, 96, 96, 96) con 3 clases de salida (CN, MCI, AD).

Uso:
    from src.model import Simple3DCNN  # alias retrocompatible
    model = Simple3DCNN()
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificacion rapida:
    python -m src.model
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet10

from src.config import cfg


class AlzheimerResNet(nn.Module):
    """
    3D ResNet-10 para clasificacion de volumenes cerebrales MRI.

    Wrapper alrededor de monai.networks.nets.resnet10 configurado para:
    - Entrada 3D monocanal (spatial_dims=3, n_input_channels=1)
    - 3 clases de salida (CN, MCI, AD)

    Args:
        num_classes: Numero de clases de salida.
    """

    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        self.net = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Simple3DCNN = AlzheimerResNet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlzheimerResNet().to(device)

    dummy = torch.randn(1, 1, 96, 96, 96, device=device)
    out = model(dummy)

    print(f"Device:       {device}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output:       {out.detach().cpu()}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros totales:      {n_params:,}")
    print(f"Parametros entrenables:  {n_train:,}")

    assert out.shape == (1, cfg.NUM_CLASSES), (
        f"Shape incorrecto: {out.shape}, esperado (1, {cfg.NUM_CLASSES})"
    )
    print("\n=== MODEL CHECK PASSED ===")
