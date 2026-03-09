"""
model.py — Arquitectura Simple3DCNN para clasificación de MRI 3D.

4 bloques convolucionales seguidos de un clasificador global.

Arquitectura:
    Input (B, 1, 96, 96, 96)
    -> [Conv3d -> BN3d -> ReLU -> MaxPool3d] x4
    -> AdaptiveAvgPool3d(1) -> Dropout -> Linear(256, 3)

Uso:
    from src.model import Simple3DCNN
    model = Simple3DCNN()
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificación rápida:
    python -m src.model
"""

import torch
import torch.nn as nn

from src.config import cfg


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloque Conv3d(3x3x3, pad=1) -> BatchNorm3d -> ReLU -> MaxPool3d(2)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2),
    )


class Simple3DCNN(nn.Module):
    """
    CNN 3D vanilla para clasificación de volúmenes cerebrales.

    Progresión de canales: 1 -> 32 -> 64 -> 128 -> 256.
    Cada bloque reduce las dimensiones espaciales a la mitad con MaxPool3d(2).
    AdaptiveAvgPool3d colapsa las dimensiones espaciales a 1x1x1,
    haciendo la red agnóstica al tamaño de entrada.

    Args:
        in_channels: Canales de entrada (1 para MRI).
        num_classes: Número de clases de salida (3: CN, MCI, AD).
        dropout: Probabilidad de dropout antes de la capa lineal.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = cfg.NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DCNN().to(device)

    dummy = torch.randn(1, 1, 96, 96, 96, device=device)
    out = model(dummy)

    print(f"Device:       {device}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output:       {out.detach().cpu()}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales:      {n_params:,}")
    print(f"Parámetros entrenables:  {n_train:,}")

    assert out.shape == (1, cfg.NUM_CLASSES), (
        f"Shape incorrecto: {out.shape}, esperado (1, {cfg.NUM_CLASSES})"
    )
    print("\n=== CHECKPOINT 3.1 PASSED ===")
