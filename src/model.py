"""
model.py — Arquitectura Simple3DCNN para clasificación de MRI 3D.

5 bloques convolucionales seguidos de un clasificador global.

Arquitectura:
    Input (B, 1, 96, 96, 96)
    -> [Conv3d -> GroupNorm -> ReLU -> MaxPool3d] x5
    -> AdaptiveAvgPool3d(1) -> Dropout -> Linear(512, 3)

Mejoras v2:
    - GroupNorm(8) en lugar de BatchNorm3d (estable con batch_size pequeño).
    - 5 bloques conv (1->32->64->128->256->512) para reducción espacial suave
      (96 -> 48 -> 24 -> 12 -> 6 -> 3) antes del pooling global.
    - Inicialización Kaiming para Conv3d y bias con log-priors de clase.

Uso:
    from src.model import Simple3DCNN
    model = Simple3DCNN()
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificación rápida:
    python -m src.model
"""

import math

import torch
import torch.nn as nn

from src.config import cfg


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloque Conv3d(3x3x3, pad=1) -> GroupNorm(8) -> ReLU -> MaxPool3d(2)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2),
    )


class Simple3DCNN(nn.Module):
    """
    CNN 3D para clasificación de volúmenes cerebrales.

    Progresión de canales: 1 -> 32 -> 64 -> 128 -> 256 -> 512.
    Cada bloque reduce las dimensiones espaciales a la mitad con MaxPool3d(2).
    Con entrada 96³: 96 -> 48 -> 24 -> 12 -> 6 -> 3, y luego
    AdaptiveAvgPool3d colapsa 3³ -> 1³.

    Args:
        in_channels: Canales de entrada (1 para MRI).
        num_classes: Número de clases de salida (3: CN, MCI, AD).
        dropout: Probabilidad de dropout antes de la capa lineal.
    """

    # Proporción observada de cada clase en el split de train: CN=0.574, MCI=0.298, AD=0.128
    CLASS_PRIORS = [0.574, 0.298, 0.128]

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
            _conv_block(256, 512),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming init para Conv3d; log-prior bias para la capa final Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    log_priors = [math.log(p) for p in self.CLASS_PRIORS]
                    m.bias.data.copy_(torch.tensor(log_priors, dtype=m.bias.dtype))

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
