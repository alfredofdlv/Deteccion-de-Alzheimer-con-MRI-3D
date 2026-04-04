"""
model.py — Modelos 3D para clasificacion de MRI cerebral.

Modelos disponibles:
    - resnet10:     ResNet-10 3D de MONAI (~14.3M params)
    - simple3dcnn:  CNN manual de 4 bloques conv (~1.16M params)

Uso:
    from src.model import get_model
    model = get_model("resnet10")
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificacion rapida:
    python -m src.model
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet10, DenseNet121

from src.config import cfg


# ---------------------------------------------------------------------------
# Simple3DCNN — CNN manual de 4 bloques
# ---------------------------------------------------------------------------

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
    CNN 3D con 4 bloques convolucionales para volumenes cerebrales.

    Progresion de canales: 1 -> 32 -> 64 -> 128 -> 256.
    Con entrada 96^3: 96 -> 48 -> 24 -> 12 -> 6, luego
    AdaptiveAvgPool3d colapsa a 1^3. ~1.16M parametros.

    Args:
        in_channels: Canales de entrada (1 para MRI).
        num_classes: Numero de clases de salida (3: CN, MCI, AD).
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
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# AlzheimerResNet — ResNet-10 3D de MONAI
# ---------------------------------------------------------------------------

class AlzheimerResNet(nn.Module):
    """
    ResNet-10 3D para clasificacion de volumenes cerebrales MRI.

    Wrapper de monai.networks.nets.resnet10: entrada 3D monocanal,
    3 clases de salida. ~14.3M parametros.

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

# ---------------------------------------------------------------------------
# AlzheimerDenseNet — DenseNet-121 3D de MONAI
# ---------------------------------------------------------------------------

class AlzheimerDenseNet(nn.Module):
    """
    DenseNet-121 3D para clasificacion de volumenes cerebrales MRI.
    Wrapper de monai.networks.nets.DenseNet121: entrada 3D monocanal. ~7.9M parametros.

    Args:
        num_classes: Numero de clases de salida (ignorado si ordinal=True).
        ordinal:     Si True, la capa final emite 2 logits ordinales en lugar de
                     num_classes. Los 2 logits representan P(Y>=MCI) y P(Y>=AD).
                     Requiere OrdinalClinicalF2Loss y decodificacion con decode_preds().
    """
    uses_ordinal: bool = False  # marcador leido por train/evaluate

    def __init__(self, num_classes: int = cfg.NUM_CLASSES, ordinal: bool = False):
        super().__init__()
        self.uses_ordinal = ordinal
        out_channels = 2 if ordinal else num_classes
        self.net = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MultimodalDenseNet — DenseNet-121 3D + vector clínico
# ---------------------------------------------------------------------------

class MultimodalDenseNet(nn.Module):
    """
    Fusion Multimodal: DenseNet-121 3D (MRI) + vector clinico (Age, Sex, Educ, APOE4).

    Arquitectura:
        - Backbone: DenseNet121(out_channels=256)  -> img_emb (B, 256)
        - Fusion:   cat([img_emb, clinical_4])     -> (B, 260)
        - MLP head: Linear(260, 64) -> BN1d -> ReLU -> Dropout(0.3) -> Linear(64, out)

    El atributo de clase `uses_clinical = True` permite que train/evaluate
    detecten el modo multimodal sin acoplamiento por isinstance.

    Args:
        num_classes:  Numero de clases (ignorado si ordinal=True).
        num_clinical: Dimension del vector clinico (default: 4).
        ordinal:      Si True, la capa final emite 2 logits ordinales.
    """
    uses_clinical: bool = True   # marcador leido por train/evaluate
    uses_ordinal: bool = False   # marcador leido por train/evaluate

    def __init__(
        self,
        num_classes: int = cfg.NUM_CLASSES,
        num_clinical: int = 4,
        ordinal: bool = False,
    ):
        super().__init__()
        self.uses_ordinal = ordinal
        out_classes = 2 if ordinal else num_classes
        self.feature_extractor = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=256,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + num_clinical, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, out_classes),
        )

    def forward(self, x: torch.Tensor, clinical: torch.Tensor | None = None) -> torch.Tensor:
        img_features = self.feature_extractor(x)           # (B, 256)
        if clinical is None:
            clinical = torch.zeros(x.size(0), 4, device=x.device, dtype=x.dtype)
        fused = torch.cat([img_features, clinical], dim=1)  # (B, 260)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = ["resnet10", "simple3dcnn", "densenet121", "multimodal_densenet"]


def get_model(name: str = "resnet10", ordinal: bool = False) -> nn.Module:
    """Instancia un modelo por nombre.

    Args:
        name:    'resnet10', 'simple3dcnn', 'densenet121' o 'multimodal_densenet'.
        ordinal: Si True, los modelos DenseNet emiten 2 logits ordinales en lugar de
                 cfg.NUM_CLASSES. Ignorado para resnet10 y simple3dcnn.

    Returns:
        nn.Module listo para .to(device).
    """
    if name == "resnet10":
        return AlzheimerResNet()
    elif name == "simple3dcnn":
        return Simple3DCNN()
    elif name == "densenet121":
        return AlzheimerDenseNet(ordinal=ordinal)
    elif name == "multimodal_densenet":
        return MultimodalDenseNet(ordinal=ordinal)
    else:
        raise ValueError(f"Modelo '{name}' no reconocido. Opciones: {AVAILABLE_MODELS}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(1, 1, 96, 96, 96, device=device)
    dummy_clinical = torch.randn(1, 4, device=device)

    configs = (
        [(name, False) for name in AVAILABLE_MODELS]
        + [("densenet121", True), ("multimodal_densenet", True)]
    )

    for name, ordinal in configs:
        print(f"\n{'=' * 50}")
        print(f"Modelo: {name} | ordinal={ordinal}")
        print(f"{'=' * 50}")
        model = get_model(name, ordinal=ordinal).to(device).eval()
        if getattr(model, "uses_clinical", False):
            out = model(dummy, dummy_clinical)
        else:
            out = model(dummy)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Device:        {device}")
        print(f"  Input shape:   {dummy.shape}")
        if getattr(model, "uses_clinical", False):
            print(f"  Clinical shape:{dummy_clinical.shape}")
        print(f"  Output shape:  {out.shape}")
        print(f"  Parametros:    {n_params:,}")
        expected_out = 2 if ordinal else cfg.NUM_CLASSES
        # Solo modelos DenseNet soportan ordinal
        if ordinal and name not in ("densenet121", "multimodal_densenet"):
            pass  # sin assert para modelos que no soportan ordinal
        else:
            assert out.shape == (1, expected_out), f"Esperado (1, {expected_out}), got {out.shape}"
        print(f"  CHECK PASSED")
