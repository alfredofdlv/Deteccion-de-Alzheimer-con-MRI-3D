"""
model.py — Modelos 3D para clasificacion de MRI cerebral.

Modelos disponibles:
    - resnet10:         ResNet-10 3D de MONAI (~14.3M params)
    - simple3dcnn:      CNN manual de 4 bloques conv (~1.16M params)
    - densenet121:      DenseNet-121 3D de MONAI (~7.9M params)
    - multimodal_densenet: DenseNet-121 3D + vector clinico (Age, Sex, Educ, APOE4)
    - ultimate_fm:      ResEncL_UNet_Encoder (frozen) + LoRA 3D + Lorentz Hyperbolic Head

Uso:
    from src.model import get_model
    model = get_model("resnet10")
    out = model(torch.randn(1, 1, 96, 96, 96))  # (1, 3)

Verificacion rapida:
    python -m src.model
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet10, resnet18, DenseNet121

from src.config import cfg


# ---------------------------------------------------------------------------
# Carga flexible de checkpoints SSL (claves anidadas y prefijos tipo module./encoder.)
# ---------------------------------------------------------------------------

def _torch_load_checkpoint(path: Path) -> object:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _encoder_only_state_dict(sd: dict) -> dict:
    """Filtra tensores del encoder y quita el prefijo 'encoder.' (checkpoints nnssl)."""
    enc = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
    if enc:
        return enc
    skip_prefixes = ("decoder.", "loss_weights.", "optimizer_", "grad_scaler")
    return {
        k: v
        for k, v in sd.items()
        if torch.is_tensor(v) and not k.startswith(skip_prefixes)
    }


def _unwrap_state_dict_from_checkpoint(raw: object) -> dict:
    """Extrae un dict de tensores desde distintos formatos de checkpoint."""
    if isinstance(raw, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "encoder",
            "backbone",
            "network_weights",
        ):
            inner = raw.get(key)
            if isinstance(inner, dict) and inner:
                first = next(iter(inner.values()))
                if torch.is_tensor(first):
                    if key == "network_weights":
                        return _encoder_only_state_dict(inner)
                    return inner
        if raw and torch.is_tensor(next(iter(raw.values()))):
            return _encoder_only_state_dict(raw)
    raise ValueError(
        "Checkpoint sin state_dict reconocible "
        "(esperado network_weights/state_dict/model/... o dict plano de tensores)."
    )


def _state_dict_variants(sd: dict) -> list[dict]:
    """Genera variantes con prefijos típicos de DDP / Lightning / nnUNet."""
    out: list[dict] = [sd]
    for prefix in (
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "net.",
        "feature_extractor.",
    ):
        stripped = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        if stripped and len(stripped) >= max(1, len(sd) // 4):
            out.append(stripped)
    return out


def _pick_best_state_dict_for_module(module: nn.Module, sd: dict) -> dict:
    """Elige la variante con más claves coincidentes y formas iguales."""
    model_sd = module.state_dict()
    variants = _state_dict_variants(sd)
    best = variants[0]
    best_score = -1
    for cand in variants:
        score = sum(
            1
            for k, v in model_sd.items()
            if k in cand and v.shape == cand[k].shape
        )
        if score > best_score:
            best_score = score
            best = cand
    return best


def _load_backbone_state_dict(backbone: nn.Module, weights_file: Path) -> tuple[int, int, int]:
    """
    Carga pesos en `backbone` y devuelve
    (n_claves_cargadas_aprox, len(missing_keys), len(unexpected_keys)).
    """
    raw = _torch_load_checkpoint(weights_file)
    sd = _unwrap_state_dict_from_checkpoint(raw)
    best_sd = _pick_best_state_dict_for_module(backbone, sd)
    missing, unexpected = backbone.load_state_dict(best_sd, strict=False)
    n_params = len(backbone.state_dict())
    matched = n_params - len(missing)
    return matched, len(missing), len(unexpected)


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

_MEDICALNET_KWARGS = dict(
    feed_forward=False,
    shortcut_type="B",
    bias_downsample=False,
)

_MEDICALNET_FEATURE_DIM = {"resnet10": 512, "resnet18": 512}


class AlzheimerResNet(nn.Module):
    """
    ResNet 3D para clasificacion de volumenes cerebrales MRI.

    Wrapper de monai.networks.nets.resnet10/resnet18. Con pretrained=True carga
    pesos MedicalNet (MONAI) y anade cabeza lineal sobre features 512-d.

    Args:
        num_classes: Numero de clases de salida.
        pretrained:  Si True, backbone MedicalNet preentrenado (solo resnet10/18).
        backbone_name: 'resnet10' (default) o 'resnet18'.
    """

    def __init__(
        self,
        num_classes: int = cfg.NUM_CLASSES,
        pretrained: bool = False,
        backbone_name: str = "resnet10",
    ):
        super().__init__()
        self.pretrained = pretrained
        self.backbone_name = backbone_name
        resnet_fn = resnet18 if backbone_name == "resnet18" else resnet10

        if pretrained:
            self.backbone = resnet_fn(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=num_classes,
                pretrained=True,
                **_MEDICALNET_KWARGS,
            )
            feat_dim = _MEDICALNET_FEATURE_DIM.get(backbone_name, 512)
            self.head = nn.Linear(feat_dim, num_classes)
            self.net = None
        else:
            self.backbone = None
            self.head = None
            self.net = resnet_fn(
                spatial_dims=3,
                n_input_channels=1,
                num_classes=num_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.flatten(1)
            return self.head(features)
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
        ordinal:     Si True, cabeza CORAL (Cao et al. 2020): peso compartido
                     Linear(D, 1) + sesgos (K-1) → logits P(Y>=MCI), P(Y>=AD).
                     Requiere OrdinalClinicalF2Loss y decode_preds() unificado.
    """
    uses_ordinal: bool = False  # marcador leido por train/evaluate
    uses_coral: bool = False    # True cuando ordinal=True (CORAL estricto)
    yaware_pretrained: bool = False

    def __init__(
        self,
        num_classes: int = cfg.NUM_CLASSES,
        ordinal: bool = False,
        dropout_prob: float | None = None,
        yaware_pretrained: bool = False,
        yaware_weights_path: str | Path | None = None,
    ):
        super().__init__()
        self.uses_ordinal = ordinal
        self.uses_coral = ordinal
        self.yaware_pretrained = False
        p = cfg.DENSENET_DROPOUT if dropout_prob is None else dropout_prob
        self.net = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            dropout_prob=p,
        )
        if yaware_pretrained:
            from src.yaware_densenet import (
                default_yaware_weights_path,
                load_yaware_into_monai_densenet,
            )

            wp = Path(yaware_weights_path) if yaware_weights_path else default_yaware_weights_path()
            report = load_yaware_into_monai_densenet(self.net, wp)
            self.yaware_pretrained = True
            self.yaware_load_report = report
            print(f"[INFO] {report}")
        if ordinal:
            dim = cfg.DENSENET_EMBED_DIM
            self.coral_linear = nn.Linear(dim, 1, bias=False)
            self.coral_biases = nn.Parameter(
                torch.zeros(num_classes - 1, dtype=torch.float32)
            )

    def _gap_features(self, x: torch.Tensor) -> torch.Tensor:
        """Features 1024-d tras backbone DenseNet (antes de class_layers)."""
        h = self.net.features(x)
        h = F.relu(h, inplace=True)
        h = F.adaptive_avg_pool3d(h, 1)
        return h.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_ordinal:
            f = self.coral_linear(self._gap_features(x))  # (B, 1)
            return f + self.coral_biases                     # (B, K-1)
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
        ordinal:      No soportado (usar densenet121 / densenet121_coral).
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
        if ordinal:
            raise ValueError(
                "multimodal_densenet no soporta modo ordinal/CORAL. "
                "Usa densenet121 o densenet121_coral."
            )
        self.uses_ordinal = False
        out_classes = num_classes
        self.feature_extractor = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=256,
            dropout_prob=cfg.DENSENET_DROPOUT,
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
# UltimateNeuroFM — Foundation Model + LoRA 3D + Hyperbolic Lorentz Head
# ---------------------------------------------------------------------------

class LoRA3DConv(nn.Module):
    """
    Wrapper LoRA para nn.Conv3d: agrega matrices de bajo rango A y B.

    El forward computa: conv(x) + (lora_B * lora_A)(x) * scaling
    donde lora_A proyecta a rank y lora_B proyecta de vuelta a C_out.
    Solo lora_A y lora_B son entrenables; conv queda congelada.

    Args:
        conv:  La capa Conv3d original (ya congelada).
        rank:  Rango del adaptador LoRA (default: 4).
    """

    def __init__(self, conv: nn.Conv3d, rank: int = 4):
        super().__init__()
        self.conv = conv
        C_out = conv.out_channels
        C_in = conv.in_channels // conv.groups
        self.rank = rank
        self.scaling = 1.0 / rank
        self.lora_A = nn.Parameter(torch.randn(rank, C_in, 1, 1, 1) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(C_out, rank, 1, 1, 1))
        # Guardar metadatos de la conv original para reproducir el stride/padding
        self._stride = conv.stride
        self._groups = conv.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        # Proyeccion de bajo rango con kernel 1x1x1 (sin padding para kernel unitario).
        # El stride de la conv original controla la reduccion espacial en lora_A,
        # produciendo el mismo tamaño espacial que `base`.
        lora_mid = F.conv3d(x, self.lora_A, bias=None,
                             stride=self._stride, padding=0,
                             dilation=(1, 1, 1), groups=self._groups)
        delta = F.conv3d(lora_mid, self.lora_B, bias=None,
                          stride=(1, 1, 1), padding=(0, 0, 0)) * self.scaling
        return base + delta


def inject_lora_into_backbone(
    backbone: nn.Module,
    stages: list[str],
    rank: int = 4,
) -> None:
    """
    Sustituye las Conv3d dentro de los stages especificados por LoRA3DConv in-place.

    La convencion de nombre es: el modulo se considera parte de un stage si
    cualquiera de las cadenas en `stages` aparece como prefijo en su nombre
    completo (e.g. 'stage4' captura 'stage4.block1.conv1', etc.).

    Args:
        backbone: El modulo del backbone (ya congelado).
        stages:   Lista de prefijos de stage, e.g. ['stage3', 'stage4'].
        rank:     Rango del adaptador LoRA.
    """
    for name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Conv3d):
            continue
        in_stage = any(name.startswith(s) or f".{s}." in f".{name}." for s in stages)
        if not in_stage:
            continue
        # Navegar hasta el parent para reemplazar el atributo
        parts = name.split(".")
        parent = backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        lora_conv = LoRA3DConv(module, rank=rank)
        # lora_A y lora_B son entrenables; la conv interna permanece congelada
        setattr(parent, child_name, lora_conv)


def _build_mock_backbone(feature_dim: int = 320) -> nn.Module:
    """Backbone mock para cuando ResEncL_UNet_Encoder no esta disponible."""

    class MockResEncL(nn.Module):
        def __init__(self, out_channels: int):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(32),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.stage1 = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(64),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.stage2 = nn.Sequential(
                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(128),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.stage3 = nn.Sequential(
                nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(256),
                nn.LeakyReLU(0.01, inplace=True),
            )
            self.stage4 = nn.Sequential(
                nn.Conv3d(256, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.01, inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            return x

    return MockResEncL(out_channels=feature_dim)


class LorentzProjectionWithRadialPenalty(nn.Module):
    """
    Proyecta un vector euclideo + penalizaciones clinicas radiales al manifold de Lorentz.

    Pipeline:
        1. Linear(feature_dim -> lorentz_dim - 1): mapea el GAP vector al espacio tangente
        2. Radial penalty MLP(age, apoe4 -> 1 escalar): modifica la norma del vector tangente
        3. expmap0: proyecta el vector tangente al manifold de Lorentz

    El vector tangente final tiene dimension lorentz_dim (la primera coordenada se calcula
    por expmap0 para satisfacer la restriccion del manifold).

    Args:
        feature_dim:  Dimension del vector GAP de entrada (e.g. 320).
        lorentz_dim:  Dimension del espacio de Lorentz (e.g. 257 = 256+1).
        curvature:    Curvatura del manifold (negativa implicita en geoopt.Lorentz).
    """

    def __init__(self, feature_dim: int, lorentz_dim: int, curvature: float = 1.0):
        super().__init__()
        try:
            import geoopt
            self.manifold = geoopt.Lorentz(k=curvature)
            self._geoopt_available = True
        except ImportError:
            warnings.warn(
                "geoopt no esta instalado. LorentzProjection usara aproximacion euclidea. "
                "Instalar con: pip install geoopt",
                stacklevel=2,
            )
            self.manifold = None
            self._geoopt_available = False

        self.lorentz_dim = lorentz_dim
        # Proyeccion euclidea al espacio tangente (dim-1 porque expmap agrega la coord temporal)
        self.tangent_proj = nn.Linear(feature_dim, lorentz_dim - 1)
        # MLP para penalizacion radial basada en variables clinicas (age, apoe4)
        self.radial_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # salida en (0, 1): factor de escala del radio
        )
        self.layer_norm = nn.LayerNorm(lorentz_dim - 1)

    def forward(
        self,
        gap_vector: torch.Tensor,
        apoe4: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gap_vector: (B, feature_dim) vector GAP del backbone
            apoe4:      (B,) indicador APOE4 [0, 1, 2]
            age:        (B,) edad normalizada

        Returns:
            (B, lorentz_dim) punto en el manifold de Lorentz
        """
        tangent = self.tangent_proj(gap_vector)          # (B, lorentz_dim - 1)
        tangent = self.layer_norm(tangent)

        # Penalizacion radial: age y apoe4 modulan la magnitud del vector tangente
        clinical_input = torch.stack([
            age.float(),
            apoe4.float(),
        ], dim=1)                                         # (B, 2)
        radial_scale = self.radial_mlp(clinical_input)   # (B, 1)
        tangent = tangent * (1.0 + radial_scale)          # escala adaptativa

        if self._geoopt_available:
            # Acotar norma espacial del tangente: expmap0 usa sinh(||u||_L / sqrt(k))
            # y con LayerNorm en alta dimension ||v||_2 ~ sqrt(d) puede hacer explotar sinh.
            tn = tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            tangent = tangent / tn * 0.5
            # Construir vector tangente en el origen: (0, v) en T_o(H^n)
            tangent_origin = torch.cat([
                torch.zeros(tangent.size(0), 1, device=tangent.device, dtype=tangent.dtype),
                tangent,
            ], dim=1)                                     # (B, lorentz_dim)
            return self.manifold.expmap0(tangent_origin)  # (B, lorentz_dim)
        else:
            # Fallback: normalizacion L2 aproximando la superficie del hiperboloide
            tangent_origin = torch.cat([
                torch.ones(tangent.size(0), 1, device=tangent.device, dtype=tangent.dtype),
                tangent,
            ], dim=1)
            return F.normalize(tangent_origin, p=2, dim=1)


class LorentzLinearClassificationHead(nn.Module):
    """
    Clasificador lineal hiperbolico usando el producto interno de Minkowski.

    Cada clase k tiene un prototipo w_k en el manifold de Lorentz almacenado
    como geoopt.ManifoldParameter. El logit para la clase k es el negativo del
    producto interno de Lorentz entre el punto x y el prototipo w_k.

    logit_k = -<x, w_k>_L = -(−x_0 * w_k_0 + sum_i x_i * w_k_i)

    Args:
        lorentz_dim:  Dimension del espacio de Lorentz (incluye coord temporal).
        num_classes:  Numero de clases de salida.
        curvature:    Curvatura del manifold (debe coincidir con el projector).
    """

    def __init__(self, lorentz_dim: int, num_classes: int, curvature: float = 1.0):
        super().__init__()
        try:
            import geoopt
            manifold = geoopt.Lorentz(k=curvature)
            # Inicializar prototipos SEPARADOS en el hiperboloide.
            # Cada clase k tiene una dirección espacial distinta con norma ~1.0,
            # lo que coloca los prototipos a distancia hiperbólica cosh^{-1}(sqrt(2)) entre sí.
            # Si todos parten del origen [1,0,...,0] los logits son idénticos y el
            # gradiente inicial es prácticamente cero → el modelo nunca aprende a discriminar.
            torch.manual_seed(0)
            spatial = torch.randn(num_classes, lorentz_dim - 1)
            # Escalar a norma ~3.0 para que los prototipos estén bien separados en el
            # hiperboloide (distancia hiperbólica acosh(√10) ≈ 1.82).
            # Norma 1 → distancia ~0.88; muy pequeña → logits casi iguales y gradiente débil.
            spatial = spatial / spatial.norm(dim=-1, keepdim=True) * 3.0
            time_coord = (1.0 + spatial.pow(2).sum(dim=-1)).sqrt()  # satisface hiperboloide
            init_weight = torch.cat([time_coord.unsqueeze(1), spatial], dim=1)
            self.weight = geoopt.ManifoldParameter(init_weight, manifold=manifold)
            self._geoopt_available = True
        except ImportError:
            self.weight = nn.Parameter(torch.randn(num_classes, lorentz_dim) * 0.1)
            self._geoopt_available = False

        self.num_classes = num_classes
        self.lorentz_dim = lorentz_dim

    def _lorentz_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Producto interno de Minkowski: -x0*y0 + sum(x_i * y_i, i>0)."""
        # x: (B, D), y: (C, D)
        # Retorna (B, C)
        time_part = x[:, 0:1] * y[:, 0:1].T                # (B, C)
        space_part = x[:, 1:] @ y[:, 1:].T                  # (B, C)
        return -time_part + space_part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, lorentz_dim) puntos en el manifold de Lorentz

        Returns:
            (B, num_classes) logits euclidianos
        """
        if self._geoopt_available:
            try:
                import geoopt.manifolds.lorentz.math as lmath
                # lmath.inner requiere u y v broadcastables con la misma forma en dim=-1;
                # expandir x a (B, C, D) para emparejar cada prototipo w_k.
                b = x.size(0)
                x_exp = x.unsqueeze(1).expand(-1, self.num_classes, -1)
                w_exp = self.weight.unsqueeze(0).expand(b, -1, -1)
                inner = lmath.inner(x_exp, w_exp, dim=-1)
                return -inner
            except Exception:
                pass
        # Fallback: producto interno de Minkowski manual
        return -self._lorentz_inner(x, self.weight)


class UltimateNeuroFM(nn.Module):
    """
    UltimateNeuroFM: ResEnc-L SSL (congelado) + LoRA 3D + cabeza de clasificacion.

    Por defecto (`head="euclidean"`): GAP + fusion de 4 covariables clinicas + MLP.
    Opcional (`head="hyperbolic"`): proyeccion Lorentz + prototipos (geoopt).

    Marcadores de instancia:
        uses_clinical  → DataLoader con vector clinico (4 dims)
        uses_ordinal   → False
        is_hyperbolic  → True solo si head="hyperbolic"

    Args:
        weights_path: Checkpoint SSL resenc_l_ssl3d.pth.
        head:         "euclidean" (default) o "hyperbolic".
        lora_rank:    Rango LoRA (default: 4).
        lora_stages:  Prefijos de stage para LoRA (default: stages.4, stages.5). [] = sin LoRA.
        freeze_backbone: Si False, descongela ResEnc-L y no inyecta LoRA (full fine-tune).
        feature_dim:  Canales GAP (default: 320).
        num_clinical: Dimension del vector clinico (default: 4).
        lorentz_dim:  Solo para head="hyperbolic".
        curvature:    Solo para head="hyperbolic".
        num_classes:  Clases de salida.
    """

    uses_clinical: bool = True
    uses_ordinal: bool = False

    def __init__(
        self,
        weights_path: str | None = None,
        head: str = "euclidean",
        lora_rank: int = 4,
        lora_stages: list[str] | None = None,
        freeze_backbone: bool = True,
        feature_dim: int = 320,
        num_clinical: int = 4,
        lorentz_dim: int = 257,
        curvature: float = 1.0,
        num_classes: int = cfg.NUM_CLASSES,
    ):
        super().__init__()
        if head not in ("euclidean", "hyperbolic"):
            raise ValueError(f"head debe ser 'euclidean' o 'hyperbolic', got {head!r}")

        self.head = head
        self.is_hyperbolic = head == "hyperbolic"
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone

        if weights_path is None:
            weights_path = str(cfg.RESENC_SSL_CHECKPOINT)
        if lora_stages is None:
            # ResEnc-L real (nnssl): stages.0..5; LoRA en las dos capas más profundas
            lora_stages = ["stages.4", "stages.5"]

        # --- Backbone ---
        backbone = self._load_backbone(weights_path, feature_dim)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            if lora_stages:
                inject_lora_into_backbone(backbone, stages=lora_stages, rank=lora_rank)
                print(
                    f"[UltimateNeuroFM] Backbone congelado + LoRA en {lora_stages} "
                    f"(rank={lora_rank})"
                )
            else:
                print("[UltimateNeuroFM] Backbone congelado, sin LoRA (linear probe)")
        else:
            print("[UltimateNeuroFM] Full fine-tuning: backbone entrenable, sin LoRA")

        self.feature_extractor = backbone

        # --- Pooling ---
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        if head == "euclidean":
            self.projector = None
            fused_dim = feature_dim + num_clinical
            head_dropout = 0.5 if not freeze_backbone else 0.3
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=head_dropout),
                nn.Linear(128, num_classes),
            )
        else:
            self.projector = LorentzProjectionWithRadialPenalty(
                feature_dim=feature_dim,
                lorentz_dim=lorentz_dim,
                curvature=curvature,
            )
            self.classifier = LorentzLinearClassificationHead(
                lorentz_dim=lorentz_dim,
                num_classes=num_classes,
                curvature=curvature,
            )

    @staticmethod
    def _load_backbone(weights_path: str, feature_dim: int) -> nn.Module:
        """Carga ResEncL_UNet_Encoder o usa un mock si no esta disponible."""
        weights_file = Path(weights_path).expanduser()
        if not weights_file.is_file():
            for candidate in (
                cfg.PROJECT_ROOT / weights_path,
                cfg.WEIGHTS_DIR / Path(weights_path).name,
                cfg.RESENC_SSL_CHECKPOINT,
            ):
                if candidate.is_file():
                    weights_file = candidate
                    break

        # Intentar importar desde BrainFM4Challenges (debe estar en PYTHONPATH o clonado)
        try:
            from BrainFM4Challenges.models import ResEncL_UNet_Encoder  # type: ignore
            backbone = ResEncL_UNet_Encoder(out_channels=feature_dim)
            if weights_file.is_file():
                try:
                    matched, n_miss, n_unexp = _load_backbone_state_dict(backbone, weights_file)
                    print(
                        f"[UltimateNeuroFM] Backbone cargado desde {weights_file.resolve()} "
                        f"(tensores_cargados={matched}/{len(backbone.state_dict())}, "
                        f"missing={n_miss}, unexpected={n_unexp})"
                    )
                    if n_miss > 0:
                        print(
                            f"[UltimateNeuroFM] INFO: {n_miss} capas sin peso en el checkpoint "
                            "(normal si el encoder no coincide del todo con el SSL real)."
                        )
                except Exception as e:
                    warnings.warn(
                        f"No se pudo leer el checkpoint {weights_file}: {e}. "
                        "Usando pesos aleatorios del backbone.",
                        stacklevel=2,
                    )
            else:
                print(
                    f"[UltimateNeuroFM] WARN: checkpoint no encontrado "
                    f"(buscado: {weights_path}, {cfg.RESENC_SSL_CHECKPOINT}). "
                    "Backbone aleatorio (sin SSL)."
                )
            return backbone

        except ImportError:
            warnings.warn(
                "No se pudo importar BrainFM4Challenges.models.ResEncL_UNet_Encoder. "
                "Usando backbone interno (_build_mock_backbone). "
                "Instala el paquete local: cd BrainFM4Challenges && pip install -e . "
                "(el repo upstream no incluye pyproject.toml; el proyecto añade un stub instalable).",
                stacklevel=3,
            )
            backbone = _build_mock_backbone(feature_dim=feature_dim)
            if weights_file.exists():
                print(
                    f"[UltimateNeuroFM] WARN: {weights_file} encontrado pero el backbone "
                    "es mock (BrainFM4Challenges no importable). Pesos ignorados."
                )
            return backbone

    def forward(self, x: torch.Tensor, clinical: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (B, 1, D, H, W) tensor MRI 3D
            clinical: (B, 4) vector [age_norm, sex, educ, apoe4]; si None se usa ceros

        Returns:
            (B, num_classes) logits
        """
        if clinical is None:
            clinical = torch.zeros(x.size(0), 4, device=x.device, dtype=x.dtype)

        features_3d = self.feature_extractor(x)
        gap_vector = self.pool(features_3d)  # (B, feature_dim)

        if self.head == "euclidean":
            fused = torch.cat([gap_vector, clinical], dim=1)
            return self.classifier(fused)

        age = clinical[:, 0]
        apoe4 = clinical[:, 3]
        hyp_features = self.projector(gap_vector, apoe4, age)
        return self.classifier(hyp_features)


# ---------------------------------------------------------------------------
# EfficientNet25D — cortes axiales 2D + soft attention (paradigma AXIAL / 2.5D)
# ---------------------------------------------------------------------------

class EfficientNet25DAttention(nn.Module):
    """
    Volumen 3D -> N cortes axiales -> EfficientNet-B0 (ImageNet) -> atencion suave -> logits.

    Inspirado en AXIAL (Lozupone et al.): alternativa 2.5D al coste de CNN 3D completa.
    """

    def __init__(
        self,
        num_classes: int = cfg.NUM_CLASSES,
        num_slices: int | None = None,
        slice_size: int | None = None,
        pretrained_2d: bool = True,
        dropout_prob: float | None = None,
        freeze_blocks: int | None = None,
    ):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        self.num_slices = num_slices or cfg.EFFICIENTNET25D_NUM_SLICES
        self.slice_size = slice_size or cfg.EFFICIENTNET25D_SLICE_SIZE
        self.embed_dim = cfg.EFFICIENTNET25D_EMBED_DIM
        drop = cfg.EFFICIENTNET25D_DROPOUT if dropout_prob is None else dropout_prob
        freeze_n = (
            cfg.EFFICIENTNET25D_FREEZE_BLOCKS if freeze_blocks is None else freeze_blocks
        )
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_2d else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        if freeze_n > 0:
            for i, block in enumerate(self.features):
                if i < freeze_n:
                    for p in block.parameters():
                        p.requires_grad = False
        self.attention = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.Tanh(),
            nn.Dropout(p=drop),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(self.embed_dim, num_classes),
        )

    def _slice_indices(self, depth: int, device: torch.device) -> torch.Tensor:
        if depth <= self.num_slices:
            return torch.arange(depth, device=device)
        return torch.linspace(0, depth - 1, self.num_slices, device=device).long()

    def encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, embed_dim) features por corte axial."""
        b, _, depth, h, w = x.shape
        idx = self._slice_indices(depth, x.device)
        slices = x[:, 0, idx, :, :]
        s = slices.shape[1]
        flat = slices.reshape(b * s, 1, h, w)
        flat = F.interpolate(
            flat,
            size=(self.slice_size, self.slice_size),
            mode="bilinear",
            align_corners=False,
        )
        flat = flat.repeat(1, 3, 1, 1)
        feats = self.features(flat)
        feats = self.avgpool(feats).flatten(1)
        return feats.view(b, s, -1)

    def pool_attention(self, slice_feats: torch.Tensor) -> torch.Tensor:
        """(B, embed_dim) embedding volumetrico ponderado."""
        attn_logits = self.attention(slice_feats).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        return (slice_feats * attn).sum(dim=1)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_attention(self.encode_slices(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_embedding(x))


def adapt_efficientnet25d_state_dict(state_dict: dict) -> dict:
    """
    Migra checkpoints entrenados sin Dropout (Linear directo) a la cabeza actual.

    Antes: attention[0,1,2]=Linear,Tanh,Linear | classifier=Linear
    Ahora: attention[0,1,2,3]=Linear,Tanh,Dropout,Linear | classifier=Dropout,Linear
    """
    sd = dict(state_dict)
    if "classifier.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd.pop("classifier.weight")
        sd["classifier.1.bias"] = sd.pop("classifier.bias")
    if "attention.2.weight" in sd and "attention.3.weight" not in sd:
        sd["attention.3.weight"] = sd.pop("attention.2.weight")
        sd["attention.3.bias"] = sd.pop("attention.2.bias")
    return sd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "resnet10",
    "simple3dcnn",
    "densenet121",
    "densenet121_coral",
    "multimodal_densenet",
    "ultimate_fm",
    "efficientnet25d",
]


def get_model(
    name: str = "resnet10",
    ordinal: bool = False,
    num_classes: int | None = None,
    weights_path: str | None = None,
    lora_stages: list[str] | None = None,
    freeze_backbone: bool | None = None,
    pretrained: bool = False,
    backbone_name: str = "resnet10",
    dropout_prob: float | None = None,
    yaware_pretrained: bool = False,
    yaware_weights_path: str | None = None,
) -> nn.Module:
    """Instancia un modelo por nombre.

    Args:
        name:         'resnet10', 'simple3dcnn', 'densenet121', 'multimodal_densenet'
                      o 'ultimate_fm'.
        ordinal:      Si True, los modelos DenseNet emiten 2 logits ordinales en lugar de
                      cfg.NUM_CLASSES. Ignorado para resnet10, simple3dcnn y ultimate_fm.
        weights_path: Ruta al checkpoint del backbone pre-entrenado para 'ultimate_fm'.
        lora_stages:    None = default LoRA; [] = linear probe sin LoRA.
        freeze_backbone: None = True; False = full fine-tune del ResEnc-L.
        pretrained:      MedicalNet 3D para resnet10/resnet18 (cabecera nueva).
        backbone_name:   'resnet10' o 'resnet18' (con --pretrained).
        yaware_pretrained: Carga encoder DenseNet121 desde checkpoint y-Aware BHB-10K.
        yaware_weights_path: Ruta al .pth (default: cfg.YAWARE_DENSENET_CHECKPOINT).

    Returns:
        nn.Module listo para .to(device).
    """
    out_classes = num_classes if num_classes is not None else cfg.NUM_CLASSES
    if name == "resnet10":
        return AlzheimerResNet(
            num_classes=out_classes,
            pretrained=pretrained,
            backbone_name=backbone_name if pretrained else "resnet10",
        )
    elif name == "simple3dcnn":
        return Simple3DCNN(num_classes=out_classes)
    elif name == "densenet121":
        return AlzheimerDenseNet(
            ordinal=ordinal,
            num_classes=out_classes,
            dropout_prob=dropout_prob,
            yaware_pretrained=yaware_pretrained,
            yaware_weights_path=yaware_weights_path,
        )
    elif name == "densenet121_coral":
        return AlzheimerDenseNet(
            ordinal=True,
            num_classes=out_classes,
            dropout_prob=dropout_prob,
            yaware_pretrained=yaware_pretrained,
            yaware_weights_path=yaware_weights_path,
        )
    elif name == "multimodal_densenet":
        if ordinal:
            raise ValueError(
                "multimodal_densenet no soporta --ordinal/CORAL. "
                "Usa densenet121_coral."
            )
        return MultimodalDenseNet(ordinal=False, num_classes=out_classes)
    elif name == "ultimate_fm":
        fm_kwargs: dict = {"weights_path": weights_path, "num_classes": out_classes}
        if lora_stages is not None:
            fm_kwargs["lora_stages"] = lora_stages
        if freeze_backbone is not None:
            fm_kwargs["freeze_backbone"] = freeze_backbone
        return UltimateNeuroFM(**fm_kwargs)
    elif name == "efficientnet25d":
        if ordinal:
            raise ValueError("efficientnet25d no soporta --ordinal/CORAL.")
        return EfficientNet25DAttention(
            num_classes=out_classes,
            dropout_prob=dropout_prob,
        )
    else:
        raise ValueError(f"Modelo '{name}' no reconocido. Opciones: {AVAILABLE_MODELS}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(1, 1, 96, 96, 96, device=device)
    dummy_clinical = torch.randn(1, 4, device=device)

    configs: list[tuple[str, bool]] = [
        (name, False) for name in AVAILABLE_MODELS if name != "densenet121_coral"
    ]
    configs += [("densenet121", True), ("densenet121_coral", False)]

    for name, ordinal in configs:
        print(f"\n{'=' * 50}")
        print(f"Modelo: {name} | ordinal={ordinal}")
        print(f"{'=' * 50}")
        try:
            model = get_model(name, ordinal=ordinal).to(device).eval()
        except ValueError as exc:
            print(f"  SKIP: {exc}")
            continue
        if getattr(model, "uses_clinical", False):
            out = model(dummy, dummy_clinical)
        else:
            out = model(dummy)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Device:        {device}")
        print(f"  Input shape:   {dummy.shape}")
        if getattr(model, "uses_clinical", False):
            print(f"  Clinical shape:{dummy_clinical.shape}")
        print(f"  Output shape:  {out.shape}")
        print(f"  Parametros:    {n_params:,}  (entrenables: {n_trainable:,})")
        uses_ord = getattr(model, "uses_ordinal", False)
        expected_out = cfg.NUM_CLASSES - 1 if uses_ord else cfg.NUM_CLASSES
        assert out.shape == (1, expected_out), f"Esperado (1, {expected_out}), got {out.shape}"
        if uses_ord:
            assert getattr(model, "uses_coral", False), "ordinal requiere cabeza CORAL"
        print(f"  CHECK PASSED")
