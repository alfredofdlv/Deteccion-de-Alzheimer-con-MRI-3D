"""
losses.py — Pérdidas personalizadas para clasificación ordinal de Alzheimer.

OrdinalClinicalF2Loss
---------------------
Combina dos términos:
  1. BCE ordinal  — dos umbrales binarios que respetan la ordenación CN < MCI < AD.
  2. Soft F2 clínico — aproximación diferenciable del F2 ponderado (60% AD, 30% MCI).

La representación ordinal mapea las etiquetas (B,) a umbrales binarios (B, 2):
    CN  (0) → [0, 0]   (no supera ningún umbral)
    MCI (1) → [1, 0]   (supera P(Y≥MCI), no supera P(Y≥AD))
    AD  (2) → [1, 1]   (supera ambos umbrales)

Probabilidades de clase derivadas de los umbrales (p1, p2 = sigmoid(logits)):
    P(CN)  = 1 - p1
    P(MCI) = relu(p1 - p2)   ← ReLU evita negativos si p2 > p1 en early training
    P(AD)  = p2

ADVERTENCIA: la restricción p1 >= p2 no está forzada. Si el modelo viola la
restricción durante las primeras épocas, P(MCI) → 0 y su gradiente desaparece.
Síntoma: val F2 de MCI bloqueado en 0 durante muchas épocas. Si ocurre, añadir
el término de penalización (monotonicity_penalty) o reducir alpha.

Uso:
    from src.losses import OrdinalClinicalF2Loss
    criterion = OrdinalClinicalF2Loss(
        weights=cfg.CLINICAL_F2_WEIGHTS,
        alpha=0.5,
        pos_weight=compute_pos_weight(dataset),
    )
    loss = criterion(logits, labels)   # logits (B, 2), labels (B,) en {0, 1, 2}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalClinicalF2Loss(nn.Module):
    """
    Pérdida ordinal BCE + soft clinical F2 para clasificación CN/MCI/AD.

    Args:
        weights:             Dict {0: w_cn, 1: w_mci, 2: w_ad} con pesos clínicos
                             para el soft F2 (p. ej. cfg.CLINICAL_F2_WEIGHTS).
        alpha:               Balance entre BCE y soft F2. Loss = bce + alpha * f2_loss.
                             Valores recomendados: 0.3–0.6. Subir si el modelo ignora AD/MCI.
        pos_weight:          Tensor (2,) con N_neg/N_pos para cada umbral ordinal.
                             Compénsala la imbalanza en la parte BCE.
                             Calcular con compute_pos_weight() en train.py.
        monotonicity_lambda: Penalización por violación p2 > p1 (default: 0.0 = desactivado).
                             Activar con λ ≈ 0.1 si F2 de MCI se bloquea en 0.
        smooth:              Epsilon de suavizado en TP/FP/FN para evitar división por cero.
    """

    def __init__(
        self,
        weights: dict[int, float],
        alpha: float = 0.5,
        pos_weight: torch.Tensor | None = None,
        monotonicity_lambda: float = 0.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w = [weights[0], weights[1], weights[2]]
        self.alpha = alpha
        self.monotonicity_lambda = monotonicity_lambda
        self.smooth = smooth

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 2) — salidas crudas del modelo ordinal.
            targets: (B,)   — etiquetas en {0=CN, 1=MCI, 2=AD}.

        Returns:
            Escalar de pérdida.
        """
        B = targets.size(0)
        device = logits.device

        # --- 1. Codificación ordinal de targets (B, 2) ---
        # CN=0 → [0,0]; MCI=1 → [1,0]; AD=2 → [1,1]
        t = targets.long()
        ordinal_targets = torch.zeros(B, 2, device=device, dtype=torch.float32)
        ordinal_targets[:, 0] = (t >= 1).float()   # P(Y >= MCI)
        ordinal_targets[:, 1] = (t >= 2).float()   # P(Y >= AD)

        # --- 2. BCE ordinal con pos_weight opcional ---
        pw = self.pos_weight.to(device) if self.pos_weight is not None else None
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, ordinal_targets, pos_weight=pw, reduction="mean"
        )

        # --- 3. Probabilidades de clase derivadas ---
        probs = torch.sigmoid(logits)          # (B, 2)
        p1 = probs[:, 0]                       # P(Y >= MCI)
        p2 = probs[:, 1]                       # P(Y >= AD)

        p_cn  = 1.0 - p1                       # (B,)
        p_mci = F.relu(p1 - p2)               # (B,)  — ReLU por si p2 > p1
        p_ad  = p2                             # (B,)

        p_classes = torch.stack([p_cn, p_mci, p_ad], dim=1)  # (B, 3)

        # --- 4. One-hot de targets (B, 3) ---
        targets_oh = F.one_hot(t, num_classes=3).float()  # (B, 3)

        # --- 5. TP / FP / FN suaves por clase ---
        # Se calculan sobre el batch completo (suma sobre dim 0)
        tp = (p_classes * targets_oh).sum(dim=0)           # (3,)
        fp = (p_classes * (1.0 - targets_oh)).sum(dim=0)   # (3,)
        fn = ((1.0 - p_classes) * targets_oh).sum(dim=0)   # (3,)

        # --- 6. Soft F2 por clase (β=2 → 5·TP / (5·TP + 4·FN + FP)) ---
        beta2 = 4.0   # β²
        f2_per_class = (
            (1.0 + beta2) * tp
            / ((1.0 + beta2) * tp + beta2 * fn + fp + self.smooth)
        )  # (3,)

        # Pérdida clínica: sum_c w_c * (1 - F2_c)
        w = torch.tensor(self.w, device=device, dtype=torch.float32)
        f2_loss = (w * (1.0 - f2_per_class)).sum()

        # --- 7. Penalización de monotonicidad (opcional) ---
        # Fuerza p1 >= p2: penaliza relu(p2 - p1)
        mono_loss = 0.0
        if self.monotonicity_lambda > 0.0:
            mono_loss = self.monotonicity_lambda * F.relu(p2 - p1).mean()

        return bce_loss + self.alpha * f2_loss + mono_loss
