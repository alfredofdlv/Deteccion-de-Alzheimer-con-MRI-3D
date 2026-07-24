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


class ExpectedCostLoss(nn.Module):
    """
    Pérdida sensible al coste: CrossEntropy + lambda * coste esperado (riesgo).

    Forma combinada (recomendada frente a la versión "pura"): mantiene la CE como
    término de estimación de probabilidades bien calibradas (proper scoring rule) y
    añade un término que penaliza asignar masa de probabilidad a clases con alto
    coste de error según la matriz C:

        L = ce_weight * CE(logits, y)  +  lambda * mean_b sum_j P(j|x_b) * C[y_b, j]

    donde P = softmax(logits) y C[i, j] es el coste de predecir j siendo la verdad i.
    Como C[i, i] = 0, el término de coste es 0 cuando la red pone toda la masa en la
    clase correcta, y crece si la reparte hacia errores caros (p. ej. AD -> CN).

    ADVERTENCIA (ver Docs/Explicacion-Metricas): usada en solitario (ce_weight=0) esta
    pérdida NO es un proper scoring rule y puede degenerar/colapsar hacia la clase que
    minimiza el coste dado el desbalance (típicamente sobre-predecir MCI/AD y hundir la
    especificidad). Por eso el default combina CE + lambda*coste. Vigilar colapso con
    la matriz de confusión de validación y con val_balanced_accuracy (ver train.py).

    Args:
        cost_matrix:     Matriz KxK (lista/tupla/tensor). C[i, j] = coste real i -> pred j.
        weight:          Pesos por clase para la CE (imbalance). NO usar multiplicadores
                         clínicos aquí para evitar doble contabilización de la asimetría.
        label_smoothing: Suavizado de etiquetas de la CE (alineado con el proyecto).
        lam:             Peso del término de coste esperado (lambda).
        ce_weight:       Peso de la CE (default 1.0; poner 0.0 = coste puro, arriesgado).
    """

    def __init__(
        self,
        cost_matrix,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
        lam: float = 0.5,
        ce_weight: float = 1.0,
    ) -> None:
        super().__init__()
        C = torch.as_tensor(cost_matrix, dtype=torch.float32)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"cost_matrix debe ser cuadrada KxK, recibido {tuple(C.shape)}")
        self.register_buffer("C", C)
        self.lam = lam
        self.ce_weight = ce_weight
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, K) — salidas crudas del clasificador.
            targets: (B,)   — etiquetas enteras en {0, ..., K-1}.
        """
        targets = targets.long()
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets, weight=w, label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits, dim=1)               # (B, K)
        batch_costs = self.C.to(logits.device)[targets]  # (B, K): fila C[y_b, :]
        expected = (probs * batch_costs).sum(dim=1).mean()
        return self.ce_weight * ce + self.lam * expected


class FocalLoss(nn.Module):
    """
    Focal loss multiclase (Lin et al., 2017) sobre cross-entropy.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        weight:      Pesos por clase (mismo uso que en CrossEntropyLoss).
        gamma:       Factor de enfoque en ejemplos dificiles (default 2.0).
        label_smoothing: Suavizado de etiquetas (alineado con CE del proyecto).
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) — salidas crudas del clasificador.
            targets: (B,)   — etiquetas enteras en {0, ..., C-1}.
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        if self.label_smoothing > 0.0:
            with torch.no_grad():
                smooth = self.label_smoothing / (num_classes - 1)
                target_dist = torch.full_like(log_probs, smooth)
                target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(target_dist * log_probs).sum(dim=1)
            p_t = (probs * target_dist).sum(dim=1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = focal_factor * ce

        if self.weight is not None:
            w = self.weight.to(logits.device)
            class_w = w[targets]
            loss = loss * class_w

        return loss.mean()
