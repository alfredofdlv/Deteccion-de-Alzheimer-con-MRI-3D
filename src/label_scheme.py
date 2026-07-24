"""
label_scheme.py — Esquemas de clasificación (multiclase vs binario CN vs Impaired).

Uso:
    from src.label_scheme import get_scheme, remap_label
    scheme = get_scheme("binary_cn_imp")
    y = remap_label(2, scheme)  # AD -> Impaired (1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.config import cfg

LABEL_SCHEME_CHOICES = ("multiclass", "binary_cn_imp", "binary_mci_ad")


@dataclass(frozen=True)
class LabelScheme:
    name: str
    num_classes: int
    class_names: List[str]
    clinical_f2_weights: Dict[int, float]
    clinical_weight_multipliers: Dict[int, float]
    _remap: Dict[int, int]
    include_raw_labels: tuple[int, ...] | None = None

    def remap(self, label: int) -> int:
        return self._remap.get(int(label), int(label))

    def validate_label(self, label: int) -> None:
        if label < 0 or label >= self.num_classes:
            raise ValueError(
                f"Label {label} fuera de rango para esquema {self.name!r} "
                f"(0..{self.num_classes - 1})"
            )


_SCHEMES: Dict[str, LabelScheme] = {
    "multiclass": LabelScheme(
        name="multiclass",
        num_classes=cfg.NUM_CLASSES,
        class_names=["CN", "MCI", "AD"],
        clinical_f2_weights=dict(cfg.CLINICAL_F2_WEIGHTS),
        clinical_weight_multipliers=dict(cfg.CLINICAL_WEIGHT_MULTIPLIERS),
        _remap={0: 0, 1: 1, 2: 2},
    ),
    "binary_cn_imp": LabelScheme(
        name="binary_cn_imp",
        num_classes=2,
        class_names=["CN", "Impaired"],
        clinical_f2_weights={0: 0.25, 1: 0.75},
        clinical_weight_multipliers={0: 1.0, 1: 2.0},
        _remap={0: 0, 1: 1, 2: 1},
    ),
    "binary_mci_ad": LabelScheme(
        name="binary_mci_ad",
        num_classes=2,
        class_names=["MCI", "AD"],
        clinical_f2_weights={0: 0.5, 1: 0.5},
        clinical_weight_multipliers={0: 1.0, 1: 2.0},
        _remap={1: 0, 2: 1},
        include_raw_labels=(1, 2),
    ),
}


def get_scheme(name: str = "multiclass") -> LabelScheme:
    if name not in _SCHEMES:
        raise ValueError(
            f"Esquema de labels desconocido: {name!r}. "
            f"Opciones: {list(_SCHEMES)}"
        )
    return _SCHEMES[name]


def remap_label(label: int, scheme_name: str = "multiclass") -> int:
    return get_scheme(scheme_name).remap(label)


def compute_clinical_f2_for_scheme(
    labels: list[int],
    preds: list[int],
    scheme_name: str = "multiclass",
    beta: float = 2.0,
) -> float:
    from sklearn.metrics import fbeta_score

    scheme = get_scheme(scheme_name)
    f2_per_class = fbeta_score(
        labels,
        preds,
        beta=beta,
        average=None,
        labels=list(range(scheme.num_classes)),
        zero_division=0,
    )
    return float(
        sum(scheme.clinical_f2_weights[c] * f2_per_class[c] for c in range(scheme.num_classes))
    )
