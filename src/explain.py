"""
explain.py — Grad-CAM 3D para visualización de regiones cerebrales relevantes.

Genera mapas de activación de clase ponderados por gradiente (Grad-CAM) sobre
muestras del test set correctamente clasificadas como AD o MCI, mostrando qué
regiones cerebrales priorizan los modelos para la detección de Alzheimer.

Genera en outputs/<run_name>/explain/:
    gradcam_sample_00_label{L}_pred{P}.png  — figura 3 paneles por muestra

Uso:
    python -m src.explain --run full_100ep --dataset oasis3
    python -m src.explain --run full_100ep --dataset oasis3 --n-samples 3 --classes 2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.visualize import GradCAM

from src.config import cfg
from src.dataset import get_dataloader
from src.model import AVAILABLE_MODELS, get_model


CLASS_NAMES = ["CN", "MCI", "AD"]

# Capa objetivo para Grad-CAM por arquitectura.
# DenseNet: net.features.denseblock4 es el último bloque denso con mapa espacial
# presente (~3x3x3 para entrada 96³). net.class_layers viene DESPUÉS del avg_pool
# (colapsa dimensiones espaciales), por lo que Grad-CAM sobre ella produce heatmaps
# uniformes sin localización.
TARGET_LAYERS: dict[str, str] = {
    "resnet10":    "net.layer4",
    "densenet121": "net.features.denseblock4",
}

# Mapeo de clase nominal (0=CN,1=MCI,2=AD) al índice de logit ordinal.
# En modo ordinal el modelo produce 2 logits: [P(Y>=MCI), P(Y>=AD)].
# Para Grad-CAM usamos el logit más discriminativo para la clase pedida.
_ORDINAL_CLASS_IDX: dict[int, int] = {
    0: 0,   # CN  → umbral P(Y>=MCI); activación baja = CN
    1: 0,   # MCI → umbral P(Y>=MCI)
    2: 1,   # AD  → umbral P(Y>=AD)
}


def _resolve_class_idx(class_idx: int, uses_ordinal: bool) -> int:
    """Traduce class_idx nominal al índice de logit correcto según el modo del modelo."""
    if uses_ordinal:
        return _ORDINAL_CLASS_IDX.get(class_idx, class_idx)
    return class_idx


# ---------------------------------------------------------------------------
# Grad-CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(
    cam: GradCAM,
    image: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    """
    Calcula el mapa Grad-CAM para una imagen y clase dadas.

    Args:
        cam:       Instancia de monai.visualize.GradCAM ya inicializada.
        image:     Tensor (1, 1, D, H, W) en el device del modelo.
        class_idx: Índice de clase para el que calcular el mapa.

    Returns:
        heatmap: np.ndarray (D, H, W) normalizado a [0, 1].
    """
    image = image.clone().requires_grad_(True)
    with torch.enable_grad():
        result = cam(x=image, class_idx=class_idx)  # (1, 1, D, H, W)

    heatmap = result[0, 0].detach().cpu().numpy()
    # Normalizar a [0, 1]
    lo, hi = heatmap.min(), heatmap.max()
    heatmap = (heatmap - lo) / (hi - lo + 1e-8)
    return heatmap


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_figure(
    img_np: np.ndarray,
    heatmap: np.ndarray,
    true_label: int,
    pred_label: int,
    sample_idx: int,
    save_path: Path,
) -> None:
    """
    Genera y guarda una figura de 3 paneles (Axial, Coronal, Sagital)
    con el MRI en escala de grises y el heatmap Grad-CAM superpuesto.

    Args:
        img_np:     Volumen MRI (D, H, W) con valores en [0, 1].
        heatmap:    Mapa Grad-CAM (D, H, W) normalizado a [0, 1].
        true_label: Etiqueta real (índice de clase).
        pred_label: Predicción del modelo (índice de clase).
        sample_idx: Número de muestra para el título.
        save_path:  Ruta donde guardar el PNG.
    """
    D, H, W = img_np.shape
    d_mid, h_mid, w_mid = D // 2, H // 2, W // 2

    slices = {
        "Axial (D/2)":    (img_np[d_mid, :, :],  heatmap[d_mid, :, :]),
        "Coronal (H/2)":  (img_np[:, h_mid, :],  heatmap[:, h_mid, :]),
        "Sagital (W/2)":  (img_np[:, :, w_mid],  heatmap[:, :, w_mid]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Grad-CAM | Muestra {sample_idx + 1} | "
        f"Real: {CLASS_NAMES[true_label]}  Pred: {CLASS_NAMES[pred_label]}",
        fontsize=13, fontweight="bold",
    )

    for ax, (title, (mri_slice, cam_slice)) in zip(axes, slices.items()):
        ax.imshow(mri_slice, cmap="gray", origin="lower")
        im = ax.imshow(cam_slice, cmap="viridis", alpha=0.45, origin="lower",
                       vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Colorbar compartida
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Activación Grad-CAM")

    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def explain_model(
    run_name: str,
    dataset: str = "oasis3",
    n_samples: int = 5,
    model_name: str | None = None,
    target_classes: list[int] | None = None,
) -> None:
    """
    Genera visualizaciones Grad-CAM para muestras correctamente clasificadas.

    Args:
        run_name:       Nombre de la carpeta en outputs/ con best_model.pth.
        dataset:        'oasis3' o 'oasis1'.
        n_samples:      Número de muestras a visualizar.
        model_name:     Nombre del modelo (auto-detectado del checkpoint si None).
        target_classes: Clases de interés (default: [1, 2] = MCI, AD).
    """
    if target_classes is None:
        target_classes = [1, 2]  # MCI=1, AD=2

    run_dir = cfg.OUTPUTS_DIR / run_name
    model_path = run_dir / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}\n"
            f"Asegúrate de haber entrenado con --run {run_name}"
        )

    explain_dir = run_dir / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --- Cargar checkpoint y modelo (mismo patrón que evaluate.py) ---
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    resolved_model = model_name or checkpoint.get("model_name", "resnet10")
    uses_ordinal = checkpoint.get("uses_ordinal", False)
    use_clinical = checkpoint.get("use_clinical", False)
    print(f"[INFO] Modelo: {resolved_model} | ordinal={uses_ordinal}")

    if resolved_model not in TARGET_LAYERS:
        raise ValueError(
            f"Grad-CAM no soportado para '{resolved_model}'. "
            f"Modelos compatibles: {list(TARGET_LAYERS.keys())}"
        )

    model = get_model(resolved_model, ordinal=uses_ordinal).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"[INFO] Checkpoint: epoch {checkpoint['epoch']} | "
        f"val_loss={checkpoint['val_loss']:.4f} | "
        f"val_acc={checkpoint['val_accuracy']:.2%}"
    )

    # --- Grad-CAM ---
    target_layer = TARGET_LAYERS[resolved_model]
    print(f"[INFO] Target layer: {target_layer}")
    cam = GradCAM(nn_module=model, target_layers=target_layer)

    # --- DataLoader test set (batch_size=1 para procesar muestra a muestra) ---
    loader = get_dataloader(
        "test",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        dataset=dataset,
        use_clinical=use_clinical,
    )
    print(
        f"[INFO] Test set: {len(loader.dataset)} muestras | "
        f"Clases objetivo: {[CLASS_NAMES[c] for c in target_classes]}"
    )

    collected = 0
    total_scanned = 0

    for batch in loader:
        if collected >= n_samples:
            break

        total_scanned += 1
        image = batch["image"].to(device)   # (1, 1, 96, 96, 96)
        clinical = batch.get("clinical")
        if clinical is not None:
            clinical = clinical.to(device)
        true_label = int(batch["label"].item())

        # Predicción sin gradientes — decodificación compatible con modo ordinal
        with torch.no_grad():
            logits = model(image, clinical) if use_clinical else model(image)
        if uses_ordinal:
            pred_label = int((torch.sigmoid(logits) > 0.5).sum(dim=1).item())
        else:
            pred_label = int(logits.argmax(dim=1).item())

        # Solo muestras correctamente clasificadas de las clases objetivo
        if pred_label != true_label or pred_label not in target_classes:
            continue

        print(
            f"  [Muestra {collected + 1}/{n_samples}] "
            f"Real={CLASS_NAMES[true_label]} Pred={CLASS_NAMES[pred_label]} "
            f"(escaneadas: {total_scanned})"
        )

        # Grad-CAM: traducir class_idx al logit correcto según el modo del modelo
        gradcam_idx = _resolve_class_idx(pred_label, uses_ordinal)
        heatmap = compute_gradcam(cam, image, class_idx=gradcam_idx)  # (D, H, W)

        # Extraer volumen MRI como numpy para visualización
        img_np = image[0, 0].detach().cpu().numpy()  # (D, H, W)

        # Guardar figura
        fname = (
            f"gradcam_sample_{collected:02d}"
            f"_label{true_label}"
            f"_pred{pred_label}.png"
        )
        save_path = explain_dir / fname
        _save_figure(img_np, heatmap, true_label, pred_label, collected, save_path)
        print(f"    → Guardado: {save_path}")

        collected += 1

    if collected == 0:
        print(
            "[WARN] No se encontraron muestras correctamente clasificadas "
            f"para las clases {[CLASS_NAMES[c] for c in target_classes]}.\n"
            "       Prueba con --classes 0,1,2 para incluir CN."
        )
    else:
        print(
            f"\n[OK] {collected} visualizaciones Grad-CAM guardadas en: {explain_dir}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM 3D para visualización de regiones cerebrales"
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Nombre de la carpeta en outputs/ (ej. full_100ep)",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=["oasis1", "oasis3"],
        help="Dataset a utilizar (default: oasis3)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Número de muestras AD/MCI correctas a visualizar (default: 5)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(TARGET_LAYERS.keys()),
        help="Modelo (default: auto-detectar del checkpoint)",
    )
    parser.add_argument(
        "--classes", type=str, default="1,2",
        help="Clases objetivo separadas por coma: 0=CN,1=MCI,2=AD (default: '1,2')",
    )
    args = parser.parse_args()

    target_classes = [int(c.strip()) for c in args.classes.split(",")]

    explain_model(
        run_name=args.run,
        dataset=args.dataset,
        n_samples=args.n_samples,
        model_name=args.model,
        target_classes=target_classes,
    )
