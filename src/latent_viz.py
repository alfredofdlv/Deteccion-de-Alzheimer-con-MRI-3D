"""
latent_viz.py — Auditoría del espacio latente (embeddings CNN + t-SNE / UMAP).

Extrae vectores 1024-d de DenseNet121 (antes de la capa lineal) sobre el test set,
proyecta a 2D con t-SNE y UMAP, y guarda scatter plots coloreados por clase.

Genera en outputs/<run_name>/explain/:
    - latent_tsne_test.png
    - latent_umap_test.png
    - test_embeddings.npz  (embeddings, labels, class_names)

Uso:
    python -m src.latent_viz --run densenet-cropped --dataset oasis3
    python -m src.latent_viz --run densenet-cropped --dataset oasis3 --batch-size 2 --perplexity 30
"""

from __future__ import annotations

import argparse
import time
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
import torch.nn.functional as F
from sklearn.manifold import TSNE

from src.config import cfg
from src.dataset import get_dataloader
from src.inference_utils import CLASS_NAMES, load_model_from_run

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "umap-learn no está instalado. Instálalo con: pip install umap-learn"
    ) from exc


# Colores fijos por clase (colorblind-friendly)
CLASS_COLORS: dict[int, str] = {
    0: "#0072B2",  # CN — azul
    1: "#E69F00",  # MCI — naranja
    2: "#D55E00",  # AD — rojo-anaranjado
}

DENSENET_EMBED_DIM = cfg.DENSENET_EMBED_DIM


def _validate_densenet_extractor(model: torch.nn.Module) -> None:
    """Comprueba que el modelo expone net.features (DenseNet121 MONAI)."""
    if not hasattr(model, "net") or not hasattr(model.net, "features"):
        raise ValueError(
            "Este script requiere un modelo con model.net.features "
            "(p. ej. densenet121). Usa --run con un checkpoint DenseNet."
        )


def extract_embeddings_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> np.ndarray:
    """
    Extrae embeddings GAP 1024-d para un batch.

    Réplica el forward de MONAI DenseNet121 antes de class_layers
    (misma lógica que src/late_fusion.py::_extract_cnn_batch).
    """
    feat = model.net.features(images)
    feat = F.relu(feat, inplace=True)
    feat = F.adaptive_avg_pool3d(feat, 1)
    return feat.flatten(1).cpu().numpy()


def extract_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Itera el test_loader y concatena embeddings + etiquetas.

    Returns:
        embeddings: (N, 1024) float32
        labels:     (N,) int64
    """
    model.eval()
    emb_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    n_batches = len(loader)
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy()

            batch_emb = extract_embeddings_batch(model, images)
            if batch_emb.shape[1] != DENSENET_EMBED_DIM:
                raise RuntimeError(
                    f"Dimensión inesperada: {batch_emb.shape[1]} "
                    f"(esperado {DENSENET_EMBED_DIM})"
                )

            emb_list.append(batch_emb.astype(np.float32))
            label_list.append(labels.astype(np.int64))

            if i % max(1, n_batches // 10) == 0 or i == n_batches:
                elapsed = time.time() - t0
                print(
                    f"\r  Extracción [{i}/{n_batches}] {i / n_batches:>6.1%} | "
                    f"{elapsed:.0f}s",
                    end="",
                    flush=True,
                )

    print()
    embeddings = np.vstack(emb_list)
    labels = np.concatenate(label_list)
    return embeddings, labels


def _clamp_perplexity(perplexity: int, n_samples: int) -> int:
    """t-SNE requiere perplexity < n_samples; sklearn recomienda < n/3."""
    max_p = max(5, min(perplexity, (n_samples - 1) // 3))
    if max_p != perplexity:
        print(
            f"[INFO] perplexity ajustado: {perplexity} -> {max_p} "
            f"(N={n_samples})"
        )
    return max_p


def run_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    seed: int = cfg.RANDOM_SEED,
) -> np.ndarray:
    """Proyecta embeddings a 2D con t-SNE."""
    n = embeddings.shape[0]
    p = _clamp_perplexity(perplexity, n)
    print(f"[INFO] t-SNE (perplexity={p}, N={n})...")
    reducer = TSNE(
        n_components=2,
        perplexity=p,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(embeddings)


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = cfg.RANDOM_SEED,
) -> np.ndarray:
    """Proyecta embeddings a 2D con UMAP."""
    n = embeddings.shape[0]
    k = min(n_neighbors, n - 1)
    if k != n_neighbors:
        print(f"[INFO] n_neighbors ajustado: {n_neighbors} -> {k} (N={n})")
    print(f"[INFO] UMAP (n_neighbors={k}, min_dist={min_dist}, N={n})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=k,
        min_dist=min_dist,
        random_state=seed,
        metric="euclidean",
    )
    return reducer.fit_transform(embeddings)


def plot_latent_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Scatter 2D coloreado por clase con leyenda y conteos."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for class_idx in range(cfg.NUM_CLASSES):
        mask = labels == class_idx
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=CLASS_COLORS[class_idx],
            label=f"{CLASS_NAMES[class_idx]} (n={n_c})",
            alpha=0.75,
            s=36,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xlabel("Dimensión 1")
    ax.set_ylabel("Dimensión 2")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_latent_space(
    run_name: str,
    dataset: str = "oasis3",
    batch_size: int = 1,
    perplexity: int = 30,
    n_neighbors: int = 15,
    seed: int = cfg.RANDOM_SEED,
) -> None:
    """
    Orquesta extracción, proyección y guardado de artefactos de espacio latente.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model, checkpoint, run_dir = load_model_from_run(run_name, device=device)
    model_name = checkpoint.get("model_name", "densenet121")
    if model_name != "densenet121":
        print(f"[WARN] Modelo '{model_name}'; se asume extractor DenseNet net.features")

    _validate_densenet_extractor(model)
    use_clinical = checkpoint.get("use_clinical", False)
    if use_clinical:
        print("[WARN] Checkpoint multimodal: se ignoran covariables clínicas en embeddings")

    print(
        f"[INFO] Checkpoint: epoch {checkpoint['epoch']} | "
        f"val_clinical_f2={checkpoint.get('val_clinical_f2', float('nan')):.4f}"
    )

    explain_dir = run_dir / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(
        "test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        dataset=dataset,
        use_clinical=use_clinical,
    )
    n_samples = len(loader.dataset)
    print(f"[INFO] Test set: {n_samples} muestras | batch_size={batch_size}")

    print("\n[1/3] Extrayendo embeddings CNN...")
    embeddings, labels = extract_embeddings(model, loader, device)
    print(f"       Shape: {embeddings.shape} | labels: {labels.shape}")

    # Liberar VRAM antes de proyecciones CPU
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    npz_path = explain_dir / "test_embeddings.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        labels=labels,
        class_names=np.array(CLASS_NAMES),
        run_name=run_name,
        dataset=dataset,
    )
    print(f"       Guardado: {npz_path}")

    print("\n[2/3] Proyección t-SNE...")
    tsne_coords = run_tsne(embeddings, perplexity=perplexity, seed=seed)
    tsne_path = explain_dir / "latent_tsne_test.png"
    plot_latent_scatter(
        tsne_coords,
        labels,
        title=f"t-SNE — espacio latente DenseNet121 | {run_name} | N={n_samples}",
        save_path=tsne_path,
    )
    print(f"       Guardado: {tsne_path}")

    print("\n[3/3] Proyección UMAP...")
    umap_coords = run_umap(embeddings, n_neighbors=n_neighbors, seed=seed)
    umap_path = explain_dir / "latent_umap_test.png"
    plot_latent_scatter(
        umap_coords,
        labels,
        title=f"UMAP — espacio latente DenseNet121 | {run_name} | N={n_samples}",
        save_path=umap_path,
    )
    print(f"       Guardado: {umap_path}")

    counts = {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(cfg.NUM_CLASSES)}
    print(f"\n[OK] Distribución test: {counts}")
    print(f"[OK] Artefactos en: {explain_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualización t-SNE/UMAP del espacio latente DenseNet121 (test set)"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Carpeta en outputs/ con best_model.pth (ej. densenet-cropped)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="oasis3",
        choices=["oasis1", "oasis3"],
        help="Dataset (default: oasis3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size para extracción (default: 1, conservador en VRAM)",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="Perplexity t-SNE (default: 30, se acota según N)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="n_neighbors UMAP (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.RANDOM_SEED,
        help=f"Semilla reproducible (default: {cfg.RANDOM_SEED})",
    )
    args = parser.parse_args()

    visualize_latent_space(
        run_name=args.run,
        dataset=args.dataset,
        batch_size=args.batch_size,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )
