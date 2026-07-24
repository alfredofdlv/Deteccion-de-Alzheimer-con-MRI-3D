"""
qc_mni_registration.py — Control visual del pipeline MNI (skull-strip + registro afín).

Procesa N sujetos y guarda montajes axial/sagital/coronal: crudo vs brain vs MNI.

Uso:
    python scripts/qc_mni_registration.py --n 10
    python scripts/qc_mni_registration.py --n 5 --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.spatial_preprocess import (
    detect_skull_strip_backend,
    register_affine_to_mni,
    skull_strip,
)


def _middle_slices(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve cortes central axial, sagital y coronal (2D).

    El volumen está en orientación canónica RAS+, por lo que los ejes son
    (X = izquierda-derecha, Y = anterior-posterior, Z = inferior-superior):
      - Axial   = plano X-Y (fija Z): vol[:, :, z]
      - Sagital = plano Y-Z (fija X): vol[x, :, :]
      - Coronal = plano X-Z (fija Y): vol[:, y, :]
    """
    x, y, z = vol.shape
    axial = vol[:, :, z // 2]
    sagittal = vol[x // 2, :, :]
    coronal = vol[:, y // 2, :]
    return axial, sagittal, coronal


def _plot_triplet(
    raw: np.ndarray,
    brain: np.ndarray,
    registered: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    views = ("Axial", "Sagittal", "Coronal")
    stages = ("Raw RAS", "Skull-stripped", "MNI Affine")
    vols = [
        _middle_slices(raw),
        _middle_slices(brain),
        _middle_slices(registered),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(title, fontsize=11)

    for row, (stage, slices) in enumerate(zip(stages, vols)):
        for col, (view_name, slc) in enumerate(zip(views, slices)):
            ax = axes[row, col]
            vmin, vmax = np.percentile(slc, [1, 99])
            ax.imshow(slc.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(view_name, fontsize=9)
            if col == 0:
                ax.set_ylabel(stage, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="QC visual del preprocesado MNI")
    parser.add_argument("--dataset", type=str, default="oasis3", choices=["oasis3"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=10, help="Numero de sujetos a procesar")
    parser.add_argument(
        "--out-dir", type=Path, default=cfg.OUTPUTS_DIR / "qc_mni",
        help="Directorio de salida para montajes",
    )
    args = parser.parse_args()

    backend = detect_skull_strip_backend()
    print(f"[INFO] Skull-strip backend: {backend}")

    nifti_csv = cfg.DATA_SPLITS_DIR / f"{args.dataset}_{args.split}.csv"
    if not nifti_csv.exists():
        parser.error(
            f"No existe {nifti_csv}. "
            "Genera splits NIfTI con scripts/prepare_oasis3_nifti_splits.py"
        )
    nifti_rows = pd.read_csv(nifti_csv).head(args.n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"QC MNI — backend skull-strip: {backend}",
        f"Registration: {cfg.MNI_REGISTRATION_TYPE} to {cfg.MNI_TEMPLATE_NAME}",
        "",
    ]

    for i, (_, row) in enumerate(nifti_rows.iterrows(), 1):
        img_path = row["image_path"]
        subject_id = row.get("subject_id", Path(img_path).stem)
        scan_tag = Path(img_path).stem.replace(".nii", "")
        work_dir = cfg.INTERMEDIATE_DIR / "qc_mni" / scan_tag
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(nifti_rows)}] {subject_id}")
        try:
            raw_arr = nib.as_closest_canonical(
                nib.load(img_path),
            ).get_fdata(dtype=np.float32)

            brain_path, _mask_path, used_backend = skull_strip(img_path, work_dir)
            brain_arr = nib.load(str(brain_path)).get_fdata(dtype=np.float32)

            warped, _ = register_affine_to_mni(brain_path)
            reg_arr = warped.numpy().astype(np.float32)
            if reg_arr.ndim == 4:
                reg_arr = reg_arr[..., 0]

            out_png = out_dir / f"{scan_tag}_qc.png"
            _plot_triplet(
                raw_arr, brain_arr, reg_arr,
                title=f"{scan_tag} | {used_backend} + Affine MNI152",
                out_path=out_png,
            )
            summary_lines.append(f"OK  {scan_tag} -> {out_png.name}")
            print(f"  -> {out_png}")
        except Exception as e:
            summary_lines.append(f"ERR {scan_tag}: {type(e).__name__}: {e}")
            print(f"  [ERROR] {type(e).__name__}: {e}")

    summary_path = out_dir / "qc_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Resumen: {summary_path}")


if __name__ == "__main__":
    main()
