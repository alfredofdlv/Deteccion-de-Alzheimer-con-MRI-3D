"""
spatial_preprocess.py — Skull-stripping y registro afín a MNI152 (offline).

Pipeline MNI (idea A):
    Load NIfTI → Orientation RAS → SkullStrip → Affine 12-DOF → percentiles → Resize 96³

Pipeline MNI+N4 (mni_n4):
    ... → SkullStrip → N4 bias correction (mask) → Affine 12-DOF → ...

Solo registro afín (12 DOF); nunca no lineal (SyN/BSpline) para no borrar atrofia.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    Resize,
    ScaleIntensityRangePercentiles,
)

from src.config import cfg

_ALLOWED_REGISTRATION = frozenset({"Affine"})
_SYNTHSTRIP_CMD = "mri_synthstrip"
_HDBET_CMD = "hd-bet"


def detect_skull_strip_backend(requested: str | None = None) -> str:
    """
    Resuelve el backend de skull-stripping.

    auto: mri_synthstrip si está en PATH; si no, hd-bet.
    """
    backend = (requested or cfg.SKULL_STRIP_BACKEND).lower()
    if backend == "auto":
        if shutil.which(_SYNTHSTRIP_CMD):
            return "synthstrip"
        if shutil.which(_HDBET_CMD):
            return "hd-bet"
        try:
            import HD_BET  # noqa: F401
            return "hd-bet"
        except ImportError:
            pass
        raise RuntimeError(
            "No hay backend de skull-stripping disponible. "
            "Instala FreeSurfer (mri_synthstrip) o: pip install hd-bet"
        )
    if backend in ("synthstrip", "hd-bet"):
        return backend
    raise ValueError(f"Backend desconocido: {backend!r}. Use auto, synthstrip o hd-bet.")


def _resolve_hdbet_device(device: str | None = None) -> str:
    """Resuelve device HD-BET: parametro > env MNI_HDBET_DEVICE > cpu."""
    resolved = device or os.environ.get("MNI_HDBET_DEVICE", "cpu")
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HD-BET device=cuda pero torch.cuda no está disponible")
    return resolved


def _ensure_ras_nifti(input_path: str | Path, work_dir: Path) -> Path:
    """Carga NIfTI y guarda en orientación RAS (canónica)."""
    img = nib.load(str(input_path))
    ras_img = nib.as_closest_canonical(img)
    out = work_dir / "input_ras.nii.gz"
    nib.save(ras_img, str(out))
    return out


def _brain_mask_from_nifti(brain_path: Path, mask_path: Path, threshold: float = 0.0) -> Path:
    """Genera máscara binaria cerebral (voxels > threshold) cuando HD-BET no la exporta."""
    img = nib.load(str(brain_path))
    data = img.get_fdata(dtype=np.float32)
    mask_data = (data > threshold).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    nib.save(mask_img, str(mask_path))
    return mask_path


def skull_strip(
    input_nii: str | Path,
    work_dir: Path,
    backend: str | None = None,
    device: str | None = None,
) -> tuple[Path, Path, str]:
    """
    Aísla parénquima cerebral.

    Returns:
        (ruta brain.nii.gz, ruta brain_mask.nii.gz, nombre_backend)
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved = detect_skull_strip_backend(backend)
    hdbet_device = _resolve_hdbet_device(device)
    ras_path = _ensure_ras_nifti(input_nii, work_dir)
    brain_path = work_dir / "brain.nii.gz"
    mask_path = work_dir / "brain_mask.nii.gz"

    if resolved == "synthstrip":
        if not shutil.which(_SYNTHSTRIP_CMD):
            raise RuntimeError("mri_synthstrip no está en PATH")
        cmd = [
            _SYNTHSTRIP_CMD,
            "-i", str(ras_path),
            "-o", str(brain_path),
            "-m", str(mask_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    else:
        cli_device = "cpu" if hdbet_device.startswith("cpu") else "cuda"
        if shutil.which(_HDBET_CMD):
            cmd = [
                _HDBET_CMD,
                "-i", str(ras_path),
                "-o", str(brain_path),
                "-device", cli_device,
                "--disable_tta",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            from HD_BET.checkpoint_download import maybe_download_parameters
            from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

            maybe_download_parameters()
            predictor = get_hdbet_predictor(
                use_tta=False,
                device=torch.device(hdbet_device),
                verbose=False,
            )
            hdbet_predict(
                str(ras_path),
                str(brain_path),
                predictor,
                keep_brain_mask=False,
                compute_brain_extracted_image=True,
            )
        if not mask_path.exists():
            _brain_mask_from_nifti(brain_path, mask_path)

    if not brain_path.exists():
        raise FileNotFoundError(f"Skull-strip falló: {brain_path}")
    if not mask_path.exists():
        _brain_mask_from_nifti(brain_path, mask_path)
    return brain_path, mask_path, resolved


def n4_bias_correct(
    brain_nii: str | Path,
    mask_nii: str | Path,
    work_dir: Path,
) -> Path:
    """
    Corrección de bias field N4 (ANTsPy, CPU) sobre cerebro skull-stripped.

    Returns:
        Ruta a brain_n4.nii.gz
    """
    import ants

    work_dir.mkdir(parents=True, exist_ok=True)
    image = ants.image_read(str(brain_nii))
    mask = ants.image_read(str(mask_nii))
    corrected = ants.n4_bias_field_correction(
        image,
        mask=mask,
        shrink_factor=cfg.N4_SHRINK_FACTOR,
        convergence={
            "iters": cfg.N4_CONVERGENCE_ITERS,
            "tol": cfg.N4_CONVERGENCE_TOL,
        },
        verbose=False,
    )
    out_path = work_dir / "brain_n4.nii.gz"
    ants.image_write(corrected, str(out_path))
    return out_path


def _load_mni_template() -> Any:
    """Carga template MNI de referencia vía antspyx (get_ants_data)."""
    import ants

    mni_path = ants.get_ants_data("mni")
    return ants.image_read(mni_path)


def register_affine_to_mni(
    brain_nii: str | Path,
    registration_type: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Registro afín 12-DOF del cerebro al template MNI152.

    Returns:
        (ants_image_warped, metadata_dict)
    """
    import ants

    reg_type = registration_type or cfg.MNI_REGISTRATION_TYPE
    if reg_type not in _ALLOWED_REGISTRATION:
        raise ValueError(
            f"Solo se permite registro afín ({_ALLOWED_REGISTRATION}); "
            f"recibido: {reg_type!r}"
        )

    fixed = _load_mni_template()
    moving = ants.image_read(str(brain_nii))
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=reg_type,
        verbose=False,
    )
    warped = reg["warpedmovout"]
    meta = {
        "registration_type": reg_type,
        "template": cfg.MNI_TEMPLATE_NAME,
        "forward_transforms": reg.get("fwdtransforms", []),
    }
    return warped, meta


def _ants_to_channel_first_tensor(ants_image: Any) -> torch.Tensor:
    """Convierte imagen ANTs a tensor (1, D, H, W) float32."""
    arr = ants_image.numpy().astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return torch.from_numpy(arr).unsqueeze(0)


def apply_intensity_and_resize(
    tensor: torch.Tensor,
    spatial_size: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Percentiles 1-99 → [0,1] y resize al tamaño objetivo."""
    size = spatial_size or cfg.IMAGE_SIZE
    pipeline = Compose([
        ScaleIntensityRangePercentiles(
            lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True,
        ),
        Resize(spatial_size=size),
    ])
    return pipeline(tensor)


def process_image_mni_pipeline(
    image_path: str | Path,
    work_dir: Path | None = None,
    keep_intermediate: bool = False,
    skull_strip_backend: str | None = None,
    device: str | None = None,
    apply_n4: bool = False,
    spatial_size: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Pipeline MNI (opcional N4): skull-strip → [N4] → registro afín → normalización → resize.

    Args:
        apply_n4: Si True, aplica N4 tras skull-strip (variante mni_n4).

    Returns:
        (tensor (1,D,H,W), metadata)
    """
    image_path = Path(image_path)
    own_tmp = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mni_preprocess_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {"source": str(image_path)}
    timings: dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        brain_path, mask_path, ss_backend = skull_strip(
            image_path, work_dir, skull_strip_backend, device=device,
        )
        timings["skull_strip_s"] = time.perf_counter() - t0
        meta["skull_strip_backend"] = ss_backend
        meta["hdbet_device"] = _resolve_hdbet_device(device)

        register_input = brain_path
        if apply_n4:
            t0 = time.perf_counter()
            register_input = n4_bias_correct(brain_path, mask_path, work_dir)
            timings["n4_s"] = time.perf_counter() - t0
            meta["n4"] = True
            meta["n4_backend"] = "antspy"

        t0 = time.perf_counter()
        warped, reg_meta = register_affine_to_mni(register_input)
        timings["registration_s"] = time.perf_counter() - t0
        meta.update(reg_meta)

        tensor = _ants_to_channel_first_tensor(warped)
        out_size = spatial_size or cfg.IMAGE_SIZE
        tensor = apply_intensity_and_resize(tensor, spatial_size=out_size)
        meta["output_shape"] = list(tensor.shape)
        meta["output_spatial_size"] = list(out_size)
        meta["timings_s"] = timings
        return tensor.contiguous(), meta
    finally:
        if own_tmp and not keep_intermediate:
            shutil.rmtree(work_dir, ignore_errors=True)


def process_image_mni_n4_pipeline(
    image_path: str | Path,
    work_dir: Path | None = None,
    keep_intermediate: bool = False,
    skull_strip_backend: str | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pipeline MNI + N4 (atajo para variante mni_n4)."""
    return process_image_mni_pipeline(
        image_path,
        work_dir=work_dir,
        keep_intermediate=keep_intermediate,
        skull_strip_backend=skull_strip_backend,
        device=device,
        apply_n4=True,
    )


def save_registration_metadata(meta: dict[str, Any], out_path: Path) -> None:
    """Guarda metadatos de registro (sin rutas temporales de transforms)."""
    serializable = {
        k: v for k, v in meta.items()
        if k != "forward_transforms"
    }
    if "forward_transforms" in meta:
        serializable["forward_transforms"] = [
            str(p) for p in meta["forward_transforms"]
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
