"""
preprocess_to_pt.py — Preprocesar imagenes NIfTI/ANALYZE a tensores .pt.

Variantes:
  cropped (default): CropForeground + resize 96³ (baseline)
  mni: skull-strip + registro afín MNI152 + resize 96³
  mni_n4: mni + N4 bias correction (antspy CPU, tras skull-strip)
  mni_128: igual que mni pero resize 128³ (más detalle post-registro)

Genera CSVs de splits con rutas a los .pt.

Uso:
    python scripts/preprocess_to_pt.py --dataset oasis3
    python scripts/preprocess_to_pt.py --dataset adni --variant mni --workers 8 --gpus 0,1
    python scripts/preprocess_to_pt.py --dataset oasis3 --variant mni_128 --workers 4 --gpus 0
    python scripts/preprocess_to_pt.py --dataset oasis3 --variant mni_n4 --limit 50
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
from monai.transforms import (
    Compose,
    CropForeground,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    Resize,
    ScaleIntensityRangePercentiles,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

APPROX_MB_PER_IMAGE_96 = 3.4
THREADS_PER_WORKER = 2


def _threads_per_worker() -> int:
    """Hilos CPU (ANTs/ITK) por worker; override con PREPROCESS_THREADS_PER_WORKER."""
    raw = os.environ.get("PREPROCESS_THREADS_PER_WORKER")
    if raw is not None and raw.strip() != "":
        return max(1, int(raw))
    return THREADS_PER_WORKER


def _approx_mb_per_image(spatial_size: tuple[int, int, int]) -> float:
    voxels = spatial_size[0] * spatial_size[1] * spatial_size[2]
    return voxels * 4 / (1024 * 1024)


def _resolve_spatial_size(variant: str, spatial_size_arg: int | None) -> tuple[int, int, int]:
    if spatial_size_arg is not None:
        return (spatial_size_arg,) * 3
    return cfg.spatial_size_for_variant(variant)


def build_preprocess_pipeline(
    spatial_size: tuple[int, int, int] | None = None,
) -> Compose:
    """Transforms determinísticas offline (variante cropped)."""
    size = spatial_size or cfg.IMAGE_SIZE
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRangePercentiles(
            lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForeground(
            select_fn=lambda x: x > 0.1,
            margin=2,
        ),
        Resize(spatial_size=size),
    ])


def derive_pt_filename(image_path: str) -> str:
    """Genera un nombre unico .pt a partir de la ruta original."""
    p = Path(image_path)
    return p.stem.replace(".nii", "") + ".pt"


def _ensure_single_channel(tensor: torch.Tensor) -> torch.Tensor:
    """NIfTI 4D (multi-volumen) → conservar solo el primer canal/volumen."""
    if tensor.shape[0] > 1:
        tensor = tensor[0:1]
    return tensor


def _parse_gpus(gpus: str | None) -> list[int]:
    if not gpus:
        return []
    return [int(x.strip()) for x in gpus.split(",") if x.strip() != ""]


def _resolve_job_device(
    job_index: int, hdbet_device: str, gpu_list: list[int],
) -> str:
    if hdbet_device.startswith("cpu"):
        return "cpu"
    if not gpu_list:
        return hdbet_device if ":" in hdbet_device else "cuda"
    gpu_id = gpu_list[job_index % len(gpu_list)]
    return f"cuda:{gpu_id}"


def _bind_cuda_device(device: str) -> str:
    """Fija CUDA_VISIBLE_DEVICES por worker (HD-BET CLI usa cuda:0 local)."""
    if device.startswith("cuda:"):
        gpu_id = device.split(":", 1)[1]
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # Si el padre ya restringió a una sola GPU, no sobrescribir con índice físico.
        if parent_visible and "," not in parent_visible:
            return "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return "cuda"
    return device


def _set_thread_limits(workers: int) -> int:
    per_worker = _threads_per_worker()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(per_worker)
    return per_worker


def _process_one_cropped(args_tuple: tuple) -> tuple[str, str | None, str | None]:
    img_path, pt_path_str, spatial_size = args_tuple
    pt_path = Path(pt_path_str)
    if pt_path.exists():
        return img_path, str(pt_path), None
    try:
        pipeline = build_preprocess_pipeline(spatial_size=spatial_size)
        tensor = _ensure_single_channel(pipeline(img_path))
        torch.save(tensor.contiguous(), pt_path)
        return img_path, str(pt_path), None
    except Exception as e:
        return img_path, None, str(e)


def _is_spatial_variant(variant: str) -> bool:
    return variant in cfg.SPATIAL_PREPROCESS_VARIANTS


def _process_one_spatial(args_tuple: tuple) -> tuple[str, str | None, str | None]:
    (
        img_path, pt_path_str, intermediate_root, keep_intermediate,
        device, variant, spatial_size,
    ) = args_tuple
    pt_path = Path(pt_path_str)
    if pt_path.exists():
        return img_path, str(pt_path), None
    try:
        from src.spatial_preprocess import (
            process_image_mni_pipeline,
            save_registration_metadata,
        )

        device = _bind_cuda_device(device)
        stem = Path(img_path).stem.replace(".nii", "")
        work_dir = Path(intermediate_root) / stem
        apply_n4 = variant == "mni_n4"
        tensor, meta = process_image_mni_pipeline(
            img_path,
            work_dir=work_dir,
            keep_intermediate=keep_intermediate,
            device=device,
            apply_n4=apply_n4,
            spatial_size=spatial_size,
        )
        meta["preprocess_variant"] = variant
        torch.save(tensor, pt_path)
        meta_path = pt_path.with_suffix(".json")
        save_registration_metadata(meta, meta_path)
        if not keep_intermediate and work_dir.exists():
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        return img_path, str(pt_path), None
    except Exception as e:
        return img_path, None, str(e)


def _split_csv_paths(dataset: str) -> dict[str, Path]:
    splits = ["train", "val", "test"]
    if dataset == "oasis1":
        return {s: cfg.DATA_SPLITS_DIR / f"{s}.csv" for s in splits}
    return {s: cfg.DATA_SPLITS_DIR / f"{dataset}_{s}.csv" for s in splits}


def _output_csv_path(dataset: str, split: str, variant: str) -> Path:
    return cfg.DATA_SPLITS_DIR / cfg.split_csv_name(dataset, split, variant)


def _row_to_pt_record(row: pd.Series, pt_path: str) -> dict:
    record = {
        "subject_id": row["subject_id"],
        "image_path": pt_path,
        "label": row["label"],
    }
    for col in ("GENDER", "EDUC"):
        if col in row.index and pd.notna(row[col]):
            record[col] = row[col]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesar imagenes a tensores .pt",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=["oasis1", "oasis3", "adni"],
    )
    parser.add_argument(
        "--variant", type=str, default=cfg.PREPROCESS_VARIANT_DEFAULT,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="cropped (baseline), mni, mni_n4 o mni_128 (skull-strip + registro afín + resize)",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        metavar="N",
        help="Override del resize final (p. ej. 128). Con variant=mni equivale a mni_128.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Procesos paralelos (recomendado 4 por GPU para variant=mni)",
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="IDs GPU comma-separated para HD-BET (ej. 0,1). Round-robin por worker.",
    )
    parser.add_argument(
        "--hdbet-device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo HD-BET (variantes espaciales: mni, mni_n4)",
    )
    parser.add_argument(
        "--intermediate-local", action="store_true",
        help="Intermedios MNI en /dev/shm (más rápido en NAS)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Procesar solo las primeras N imagenes unicas (piloto QC)",
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Conservar NIfTI intermedios (solo variantes espaciales)",
    )
    args = parser.parse_args()

    variant = args.variant
    if args.spatial_size == cfg.IMAGE_SIZE_HIGH[0] and variant == "mni":
        variant = "mni_128"

    spatial_size = _resolve_spatial_size(variant, args.spatial_size)

    out_dir = cfg.preprocessed_dir(args.dataset, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _split_csv_paths(args.dataset)
    all_images: dict[str, dict] = {}
    for split, csv_path in csv_paths.items():
        if not csv_path.exists():
            print(f"[WARN] No existe {csv_path}, saltando split '{split}'")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = row["image_path"]
            if img_path not in all_images:
                all_images[img_path] = {
                    "subject_id": row["subject_id"],
                    "label": row["label"],
                    "splits": [],
                }
            all_images[img_path]["splits"].append(split)

    unique_paths = list(all_images.keys())
    if args.limit is not None:
        unique_paths = unique_paths[: args.limit]

    total = len(unique_paths)
    gpu_list = _parse_gpus(args.gpus)

    print(f"[INFO] Variante: {variant}")
    print(f"[INFO] Spatial size: {spatial_size}")
    print(f"[INFO] {total} imagenes unicas a preprocesar para '{args.dataset}'")
    print(f"[INFO] Destino: {out_dir}")
    print(f"[INFO] Workers: {args.workers}")
    if _is_spatial_variant(args.variant):
        print(f"[INFO] HD-BET device: {args.hdbet_device}")
        if args.variant == "mni_n4":
            print("[INFO] N4 backend: antspy (CPU, tras skull-strip)")
        if gpu_list:
            print(f"[INFO] GPUs: {gpu_list}")

    mb_per = _approx_mb_per_image(spatial_size)
    if _is_spatial_variant(args.variant) and args.intermediate_local:
        intermediate_root = Path(
            os.environ.get("PREPROCESS_INTERMEDIATE_DIR", "/dev/shm/mni_preprocess"),
        )
    else:
        intermediate_root = cfg.INTERMEDIATE_DIR / cfg.preprocessed_subdir(
            args.dataset, variant,
        )
    if _is_spatial_variant(args.variant):
        intermediate_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Intermedios: {intermediate_root}")

    print(f"[INFO] Tamaño estimado: ~{total * mb_per / 1024:.1f} GB\n")

    jobs: list[tuple] = []
    for job_index, img_path in enumerate(unique_paths):
        pt_name = derive_pt_filename(img_path)
        pt_path = out_dir / pt_name
        if args.variant == "cropped":
            jobs.append((img_path, str(pt_path), spatial_size))
        elif _is_spatial_variant(args.variant):
            device = _resolve_job_device(job_index, args.hdbet_device, gpu_list)
            jobs.append((
                img_path,
                str(pt_path),
                str(intermediate_root),
                args.keep_intermediate,
                device,
                variant,
                spatial_size,
            ))
        else:
            raise ValueError(f"Variante no soportada: {args.variant!r}")

    t_start = time.time()
    success = 0
    errors: list[tuple[str, str]] = []
    pt_map: dict[str, str] = {}

    worker_fn = _process_one_cropped if args.variant == "cropped" else _process_one_spatial
    use_spawn = _is_spatial_variant(args.variant) and args.hdbet_device == "cuda"

    if args.workers <= 1:
        if _is_spatial_variant(args.variant):
            _set_thread_limits(1)
        for i, job in enumerate(jobs, 1):
            img_path, pt_path, err = worker_fn(job)
            if err:
                errors.append((img_path, err))
                print(f"\n  [ERROR] {img_path}: {err}")
            elif pt_path:
                pt_map[img_path] = pt_path
                success += 1
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (total - i)
            avg = elapsed / i
            print(
                f"\r  [{i}/{total}] {i/total:>6.1%} | "
                f"{elapsed:.0f}s / ETA {timedelta(seconds=int(eta))} | "
                f"avg: {avg:.2f}s/img | ok: {success} err: {len(errors)}",
                end="", flush=True,
            )
    else:
        threads = _set_thread_limits(args.workers)
        if use_spawn:
            print("[INFO] mp_context=spawn (HD-BET CUDA)")
        print(f"[INFO] Threads por worker: {threads}")

        mp_ctx = mp.get_context("spawn") if use_spawn else None
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp_ctx,
        ) as pool:
            for i, (img_path, pt_path, err) in enumerate(
                pool.map(worker_fn, jobs, chunksize=1), 1,
            ):
                if err:
                    errors.append((img_path, err))
                    print(f"\n  [ERROR] {img_path}: {err}")
                elif pt_path:
                    pt_map[img_path] = pt_path
                    success += 1
                elapsed = time.time() - t_start
                eta = (elapsed / i) * (total - i)
                avg = elapsed / i
                print(
                    f"\r  [{i}/{total}] {i/total:>6.1%} | "
                    f"{elapsed:.0f}s / ETA {timedelta(seconds=int(eta))} | "
                    f"avg: {avg:.2f}s/img | ok: {success} err: {len(errors)}",
                    end="", flush=True,
                )

    elapsed_total = time.time() - t_start
    print(f"\n\n{'='*60}")
    print("PREPROCESAMIENTO COMPLETADO")
    print(f"  Variante:  {variant}")
    print(f"  Spatial:   {spatial_size}")
    print(f"  Exitosas:  {success}/{total}")
    print(f"  Errores:   {len(errors)}")
    print(f"  Tiempo:    {timedelta(seconds=int(elapsed_total))}")
    print(f"  Destino:   {out_dir}")
    print(f"{'='*60}\n")

    print("[INFO] Generando CSVs de splits con rutas .pt...")
    for split, csv_path in csv_paths.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        new_rows = []
        for _, row in df.iterrows():
            orig = row["image_path"]
            pt_path = pt_map.get(orig)
            if pt_path is None:
                pt_name = derive_pt_filename(orig)
                candidate = out_dir / pt_name
                if candidate.exists():
                    pt_path = str(candidate)
            if pt_path:
                new_rows.append(_row_to_pt_record(row, pt_path))
        new_df = pd.DataFrame(new_rows)
        out_csv = _output_csv_path(args.dataset, split, variant)
        new_df.to_csv(out_csv, index=False)
        print(f"  {out_csv.name}: {len(new_df)} samples")

    if errors:
        print("\n[WARN] Imagenes con errores:")
        for path, err in errors[:50]:
            print(f"  {path}: {err}")
        if len(errors) > 50:
            print(f"  ... y {len(errors) - 50} más")


if __name__ == "__main__":
    main()
