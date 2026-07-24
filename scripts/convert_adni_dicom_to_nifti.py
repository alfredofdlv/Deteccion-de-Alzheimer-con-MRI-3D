"""
convert_adni_dicom_to_nifti.py — DICOM ADNI → NIfTI y actualización de splits.

Busca DICOM en data/raw/ADNI/ (layout IDA: por image_id o por carpetas de sujeto),
convierte con dcm2niix y escribe NIfTI en data/raw/ADNI-nifti/.
Actualiza image_path en adni_{train,val,test}.csv si el NIfTI existe.

Uso:
    python scripts/convert_adni_dicom_to_nifti.py
    python scripts/convert_adni_dicom_to_nifti.py --dicom-root data/raw/ADNI --workers 4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg


def find_dcm2niix() -> str | None:
    bundled = PROJECT_ROOT / "scripts" / "ADNI" / "tools" / "dcm2niix"
    if bundled.is_file():
        return str(bundled)
    for name in ("dcm2niix", "dcm2niix_bin"):
        path = shutil.which(name)
        if path:
            return path
    return None


def expected_nifti_name(subject_id: str, viscode: str, image_id: str) -> str:
    return f"{subject_id}_{viscode}_{image_id}.nii.gz"


def _dir_has_dcm(path: Path) -> bool:
    return any(path.glob("*.dcm")) or any(path.glob("*.DCM"))


def find_dicom_dir(
    dicom_root: Path,
    image_id: str,
    subject_id: str,
    *,
    series_instance_uid: str | None = None,
) -> Path | None:
    """Localiza carpeta DICOM de UNA serie (layout IDA: ADNI/{PTID}/.../I{IMAGEID}/)."""
    iid = str(image_id)
    sid = str(subject_id)
    i_folder = f"I{iid}"

    def _match_series_uid(dicom_dir: Path) -> bool:
        if not series_instance_uid:
            return True
        try:
            import pydicom
            for dcm in dicom_dir.glob("*.dcm"):
                hdr = pydicom.dcmread(dcm, stop_before_pixels=True)
                if str(getattr(hdr, "SeriesInstanceUID", "")) == str(series_instance_uid):
                    return True
            for dcm in dicom_dir.glob("*.DCM"):
                hdr = pydicom.dcmread(dcm, stop_before_pixels=True)
                if str(getattr(hdr, "SeriesInstanceUID", "")) == str(series_instance_uid):
                    return True
        except Exception:
            return False
        return False

    def _accept(path: Path) -> Path | None:
        if path.is_dir() and _dir_has_dcm(path) and _match_series_uid(path):
            return path
        return None

    # Layout habitual tras descarga IDA: data/raw/ADNI/ADNI/{PTID}/.../I{IMAGEID}/
    subject_roots = [
        dicom_root / "ADNI" / sid,
        dicom_root / sid,
    ]
    for subj_root in subject_roots:
        if not subj_root.is_dir():
            continue
        exact = subj_root / i_folder
        if (hit := _accept(exact)):
            return hit
        for hit_path in subj_root.rglob(i_folder):
            if hit_path.is_dir() and (hit := _accept(hit_path)):
                return hit

    # Candidatos planos legacy
    for c in (
        dicom_root / i_folder,
        dicom_root / sid / i_folder,
        dicom_root / sid / iid,
        dicom_root / iid,
    ):
        if hit := _accept(c):
            return hit

    return None


def convert_one(
    dcm2niix: str,
    dicom_dir: Path,
    out_path: Path,
) -> tuple[bool, str]:
    if out_path.exists():
        return True, "already exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            dcm2niix,
            "-z", "y",
            "-f", out_path.stem,
            "-o", str(tmp_dir),
            str(dicom_dir),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout or "dcm2niix failed"
        niftis = list(tmp_dir.glob("*.nii.gz"))
        if not niftis:
            return False, "no NIfTI produced"
        shutil.move(str(niftis[0]), str(out_path))
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def update_split_csvs() -> None:
    splits = ["train", "val", "test"]
    for split in splits:
        csv_path = cfg.DATA_SPLITS_DIR / f"adni_{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        updated = 0
        for i, row in df.iterrows():
            nifti = cfg.ADNI_NIFTI_DIR / expected_nifti_name(
                str(row["subject_id"]),
                str(row["viscode"]),
                str(row["image_id"]),
            )
            if nifti.exists():
                df.at[i, "image_path"] = str(nifti.resolve())
                updated += 1
        df.to_csv(csv_path, index=False)
        print(f"  {csv_path.name}: {updated}/{len(df)} con NIfTI existente")


def _convert_job(args_tuple: tuple) -> tuple[str, bool, str]:
    """Worker: (dcm2niix, dicom_dir, out_path, image_id) -> (image_id, ok, msg)."""
    dcm2niix, dicom_dir, out_path, image_id = args_tuple
    success, msg = convert_one(
        dcm2niix, Path(dicom_dir), Path(out_path),
    )
    return image_id, success, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir DICOM ADNI a NIfTI")
    parser.add_argument(
        "--dicom-root", type=Path, default=cfg.ADNI_RAW_DIR,
        help=f"Raíz DICOM (default: {cfg.ADNI_RAW_DIR})",
    )
    parser.add_argument(
        "--nifti-root", type=Path, default=cfg.ADNI_NIFTI_DIR,
        help=f"Salida NIfTI (default: {cfg.ADNI_NIFTI_DIR})",
    )
    parser.add_argument(
        "--manifest", type=Path, default=cfg.ADNI_DICOM_MANIFEST,
        help="CSV con image_id (default: adni_download_list.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Convertir solo las primeras N filas del manifest (QC piloto)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Procesos paralelos para dcm2niix (default: 1)",
    )
    args = parser.parse_args()

    dcm2niix = find_dcm2niix()
    if dcm2niix is None:
        print(
            "[ERROR] dcm2niix no encontrado en PATH.\n"
            "  Instalar: sudo apt install dcm2niix  o  conda install -c conda-forge dcm2niix",
        )
        sys.exit(1)

    if not args.manifest.exists():
        print(f"[ERROR] No existe manifest: {args.manifest}")
        print("  Ejecuta primero: python scripts/build_adni_manifest.py")
        sys.exit(1)

    if not args.dicom_root.exists():
        print(f"[WARN] No existe {args.dicom_root} — creando directorio vacío.")
        args.dicom_root.mkdir(parents=True, exist_ok=True)
        print("  Descarga DICOM del IDA antes de convertir (scripts/download_adni_pilot.py).")
        sys.exit(0)

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    args.nifti_root.mkdir(parents=True, exist_ok=True)

    ok, fail, skip = 0, 0, 0
    errors: list[tuple[str, str]] = []
    jobs: list[tuple] = []

    print(f"[INFO] dcm2niix: {dcm2niix}")
    print(f"[INFO] DICOM: {args.dicom_root}")
    print(f"[INFO] NIfTI: {args.nifti_root}")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] {len(manifest)} entradas en manifest\n")

    for _, row in manifest.iterrows():
        image_id = str(row["image_id"])
        subject_id = str(row["subject_id"])
        viscode = str(row["viscode"])
        out_name = expected_nifti_name(subject_id, viscode, image_id)
        out_path = args.nifti_root / out_name

        if out_path.exists():
            skip += 1
            continue

        series_uid = row.get("series_instance_uid")
        series_uid = None if pd.isna(series_uid) else str(series_uid)
        dicom_dir = find_dicom_dir(
            args.dicom_root, image_id, subject_id,
            series_instance_uid=series_uid,
        )
        if dicom_dir is None:
            fail += 1
            errors.append((image_id, "DICOM no encontrado"))
            continue

        jobs.append((dcm2niix, str(dicom_dir), str(out_path), image_id))

    total_jobs = len(jobs)
    if total_jobs == 0:
        print("[INFO] Nada que convertir (todos existentes o sin DICOM).")
    elif args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            image_id, success, msg = _convert_job(job)
            if success:
                ok += 1
            else:
                fail += 1
                errors.append((image_id, msg))
            if i % 10 == 0 or i == total_jobs:
                print(f"\r  [{i}/{total_jobs}] ok={ok} fail={fail}", end="", flush=True)
        if total_jobs:
            print()
    else:
        with Pool(processes=args.workers) as pool:
            for i, (image_id, success, msg) in enumerate(
                pool.imap_unordered(_convert_job, jobs), 1,
            ):
                if success:
                    ok += 1
                else:
                    fail += 1
                    errors.append((image_id, msg))
                if i % 10 == 0 or i == total_jobs:
                    print(f"\r  [{i}/{total_jobs}] ok={ok} fail={fail}", end="", flush=True)
        if total_jobs:
            print()

    print(f"\n{'='*60}")
    print(f"  Convertidos: {ok}  |  Ya existían: {skip}  |  Fallos: {fail}")
    print(f"{'='*60}\n")

    print("[INFO] Actualizando splits ADNI...")
    update_split_csvs()

    if errors and fail <= 20:
        print("\n[WARN] Errores (muestra):")
        for iid, err in errors[:20]:
            print(f"  {iid}: {err}")
    elif fail > 20:
        print(f"\n[WARN] {fail} conversiones fallidas (DICOM aún no descargado?)")

    if errors:
        err_log = PROJECT_ROOT / "outputs" / "adni_dicom_convert_errors.csv"
        err_log.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors, columns=["image_id", "error"]).to_csv(err_log, index=False)
        print(f"[INFO] Log completo de errores: {err_log} ({len(errors)} filas)")


if __name__ == "__main__":
    main()
