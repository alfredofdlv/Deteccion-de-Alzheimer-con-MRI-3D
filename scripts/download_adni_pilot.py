"""
download_adni_pilot.py — Asistente de descarga DICOM ADNI piloto (IDA / IdaDownloader).

Genera instrucciones y, si se proporciona URL de descarga del IDA, invoca IdaDownloader.jar.

Pasos manuales en LONI IDA (si no tienes URL batch):
  1. Search and Download → Advanced Image Search (o ARC Builder)
  2. Filtrar: Modality=MRI, Series Type=T1w, Visit=bl (baseline)
  3. Importar image_ids desde data/splits/adni_download_list.csv
  4. Add to Download Queue → Download → copiar URL del job
  5. Ejecutar este script con --ida-url <URL>

Uso:
    python scripts/download_adni_pilot.py --instructions
    python scripts/download_adni_pilot.py --ida-url 'https://...' --output data/raw/ADNI
    python scripts/download_adni_pilot.py --check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

ADNI_SCRIPTS = PROJECT_ROOT / "scripts" / "ADNI"


def find_ida_jar() -> Path | None:
    jars = sorted(ADNI_SCRIPTS.glob("IdaDownloader_*.jar"))
    return jars[-1] if jars else None


def print_instructions(manifest_path: Path, n_images: int) -> None:
    print(
        f"""
=== Descarga piloto ADNI ({n_images} imágenes) ===

Manifest: {manifest_path}

Opción A — ARC Builder / Advanced Image Search (recomendado para piloto):
  1. Inicia sesión en LONI IDA (ADNI).
  2. Search and Download → Advanced Image Search.
  3. Filtros sugeridos: MRI, T1w, Visit Code = bl.
  4. Opcional: pegar lista de IMAGEID desde la columna image_id del manifest.
  5. Add to Cart / Download Queue → iniciar descarga.
  6. Extrae el ZIP en: {cfg.ADNI_RAW_DIR}

Opción B — IdaDownloader.jar (URL del job IDA):
  java -jar IdaDownloader_*.jar --directory={cfg.ADNI_RAW_DIR} --chunks=10 <URL>

  Obtén <URL> desde la página de descarga del IDA tras encolar las imágenes.

Opción C — Este script con URL:
  python scripts/download_adni_pilot.py --ida-url '<URL>' --output {cfg.ADNI_RAW_DIR}

Tras la descarga:
  python scripts/convert_adni_dicom_to_nifti.py
  python scripts/preprocess_to_pt.py --dataset adni
"""
    )


def run_ida_download(jar: Path, url: str, output: Path, chunks: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", "-jar", str(jar),
        f"--directory={output}",
        f"--chunks={chunks}",
        url,
    ]
    print(f"[INFO] Ejecutando: {' '.join(cmd[:4])} ... <URL>")
    return subprocess.call(cmd)


MIN_ZIP_BYTES = 500_000_000  # 500 MB; stubs IDA ~290 B con README de error IP


def _validate_zip(dest: Path) -> str | None:
    """Devuelve mensaje de error si el ZIP no parece válido."""
    if not dest.exists():
        return "archivo no creado"
    size = dest.stat().st_size
    if size < MIN_ZIP_BYTES:
        import zipfile
        msg = f"tamaño sospechoso ({size} bytes, esperado >>500 MB)"
        try:
            with zipfile.ZipFile(dest, "r") as zf:
                for name in zf.namelist():
                    if name.upper().endswith("README.TXT"):
                        readme = zf.read(name).decode("utf-8", errors="replace")
                        if "IP address" in readme or "download" in readme.lower():
                            return f"stub IDA: {readme.strip()[:200]}"
        except zipfile.BadZipFile:
            pass
        return msg
    return None


def download_urls_file(
    urls_file: Path,
    output: Path,
    *,
    extract: bool = True,
    skip_existing: bool = True,
    delete_zip_after_extract: bool = False,
) -> int:
    """Descarga ZIPs del IDA (curl con reanudación) y opcionalmente extrae."""
    import zipfile

    urls = [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        print(f"[ERROR] Sin URLs en {urls_file}")
        return 1

    zip_dir = output / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    failed = 0
    for i, url in enumerate(urls, 1):
        name = url.rstrip("/").split("/")[-1]
        dest = zip_dir / name
        print(f"\n[{i}/{len(urls)}] {name}")
        if skip_existing and dest.exists():
            err = _validate_zip(dest)
            if err is None:
                print(f"  [SKIP] Ya existe ({dest.stat().st_size / 1e9:.2f} GB)")
            else:
                print(f"  [RETRY] ZIP previo inválido: {err}")
                dest.unlink(missing_ok=True)
        if not (skip_existing and dest.exists() and _validate_zip(dest) is None):
            cmd = [
                "curl", "-L", "-C", "-", "--fail", "--retry", "5", "--retry-delay", "10",
                "-o", str(dest), url,
            ]
            print(f"  [DOWN] curl → {dest}")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"  [ERROR] Descarga fallida (rc={rc})")
                failed += 1
                continue
            err = _validate_zip(dest)
            if err:
                print(f"  [ERROR] ZIP inválido: {err}")
                failed += 1
                continue

        if extract:
            print(f"  [UNZIP] → {output}")
            # unzip del sistema suele ser más rápido que zipfile en ZIPs grandes
            unzip_rc = subprocess.call(
                ["unzip", "-o", "-q", str(dest), "-d", str(output)],
            )
            if unzip_rc != 0:
                try:
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(output)
                except zipfile.BadZipFile as e:
                    print(f"  [ERROR] ZIP corrupto: {e}")
                    failed += 1
                    continue
            if delete_zip_after_extract:
                dest.unlink()
                print(f"  [DEL] ZIP eliminado tras extraer")

    print(f"\n[OK] Descargas completadas. Fallos: {failed}/{len(urls)}")
    return 1 if failed else 0


def check_status(manifest_path: Path, dicom_root: Path, nifti_root: Path) -> None:
    import pandas as pd

    if not manifest_path.exists():
        print(f"[ERROR] Manifest no encontrado: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    n = len(df)
    dicom_found = 0
    nifti_found = 0
    zip_dir = dicom_root / "_zips"
    if zip_dir.is_dir():
        zips = list(zip_dir.glob("*.zip"))
        total_gb = sum(z.stat().st_size for z in zips) / 1e9
        print(f"ZIPs en _zips/: {len(zips)} archivos, {total_gb:.2f} GB")

    for _, row in df.iterrows():
        iid = str(row["image_id"])
        sid = str(row["subject_id"])
        vc = str(row.get("viscode", row.get("image_visit", "bl")))
        nifti = nifti_root / f"{sid}_{vc}_{iid}.nii.gz"
        if nifti.exists():
            nifti_found += 1
        if dicom_root.exists():
            found = False
            for pat in (f"I{iid}", iid, sid):
                for p in dicom_root.rglob(f"*{pat}*"):
                    if p.is_dir() and (any(p.rglob("*.dcm")) or any(p.rglob("*.DCM"))):
                        found = True
                        break
                if found:
                    break
            if found:
                dicom_found += 1

    print(f"Manifest:     {n} imágenes")
    print(f"DICOM root:   {dicom_root} ({'existe' if dicom_root.exists() else 'NO existe'})")
    print(f"DICOM ~found: {dicom_found}/{n}")
    print(f"NIfTI root:   {nifti_root} ({'existe' if nifti_root.exists() else 'NO existe'})")
    print(f"NIfTI listos: {nifti_found}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga piloto ADNI (IDA)")
    parser.add_argument("--instructions", action="store_true", help="Mostrar guía IDA")
    parser.add_argument("--check", action="store_true", help="Estado descarga/conversión")
    parser.add_argument("--ida-url", type=str, default=None, help="URL de descarga del IDA")
    parser.add_argument(
        "--urls-file", type=Path, default=None,
        help="Archivo con una URL por línea (ZIPs IDA, p. ej. urls_adni.txt)",
    )
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Solo descargar ZIPs, no extraer",
    )
    parser.add_argument(
        "--delete-zip-after-extract", action="store_true",
        help="Borrar ZIP tras extraer (ahorra espacio en descargas grandes)",
    )
    parser.add_argument(
        "--output", type=Path, default=cfg.ADNI_RAW_DIR,
        help=f"Directorio destino DICOM (default: {cfg.ADNI_RAW_DIR})",
    )
    parser.add_argument("--chunks", type=int, default=10, help="Chunks IdaDownloader (1-20)")
    args = parser.parse_args()

    manifest = cfg.ADNI_DICOM_MANIFEST
    n_images = 0
    if manifest.exists():
        import pandas as pd
        n_images = len(pd.read_csv(manifest))

    if args.check:
        check_status(manifest, args.output, cfg.ADNI_NIFTI_DIR)
        return

    if args.urls_file is not None:
        if not args.urls_file.exists():
            print(f"[ERROR] No existe: {args.urls_file}")
            sys.exit(1)
        rc = download_urls_file(
            args.urls_file,
            args.output,
            extract=not args.no_extract,
            delete_zip_after_extract=args.delete_zip_after_extract,
        )
        sys.exit(rc)

    if args.instructions or args.ida_url is None:
        if not manifest.exists():
            print("[WARN] Ejecuta primero: python scripts/build_adni_manifest.py")
        print_instructions(manifest, n_images)
        if args.ida_url is None:
            return

    jar = find_ida_jar()
    if jar is None:
        print(f"[ERROR] No se encontró IdaDownloader_*.jar en {ADNI_SCRIPTS}")
        sys.exit(1)

    rc = run_ida_download(jar, args.ida_url, args.output, args.chunks)
    sys.exit(rc)


if __name__ == "__main__":
    main()
