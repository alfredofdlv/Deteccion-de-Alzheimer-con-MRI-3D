"""
download_oasis3.py — Descarga sesiones MR de OASIS-3 desde NITRC IR.

Reemplaza al script bash oficial para evitar problemas de codificación
con caracteres no-ASCII en contraseñas (ej: Ñ) en Windows.

Descarga solo scans T1w y organiza en estructura BIDS dentro de data/OASIS-4/.

Uso:
    python download_oasis3.py data/processed/download_test_10.csv
    python download_oasis3.py data/processed/download_test_10.csv --scan-type T1w,T2w
    python download_oasis3.py data/processed/oasis3_matched_dataset.csv --column mr_session
"""

import argparse
import getpass
import gzip
import io
import tarfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "OASIS-4"
NITRC_BASE = "https://www.nitrc.org/ir"


def authenticate(username: str, password: str) -> requests.Session:
    """Crea una sesión autenticada contra NITRC IR."""
    session = requests.Session()
    session.verify = False

    resp = session.get(
        f"{NITRC_BASE}/data/JSESSION",
        auth=(username, password),
    )

    if resp.status_code != 200 or len(resp.text.strip()) < 10:
        raise RuntimeError(
            f"Autenticación fallida (HTTP {resp.status_code}). "
            "Verifica usuario y contraseña."
        )

    print(f"[OK] Sesión autenticada (token: {resp.text.strip()[:8]}...)")
    return session


def detect_project(experiment_id: str) -> str:
    """Detecta el proyecto NITRC basándose en el ID del experimento."""
    if experiment_id.startswith("OAS4"):
        return "OASIS4"
    if "_AV1451" in experiment_id:
        return "OASIS3_AV1451"
    return "OASIS3"


def download_scan(
    session: requests.Session,
    experiment_id: str,
    output_dir: Path,
    scan_type: str = "T1w",
) -> bool:
    """Descarga y extrae un scan de NITRC IR. Retorna True si tuvo éxito."""
    subject_id = experiment_id.split("_")[0]
    project_id = detect_project(experiment_id)

    url = (
        f"{NITRC_BASE}/data/archive/projects/{project_id}"
        f"/subjects/{subject_id}/experiments/{experiment_id}"
        f"/scans/{scan_type}/files?format=tar.gz"
    )

    resp = session.get(url, stream=True)

    if resp.status_code != 200:
        print(f"  [SKIP] HTTP {resp.status_code} — sin datos para {experiment_id}")
        return False

    content = resp.content
    if len(content) < 1000:
        print(f"  [SKIP] Respuesta muy pequeña ({len(content)} bytes) — probablemente sin scan")
        return False

    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                filename = Path(member.name).name

                # Determinar subdirectorio BIDS
                scan_label = filename.split("_")[-1].split(".")[0]  # T1w, T2w, bold, etc.
                bids_type = _scan_to_bids_folder(scan_label)

                # Extraer subject y session del filename
                # sub-OAS30709_ses-d1365_run-01_T1w.nii.gz
                parts = filename.split("_")
                sub_part = parts[0] if parts[0].startswith("sub-") else f"sub-{subject_id}"
                ses_part = next((p for p in parts if p.startswith("ses-")), None)
                if ses_part is None:
                    days = experiment_id.split("_d")[-1] if "_d" in experiment_id else "0000"
                    ses_part = f"ses-d{int(days):04d}"

                dest_dir = output_dir / sub_part / ses_part / bids_type
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_file = dest_dir / filename
                with tar.extractfile(member) as src:
                    dest_file.write_bytes(src.read())

        print(f"  [OK] {experiment_id} extraído")
        return True

    except (tarfile.TarError, gzip.BadGzipFile) as e:
        print(f"  [ERROR] Archivo corrupto para {experiment_id}: {e}")
        return False


def _scan_to_bids_folder(scan_label: str) -> str:
    """Mapea un tipo de scan a su carpeta BIDS."""
    mapping = {
        "T1w": "anat", "T2w": "anat", "FLAIR": "anat", "T2star": "anat", "angio": "anat",
        "bold": "func", "asl": "func",
        "fieldmap": "fmap",
        "dwi": "dwi", "dti": "dwi",
        "pet": "pet",
        "swi": "swi", "minIP": "swi", "GRE": "swi",
    }
    return mapping.get(scan_label, "other")


def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = argparse.ArgumentParser(description="Descargar scans MR de OASIS-3 desde NITRC IR")
    parser.add_argument("csv", type=str, help="CSV con los experiment IDs (ej: download_test_10.csv)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Directorio de salida")
    parser.add_argument("--username", type=str, default=None, help="Usuario NITRC (si no se da, se pide)")
    parser.add_argument("--scan-type", type=str, default="T1w", help="Tipo de scan (default: T1w)")
    parser.add_argument("--password", type=str, default=None, help="Contraseña NITRC (si no se da, se pide)")
    parser.add_argument("--column", type=str, default=None,
                        help="Nombre de la columna con los experiment IDs (autodetect si no se da)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el archivo: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Autodetectar columna con experiment IDs
    if args.column:
        id_col = args.column
    else:
        import re
        mr_pattern = re.compile(r"OAS3\d+_MR_d\d+")
        id_col = None
        for col in df.columns:
            if df[col].astype(str).str.match(mr_pattern).any():
                id_col = col
                break
        if id_col is None:
            id_col = df.columns[0]

    experiment_ids = df[id_col].dropna().astype(str).tolist()
    experiment_ids = [eid.strip() for eid in experiment_ids if eid.startswith("OAS3")]
    print(f"[INFO] {len(experiment_ids)} sesiones a descargar desde columna '{id_col}'")

    username = args.username or input("Usuario NITRC: ")
    password = args.password or getpass.getpass("Contraseña NITRC: ")

    session = authenticate(username, password)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for i, eid in enumerate(experiment_ids, 1):
        print(f"\n[{i}/{len(experiment_ids)}] {eid}")
        if download_scan(session, eid, output_dir, scan_type=args.scan_type):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"DESCARGA COMPLETADA")
    print(f"  Exitosas:  {ok}")
    print(f"  Fallidas:  {fail}")
    print(f"  Destino:   {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
