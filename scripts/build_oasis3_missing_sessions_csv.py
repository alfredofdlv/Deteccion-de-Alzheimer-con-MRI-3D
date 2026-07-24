#!/usr/bin/env python3
"""
build_oasis3_missing_sessions_csv.py

Genera un CSV con las sesiones OASIS-3 que siguen pendientes de descarga
según el contenido actual de data/raw/OASIS-3/.

Entrada:
  - data/oasis3_sessions_to_download.csv
      * Columna: experiment_id (p. ej. OAS30001_MR_d0129)
  - Directorio de descargas:
      * data/raw/OASIS-3/<experiment_id>/...

Criterio de "descargado OK":
  - Existe la carpeta data/raw/OASIS-3/<experiment_id>/
  - Dentro hay al menos un fichero *.nii.gz cuyo nombre contiene "T1w"

Salida:
  - data/oasis3_sessions_to_download_missing.csv
      * Columna: experiment_id
      * Solo IDs que NO cumplen el criterio anterior
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_CSV = PROJECT_ROOT / "data" / "oasis3_sessions_to_download.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "OASIS-3"
DEFAULT_OUT = PROJECT_ROOT / "data" / "oasis3_sessions_to_download_missing.csv"


def has_t1w_nii(experiment_dir: Path) -> bool:
    """
    Devuelve True si dentro de experiment_dir hay al menos un .nii.gz
    con 'T1w' en el nombre del fichero.
    """
    if not experiment_dir.is_dir():
        return False

    # Recorremos recursivamente buscando .nii.gz con 'T1w'
    for nii in experiment_dir.rglob("*.nii.gz"):
        if "t1w" in nii.name.lower():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera data/oasis3_sessions_to_download_missing.csv "
                    "con experiment_id pendientes de descarga T1w."
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        default=SESSIONS_CSV,
        help="CSV con columna experiment_id (por defecto: data/oasis3_sessions_to_download.csv)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directorio raíz de descargas (por defecto: data/raw/OASIS-3/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="CSV de salida con experiment_id pendientes "
             "(por defecto: data/oasis3_sessions_to_download_missing.csv)",
    )
    args = parser.parse_args()

    if not args.sessions.exists():
        raise SystemExit(f"No existe {args.sessions}")

    # Leer lista completa de experiment_id
    df = pd.read_csv(args.sessions)
    if "experiment_id" not in df.columns:
        raise SystemExit(f"Columna 'experiment_id' no encontrada en {args.sessions}")
    all_ids = sorted(set(df["experiment_id"].astype(str).str.strip()))

    print(f"[INFO] Total experiment_id objetivo: {len(all_ids)}")
    print(f"[INFO] Directorio RAW: {args.raw_dir}")

    # Clasificar cada ID como completado/pediente
    completed = []
    missing = []

    for eid in all_ids:
        exp_dir = args.raw_dir / eid
        if has_t1w_nii(exp_dir):
            completed.append(eid)
        else:
            missing.append(eid)

    print(f"[INFO] Sesiones con al menos un T1w (*.nii.gz): {len(completed)}")
    print(f"[INFO] Sesiones pendientes (sin T1w encontrado): {len(missing)}")

    # Escribir CSV de pendientes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"experiment_id": sorted(missing)}).to_csv(
        args.out, index=False, lineterminator="\n"
    )
    print(f"[OK] Escrito {args.out} con {len(missing)} experiment_id pendientes")


if __name__ == "__main__":
    main()