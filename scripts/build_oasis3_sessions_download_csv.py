#!/usr/bin/env python3
"""
Genera data/oasis3_sessions_to_download.csv con experiment_id (formato OAS30001_MR_d0129)
a partir de los splits oasis3_*_pt.csv, para download_oasis_scans.sh de NrgXnat/oasis-scripts.

Uso:
    python scripts/build_oasis3_sessions_download_csv.py
    python scripts/build_oasis3_sessions_download_csv.py --validate
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Raíz del repo (scripts/ -> parent.parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_NAMES = ("train", "val", "test")
SESSION_RE = re.compile(
    r"sub-(?P<sub>OAS\d+)_(?:ses|sess)-d(?P<days>\d+)",
    re.IGNORECASE,
)


def basename_from_image_path(image_path: str) -> str:
    s = str(image_path).replace("\\", "/")
    return s.split("/")[-1]


def path_to_experiment_id(image_path: str, subject_id: str) -> str:
    base = basename_from_image_path(image_path)
    m = SESSION_RE.search(base)
    if not m:
        raise ValueError(f"No se pudo derivar sesión desde: {base!r} (fila subject_id={subject_id})")
    sub = m.group("sub").upper()
    if sub != str(subject_id).strip().upper():
        raise ValueError(
            f"Inconsistencia subject_id={subject_id!r} vs nombre fichero sub={sub!r} en {base!r}"
        )
    d = int(m.group("days"))
    return f"{sub}_MR_d{d:04d}"


def load_experiment_ids_from_splits() -> list[str]:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    seen: dict[str, None] = {}
    errors: list[str] = []

    for split in SPLIT_NAMES:
        path = splits_dir / f"oasis3_{split}_pt.csv"
        if not path.exists():
            errors.append(f"Falta {path}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            try:
                eid = path_to_experiment_id(row["image_path"], row["subject_id"])
            except ValueError as ex:
                errors.append(str(ex))
                continue
            seen[eid] = None

    if errors:
        for e in errors[:20]:
            print(e, file=sys.stderr)
        if len(errors) > 20:
            print(f"... y {len(errors) - 20} errores más", file=sys.stderr)
        raise SystemExit(1)

    return sorted(seen.keys())


def validate_against_inventory(experiment_ids: list[str], inventory_csv: Path) -> None:
    df = pd.read_csv(inventory_csv)
    if "MR ID" not in df.columns:
        raise SystemExit(f"Columna 'MR ID' no encontrada en {inventory_csv}")
    inventory = set(df["MR ID"].astype(str).str.strip())
    needed = set(experiment_ids)
    missing = sorted(needed - inventory)
    extra_unused = len(inventory & needed)  # sanity
    print(f"[validate] experiment_id en splits: {len(needed)}")
    print(f"[validate] MR ID en inventario:    {len(inventory)}")
    print(f"[validate] intersección:           {len(needed & inventory)}")
    if missing:
        print(f"[validate] ⚠ Faltan en inventario ({len(missing)}):", file=sys.stderr)
        for m in missing[:30]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 30:
            print(f"  ... +{len(missing) - 30} más", file=sys.stderr)
    # T1w en columna Scans
    if "Scans" in df.columns:
        mr_to_scans = dict(zip(df["MR ID"].astype(str).str.strip(), df["Scans"].astype(str)))
        no_t1w = [eid for eid in experiment_ids if "T1w" not in mr_to_scans.get(eid, "")]
        if no_t1w:
            print(f"[validate] ⚠ Sesiones sin 'T1w' en columna Scans ({len(no_t1w)}):", file=sys.stderr)
            for m in no_t1w[:20]:
                print(f"  {m} -> {mr_to_scans.get(m, 'N/A')!r}", file=sys.stderr)
            if len(no_t1w) > 20:
                print(f"  ... +{len(no_t1w) - 20} más", file=sys.stderr)
    _ = extra_unused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "oasis3_sessions_to_download.csv",
        help="Ruta del CSV de salida",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cruzar con data/uo293619_3_10_2026_17_33_22.csv (MR ID / Scans)",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT / "data" / "uo293619_3_10_2026_17_33_22.csv",
    )
    args = parser.parse_args()

    eids = load_experiment_ids_from_splits()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"experiment_id": eids}).to_csv(args.out, index=False, lineterminator="\n")
    print(f"[OK] {args.out} — {len(eids)} experiment_id únicos")

    if args.validate:
        if not args.inventory.exists():
            print(f"[validate] No existe {args.inventory}, omitiendo.", file=sys.stderr)
        else:
            validate_against_inventory(eids, args.inventory)


if __name__ == "__main__":
    main()
