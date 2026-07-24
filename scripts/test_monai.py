"""
test_monai.py — Validación de que los NIfTI descargados de OASIS-3 son
compatibles con el pipeline MONAI del proyecto.

Busca un archivo T1w .nii.gz en data/OASIS-4/ y lo pasa por el mismo
pipeline de transforms que usa el entrenamiento (sin augmentation).

Uso:
    python test_monai.py
"""

from pathlib import Path

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    Resize,
    ScaleIntensityRangePercentiles,
)

OASIS3_DIR = Path(__file__).resolve().parent / "data" / "OASIS-4"
EXPECTED_SHAPE = (1, 96, 96, 96)


def find_t1w_nifti(base_dir: Path) -> Path | None:
    """Busca el primer archivo *T1w*.nii.gz dentro del directorio."""
    matches = sorted(base_dir.rglob("*T1w*.nii.gz"))
    return matches[0] if matches else None


def main() -> None:
    nifti_path = find_t1w_nifti(OASIS3_DIR)

    if nifti_path is None:
        print(
            "[ERROR] No se encontró ningún archivo *T1w*.nii.gz en:\n"
            f"  {OASIS3_DIR}\n\n"
            "Asegúrate de haber ejecutado el script de descarga primero.\n"
            "Consulta las instrucciones en README.md o Docs/Context.md."
        )
        return

    print(f"[OK] Archivo encontrado: {nifti_path.relative_to(OASIS3_DIR.parent.parent)}")

    # Pipeline idéntico al de src/dataset.py (versión no-diccionario)
    loader = LoadImage(image_only=True)
    img_raw = loader(str(nifti_path))
    print(f"[INFO] Shape original (tras LoadImage): {tuple(img_raw.shape)}")

    pipeline = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRangePercentiles(
            lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True,
        ),
        Resize(spatial_size=(96, 96, 96)),
    ])

    img_out = pipeline(img_raw)
    final_shape = tuple(img_out.shape)
    print(f"[INFO] Shape final (tras pipeline):     {final_shape}")

    if final_shape == EXPECTED_SHAPE:
        print(f"\n[PASS] Shape final correcto: {final_shape}")
        print("Los NIfTI de OASIS-3 son compatibles con el pipeline.")
    else:
        print(f"\n[FAIL] Shape esperado {EXPECTED_SHAPE}, obtenido {final_shape}")
        print("Revisar el pipeline o el archivo descargado.")


if __name__ == "__main__":
    main()
