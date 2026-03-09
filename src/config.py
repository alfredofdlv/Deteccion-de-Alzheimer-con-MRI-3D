"""
config.py — Configuración centralizada del proyecto.

Todas las variables globales (rutas, hiperparámetros, semillas) se definen aquí
para garantizar reproducibilidad y evitar "magic numbers" dispersos por el código.

Uso:
    from src.config import cfg
    print(cfg.IMAGE_SIZE)
    print(cfg.DATA_RAW_DIR)
"""

from pathlib import Path


class ProjectConfig:
    """Configuración centralizada del proyecto TFG."""

    # ========================
    # Semilla de reproducibilidad
    # ========================
    RANDOM_SEED: int = 42

    # ========================
    # Rutas del proyecto
    # ========================
    # Raíz del proyecto (dos niveles arriba de este archivo: src/config.py -> tfg/)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = PROJECT_ROOT / "data"
    DATA_RAW_DIR: Path = DATA_DIR / "raw"
    DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
    DATA_SPLITS_DIR: Path = DATA_DIR / "splits"

    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

    # ========================
    # Rutas específicas OASIS-1
    # ========================
    OASIS_RAW_DIR: Path = DATA_DIR / "OASIS-1" / "raw"
    OASIS_CLINICAL_FILE: Path = OASIS_RAW_DIR / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
    PROCESSED_IMAGES_DIR: Path = DATA_PROCESSED_DIR / "images"
    MASTER_CSV_PATH: Path = DATA_PROCESSED_DIR / "dataset_master.csv"

    # ========================
    # Ratios de partición
    # ========================
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # ========================
    # Parámetros de imagen 3D
    # ========================
    # Tamaño objetivo para las MRI 3D tras preprocesamiento.
    # (96, 96, 96) es un buen compromiso entre resolución y consumo de VRAM
    # para GPUs con ≤8 GB (ej. RTX 3060, Colab T4).
    IMAGE_SIZE: tuple = (96, 96, 96)

    # ========================
    # Hiperparámetros de entrenamiento
    # ========================
    BATCH_SIZE: int = 4          # Conservador para GPUs con poca VRAM
    NUM_WORKERS: int = 2         # DataLoader workers (ajustar según CPU)
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4         # Regularización L2 (penalización sobre norma de pesos)
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 10  # Epochs sin mejora en val_loss antes de parar

    # ========================
    # Clases del dataset OASIS-1
    # ========================
    # CDR (Clinical Dementia Rating):
    #   0   = Sin demencia          -> Clase 0 (CN)
    #   0.5 = Demencia muy leve     -> Clase 1 (MCI)
    #   1+  = Demencia leve/mod.    -> Clase 2 (AD)
    NUM_CLASSES: int = 3
    CLASS_LABELS: dict = {
        0: "CN (Cognitively Normal)",
        1: "MCI (Mild Cognitive Impairment)",
        2: "AD (Alzheimer's Disease)",
    }

    def __repr__(self) -> str:
        return (
            f"ProjectConfig(\n"
            f"  RANDOM_SEED         = {self.RANDOM_SEED}\n"
            f"  IMAGE_SIZE          = {self.IMAGE_SIZE}\n"
            f"  BATCH_SIZE          = {self.BATCH_SIZE}\n"
            f"  NUM_CLASSES         = {self.NUM_CLASSES}\n"
            f"  OASIS_RAW_DIR       = {self.OASIS_RAW_DIR}\n"
            f"  PROCESSED_IMAGES_DIR= {self.PROCESSED_IMAGES_DIR}\n"
            f"  OUTPUTS_DIR         = {self.OUTPUTS_DIR}\n"
            f")"
        )


# Instancia global — importar directamente:  from src.config import cfg
cfg = ProjectConfig()

