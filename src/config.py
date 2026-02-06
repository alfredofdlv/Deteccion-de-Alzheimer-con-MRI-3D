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
    NUM_EPOCHS: int = 50

    # ========================
    # Clases del dataset OASIS-1
    # ========================
    # CDR (Clinical Dementia Rating):
    #   0   = Sin demencia
    #   0.5 = Demencia muy leve
    #   1   = Demencia leve
    #   2   = Demencia moderada
    NUM_CLASSES: int = 2  # Binario: sano vs. demencia (simplificación inicial)
    CLASS_LABELS: dict = {0: "CN (Cognitively Normal)", 1: "AD (Alzheimer's Disease)"}

    def __repr__(self) -> str:
        return (
            f"ProjectConfig(\n"
            f"  RANDOM_SEED    = {self.RANDOM_SEED}\n"
            f"  IMAGE_SIZE     = {self.IMAGE_SIZE}\n"
            f"  BATCH_SIZE     = {self.BATCH_SIZE}\n"
            f"  NUM_CLASSES    = {self.NUM_CLASSES}\n"
            f"  DATA_RAW_DIR   = {self.DATA_RAW_DIR}\n"
            f"  OUTPUTS_DIR    = {self.OUTPUTS_DIR}\n"
            f")"
        )


# Instancia global — importar directamente:  from src.config import cfg
cfg = ProjectConfig()

