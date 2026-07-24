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
    PREPROCESSED_DIR: Path = DATA_DIR / "preprocessed"

    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

    # Pesos pre-entrenados (UltimateNeuroFM / ResEncL SSL). Coloca aquí resenc_l_ssl3d.pth.
    WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
    RESENC_SSL_CHECKPOINT: Path = WEIGHTS_DIR / "resenc_l_ssl3d.pth"
    YAWARE_DENSENET_CHECKPOINT: Path = (
        WEIGHTS_DIR / "DenseNet121_BHB-10K_yAwareContrastive.pth"
    )

    # ========================
    # Rutas específicas OASIS-1
    # ========================
    OASIS_RAW_DIR: Path = DATA_DIR / "OASIS-1" / "raw"
    OASIS_CLINICAL_FILE: Path = OASIS_RAW_DIR / "oasis_cross-sectional-5708aa0a98d82080.xlsx"
    PROCESSED_IMAGES_DIR: Path = DATA_PROCESSED_DIR / "images"
    MASTER_CSV_PATH: Path = DATA_PROCESSED_DIR / "dataset_master.csv"

    # ========================
    # Rutas específicas OASIS-3 (Linux / NAS)
    # ========================
    # T1w descargados con oasis-scripts → data/raw/OASIS-3/<OAS3XXXX_MR_dYYYY>/anatN/...
    OASIS3_RAW_DIR: Path = DATA_DIR / "raw" / "OASIS-3"
    OASIS3_CLINICAL_CSV: Path = DATA_DIR / "oasis3_master_clinical.csv"

    # ========================
    # Rutas específicas ADNI (Linux / NAS)
    # ========================
    ADNI_DOCS_DIR: Path = PROJECT_ROOT / "Docs" / "ADNI-Context"
    ADNI_RAW_DIR: Path = DATA_DIR / "raw" / "ADNI"          # DICOM descargados del IDA
    ADNI_NIFTI_DIR: Path = DATA_DIR / "raw" / "ADNI-nifti"  # NIfTI tras dcm2niix
    ADNI_DICOM_MANIFEST: Path = DATA_SPLITS_DIR / "adni_download_list.csv"
    ADNI_COHORT_MANIFEST: Path = (
        PROJECT_ROOT / "Docs" / "ADNI-Context" / "03_imaging" / "cohort"
        / "Clinical_T1w_Imaging_Cohort_Manifest_17Jun2026.csv"
    )
    ADNI_URLS_FILE: Path = PROJECT_ROOT / "urls_adni.txt"

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
    # Resolución alta whole-brain (requiere .pt a 128³ o upsample online desde 96³).
    IMAGE_SIZE_HIGH: tuple = (128, 128, 128)
    HIGH_RES_BATCH_SIZE: int = 2

    # ROI lóbulo temporal medial (MTL) sobre tensores MNI 96³ RAS.
    # Bbox ~48³ centrado en hipocampo bilateral + entorhinal (Propuestas.md §B).
    MTL_ROI_START: tuple = (24, 36, 20)
    MTL_ROI_END: tuple = (72, 84, 68)
    MTL_OUTPUT_SIZE: tuple = (64, 64, 64)
    MTL_BATCH_SIZE: int = 4

    # ========================
    # Variantes de preprocesado offline
    # ========================
    # cropped: CropForeground + resize (baseline densenet-cropped)
    # mni: skull-strip + registro afín 12-DOF a MNI152 + resize
    # mni_n4: mni + N4 bias correction (antspy CPU, tras skull-strip)
    PREPROCESS_VARIANT_DEFAULT: str = "cropped"
    PREPROCESS_VARIANTS: dict = {
        "cropped": "oasis3",
        "mni": "oasis3_mni",
        "mni_n4": "oasis3_mni_n4",
        "mni_128": "oasis3_mni_128",
    }
    MNI_TEMPLATE_NAME: str = "MNI152"
    MNI_REGISTRATION_TYPE: str = "Affine"  # 12 DOF; nunca SyN/BSpline
    SKULL_STRIP_BACKEND: str = "auto"  # auto | synthstrip | hd-bet
    INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"
    # N4 bias field (ants.n4_bias_field_correction)
    N4_SHRINK_FACTOR: int = 4
    N4_CONVERGENCE_ITERS: list = [50, 50, 50, 50]
    N4_CONVERGENCE_TOL: float = 1e-7
    # Variantes espaciales (mni, mni_n4, mni_128) comparten pipeline HD-BET + ANTs
    SPATIAL_PREPROCESS_VARIANTS: frozenset = frozenset({"mni", "mni_n4", "mni_128"})

    # ========================
    # Hiperparámetros de entrenamiento
    # ========================
    BATCH_SIZE: int = 4          # Ajustado según VRAM disponible (TITAN Xp reservada)
    NUM_WORKERS: int = 4         # DataLoader workers (ajustar según CPU)
    # GPU física reservada en nserver1 (índice CUDA). Scripts: CUDA_VISIBLE_DEVICES=cfg.DEFAULT_CUDA_DEVICE_ID
    DEFAULT_CUDA_DEVICE_ID: int = 1
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 3e-3         # Regularización L2
    DENSENET_DROPOUT: float = 0.2      # Dropout en bloques DenseNet (MONAI dropout_prob)
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 15  # Epochs sin mejora en la métrica de early stop antes de parar
    # Métrica para early stopping y guardado de best_model.pth.
    # val_loss: detecta overfitting antes que F2; val_clinical_f2: prioriza rendimiento clínico.
    EARLY_STOP_METRIC: str = "val_clinical_f2"
    CHECKPOINT_METRIC: str = "val_clinical_f2"

    # ========================
    # Prioridad clinica
    # ========================
    # Multiplicadores sobre los class weights de la loss (penalizacion asimetrica).
    # Fuerzan a la red a "sufrir" mas cuando falla en clases criticas.
    CLINICAL_WEIGHT_MULTIPLIERS: dict = {0: 1.0, 1: 1.5, 2: 2.0}

    # Pesos de la metrica clinical F2 (β=2) para seleccion de modelo.
    # Priorizan deteccion de AD (60%) sobre MCI (30%) y CN (10%).
    CLINICAL_F2_WEIGHTS: dict = {0: 0.10, 1: 0.30, 2: 0.60}

    # Matriz de coste clinico para Expected Cost (src/metrics.py).
    # cost[i][j] = coste de predecir j siendo la clase real i. Filas/cols: CN, MCI, AD.
    # Captura la asimetria y direccionalidad del error (p.ej. AD->CN penaliza mas
    # que AD->MCI). Valores ILUSTRATIVOS (DeepResearch); consensuar con el tutor.
    COST_MATRIX: tuple = (
        (0, 1, 5),    # real CN  -> pred CN/MCI/AD
        (3, 0, 1),    # real MCI -> pred CN/MCI/AD
        (10, 4, 0),   # real AD  -> pred CN/MCI/AD
    )

    # ========================
    # Clases del dataset
    # ========================
    # CDR (Clinical Dementia Rating):
    #   0   = Sin demencia          -> Clase 0 (CN)
    #   0.5 = Demencia muy leve     -> Clase 1 (MCI)
    #   1+  = Demencia leve/mod.    -> Clase 2 (AD)
    NUM_CLASSES: int = 3
    # Dimensión del embedding GAP de DenseNet-121 (MONAI) antes de la cabeza lineal.
    DENSENET_EMBED_DIM: int = 1024
    # 2.5D EfficientNet-B0 (cortes axiales + soft attention, estilo AXIAL).
    EFFICIENTNET25D_NUM_SLICES: int = 16
    EFFICIENTNET25D_SLICE_SIZE: int = 224
    EFFICIENTNET25D_EMBED_DIM: int = 1280
    EFFICIENTNET25D_BATCH_SIZE: int = 2
    EFFICIENTNET25D_DROPOUT: float = 0.3
    # Congelar los primeros N bloques de EfficientNet-B0.features (0 = todo entrenable).
    EFFICIENTNET25D_FREEZE_BLOCKS: int = 4
    EFFICIENTNET25D_WEIGHT_DECAY: float = 5e-3
    # cRT (classifier re-training): épocas y LR de la cabeza con batches balanceados.
    CRT_NUM_EPOCHS: int = 30
    CRT_HEAD_LR: float = 1e-3
    CRT_PATIENCE: int = 10
    # Normalización eTIV / proxy de volumen intracraneal (mm³ o conteo de voxels).
    ETIV_NORM_DIVISOR: float = 1_500_000.0
    ETIV_FOREGROUND_THRESHOLD: float = 0.05
    ETIV_CACHE_CSV: Path = DATA_DIR / "etiv_cache.csv"
    CLASS_LABELS: dict = {
        0: "CN (Cognitively Normal)",
        1: "MCI (Mild Cognitive Impairment)",
        2: "AD (Alzheimer's Disease)",
    }

    def spatial_size_for_variant(self, variant: str | None = None) -> tuple[int, int, int]:
        """Tamaño de salida del tensor según variante de preprocesado."""
        if variant == "mni_128":
            return self.IMAGE_SIZE_HIGH
        return self.IMAGE_SIZE

    def preprocessed_subdir(self, dataset: str, variant: str | None = None) -> str:
        """Subcarpeta bajo PREPROCESSED_DIR según dataset y variante."""
        variant = variant or self.PREPROCESS_VARIANT_DEFAULT
        if variant == "cropped":
            return dataset
        if variant == "mni":
            return f"{dataset}_mni"
        if variant == "mni_n4":
            return f"{dataset}_mni_n4"
        if variant == "mni_128":
            return f"{dataset}_mni_128"
        if variant in self.PREPROCESS_VARIANTS:
            return self.PREPROCESS_VARIANTS[variant]
        raise ValueError(
            f"Variante de preprocesado desconocida: {variant!r}. "
            f"Opciones: {list(self.PREPROCESS_VARIANTS)}"
        )

    def preprocessed_dir(self, dataset: str, variant: str | None = None) -> Path:
        """Ruta absoluta a tensores .pt preprocesados."""
        return self.PREPROCESSED_DIR / self.preprocessed_subdir(dataset, variant)

    def split_csv_name(
        self, dataset: str, split: str, variant: str | None = None,
    ) -> str:
        """Nombre del CSV de split (con o sin sufijo _mni)."""
        variant = variant or self.PREPROCESS_VARIANT_DEFAULT
        if dataset == "oasis1":
            base = f"{split}_pt" if variant == "cropped" else f"{split}_pt_{variant}"
            return f"{base}.csv"
        suffix = "" if variant == "cropped" else f"_{variant}"
        return f"{dataset}_{split}_pt{suffix}.csv"

    def __repr__(self) -> str:
        return (
            f"ProjectConfig(\n"
            f"  RANDOM_SEED         = {self.RANDOM_SEED}\n"
            f"  IMAGE_SIZE          = {self.IMAGE_SIZE}\n"
            f"  BATCH_SIZE          = {self.BATCH_SIZE}\n"
            f"  NUM_WORKERS         = {self.NUM_WORKERS}\n"
            f"  NUM_CLASSES         = {self.NUM_CLASSES}\n"
            f"  PREPROCESSED_DIR    = {self.PREPROCESSED_DIR}\n"
            f"  OUTPUTS_DIR         = {self.OUTPUTS_DIR}\n"
            f"  RESENC_SSL_CHECKPOINT = {self.RESENC_SSL_CHECKPOINT}\n"
            f")"
        )


# Instancia global — importar directamente:  from src.config import cfg
cfg = ProjectConfig()

