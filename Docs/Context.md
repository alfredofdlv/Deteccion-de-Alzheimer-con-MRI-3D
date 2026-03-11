# Contexto del Proyecto: Deteccion de Alzheimer con 3D MRI

> **Fuente de verdad.** Este archivo describe el estado actual del proyecto. Los agentes AI deben consultarlo al inicio de cada sesion.
>
> Ultima actualizacion: 2026-03-11

---

## 1. Definicion del proyecto

| Campo | Valor |
|-------|-------|
| **Tipo** | Trabajo de Fin de Grado (TFG) |
| **Objetivo** | Clasificar MRI cerebrales 3D en CN / MCI / AD usando Deep Learning |
| **Filosofia** | "Low-Resource & Efficient" — debe correr en GPUs de consumo (RTX 2060, Colab T4) |
| **Prioridad** | Rigor metodologico y codigo limpio por encima de SOTA accuracy |

## 2. Stack tecnico

| Componente | Tecnologia |
|------------|-----------|
| Lenguaje | Python 3.10+ |
| Framework DL | PyTorch |
| Imagen medica | MONAI, NiBabel |
| Data Science | Pandas, NumPy, scikit-learn |
| Visualizacion | Matplotlib, Seaborn |

## 3. Datos

### 3.1 Dataset OASIS-1 (Cross-Sectional)

- **416 sujetos**, rango 18-96 anos
- Archivos: `*_masked_gfc.img/.hdr` (skull-stripped, gain-field corrected, atlas-registered)
- **235 sujetos** con CDR valido tras filtrar
- Splits: Train 164 / Val 35 / Test 36

### 3.2 Dataset OASIS-3 (Longitudinal) — Dataset activo

- **2450 sesiones MRI** T1w de sujetos con CDR evaluado
- Formato BIDS: `data/OASIS-3/sub-*/ses-*/*T1w.nii.gz`
- Splits (subject-level, sin data leakage): **Train 1731 / Val 369 / Test 350**
- Preprocesado offline a tensores `.pt` en `data/preprocessed/oasis3/`

### 3.3 Clases (basadas en CDR)

| Clase | Etiqueta | CDR |
|-------|----------|-----|
| 0 | CN (Cognitively Normal) | 0.0 |
| 1 | MCI (Mild Cognitive Impairment) | 0.5 |
| 2 | AD (Alzheimer's Disease) | >= 1.0 |

## 4. Arquitectura del codigo

```
tfg/
├── src/
│   ├── config.py        # Configuracion centralizada (ProjectConfig)
│   ├── data_prepare.py  # ETL OASIS-1: extraccion, CSV maestro, split estratificado
│   ├── data_utils.py    # Carga de splits (auto-detecta .pt vs NIfTI)
│   ├── dataset.py       # MONAI Dataset, transforms (NIfTI y .pt), DataLoaders
│   ├── model.py         # AlzheimerResNet (ResNet-10 3D de MONAI)
│   ├── train.py         # Training loop con early stopping por macro F1
│   └── evaluate.py      # Evaluacion en test set, confusion matrix
├── run_pipeline.py      # CLI: train -> evaluate -> export
├── benchmark.py         # Medicion de tiempos del pipeline
├── preprocess_to_pt.py  # Preprocesamiento offline NIfTI -> .pt
├── export_context.py    # Exportar codigo a Markdown
├── scripts/             # Scripts auxiliares (descarga, exploracion)
├── notebooks/           # Pipeline OASIS-3, sanity checks
├── outputs/             # Resultados por experimento
└── data/                # OASIS-1/3 raw + processed + splits + preprocessed
```

## 5. Pipeline actual

### 5.1 Preprocesamiento

**Offline (preprocess_to_pt.py):** Aplica transforms deterministicos una sola vez y guarda tensores `.pt`:
1. `LoadImage` — carga NIfTI/ANALYZE
2. `EnsureChannelFirst` — (D,H,W) -> (1,D,H,W)
3. `Orientation("RAS")` — orientacion estandar
4. `ScaleIntensityRangePercentiles` — normalizacion percentil 1-99 -> [0,1]
5. `Resize(96,96,96)` — reduccion para caber en VRAM

**On-the-fly (dataset.py):** Si los datos son `.pt`, solo carga el tensor. Si son NIfTI, aplica el pipeline completo.

### 5.2 Data augmentation (solo train, on-the-fly)

- `RandFlipd` — flip LR, prob=0.5
- `RandRotated` — rotacion 3D, rango 0.2 rad, prob=0.3
- `RandGaussianNoised` — ruido gaussiano, std=0.05, prob=0.3
- `RandShiftIntensityd` — shift de intensidad, prob=0.3

### 5.3 Modelo: AlzheimerResNet

- **ResNet-10 3D** de `monai.networks.nets.resnet10`
- Configuracion: `spatial_dims=3, n_input_channels=1, num_classes=3`
- **~14.3M parametros** entrenables
- Alias retrocompatible: `Simple3DCNN = AlzheimerResNet`

### 5.4 Entrenamiento

| Parametro | Valor actual |
|-----------|-------------|
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | CrossEntropyLoss + class weights + label_smoothing=0.1 |
| Batch size | 4 |
| num_workers | 4 |
| Early stopping | patience=25 (sobre **macro F1**) |
| Seleccion de modelo | **Mejor val macro F1** |
| Metricas logueadas | loss, accuracy, macro F1 (train y val) |

### 5.5 Benchmark de tiempos (RTX 2060, 6.4GB VRAM)

Con datos `.pt` preprocesados + num_workers=4:
- Carga batch: ~0.05s
- Train step (ResNet10): ~2.6s/batch (excluyendo outliers)
- Estimacion por epoch: ~20 min
- 50 epochs: ~16.5 horas

## 6. Historial de experimentos

Ver `DIARIO.md` para el historial completo. Resumen:

| Fecha | Hito |
|-------|------|
| 2026-03-09 | Primer entrenamiento OASIS-1 (Simple3DCNN, 52.78% test acc) |
| 2026-03-10 | 6 experimentos: augmentation, label smoothing, ordinal regression |
| 2026-03-11 | Migracion a OASIS-3, preprocesamiento .pt, upgrade a ResNet-10, macro F1 |

## 7. Progreso por Sprint

| Sprint | Tema | Estado |
|--------|------|--------|
| 1 | Data Engineering (ETL, splits) | COMPLETADO |
| 2 | MONAI Pipeline (transforms, DataLoaders) | COMPLETADO |
| 3 | Modelado (Simple3DCNN, overfit test) | COMPLETADO |
| 4 | Evaluacion y Refinamiento (OASIS-3, ResNet, macro F1) | COMPLETADO |
| 5 | Explainability (Grad-CAM) | PENDIENTE |

## 8. Proximos pasos prioritarios

1. Entrenar ResNet-10 sobre OASIS-3 completo y evaluar resultados
2. Implementar Grad-CAM para explicabilidad
3. Considerar cross-validation para estimaciones mas robustas
4. Explorar fine-tuning de hiperparametros (batch size, learning rate)

## 9. Reglas para agentes AI

1. **No data leakage:** nunca mezclar sujetos entre splits
2. **No magic numbers:** usar `src/config.py` para todo
3. **Memory first:** asumir VRAM limitada, batch_size max 4-8
4. **Validez medica:** no distorsionar aspecto del cerebro; resize isotropico
5. **No datos sinteticos:** no GANs/Diffusion en esta fase
6. **Idioma:** responder siempre en espanol
7. **Antes de proponer cambios:** leer el estado actual de `src/train.py` y `src/model.py`
