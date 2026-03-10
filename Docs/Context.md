# Contexto del Proyecto: Detección de Alzheimer con 3D MRI

> **Fuente de verdad.** Este archivo describe el estado actual del proyecto. Los agentes AI deben consultarlo al inicio de cada sesión.
>
> Ultima actualización: 2026-03-10

---

## 1. Definición del proyecto

| Campo | Valor |
|-------|-------|
| **Tipo** | Trabajo de Fin de Grado (TFG) |
| **Objetivo** | Clasificar MRI cerebrales 3D en CN / MCI / AD usando Deep Learning |
| **Filosofía** | "Low-Resource & Efficient" — debe correr en GPUs de consumo (Colab T4 / RTX 3060) |
| **Prioridad** | Rigor metodológico y código limpio por encima de SOTA accuracy |

## 2. Stack técnico

| Componente | Tecnología |
|------------|-----------|
| Lenguaje | Python 3.10+ |
| Framework DL | PyTorch |
| Imagen médica | MONAI, NiBabel |
| Data Science | Pandas, NumPy, scikit-learn |
| Visualización | Matplotlib, Seaborn |

## 3. Datos

### 3.1 Dataset primario: OASIS-1 (Cross-Sectional)

- **416 sujetos**, rango 18-96 años
- Archivos usados: `*_masked_gfc.img/.hdr` (skull-stripped, gain-field corrected, atlas-registered)
- **235 sujetos** con CDR válido tras filtrar

### 3.2 Clases (basadas en CDR)

| Clase | Etiqueta | CDR | Train | Val | Test |
|-------|----------|-----|-------|-----|------|
| 0 | CN (Cognitively Normal) | 0.0 | 94 | 20 | 21 |
| 1 | MCI (Mild Cognitive Impairment) | 0.5 | 49 | 10 | 11 |
| 2 | AD (Alzheimer's Disease) | >= 1.0 | 21 | 5 | 4 |
| | **Total** | | **164** | **35** | **36** |

Class weights: `[0.58, 1.12, 2.60]`

### 3.3 Dataset secundario: OASIS-3/4 (en exploración)

Se han descargado los datos clínicos (Non-Imaging Data) de OASIS-3/4 con 34 CSVs que contienen datos demográficos, CDR, evaluaciones cognitivas, FreeSurfer, PET, etc. Todavía no se usan en el pipeline.

## 4. Arquitectura del código

```
tfg/
├── src/
│   ├── config.py        # Configuración centralizada (ProjectConfig)
│   ├── data_prepare.py  # ETL: extracción, CSV maestro, split estratificado
│   ├── data_utils.py    # Utilidades de carga de splits
│   ├── dataset.py       # MONAI Dataset, transforms, DataLoaders
│   ├── model.py         # Simple3DCNN (arquitectura)
│   ├── train.py         # Training loop con early stopping, logging
│   └── evaluate.py      # Evaluación en test set, confusion matrix
├── run_pipeline.py      # CLI: train -> evaluate -> export
├── notebooks/           # Sanity checks y exploración
├── outputs/             # Resultados por experimento
└── data/                # OASIS-1 raw + processed + splits
```

## 5. Pipeline actual

### 5.1 Preprocesamiento (MONAI, on-the-fly)

1. `LoadImaged` — carga par .img/.hdr
2. `EnsureChannelFirstd` — (D,H,W) -> (1,D,H,W)
3. `Orientationd("RAS")` — orientación estándar
4. `ScaleIntensityRangePercentilesd` — normalización percentil 1-99 -> [0,1]
5. `Resized(96,96,96)` — reducción para caber en VRAM

### 5.2 Data augmentation (solo train)

6. `RandFlipd` — flip LR, prob=0.5
7. `RandRotated` — rotación 3D, rango 0.2 rad, prob=0.3
8. `RandGaussianNoised` — ruido gaussiano, std=0.05, prob=0.3
9. `RandShiftIntensityd` — shift de intensidad, prob=0.3

### 5.3 Modelo: Simple3DCNN

- **4 bloques** Conv3d(3x3x3) -> BatchNorm3d -> ReLU -> MaxPool3d(2)
- Progresión de canales: 1 -> 32 -> 64 -> 128 -> 256
- AdaptiveAvgPool3d(1) -> Dropout(0.5) -> Linear(256, 3)
- ~1.16M parámetros

### 5.4 Entrenamiento

| Parámetro | Valor actual |
|-----------|-------------|
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | CrossEntropyLoss + class weights + label_smoothing=0.1 |
| Batch size | 4 |
| Early stopping | patience=25 (sobre val_acc) |
| Selección de modelo | Mejor val_acc |

## 6. Historial de experimentos

| Run | Fecha | Variante | Mejor Val Acc | Test Acc | Nota clave |
|-----|-------|----------|---------------|----------|-----------|
| `full_100ep` | 2026-03-09 | Baseline sin augmentation (CPU) | 57.14% | 52.78% | Overfitting severo. Train 93%, Val 57% |
| `version-overfit-base` | 2026-03-10 | 5 bloques + GroupNorm (CPU) | 57.14% | -- | Modelo colapsado: predice solo CN |
| `new-version-4conv+dataaug` | 2026-03-10 | 4 bloques + data augmentation (GPU) | 65.71% | 52.78% | Data aug reduce overfitting |
| `label-smoothing` | 2026-03-10 | + label_smoothing=0.1 (val_loss) | 68.57%* | 47.22% | *val_acc del ultimo epoch, mejor val_loss en ep32 |
| `label-smoothing2` | 2026-03-10 | + label_smoothing=0.1 (val_acc) | 65.71% | 55.56% | Mejor distribución de predicciones |
| `ordinal-reg` | 2026-03-10 | Regresión ordinal (BCEWithLogits) | 62.86% | 58.33% | Predice solo CN. Enfoque descartado |

### Conclusiones de los experimentos

1. **Overfitting** sigue siendo el problema principal (164 train samples, 1.16M params)
2. **Data augmentation** reduce la divergencia train/val pero no la elimina
3. **Label smoothing** ayuda a repartir mejor las predicciones entre clases
4. **Regresión ordinal** colapsa y predice una sola clase
5. **Val accuracy oscila** mucho con solo 35 muestras (valores discretos)
6. Mejor test accuracy real: **55-58%**, macro F1: ~0.43

## 7. Progreso por Sprint

| Sprint | Tema | Estado |
|--------|------|--------|
| 1 | Data Engineering (ETL, splits) | COMPLETADO |
| 2 | MONAI Pipeline (transforms, DataLoaders) | COMPLETADO |
| 3 | Modelado (Simple3DCNN, overfit test) | COMPLETADO |
| 4 | Evaluación y Refinamiento (augmentation, métricas) | EN PROGRESO |
| 5 | Explainability (Grad-CAM) | PENDIENTE |

## 8. Próximos pasos prioritarios

1. Explorar el dataset OASIS-3/4 y su posible integración (más datos = menos overfitting)
2. Probar transfer learning con DenseNet121-3D de MONAI
3. Implementar Grad-CAM sobre la última capa conv
4. Considerar cross-validation para estimaciones más robustas

## 9. Reglas para agentes AI

1. **No data leakage:** nunca mezclar sujetos entre splits
2. **No magic numbers:** usar `src/config.py` para todo
3. **Memory first:** asumir VRAM limitada, batch_size max 4-8
4. **Validez médica:** no distorsionar aspecto del cerebro; resize isotrópico
5. **No datos sintéticos:** no GANs/Diffusion en esta fase
6. **Idioma:** responder siempre en español
7. **Antes de proponer cambios:** leer el estado actual de `src/train.py` y `src/model.py`, que evolucionan entre experimentos
