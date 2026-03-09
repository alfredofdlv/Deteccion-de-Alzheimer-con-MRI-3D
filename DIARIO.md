# Diario de Desarrollo — TFG Detección de Alzheimer con MRI 3D

---

## 2026-03-09 — Situación inicial: Overfitting severo

### Estado del proyecto

Primer entrenamiento completo de la red `Simple3DCNN` sobre el dataset OASIS-1 (clasificación de MRI 3D en 3 clases: CN, MCI, AD).

### Datos

- **Dataset**: OASIS-1 (Cross-Sectional)
- **Sujetos totales**: 235 (tras filtrar los que no tienen CDR)
- **Partición**: 164 train / 35 val / 36 test (70/15/15%)
- **Clases**: CN (CDR=0), MCI (CDR=0.5), AD (CDR≥1)
- **Distribución desbalanceada**: los class weights calculados son `[0.58, 1.12, 2.60]`, lo que indica que CN es la clase mayoritaria y AD la minoritaria

### Arquitectura

- **Modelo**: `Simple3DCNN` — 4 bloques convolucionales 3D
- **Progresión de canales**: 1 → 32 → 64 → 128 → 256
- **Bloques**: Conv3d(3×3×3, pad=1) → BatchNorm3d → ReLU → MaxPool3d(2)
- **Cabeza**: AdaptiveAvgPool3d(1) → Flatten → Dropout(0.5) → Linear(256, 3)
- **Parámetros entrenables**: 1,164,291

### Hiperparámetros

| Parámetro      | Valor        |
|----------------|--------------|
| Learning rate  | 1e-4         |
| Batch size     | 4            |
| Optimizer      | Adam         |
| Image size     | 96×96×96     |
| Epochs         | 100          |
| Early stopping | No se activó (patience=10 pero val_loss oscilaba lo suficiente para resetear) |
| Loss           | CrossEntropyLoss con class weights |

### Transforms (sin data augmentation)

1. LoadImaged
2. EnsureChannelFirstd
3. Orientationd (RAS)
4. ScaleIntensityRangePercentilesd (percentiles 1-99 → [0, 1])
5. Resized (96×96×96)

### Resultados del run `full_100ep`

| Métrica         | Mejor epoch (12) | Último epoch (100) |
|-----------------|------------------|--------------------|
| Train Loss      | 0.9522           | 0.2198             |
| Train Accuracy  | 56.10%           | 92.68%             |
| Val Loss        | 0.8927           | 3.6853             |
| Val Accuracy    | 57.14%           | 60.00%             |

**Tiempo**: ~167 minutos en CPU (~100s/epoch)

### Diagnóstico: Overfitting severo

El modelo presenta un caso claro de **overfitting**:

1. **Train accuracy sube continuamente** de ~52% a ~93%, mientras que **val accuracy se estanca en ~57-60%** y oscila erráticamente.
2. **Val loss se dispara** de 0.89 (epoch 12) a 3.69 (epoch 100), mientras que train loss baja constantemente.
3. **La divergencia empieza desde muy pronto** (alrededor del epoch 20-30), pero el early stopping no se activó porque val_loss oscila mucho entre epochs (varianza altísima con solo 35 muestras de validación).
4. **Val accuracy oscila entre valores discretos** (14.3%, 28.6%, 57.1%, 60.0%) — esto es aritmética de 35 muestras: el modelo mueve muy pocas predicciones entre epochs.

### Causas probables

- **Dataset muy pequeño**: 164 sujetos de entrenamiento para una CNN 3D con >1M parámetros es claramente insuficiente para generalizar.
- **Sin data augmentation**: no se aplican transformaciones de aumento de datos, por lo que el modelo ve las mismas 164 imágenes en cada epoch.
- **Modelo posiblemente sobredimensionado** para tan pocos datos (1.16M parámetros vs 164 muestras).
- **Set de validación muy pequeño** (35 muestras), lo que genera métricas de validación con alta varianza.

### Próximos pasos a considerar

- [ ] Añadir **data augmentation** (flips, rotaciones, intensidad, ruido)
- [ ] Probar **reducir la capacidad** del modelo (menos canales o menos capas)
- [ ] Implementar **regularización** adicional (weight decay, más dropout)
- [ ] Evaluar si el **learning rate scheduler** ayudaría
- [ ] Considerar **transfer learning** o arquitecturas preentrenadas
- [ ] Explorar **cross-validation** en lugar de un único split para usar mejor los pocos datos

---
