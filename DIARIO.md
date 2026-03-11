# Diario de Desarrollo — TFG Deteccion de Alzheimer con MRI 3D

> Historial consolidado de todos los experimentos y decisiones del proyecto.
> Cada sesion de trabajo anade una entrada nueva al inicio.

---

## 2026-03-11 — Migracion a OASIS-3, ResNet-10 y macro F1

### Resumen

Sesion completa de upgrade del pipeline: migracion del dataset OASIS-1 (235 sujetos) a OASIS-3 (2450 sesiones), reemplazo de la arquitectura Simple3DCNN por ResNet-10 de MONAI, y cambio de la metrica de seleccion de modelo de accuracy a macro F1-score. Tambien se implemento preprocesamiento offline a .pt y limpieza general del codigo.

### Cambios realizados

**Datos y preprocesamiento:**

- Integrado dataset OASIS-3 con splits subject-level: Train 1731 / Val 369 / Test 350
- Creado `preprocess_to_pt.py` para preprocesar NIfTI a tensores `.pt` offline
- Adaptado `src/dataset.py` para auto-detectar `.pt` vs NIfTI y aplicar transforms correspondientes
- Adaptado `src/data_utils.py` para buscar automaticamente CSVs `_pt.csv` preprocesados

**Modelo:**

- Reemplazado `Simple3DCNN` (~1.16M params) por `AlzheimerResNet` (ResNet-10 3D de MONAI, ~14.3M params)
- Alias `Simple3DCNN = AlzheimerResNet` para retrocompatibilidad

**Metrica de seleccion:**

- Cambiada la metrica de seleccion de modelo de `val_acc` a **macro F1-score**
- `EarlyStopping` ahora rastrea `best_f1` en vez de `best_acc`
- `is_best` se determina por `val_metrics["macro_f1"] > best_val_f1`
- CSV de training log ahora incluye columnas `train_f1` y `val_f1`
- Nueva grafica `curves_f1.png` generada en cada run
- `training_summary.txt` muestra mejor epoch por macro F1

**Logging y benchmark:**

- Logging intra-epoch con progreso, ETA y metricas en tiempo real
- Creado `benchmark.py` para medir tiempos y generar `Docs/benchmark.md`
- Benchmark comparativo: NIfTI baseline vs .pt + 4 workers vs ResNet-10

**Limpieza de codigo:**

- Scripts auxiliares movidos a `scripts/` (download_oasis3.py, test_monai.py, explorar_non_imaging.py)
- Eliminadas funciones no usadas en `data_utils.py`
- Corregido variable shadowing en `dataset.py`
- Emojis reemplazados por texto plano en `data_prepare.py`
- Constantes magicas extraidas a variables con nombre
- Documentacion actualizada: README.md, Docs/Context.md, Cursor rules

### Resultados

- Benchmark ResNet-10 + .pt + 4 workers: ~2.6s/batch train step, ~20 min/epoch estimado
- Verificacion con `--subset 20 --epochs 3`: pipeline completo funciona correctamente
- Entrenamiento completo sobre OASIS-3 pendiente

### Decisiones tomadas

1. **ResNet-10 en vez de Simple3DCNN** — modelo probado en literatura medica, mas capacidad para 2450 sesiones
2. **Macro F1 como metrica de seleccion** — accuracy es enganosa con clases desbalanceadas (CN domina)
3. **Preprocesamiento offline** — reduce I/O de ~1.4s a ~0.05s por batch
4. **OASIS-3 como dataset principal** — 10x mas datos que OASIS-1, reduce overfitting

### Hiperparametros actuales del codigo

```
Modelo:            AlzheimerResNet (ResNet-10 3D, ~14.3M params)
Learning rate:     1e-4
Weight decay:      1e-4
Batch size:        4
num_workers:       4
Early stopping:    patience=25 (sobre macro F1)
Seleccion modelo:  Mejor val macro F1
Loss:              CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
Scheduler:         ReduceLROnPlateau(factor=0.5, patience=5)
```

### Proximos pasos 

- Lanzar entrenamiento completo de ResNet-10 sobre OASIS-3
- Analizar resultados: ¿mejora el F1 en clases minoritarias (MCI/AD)?
- Implementar Grad-CAM para explicabilidad

---

## 2026-03-10 — Sesion de experimentacion intensiva

### Experimentos realizados (6 runs en un día)

#### Run: `new-version-2-dataaug` (CPU, 5 bloques + GroupNorm, sin augmentation)

- **Modelo v2**: 5 bloques conv (1->32->64->128->256->512), GroupNorm(8), Kaiming init, log-prior bias
- **Parámetros**: 4,705,539 (~4x más que v1)
- **Resultado**: El modelo colapsa. Val acc queda fija en 57.14% (predice solo CN) durante 54 epochs.
- **Diagnóstico**: Modelo sobredimensionado para 164 muestras sin augmentation. GroupNorm no compensó.

#### Run: `new-version-4conv+dataaug` (GPU, 4 bloques + data augmentation)

- Vuelta a 4 bloques + BatchNorm3d (1.16M params)
- Se añade data augmentation: RandFlip, RandRotate, RandGaussianNoise, RandShiftIntensity
- **Mejor epoch**: 32 (val_loss=0.874, val_acc=65.71%)
- **Test**: Accuracy 52.78%, F1 macro 0.44
- **Progreso**: Data augmentation reduce la divergencia train/val significativamente

#### Run: `label-smoothing` (GPU, label_smoothing=0.1, selección por val_loss)

- Se añade `label_smoothing=0.1` a CrossEntropyLoss
- Selección del mejor modelo por `val_loss` mínima
- **Mejor epoch** (val_loss): 32 (val_loss=1.019, val_acc=57.14%)
- **Test**: Accuracy 47.22% — peor resultado
- **Diagnóstico**: Val_loss mínima no correlaciona con mejor generalización en test

#### Run: `label-smoothing2` (GPU, label_smoothing=0.1, selección por val_acc)

- Mismo setup pero selección por `val_acc` máxima
- **Mejor epoch**: 25 (val_acc=65.71%, val_loss=1.050)
- **Test**: Accuracy 55.56%, macro F1 0.43
- Mejor distribución de predicciones: CN recall 76%, AD recall 75%, MCI recall 9%
- **Conclusión**: Label smoothing + val_acc selection funciona mejor

#### Run: `ordinal-reg` (GPU, regresión ordinal)

- Cambio de paradigma: `BCEWithLogitsLoss` con targets ordinales [0,0], [1,0], [1,1]
- Modelo con `num_classes-1 = 2` salidas
- **Mejor epoch**: 7 (val_acc=62.86%)
- **Test**: Accuracy 58.33% pero confusion matrix revela que predice TODO como CN
- **Conclusión**: Regresión ordinal colapsa con tan pocos datos. Enfoque descartado.

### Decisiones tomadas hoy

1. **Volver a 4 bloques + BatchNorm3d** — el modelo de 5 bloques con GroupNorm no aportó mejora
2. **Data augmentation es necesaria** — mejora clara en la convergencia train/val
3. **Label smoothing 0.1 se mantiene** — ayuda a repartir predicciones
4. **Seleccionar modelo por val_acc**, no val_loss — correlaciona mejor con test performance
5. **Regresión ordinal descartada** — no funciona con tan pocas muestras
6. **Explorar OASIS-3/4** — se descargaron datos clínicos para posible integración

### Hiperparámetros actuales del código

```
Learning rate:     1e-4
Weight decay:      1e-4
Batch size:        4
Early stopping:    patience=25 (sobre val_acc)
Loss:              CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
Scheduler:         ReduceLROnPlateau(factor=0.5, patience=5)
```

---

## 2026-03-09 — Primer entrenamiento completo + diagnóstico

### Run: `full_100ep` (CPU, sin augmentation, sin early stopping)

- Primer entrenamiento completo de la red `Simple3DCNN`
- **100 epochs** en CPU (~167 minutos, ~100s/epoch)
- **Resultado**:
  - Train accuracy: 52% -> 93% (sube continuamente)
  - Val accuracy: 57% -> 60% (oscila sin mejorar)
  - Val loss se dispara de 0.89 a 3.69
- **Diagnóstico**: **Overfitting severo** desde epoch 20-30
- **Mejor epoch**: 12 (val_loss=0.893, val_acc=57.14%)
- **Test**: Accuracy 52.78%, mejor que random (33%) pero lejos de útil

### Causas identificadas del overfitting

1. 164 muestras de train para 1.16M parámetros (ratio 7000:1)
2. Sin data augmentation
3. Sin early stopping efectivo
4. Val set muy pequeño (35 muestras) -> métricas ruidosas

### Acciones definidas

- Añadir data augmentation (RandFlip, RandRotate, RandNoise)
- Implementar early stopping
- Añadir weight decay y LR scheduler
- Considerar modelo más simple o transfer learning
