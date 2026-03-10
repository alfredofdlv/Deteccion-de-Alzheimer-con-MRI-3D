# Diario de Desarrollo — TFG Detección de Alzheimer con MRI 3D

> Historial consolidado de todos los experimentos y decisiones del proyecto.
> Cada sesión de trabajo añade una entrada nueva al inicio.

---

## 2026-03-10 — Sesión de experimentación intensiva

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
