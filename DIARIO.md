# Diario de Desarrollo — TFG Deteccion de Alzheimer con MRI 3D

> Historial consolidado de todos los experimentos y decisiones del proyecto.
> Las entradas nuevas se añaden **arriba** (mas reciente primero).
>
> **Para redactar la memoria / documentacion:** usar el **indice cronologico** (abajo, orden antiguo a reciente) para narrar la evolucion. Cada run en `outputs/<nombre>/` tiene fecha en la primera linea de `pipeline.log` (`Inicio: YYYY-MM-DD HH:MM:SS`). Resumen numerico en `training_summary.txt` cuando el entrenamiento termino bien.

---

## 2026-04-03 — Pérdida ordinal (OrdinalClinicalF2Loss) y fix Grad-CAM DenseNet

### Resumen

Dos mejoras independientes: (1) corrección del target layer de Grad-CAM para DenseNet121 (`net.class_layers.relu` → `net.features.denseblock4`, que mantiene mapa espacial 3D antes del avg_pool); (2) implementación de pérdida ordinal BCE + soft F2 clínico que respeta la progresión CN→MCI→AD, disponible como flag `--ordinal`.

### Cambios realizados

- [`src/explain.py`](src/explain.py): target layer corregido; función `_resolve_class_idx` para mapear class_idx nominal al logit ordinal correcto; carga de `uses_ordinal` del checkpoint; decodificación de predicciones y `use_clinical` consistentes con `evaluate.py`
- [`src/losses.py`](src/losses.py): nuevo — `OrdinalClinicalF2Loss` con BCE ordinal (2 umbrales binarios) + soft F2 clínico ponderado + penalización de monotonicidad opcional (`monotonicity_lambda`)
- [`src/model.py`](src/model.py): flag `ordinal=False` en `AlzheimerDenseNet` y `MultimodalDenseNet`; `get_model` lo propaga; `uses_ordinal` como marcador de instancia; `AlzheimerResNet` sin cambios
- [`src/train.py`](src/train.py): `compute_pos_weight()` para BCE ordinal balanceado; `decode_preds()` compatible con ambos modos; criterion seleccionado automáticamente según `uses_ordinal`; `--ordinal` en CLI; `uses_ordinal` guardado en checkpoint
- [`src/evaluate.py`](src/evaluate.py): lee `uses_ordinal` del checkpoint; pasa `ordinal` a `get_model`; usa `decode_preds` importado de `train.py`

### Decisiones tomadas

- Ordinal opt-in (`--ordinal`) para no romper ResNet ni checkpoints históricos
- `pos_weight` calculado desde frecuencias reales del split de train (no hardcodeado)
- `monotonicity_lambda=0.0` por defecto — activar con 0.1 si F2 de MCI se queda en 0 durante muchas épocas (síntoma de violación p1 < p2 en early training)
- `alpha=0.5` como punto de partida; ajustar si BCE domina (bajar a 0.3) o si MCI/AD son ignorados (subir a 0.6)

### Advertencias

- Runs con `--ordinal` y sin `--ordinal` **no son comparables por loss**; solo comparar por `val_clinical_f2`
- Grad-CAM con modelo ordinal usa `class_idx=1` para AD, `class_idx=0` para MCI (resuelta en `_resolve_class_idx`)

### Próximos pasos sugeridos

- Lanzar `python run_pipeline.py densenet-ordinal --dataset oasis3 --model densenet121 --ordinal`
- Comparar `val_clinical_f2` vs `densenet-cropped` (baseline estándar)
- Si F2_MCI se queda en 0, activar `monotonicity_lambda=0.1` en `OrdinalClinicalF2Loss`

---

## 2026-04-03 — Rediseño pipeline OASIS-3 para Linux (NIfTI → .pt)

### Resumen

Se elimina la dependencia de CSVs con rutas Windows obsoletas: nuevo script `prepare_oasis3_nifti_splits.py` escanea `data/raw/OASIS-3/` (layout `OAS3XXXX_MR_dYYYY/anatN/`), elige un T1w por sesión (menor `run`), hace matching clínico y escribe `oasis3_{train,val,test}.csv` con rutas absolutas Linux. Constantes `cfg.OASIS3_RAW_DIR` y `cfg.OASIS3_CLINICAL_CSV` en `src/config.py`. Documentación alineada en `Docs/Context.md`, `README.md` y docstrings de `preprocess_to_pt.py` / `prepare_oasis3_splits.py`.

### Cambios realizados

- Nuevo [`prepare_oasis3_nifti_splits.py`](prepare_oasis3_nifti_splits.py); reutiliza `match_labels` y `stratified_subject_split` desde [`prepare_oasis3_splits.py`](prepare_oasis3_splits.py)
- [`src/config.py`](src/config.py): `OASIS3_RAW_DIR`, `OASIS3_CLINICAL_CSV`
- [`preprocess_to_pt.py`](preprocess_to_pt.py): docstring del flujo OASIS-3 Linux
- [`prepare_oasis3_splits.py`](prepare_oasis3_splits.py): cuándo usar vs `prepare_oasis3_nifti_splits.py`; usa `cfg.OASIS3_CLINICAL_CSV`

### Resultados (si aplica)

- Tras ejecutar el script en el NAS: **2450** sesiones matcheadas; splits **1737 / 359 / 354** (train/val/test)

### Decisiones tomadas

- Un solo T1w por carpeta de sesión para evitar duplicados intra-sesión y alinear con el inventario de 2450 sesiones

### Próximos pasos sugeridos

- `python preprocess_to_pt.py --dataset oasis3` y entrenar con `run_pipeline.py`

---

## 2026-04-03 — Lista de descarga OASIS-3 T1w y documentacion NITRC

### Resumen

Tras perder los NIfTI crudos, se automatizo la extraccion de **2450** `experiment_id` (formato `OASXXXXX_MR_dYYYY`) desde los splits `oasis3_*_pt.csv`, se genero `data/oasis3_sessions_to_download.csv` compatible con `download_oasis_scans.sh`, se valido contra `data/uo293619_3_10_2026_17_33_22.csv` (interseccion completa; todos con `T1w` en `Scans`), y se documento el comando bash y el desajuste de layout descarga vs BIDS esperado por `oasis3_*.csv`.

### Cambios realizados

- Nuevo [`scripts/build_oasis3_sessions_download_csv.py`](scripts/build_oasis3_sessions_download_csv.py) (`--validate` opcional)
- Documentacion [`Docs/DOWNLOAD_OASIS3_T1w.md`](Docs/DOWNLOAD_OASIS3_T1w.md)
- CSV local `data/oasis3_sessions_to_download.csv` (carpeta `data/` no versionada)

### Proximos pasos sugeridos

- Ejecutar descarga a `data/raw/OASIS-3/` y alinear arbol de ficheros con rutas de `data/splits/oasis3_{train,val,test}.csv` antes de `preprocess_to_pt.py`.

---

## Indice cronologico de runs OASIS-3 (GPU, marzo 2026)

Orden **de mas antiguo a mas reciente**. Dataset: OASIS-3, splits ~1737/359 train/val salvo nota. Metrica de seleccion segun log: **Clinical F1** en los dos primeros runs; a partir de `resnet-f2`, **Clinical F2** (beta=2, pesos en `config`).

| Inicio (`pipeline.log`) | Carpeta `outputs/` | Modelo | Mejor val (training_summary) | Estado |
|-------------------------|-------------------|--------|------------------------------|--------|
| 2026-03-23 12:07 | `resnet10-oasis3-100ep` | resnet10 | Clinical F1 val **0.2988** (epoch 26) | Completo |
| 2026-03-23 14:43 | `resnet10-2-full` | resnet10 | Clinical F1 val **0.3273** (epoch 46); WD 1e-3 | Completo |
| 2026-03-23 18:20 | `resnet-f2` | resnet10 | Clinical F2 val **0.3939** (epoch 42) | Completo |
| 2026-03-23 21:25 | `resnet-focalloss` | resnet10 | (sin `training_summary`) | Log truncado (~14 epochs) |
| 2026-03-24 06:41 | `resnet-2focalloss` | resnet10 | Clinical F2 val **0.3308** (epoch 36) | Completo |
| 2026-03-24 11:24 | `resnet-cross+dataug` | resnet10 | Clinical F2 val **0.4631** (epoch 56) | Completo |
| 2026-03-24 15:34 | `densenet121` | densenet121 | Clinical F2 val **0.4924** (epoch 39) | Completo |
| 2026-03-24 18:40 | `multimodal-densenet` | multimodal_densenet | Clinical F2 val **0.4227** (epoch 33) | Completo |
| 2026-03-25 09:39 | `densenet121-cropped` | densenet121 | Clinical F2 val **0.4706** (epoch 22) | Completo |
| 2026-03-25 10:44 | `resnet-cropped` | resnet10 | — | Incompleto (solo carga datos en log) |
| 2026-03-25 13:48 | `densenet-cropped` | densenet121 | Clinical F2 val **0.5161** (epoch 24) | Completo |

**Lectura evolutiva (para la memoria):** (1) Primeros entrenamientos completos OASIS-3 con ResNet-10 y metrica clinical F1; (2) refuerzo de regularizacion (weight decay) y paso a **clinical F2** como criterio alineado con recall en AD/MCI; (3) pruebas de **focal loss** (runs parciales o peores val en resumen); (4) **augmentation** adicional en `resnet-cross+dataug`; (5) **DenseNet121** y **multimodal**; (6) entradas **cropped** — la mejor val clinical F2 en esta tabla es `densenet-cropped` (0.5161). Test final: ver `classification_report_test.txt` en cada carpeta.

---

## 2026-04-02 — Limpieza `.claude`, reglas Cursor Linux y alineacion de metricas

### Resumen

Se elimino la configuracion de Claude Code (`.claude/`), se añadio `.claude/` a `.gitignore`, se migro el contenido operativo a una regla Cursor dedicada al entorno Linux/NAS/GPU, se actualizo el contexto del proyecto para reflejar la seleccion del modelo por **val clinical F2** (codigo actual en `src/train.py`), y se sustituyo `CLAUDE.md` por `AGENTS.md` apuntando a `.cursor/rules/`.

### Cambios realizados

- Eliminada carpeta `.claude/`; `.gitignore`: entrada `.claude/`
- Nuevo [`.cursor/rules/environment-linux.mdc`](.cursor/rules/environment-linux.mdc)
- Actualizados [`.cursor/rules/project-context.mdc`](.cursor/rules/project-context.mdc) y [`.cursor/rules/session-log.mdc`](.cursor/rules/session-log.mdc) (`globs: []`, metricas y multimodal)
- [`Docs/Context.md`](Docs/Context.md): tabla de entrenamiento y arbol de codigo alineados con clinical F2
- [`AGENTS.md`](AGENTS.md) en lugar de `CLAUDE.md`; ajustes menores en [`README.md`](README.md) (metrica de seleccion)

### Decisiones tomadas

- La documentacion y las rules deben coincidir con `train.py`: early stopping y mejor checkpoint por **clinical F2**, no por macro F1 exclusivamente.

### Proximos pasos sugeridos

- Mantener `Docs/Context.md` sincronizado si cambia la metrica o el stack.

---

## 2026-03-23 a 2026-03-25 — Experimentos OASIS-3 en servidor (GPU, logs fechados)

### Resumen

Bloque de entrenamientos sobre OASIS-3 completos o parciales en `outputs/`, con inicio de pipeline entre el **23 y el 25 de marzo de 2026** (fechas tomadas de `pipeline.log`). Evolucion: de **clinical F1** a **clinical F2** como metrica de seleccion; exploracion de focal loss, data augmentation extra, DenseNet121, modelo multimodal (MRI + clinicas) y variantes con recorte espacial (*cropped*).

### Resultados clave (validacion, del mejor epoch)

- Mejor **val clinical F2** entre runs con resumen completo: **0.5161** — `densenet-cropped` (epoch 24).
- Mejor **ResNet-10** en bloque *cross+dataug*: val clinical F2 **0.4631** (epoch 56).
- **DenseNet121** base (`densenet121`): val clinical F2 **0.4924** (epoch 39).
- **Multimodal** (`multimodal-densenet`): val clinical F2 **0.4227** (epoch 33) — por debajo de DenseNet solo-imagen en val.

### Decisiones / observaciones

- Los runs con **focal loss** (`resnet-focalloss`, `resnet-2focalloss`) no superaron en val al mejor modelo con CE + pesos de clase en este bloque (comparar tablas en indice).
- `resnet-cropped` y `resnet-focalloss` quedaron **sin `training_summary.txt`** o con log corto: tratarlos como experimentos incompletos o no decisivos para la memoria salvo analisis adicional.

### Proximos pasos sugeridos

- Citar en la documentacion la **tabla del indice** y las rutas `outputs/<run>/` como anexo reproducible.
- Alinear texto de la memoria con metrica actual en codigo (**clinical F2**) y con test en `classification_report_test.txt`.

---

## 2026-03-11 — Migracion a OASIS-3, ResNet-10 y macro F1

### Nota historica (lectura para la memoria)

Esta entrada describe el estado del **11 de marzo de 2026** (pipeline y decisiones de ese dia). Los entrenamientos largos sobre OASIS-3 en GPU con **fechas y metricas clinicas F1/F2** estan tabulados en el **Indice cronologico** al inicio del diario. El codigo actual selecciona el modelo por **val clinical F2** (ver `src/train.py`); la mencion a **macro F1** en los apartados siguientes es la narrativa de esa migracion, no la definicion vigente de todo el proyecto.

### Resumen

Sesion completa de upgrade del pipeline: migracion del dataset OASIS-1 (235 sujetos) a OASIS-3 (2450 sesiones), reemplazo de la arquitectura Simple3DCNN por ResNet-10 de MONAI, y cambio de la metrica de seleccion de modelo de accuracy hacia metricas basadas en F-score (en evolucion: macro F1 en esta sesion; mas adelante clinical F1/F2 en servidor). Tambien se implemento preprocesamiento offline a .pt y limpieza general del codigo.

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
- Entrenamientos completos OASIS-3 en GPU: **marzo 2026**, ver Indice cronologico e `outputs/` (p. ej. `resnet10-oasis3-100ep` desde 2026-03-23)

### Decisiones tomadas

1. **ResNet-10 en vez de Simple3DCNN** — modelo probado en literatura medica, mas capacidad para 2450 sesiones
2. **Metricas tipo F-score frente a accuracy pura** — accuracy es enganosa con clases desbalanceadas (CN domina); el criterio concreto evoluciono despues a F1/F2 clinicos ponderados
3. **Preprocesamiento offline** — reduce I/O de ~1.4s a ~0.05s por batch
4. **OASIS-3 como dataset principal** — 10x mas datos que OASIS-1, reduce overfitting

### Hiperparametros (instantanea del 11 mar; ver `src/config.py` para valores actuales)

```
Modelo:            AlzheimerResNet (ResNet-10 3D, ~14.3M params)
Learning rate:     1e-4
Weight decay:      1e-4  (en experimentos posteriores se probo 1e-3, ver training_summary en outputs)
Batch size:        4
num_workers:       4
Early stopping:    patience=25 (sobre metrica de seleccion en train.py; en esta sesion: orientacion macro F1)
Seleccion modelo:  Criterio F-score (evolucion documentada en indice marzo 2026)
Loss:              CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
Scheduler:         ReduceLROnPlateau(factor=0.5, patience=5)
```

### Proximos pasos 

- Analizar resultados en test por run: `outputs/<nombre>/classification_report_test.txt`
- Comparar arquitecturas y recorte (cropped) usando el indice cronologico
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
