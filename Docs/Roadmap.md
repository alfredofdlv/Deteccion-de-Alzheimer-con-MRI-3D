
### 📅 Sprint 1: Data Engineering & "Plumbing" (Semana 1)

**Objetivo:** Tener los datos limpios, organizados y mapeados en un CSV fiable.

#### 1.1. Extracción y Limpieza de Archivos

* **Tarea:** Script Python para recorrer los `.tar.gz` de OASIS-1.
* **Lógica:** Extraer solo `*_masked_gfc.img` y `*.hdr`. Renombrar a `OAS1_XXXX_MR1.img` en una carpeta plana `data/raw/images`.
* **✅ Checkpoint:** La carpeta `data/raw/images` debe tener **416 pares** de archivos (.img/.hdr). Ningún archivo corrupto.

#### 1.2. Generación del CSV Maestro (Etiquetado)

* **Tarea:** Procesar `oasis_cross-sectional.csv`.
* **Lógica de Mapeo (Crucial):**
  * Filtrar sujetos sin CDR (NaN).
  * `CDR 0` -> **Clase 0 (CN)**
  * `CDR 0.5` -> **Clase 1 (MCI)**
  * `CDR 1, 2, 3` -> **Clase 2 (AD)**
* **✅ Checkpoint:** Un archivo `data/processed/dataset_master.csv` con columnas: `['subject_id', 'path_img', 'label', 'age']`.
  * *Validación:* Ejecutar `df['label'].value_counts()`. Debes ver las 3 clases y no debe haber nulos.

#### 1.3. Estrategia de Split (Anti-Leakage)

* **Tarea:** Dividir en Train (70%) / Val (15%) / Test (15%).
* **Requisito:** Usar `stratify=df['label']` para mantener la proporción de Alzheimer en todos los sets.
* **✅ Checkpoint:** 3 archivos CSV en `data/splits/`.
  * *Validación:* Asegúrate de que **ningún** `subject_id` de Test aparece en Train.

---

### 📅 Sprint 2: MONAI Pipeline & Visualización (Semana 2)

**Objetivo:** Que los datos entren a la GPU con la forma correcta `(Batch, Channel, D, H, W)`.

#### 2.1. Dataset y DataLoader

* **Tarea:** Implementar `monai.data.Dataset` (o `CacheDataset` si tienes RAM > 16GB).
* **Pipeline:** `LoadImaged` -> `EnsureChannelFirstd` -> `Orientationd(RAS)` -> `ScaleIntensityRangePercentilesd` -> `Resized(96, 96, 96)`.
* **✅ Checkpoint:** Iterar el DataLoader y extraer un batch.
  * `batch['image'].shape` debe ser **exactamente** `(B, 1, 96, 96, 96)`.
  * `batch['label'].shape` debe ser `(B,)`.

#### 2.2. Sanity Check Visual

* **Tarea:** Renderizar el corte central del tensor procesado.
* **✅ Checkpoint:** Ver una imagen en gris, centrada, sin cráneo.
  * *Fail condition:* Si ves ruido estático o una imagen negra, revisa la normalización. Si ves el cerebro rotado 90 grados, revisa `Orientationd`.

---

### 📅 Sprint 3: Modelado & Training Loop (Semana 3)

**Objetivo:** Lograr que la Loss baje (convergencia).

#### 3.1. Arquitectura "Vanilla" 3D

* **Tarea:** Crear una clase `Simple3DCNN` en `src/model.py`.
* **Estructura:** 4 bloques de `Conv3d` -> `BatchNorm` -> `ReLU` -> `MaxPool3d`.
* **Flatten:** Al final, un `AdaptiveAvgPool3d` + `Linear` (salida = 3 neuronas).
* **✅ Checkpoint:** Pasar un tensor dummy `torch.randn(1, 1, 96, 96, 96)` por el modelo y que no dé error de dimensiones.

#### 3.2. El Test de "Overfit un Batch" (CRÍTICO)

* **Tarea:** Coger solo 4 imágenes del dataset. Entrenar por 100 épocas.
* **✅ Checkpoint:**
  * Training Accuracy debe llegar al  **100%** .
  * Training Loss debe llegar a  **~0.0** .
  * *Si esto no pasa, tienes un bug en el código (no en los datos).*

#### 3.3. Entrenamiento Completo (V1)

* **Tarea:** Entrenar con todo el Train set. Usar `CrossEntropyLoss` (quizás con `weight` si las clases están muy desbalanceadas). Optimizado: `Adam` (lr=1e-4).
* **✅ Checkpoint:** El modelo termina 20-30 épocas sin OOM (Out Of Memory). Guardar pesos `.pth`.

---

### 📅 Sprint 4: Evaluación & Refinamiento (Semana 4)

**Objetivo:** Mejorar métricas y preparar entregables.

#### 4.1. Métricas Reales

* **Tarea:** Implementar cálculo de Accuracy, Precision, Recall y F1-Score (Macro Average).
* **✅ Checkpoint:** Matriz de Confusión generada sobre el Test Set.
  * *Expectativa:* CN y AD se separan bien. MCI se confundirá con ambos.

#### 4.2. Data Augmentation

* **Tarea:** Activar transformaciones aleatorias en MONAI (`RandRotate90`, `RandFlip`, `RandGaussianNoise`) *solo* en el pipeline de Train.
* **✅ Checkpoint:** Volver a entrenar. El Training Accuracy bajará un poco, pero el Validation Accuracy debería subir (menos overfitting).

#### 4.3. (Opcional) Upgrade a DenseNet121-3D

* **Tarea:** Si la CNN simple no pasa del 60-70%, cambiar `Simple3DCNN` por `monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3)`.
* **✅ Checkpoint:** Ajustar `batch_size` (probablemente tendrás que bajarlo a 2 o usar Gradient Accumulation) para que quepa en memoria.

---

### 🏁 Sprint 5: Explicabilidad (XAI)

**Objetivo:** Generar las imágenes para la memoria del TFG.

#### 5.1. Implementar Grad-CAM

* **Tarea:** Usar `monai.visualize.GradCAM`. Engancharlo a la última capa convolucional.
* **✅ Checkpoint:** Generar un mapa de calor superpuesto a un corte de cerebro donde las zonas rojas (activación) coincidan con ventrículos agrandados o hipocampo (zonas típicas de Alzheimer).
