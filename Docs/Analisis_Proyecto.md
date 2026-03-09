# Análisis Detallado del Proyecto: Detección Temprana de Alzheimer mediante MRI 3D

---

## Índice

1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Datos y Dataset (OASIS-1)](#2-datos-y-dataset-oasis-1)
3. [Pipeline ETL](#3-pipeline-etl)
4. [Pipeline de Preprocesamiento](#4-pipeline-de-preprocesamiento)
5. [Arquitectura del Modelo](#5-arquitectura-del-modelo)
6. [Estrategia de Entrenamiento](#6-estrategia-de-entrenamiento)
7. [Validación Experimental](#7-validación-experimental)
8. [Configuración Centralizada](#8-configuración-centralizada)
9. [Estado Actual y Próximos Pasos](#9-estado-actual-y-próximos-pasos)
10. [Resumen de Trade-offs y Decisiones Críticas](#10-resumen-de-trade-offs-y-decisiones-críticas)

---

## 1. Visión General del Proyecto

### 1.1. Objetivo

El proyecto consiste en desarrollar un pipeline de Deep Learning capaz de clasificar imágenes de resonancia magnética estructural 3D (sMRI) en tres estadios clínicos del deterioro cognitivo:

| Clase | Etiqueta | Significado clínico |
|-------|----------|---------------------|
| 0 | CN (Cognitively Normal) | Sin deterioro cognitivo |
| 1 | MCI (Mild Cognitive Impairment) | Deterioro cognitivo leve |
| 2 | AD (Alzheimer's Disease) | Enfermedad de Alzheimer |

Se trata de un Trabajo de Fin de Grado (TFG), lo que condiciona el enfoque: se prioriza el rigor metodológico y la claridad sobre la precisión estado del arte (SOTA).

### 1.2. Filosofía "Low-Resource & Efficient"

Esta filosofía permea todas las decisiones del proyecto. La restricción fundamental es que el pipeline debe ejecutarse en hardware de consumo: GPUs con 8 GB de VRAM o menos (Google Colab T4, RTX 3060 local). Esto implica:

- Volúmenes de entrada reducidos a 96x96x96 vóxeles (en lugar de 176x208x176 o superiores).
- Batch size conservador de 4.
- Modelo ligero (CNN vanilla) en lugar de arquitecturas pesadas como ViT-3D.
- Sin técnicas costosas como ensembles o test-time augmentation en la fase inicial.

### 1.3. Stack Tecnológico

| Tecnología | Rol | Justificación |
|------------|-----|---------------|
| **Python 3.10+** | Lenguaje base | Estándar en ML/DL, ecosistema maduro |
| **PyTorch** | Framework DL | Preferido en investigación, control granular del training loop, debugging más intuitivo que TensorFlow |
| **MONAI** | Framework médico | Librería especializada en imagen médica sobre PyTorch. Proporciona transforms 3D, loaders para formatos médicos (NIfTI, ANALYZE), datasets con caché, y utilidades de visualización. Evita reimplementar lógica compleja de carga/orientación |
| **NiBabel** | Lectura neuroimagen | Librería estándar para leer/escribir formatos NIfTI y ANALYZE. MONAI la usa internamente |
| **Pandas** | Manipulación tabular | Gestión de CSVs clínicos, splits y metadatos |
| **scikit-learn** | Utilidades ML | `train_test_split` con estratificación, métricas futuras (F1, confusion matrix) |
| **Matplotlib / Seaborn** | Visualización | Gráficas de entrenamiento, visualización de cortes cerebrales |
| **NumPy** | Computación numérica | Operaciones sobre arrays, base de todo el stack |

**Decisión PyTorch vs TensorFlow:** PyTorch ofrece ejecución eager por defecto (más fácil de depurar), es el framework dominante en publicaciones de investigación en neuroimagen, y MONAI está construido exclusivamente sobre él. TensorFlow/Keras habría requerido reimplementar la lógica de carga médica o usar librerías menos maduras.

---

## 2. Datos y Dataset (OASIS-1)

### 2.1. Descripción del Dataset

El proyecto utiliza **OASIS-1** (Open Access Series of Imaging Studies), un dataset público de neuroimagen que contiene:

- **416 sujetos** con MRI cross-sectional (un solo scan por sujeto).
- Rango de edad: 18-96 años.
- Incluye sujetos sanos y con distintos grados de demencia.
- Datos clínicos asociados: edad, sexo, CDR (Clinical Dementia Rating), MMSE, eTIV, nWBV, ASF.

La naturaleza cross-sectional elimina el riesgo de data leakage por sujetos repetidos en distintos splits (a diferencia de OASIS-2/3 que son longitudinales).

### 2.2. Elección de Archivos: `*_masked_gfc`

De todos los archivos disponibles por sujeto, se utilizan exclusivamente los archivos `*_masked_gfc.img/.hdr` ubicados en el directorio `T88_111/`. Estos archivos tienen tres procesados previos aplicados por el propio OASIS:

| Procesado | Significado | Impacto |
|-----------|-------------|---------|
| **masked** | Skull-stripped (cráneo eliminado) | Elimina tejido no cerebral que añadiría ruido al modelo |
| **gfc** | Gain Field Corrected | Corrige inhomogeneidades de intensidad del scanner MRI |
| **T88_111** | Atlas-registered (Talairach 1mm) | Todos los cerebros están en el mismo espacio estándar, facilitando la comparación |

**¿Por qué no usar los archivos raw?** Los archivos crudos requerirían implementar skull-stripping (FreeSurfer/BET), corrección de campo de ganancia (N4ITK), y registro al atlas, lo cual constituye un pipeline de preprocesamiento complejo fuera del alcance de un TFG enfocado en clasificación DL.

**Formato ANALYZE 7.5:** Los archivos vienen en formato ANALYZE (.img para datos de imagen + .hdr para cabecera), que es un formato legacy. MONAI y NiBabel lo leen de forma transparente tratándolo como NIfTI.

### 2.3. Mapeo de Etiquetas: CDR a Clases

El **CDR (Clinical Dementia Rating)** es una escala clínica que evalúa la severidad de la demencia. Los valores posibles son: 0, 0.5, 1, 2, 3.

El mapeo implementado en `_map_cdr_to_label()` de `src/data_prepare.py` es:

```
CDR  0.0  →  Clase 0 (CN)     — Sin demencia
CDR  0.5  →  Clase 1 (MCI)    — Demencia muy leve
CDR  1.0+ →  Clase 2 (AD)     — Demencia leve a severa (agrupa CDR 1, 2 y 3)
```

**Decisiones y justificación:**

- **Agrupar CDR 1, 2 y 3 en una sola clase:** Los sujetos con CDR >= 1 son escasos (< 30 en total). Separarlos crearía clases con muy pocas muestras, haciendo el entrenamiento inviable. Clínicamente, la distinción más relevante es CN vs MCI vs demencia establecida.
- **Descartar sujetos sin CDR:** Los sujetos jóvenes del dataset (< 60 años) no tienen CDR asignado. Se eliminan para evitar etiquetas ruidosas. Esto reduce el dataset de 416 a ~235 sujetos con CDR válido.

### 2.4. Distribución de Clases

La distribución observada tras el split (reportada por el sanity check):

| Clase | Train | Val | Test | Total | Porcentaje |
|-------|-------|-----|------|-------|------------|
| CN (0) | 94 | 20 | 21 | 135 | ~57% |
| MCI (1) | 49 | 10 | 11 | 70 | ~30% |
| AD (2) | 21 | 5 | 4 | 30 | ~13% |
| **Total** | **164** | **35** | **36** | **235** | **100%** |

### 2.5. Análisis Crítico de los Datos

**Desbalanceo de clases:** El ratio CN:MCI:AD es aproximadamente 4.5:2.3:1. La clase AD tiene solo 21 muestras de entrenamiento, lo que genera varios problemas:

- El modelo puede aprender a predecir siempre CN y obtener ~57% de accuracy sin aprender nada útil.
- La clase MCI es inherentemente difícil: es un estado intermedio entre CN y AD que a menudo presenta solapamiento en las características cerebrales.
- 21 muestras de AD en train es insuficiente para generalización robusta.

**Tamaño del dataset:** Con solo 164 muestras de entrenamiento, el riesgo de overfitting es alto. Las redes neuronales profundas típicamente requieren miles de ejemplos por clase. Esto refuerza la necesidad de data augmentation (Sprint 4) y de considerar transfer learning.

**Un solo split fijo:** No se implementa cross-validation. Esto significa que los resultados dependen del split particular generado con seed=42. Un resultado fuerte o débil podría ser artefacto del split. La cross-validation (e.g., 5-fold estratificado) daría estimaciones más robustas pero multiplicaría el coste computacional por 5.

---

## 3. Pipeline ETL

**Archivo:** `src/data_prepare.py`

### 3.1. Arquitectura del Pipeline

El ETL sigue tres pasos secuenciales, cada uno independiente y verificable:

```
Paso 1: extract_and_standardize()
    OASIS-1 (tarballs o carpetas) → data/processed/images/OAS1_XXXX_MR1.{img,hdr}

Paso 2: generate_master_csv()
    Excel clínico + imágenes procesadas → data/processed/dataset_master.csv

Paso 3: stratified_split()
    dataset_master.csv → data/splits/{train,val,test}.csv
```

### 3.2. Paso 1: Extracción y Estandarización

**Lógica de detección de fuentes:** El script auto-detecta si los discos OASIS están como carpetas ya extraídas o como archivos `.tar.gz`. Si ambos existen para un mismo disco, prioriza la carpeta (más rápida, evita descompresión).

**Extracción selectiva:** En el caso de tarballs, no descomprime todo el archivo. Lee los nombres de los miembros del tar, filtra solo los que coinciden con el patrón `*T88_111/*_masked_gfc.(img|hdr)`, y extrae solo esos en memoria. Esto ahorra disco y tiempo.

**Estandarización de nombres:** Los archivos se renombran a formato plano `OAS1_XXXX_MR1.img` eliminando la estructura de directorios anidada original. El `subject_id` se extrae con regex `(OAS1_\d{4}_MR\d)`.

**Idempotencia:** Si el archivo destino ya existe, no se sobrescribe. El script puede ejecutarse múltiples veces sin efectos secundarios.

**Validación:** Cuenta pares .img/.hdr, detecta archivos huérfanos (sin pareja), y verifica que se alcanzan los 416 pares esperados.

### 3.3. Paso 2: CSV Maestro

Lee el Excel clínico de OASIS-1, filtra sujetos sin CDR, aplica el mapeo CDR → label, y cruza con las imágenes disponibles en disco (eliminando filas cuya imagen no existe). Genera columnas: `subject_id`, `image_path`, `label`, `Age`, `CDR`, `M/F`.

### 3.4. Paso 3: Split Estratificado

Implementa el split en dos pasos para obtener ratios 70/15/15:

```
1. train_test_split(df, test_size=0.30, stratify=label)  →  train (70%) + temp (30%)
2. train_test_split(temp, test_size=0.50, stratify=label) →  val (15%) + test (15%)
```

**¿Por qué dos pasos?** `sklearn.train_test_split` solo puede hacer una partición binaria. Para obtener tres conjuntos con ratios exactos, se necesitan dos llamadas. La primera separa el 70% de train, la segunda divide el 30% restante en mitades iguales.

**Verificación anti-leakage:** Tras generar los splits, calcula la intersección de `subject_id` entre todos los pares de conjuntos. Si hay algún sujeto compartido, lanza `RuntimeError`. Esta verificación es redundante dada la mecánica de `train_test_split`, pero constituye una buena práctica defensiva.

### 3.5. Análisis Crítico del ETL

| Aspecto | Evaluación |
|---------|------------|
| Idempotencia | Positivo: permite re-ejecución segura |
| Extracción selectiva | Positivo: eficiente en disco y tiempo |
| Anti-leakage check | Positivo: verificación explícita defensiva |
| Sin cross-validation | Riesgo: resultados dependientes de un solo split |
| `random_state=42` | Reproducible, pero un único seed no valida la robustez de los resultados |
| Columnas adicionales (Age, M/F) | Guardadas pero no usadas actualmente; podrían servir para análisis de sesgo o como features auxiliares |

---

## 4. Pipeline de Preprocesamiento

**Archivo:** `src/dataset.py`

### 4.1. Pipeline de Transforms MONAI

Cada imagen 3D pasa por la siguiente secuencia de transformaciones on-the-fly:

| Paso | Transform | Entrada | Salida | Propósito |
|------|-----------|---------|--------|-----------|
| 1 | `LoadImaged` | Ruta `.img` | Array 3D (D, H, W) | Carga el par .img/.hdr desde disco |
| 2 | `EnsureChannelFirstd` | (D, H, W) | (1, D, H, W) | Añade dimensión de canal para convolucion |
| 3 | `Orientationd("RAS")` | (1, D, H, W) | (1, D, H, W) | Reorienta a Right-Anterior-Superior |
| 4 | `ScaleIntensityRangePercentilesd` | [min, max] | [0, 1] | Normaliza intensidad |
| 5 | `Resized` | (1, D, H, W) | (1, 96, 96, 96) | Redimensiona al tamaño objetivo |

### 4.2. Decisiones de Preprocesamiento en Detalle

#### Orientación RAS

Las MRI pueden venir en distintas orientaciones según el protocolo de adquisición. RAS (Right-Anterior-Superior) es el estándar de neuroimagen que garantiza que la primera dimensión va de izquierda a derecha, la segunda de posterior a anterior, y la tercera de inferior a superior. Sin esta normalización, el mismo cerebro podría aparecer rotado 90° o reflejado según el sujeto, confundiendo al modelo.

En el caso de OASIS-1 con archivos `T88_111` (ya registrados al atlas), la orientación debería ser consistente. No obstante, aplicar `Orientationd` es una medida defensiva de bajo coste.

#### Normalización por Percentiles (1-99)

Se utiliza `ScaleIntensityRangePercentilesd` con percentiles 1 y 99, mapeando ese rango a [0, 1] y recortando (clipping) valores fuera del rango.

**¿Por qué percentiles en lugar de min-max?** Min-max es sensible a vóxeles extremos (artefactos, ruido). Un solo vóxel con intensidad atípica puede comprimir todo el rango útil. Los percentiles 1-99 eliminan el 2% de valores extremos, ofreciendo normalización más robusta.

**¿Por qué no z-score (media=0, std=1)?** La normalización z-score es común en neuroimagen, pero produce valores negativos que requieren cuidado con funciones de activación como ReLU. La normalización a [0, 1] simplifica la interpretación y es compatible con visualización directa.

**Consideración:** Se aplica la normalización por volumen individual, no por dataset. Esto significa que cada cerebro se normaliza independientemente. La alternativa sería calcular percentiles globales sobre todo el dataset de entrenamiento, lo que requeriría un paso offline.

#### Resize a 96x96x96

Las imágenes OASIS-1 en espacio Talairach tienen dimensiones originales de aproximadamente 176x208x176 vóxeles. Redimensionar a 96x96x96 supone una reducción de ~84% en el número de vóxeles (de ~6.4M a ~0.9M), lo cual es necesario para caber en VRAM con batch size 4.

**Trade-off resolución vs. memoria:**

| Tamaño | Vóxeles | VRAM aprox. (batch=4) | Resolución relativa |
|--------|---------|----------------------|---------------------|
| 64x64x64 | 262K | ~1.5 GB | Baja — pierde detalle fino |
| 96x96x96 | 884K | ~3-4 GB | Media — compromiso razonable |
| 128x128x128 | 2.1M | ~6-8 GB | Alta — límite para T4 |
| 176x208x176 | 6.4M | >16 GB | Original — inviable sin GPU profesional |

La elección de 96³ es un punto medio que permite entrenar en GPUs de consumo sin perder las estructuras anatómicas principales (ventrículos, hipocampo, corteza).

**Isotropía del resize:** Se aplica `Resized` con el mismo tamaño en las tres dimensiones. Si las dimensiones originales no son cúbicas (y no lo son: 176x208x176), el resize introduce una distorsión anisotrópica leve. En este caso, dado que los datos ya están en espacio Talairach (isotropizado a 1mm), la distorsión es menor. Aun así, un approach más riguroso sería usar `Spacingd` para re-muestrear primero a un spacing isotrópico y luego recortar (crop) o padear al tamaño deseado.

### 4.3. DataLoader

El DataLoader se configura con:

- `batch_size=4` por defecto (configurable).
- `shuffle=True` solo para train.
- `pin_memory=True` para transferencia CPU→GPU más rápida.
- `CacheDataset` opcional: precarga todos los volúmenes transformados en RAM. Recomendado con >16 GB RAM, pero consume mucha memoria (~235 volúmenes × ~3.5 MB cada uno ≈ 0.8 GB).

### 4.4. Análisis Crítico del Preprocesamiento

| Aspecto | Evaluación | Impacto |
|---------|------------|---------|
| Sin data augmentation | Negativo en esta fase | Alto riesgo de overfitting con 164 muestras train |
| Normalización per-volume | Aceptable | Puede introducir variabilidad inter-sujeto pero es estándar |
| Resize directo vs. spacing + crop | Simplificación aceptable | Distorsión menor dada la pre-registración al atlas |
| Mismas transforms train/val/test | Correcto para deterministic transforms | Augmentation debe aplicarse solo a train |
| `num_workers` en DataLoader | Configurado a 2 pero forzado a 0 en train.py | Bottleneck de I/O potencial en entrenamiento |

---

## 5. Arquitectura del Modelo

**Archivo:** `src/model.py`

### 5.1. Diseño de Simple3DCNN

La arquitectura sigue un diseño VGG-like adaptado a 3D:

```
Entrada: (B, 1, 96, 96, 96)
    │
    ▼
[Bloque 1] Conv3d(1→32, k=3, p=1) → BN3d(32) → ReLU → MaxPool3d(2)
    │       Salida: (B, 32, 48, 48, 48)
    ▼
[Bloque 2] Conv3d(32→64, k=3, p=1) → BN3d(64) → ReLU → MaxPool3d(2)
    │       Salida: (B, 64, 24, 24, 24)
    ▼
[Bloque 3] Conv3d(64→128, k=3, p=1) → BN3d(128) → ReLU → MaxPool3d(2)
    │       Salida: (B, 128, 12, 12, 12)
    ▼
[Bloque 4] Conv3d(128→256, k=3, p=1) → BN3d(256) → ReLU → MaxPool3d(2)
    │       Salida: (B, 256, 6, 6, 6)
    ▼
AdaptiveAvgPool3d(1)  →  (B, 256, 1, 1, 1)
Flatten               →  (B, 256)
Dropout(0.5)          →  (B, 256)
Linear(256, 3)        →  (B, 3)
```

### 5.2. Análisis de Decisiones Arquitectónicas

#### Progresión de canales: 1 → 32 → 64 → 128 → 256

Sigue el patrón clásico de duplicar canales en cada bloque mientras se reduce la resolución espacial. Esto es estándar en arquitecturas CNN desde VGG. La progresión es conservadora comparada con algunas redes 3D médicas que usan 16 → 32 → 64 → 128, lo que resulta en más parámetros pero potencialmente más capacidad.

#### Kernel 3x3x3 con padding=1

Kernels 3x3x3 son el estándar en CNNs 3D. Con padding=1, la resolución espacial se preserva antes del MaxPool, simplificando el cálculo de dimensiones. Kernels más grandes (5x5x5) capturarían más contexto por capa pero aumentarían significativamente los parámetros y el coste computacional.

#### MaxPool3d(2) en cada bloque

Reduce las dimensiones espaciales a la mitad en cada bloque. Después de 4 bloques: 96 → 48 → 24 → 12 → 6. Alternativas como stride=2 en la convolución (strided convolution) son usadas en arquitecturas modernas y permiten al modelo aprender el downsampling, pero MaxPool es más simple y estable.

#### AdaptiveAvgPool3d(1)

Colapsa el feature map de (B, 256, 6, 6, 6) a (B, 256, 1, 1, 1) haciendo un promedio global. Dos ventajas principales:

1. Hace la red agnóstica al tamaño de entrada — funciona con cualquier resolución, no solo 96³.
2. Actúa como regularizador al reducir drásticamente la dimensionalidad antes del clasificador.

La alternativa (Flatten directo) generaría un vector de 256 × 6 × 6 × 6 = 55.296 features, requiriendo una capa FC masiva.

#### Dropout 0.5 (solo en el clasificador)

Se aplica dropout únicamente antes de la capa lineal final. No hay dropout entre las capas convolucionales. La justificación es que Batch Normalization ya proporciona regularización implícita en las capas conv (por el ruido estocástico del batch).

Un dropout de 0.5 es agresivo: descarta la mitad de las activaciones durante entrenamiento. Con un dataset tan pequeño, esto puede ser necesario, pero también podría dificultar el aprendizaje si se combina con otras formas de regularización.

#### Sin conexiones residuales

A diferencia de ResNet, no hay skip connections. Esto simplifica la implementación pero limita la profundidad efectiva que se puede alcanzar. Con solo 4 bloques conv, el vanishing gradient no debería ser un problema grave, así que las residual connections no son estrictamente necesarias.

### 5.3. Conteo de Parámetros

Estimación aproximada por capa:

| Capa | Parámetros |
|------|------------|
| Conv3d(1, 32, 3) + BN | 32 × (1 × 27 + 1) + 64 ≈ 960 |
| Conv3d(32, 64, 3) + BN | 64 × (32 × 27 + 1) + 128 ≈ 55.488 |
| Conv3d(64, 128, 3) + BN | 128 × (64 × 27 + 1) + 256 ≈ 221.568 |
| Conv3d(128, 256, 3) + BN | 256 × (128 × 27 + 1) + 512 ≈ 885.504 |
| Linear(256, 3) | 256 × 3 + 3 = 771 |
| **Total aproximado** | **~1.16M parámetros** |

Con ~1.16M parámetros y ~164 muestras de entrenamiento, el ratio parámetros/muestras es de ~7.000:1, lo cual es extremadamente alto. En condiciones ideales se busca un ratio mucho menor. Esto subraya el riesgo de overfitting y la necesidad de regularización fuerte.

### 5.4. Comparación con Alternativas

| Arquitectura | Parámetros aprox. | Ventajas | Desventajas |
|-------------|-------------------|----------|-------------|
| **Simple3DCNN (actual)** | ~1.16M | Simple, rápida, debuggeable | Capacidad limitada, sin skip connections |
| **DenseNet121-3D** (MONAI) | ~11M | Reutilización de features, eficiente | 10x más parámetros, más VRAM |
| **ResNet-18/50-3D** | ~11-23M | Skip connections, bien estudiada | Más parámetros, requiere batch size menor |
| **EfficientNet-3D** | ~5M | Balance eficiencia-precisión | Más compleja de implementar |

El roadmap del proyecto contempla migrar a DenseNet121-3D si el baseline no supera el 60-70% de accuracy.

---

## 6. Estrategia de Entrenamiento

**Archivo:** `src/train.py`

### 6.1. Función de Pérdida: CrossEntropyLoss con Pesos de Clase

La pérdida estándar CrossEntropy está ponderada por pesos de clase calculados como:

```
weight_i = N_total / (N_classes × N_i)
```

Con la distribución observada (train: 94 CN, 49 MCI, 21 AD):

| Clase | N_i | weight_i | Interpretación |
|-------|-----|----------|----------------|
| CN (0) | 94 | 164 / (3 × 94) ≈ 0.58 | Penaliza menos los errores en CN |
| MCI (1) | 49 | 164 / (3 × 49) ≈ 1.12 | Peso cercano a 1 (neutral) |
| AD (2) | 21 | 164 / (3 × 21) ≈ 2.60 | Penaliza ~4.5x más los errores en AD vs CN |

Este esquema de pesos compensa parcialmente el desbalanceo, pero es una corrección lineal a un problema potencialmente no lineal. Alternativas más agresivas incluyen:

- Focal Loss: reduce la pérdida para ejemplos bien clasificados, enfocando el entrenamiento en los difíciles.
- Oversampling/undersampling: replicar muestras de AD o reducir muestras de CN.
- SMOTE adaptado a imágenes: generación sintética de ejemplos de la clase minoritaria (descartado por la regla "No Synthetic Data").

### 6.2. Optimizador: Adam

Se usa Adam con learning rate fijo de 1e-4. No hay learning rate scheduler.

**Justificación de Adam:** Es el optimizador por defecto en muchos proyectos de DL. Combina momentum y tasas de aprendizaje adaptativas por parámetro, funcionando bien sin mucho tuning. La alternativa SGD con momentum requiere más ajuste pero puede generalizar mejor según la literatura.

**Learning rate 1e-4:** Es un valor conservador pero estándar para Adam. Un LR demasiado alto causaría inestabilidad; demasiado bajo, convergencia lenta.

**Sin scheduler:** No se implementa reducción del learning rate durante el entrenamiento (e.g., ReduceLROnPlateau, CosineAnnealingLR). Esto significa que si el modelo se estanca en una meseta, no hay mecanismo para salir de ella reduciendo el LR. Añadir un scheduler de tipo `ReduceLROnPlateau(patience=5, factor=0.5)` sería una mejora directa.

### 6.3. Configuración de Entrenamiento

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| Batch size | 4 | Conservador para VRAM ≤8 GB |
| Epochs | 50 (por defecto) | Configurable vía CLI |
| Seed | 42 | `torch.manual_seed` + `cuda.manual_seed_all` |
| Device | CUDA si disponible, else CPU | Detección automática |
| num_workers | 0 (forzado en train.py) | Evita problemas de multiprocessing en Windows |
| Selección de mejor modelo | Menor val_loss | Guarda `best_model.pth` con estado completo |
| Early stopping | No implementado | Riesgo de overfitting |

### 6.4. Batch Size 4: Implicaciones

Con batch size 4, cada paso de optimización ve solo 4 imágenes. Esto tiene implicaciones estadísticas:

- **Alta varianza en los gradientes:** La estimación del gradiente basada en 4 muestras es ruidosa. Esto puede actuar como regularización implícita (ruido similar al de dropout), pero también puede causar oscilaciones en el entrenamiento.
- **Batch Normalization con batch=4:** BN calcula estadísticas (media, varianza) sobre el batch. Con solo 4 muestras, estas estadísticas son inestables. Alternativas como Group Normalization o Instance Normalization serían más robustas con batches tan pequeños, pero BN sigue siendo funcional.
- **Gradient Accumulation:** No implementado. Si se quisiera simular un batch size mayor (e.g., 16) sin aumentar VRAM, se podrían acumular gradientes de 4 pasos antes de hacer `optimizer.step()`.

### 6.5. Checkpointing y Logging

Cada ejecución genera un directorio `outputs/<run_name>/` con:

| Archivo | Contenido |
|---------|-----------|
| `training_log.csv` | Métricas por epoch: train/val loss, train/val accuracy, tiempo, marca `is_best` |
| `curves_loss.png` | Gráfica train loss vs val loss con línea vertical en mejor epoch |
| `curves_accuracy.png` | Gráfica train accuracy vs val accuracy |
| `best_model.pth` | Diccionario con epoch, model_state_dict, optimizer_state_dict, val_loss, val_accuracy |
| `training_summary.txt` | Resumen legible con configuración, mejor epoch y último epoch |

El checkpoint del mejor modelo incluye el estado del optimizador, permitiendo reanudar el entrenamiento si fuera necesario.

### 6.6. Análisis Crítico del Entrenamiento

| Aspecto | Estado | Impacto | Mejora sugerida |
|---------|--------|---------|-----------------|
| Sin early stopping | Ausente | Puede entrenar de más y sobreajustar | `EarlyStopping(patience=10)` |
| Sin LR scheduler | Ausente | Puede estancarse en mesetas | `ReduceLROnPlateau` o `CosineAnnealing` |
| Sin gradient clipping | Ausente | Posibles explosiones de gradiente | `torch.nn.utils.clip_grad_norm_` |
| Sin weight decay | Ausente | Sin regularización L2 | `Adam(weight_decay=1e-5)` |
| num_workers=0 | Forzado | Bottleneck de I/O, CPU no precarga datos | Investigar si num_workers > 0 funciona en el entorno |
| Sin mixed precision (AMP) | Ausente | Usa FP32 completo, más VRAM y más lento | `torch.cuda.amp.autocast` para FP16 |

---

## 7. Validación Experimental

### 7.1. Metodología por Sprints

El proyecto sigue una metodología de verificación incremental con checkpoints explícitos. Cada sprint tiene criterios de éxito binarios (pasa/no pasa):

```
Sprint 1 (Data)     →  416 pares .img/.hdr, CSV con 3 clases, splits sin leakage
Sprint 2 (Pipeline) →  Batch shape (4,1,96,96,96), intensidades en [0,1], cerebro visible
Sprint 3 (Modelo)   →  Forward pass sin error, overfit-one-batch (loss~0, acc=100%)
Sprint 4 (Eval)     →  Confusion matrix, F1, augmentation     [PENDIENTE]
Sprint 5 (XAI)      →  Grad-CAM sobre última conv              [PENDIENTE]
```

Esta metodología es sólida y recomendable. Cada sprint valida una capa del sistema antes de construir la siguiente, evitando errores que se propaguen y sean difíciles de diagnosticar.

### 7.2. Sanity Check del Pipeline (Sprint 2)

Verificaciones realizadas en `notebooks/01_sanity_check.ipynb`:

- **Shapes:** `batch['image'].shape == (4, 1, 96, 96, 96)` y `batch['label'].shape == (4,)` — PASADO.
- **Intensidades:** min=0, max=1, mean≈0.16, std≈0.30 — valores razonables para cerebros skull-stripped donde el fondo es 0.
- **Visualización:** Cortes centrales (axial, coronal, sagittal) muestran cerebro en escala de grises, centrado, sin cráneo.
- **Tamaños de splits:** train=164, val=35, test=36.

La media de 0.16 indica que el fondo (valor 0) domina el volumen, lo cual es esperable en una imagen skull-stripped: gran parte del volumen 96³ es espacio vacío alrededor del cerebro.

### 7.3. Overfit-One-Batch (Sprint 3)

El test de overfit-one-batch es una práctica recomendada por Andrej Karpathy y otros referentes de DL. Consiste en:

1. Tomar un mini-batch fijo de 4 imágenes.
2. Entrenar durante 100 epochs sobre ese mismo batch.
3. Verificar que loss → 0 y accuracy → 100%.

**Criterio de éxito:** `loss < 0.05` y `accuracy == 1.0`.

**¿Qué valida?** Si el modelo no puede memorizar 4 imágenes, hay un bug fundamental en:
- La arquitectura del modelo (e.g., dimensiones incompatibles).
- El training loop (e.g., gradientes que no fluyen, loss mal calculada).
- El pipeline de datos (e.g., labels incorrectos).

**¿Qué no valida?** No dice nada sobre la capacidad de generalización, calidad de las features, ni rendimiento esperado en validación/test.

### 7.4. Análisis Crítico de la Validación

**Fortalezas:**
- Verificación incremental por capas.
- Checkpoints binarios claros.
- Separación entre verificación de código (overfit) y rendimiento (entrenamiento completo).

**Puntos pendientes (Sprints 4-5):**
- No hay evaluación en el conjunto de test todavía.
- No se han calculado métricas por clase (precision, recall, F1 macro/weighted).
- No hay confusion matrix para analizar patrones de error.
- No se ha realizado análisis de las predicciones incorrectas.
- No hay Grad-CAM para verificar que el modelo mira regiones anatómicamente relevantes.

---

## 8. Configuración Centralizada

**Archivo:** `src/config.py`

### 8.1. Diseño

Todos los hiperparámetros, rutas y constantes están centralizados en una clase `ProjectConfig`. Se expone una instancia global `cfg` importable desde cualquier módulo.

### 8.2. Tabla Completa de Parámetros

| Categoría | Parámetro | Valor | Justificación |
|-----------|-----------|-------|---------------|
| **Reproducibilidad** | `RANDOM_SEED` | 42 | Convención estándar, fija splits y pesos iniciales |
| **Rutas** | `PROJECT_ROOT` | Raíz del repo | Derivada de `__file__`, portable |
| | `DATA_DIR` | `data/` | Todos los datos bajo un directorio |
| | `OUTPUTS_DIR` | `outputs/` | Artefactos de entrenamiento |
| **Imagen** | `IMAGE_SIZE` | (96, 96, 96) | Compromiso VRAM/resolución |
| **Entrenamiento** | `BATCH_SIZE` | 4 | Limitado por VRAM ≤8 GB |
| | `NUM_WORKERS` | 2 | Balance CPU/overhead |
| | `LEARNING_RATE` | 1e-4 | Estándar para Adam |
| | `NUM_EPOCHS` | 50 | Suficiente para convergencia inicial |
| **Datos** | `TRAIN_RATIO` | 0.70 | 70% para entrenamiento |
| | `VAL_RATIO` | 0.15 | 15% para validación |
| | `TEST_RATIO` | 0.15 | 15% para test final |
| | `NUM_CLASSES` | 3 | CN / MCI / AD |

### 8.3. Análisis del Diseño

**Positivo:** Centralizar la configuración en un solo lugar evita "magic numbers" dispersos por el código. Facilita experimentar con distintos hiperparámetros cambiando un solo archivo.

**Limitación:** La clase `ProjectConfig` usa atributos de clase (no de instancia), lo que funciona pero no es un patrón singleton robusto. Si alguien modificara `cfg.BATCH_SIZE = 8` en un punto del código, afectaría a todos los módulos que lo importen (mutable global state). Una alternativa más robusta sería usar `dataclasses.dataclass(frozen=True)` o un archivo YAML con Hydra/OmegaConf.

**Ausencia de CLI/config file:** Los hiperparámetros se fijan en código. Para experimentación sistemática, sería útil poder sobreescribirlos desde línea de comandos o desde un archivo YAML por experimento (e.g., con `argparse` extendido o Hydra).

---

## 9. Estado Actual y Próximos Pasos

### 9.1. Progreso por Sprint

| Sprint | Tema | Estado | Archivos |
|--------|------|--------|----------|
| 1 | Data Engineering | Completado | `src/data_prepare.py`, `src/data_utils.py`, `src/config.py` |
| 2 | MONAI Pipeline | Completado | `src/dataset.py`, `notebooks/01_sanity_check.ipynb` |
| 3 | Modelado y Training | Completado | `src/model.py`, `src/train.py`, `notebooks/02_overfit_one_batch.ipynb` |
| 4 | Evaluación y Refinamiento | Pendiente | — |
| 5 | Explainability (XAI) | Pendiente | — |

### 9.2. Sprint 4: Evaluación y Refinamiento (Próximo)

Tareas planificadas:

1. **Métricas detalladas:** Accuracy, Precision, Recall y F1-Score (macro average) sobre el test set.
2. **Confusion Matrix:** Visualizar qué clases se confunden. La expectativa es que CN y AD se separen razonablemente, pero MCI se confunda con ambas.
3. **Data Augmentation:** Añadir transforms aleatorias solo al pipeline de train:
   - `RandRotate90d` — rotaciones de 90° aleatorias.
   - `RandFlipd(prob=0.1)` — flip horizontal aleatorio.
   - `RandGaussianNoised` — ruido gaussiano para robustez.
4. **Opcional: DenseNet121-3D** si el baseline no supera 60-70%.

### 9.3. Sprint 5: Explainability (XAI)

- Implementar **Grad-CAM** usando `monai.visualize.GradCAM`.
- Enganchar a la última capa convolucional (`features[-1]`, el cuarto bloque conv).
- Generar mapas de calor superpuestos a cortes cerebrales.
- Verificar que las zonas de activación coinciden con regiones anatómicamente relevantes para Alzheimer: hipocampo, ventrículos laterales, corteza temporal.

### 9.4. Mejoras Potenciales No Contempladas en el Roadmap

| Mejora | Beneficio esperado | Coste |
|--------|-------------------|-------|
| Learning rate scheduler | Mejor convergencia | Bajo |
| Early stopping | Evitar overfitting | Bajo |
| Mixed precision (AMP) | 2x velocidad, -50% VRAM | Bajo |
| Gradient accumulation | Simular batch size mayor | Bajo |
| Cross-validation (5-fold) | Estimación más robusta | 5x tiempo de cómputo |
| Transfer learning (pretrained weights) | Mejor con pocos datos | Medio |
| Weight decay en Adam | Regularización L2 | Bajo |
| Group Normalization | Estabilidad con batch=4 | Bajo-Medio |

---

## 10. Resumen de Trade-offs y Decisiones Críticas

### 10.1. Tabla de Decisiones

| Decisión | Alternativa(s) | Justificación | Riesgo/Limitación |
|----------|---------------|---------------|-------------------|
| OASIS-1 cross-sectional | OASIS-2/3 (longitudinal), ADNI | Público, accesible, sin registro complejo | Dataset pequeño (~235 sujetos con CDR) |
| Archivos `masked_gfc` | Raw + pipeline FreeSurfer | Ya preprocesados (skull-strip, GFC, atlas) | Dependencia del preprocesado de OASIS |
| 3 clases (CN/MCI/AD) | Binario (CN vs demencia), 5 clases (por CDR) | Balance clínico: MCI es la clase más relevante clínicamente | MCI es inherentemente difícil de separar |
| CDR >= 1 agrupado | Separar CDR 1, 2, 3 | Muy pocas muestras por nivel | Pierde granularidad en severidad |
| Split fijo 70/15/15 | Cross-validation 5-fold | Simplicidad, reproducibilidad | Resultados dependientes de un split |
| Resize a 96³ | 64³, 128³, spacing isotrópico | Compromiso VRAM/resolución para T4/RTX 3060 | Pérdida de detalle anatómico fino |
| Normalización percentil 1-99 | Z-score, min-max global | Robustez a outliers por volumen | No normaliza globalmente entre sujetos |
| Simple3DCNN (4 bloques) | DenseNet121-3D, ResNet-3D | Debugging fácil, baseline transparente | Capacidad limitada (~1.16M params) |
| AdaptiveAvgPool | Flatten directo | Agnóstico al tamaño, regularización | Pierde información espacial |
| Dropout 0.5 (solo clasificador) | Dropout en todas las capas, DropBlock3D | Simplicidad, BN ya regulariza | Puede ser insuficiente/excesivo |
| CrossEntropyLoss + class weights | Focal Loss, oversampling | Estándar, fácil de implementar | Corrección lineal a problema no lineal |
| Adam lr=1e-4 sin scheduler | SGD+momentum, AdamW, cosine annealing | Robusto con poco tuning | Puede estancarse sin LR decay |
| Batch size 4 | 2, 8, gradient accumulation | Máximo para VRAM ≤8 GB | BN inestable, gradientes ruidosos |
| Sin early stopping | Patience-based early stopping | Simplicidad | Riesgo de overfitting |
| Sin data augmentation (por ahora) | RandRotate, RandFlip, RandNoise | Planeado para Sprint 4 | Crítico con solo 164 train samples |
| MONAI (transforms + dataset) | torchio, custom con nibabel+torch | Ecosistema integrado, mantenido por NVIDIA | Dependencia pesada, API cambiante |
| PyTorch (framework DL) | TensorFlow/Keras, JAX | MONAI lo requiere, dominante en investigación | Más verbose que Keras |

### 10.2. Riesgos Principales del Proyecto

1. **Dataset demasiado pequeño:** 164 muestras de train (21 AD) es un desafío serio para cualquier red neuronal. La mitigación principal es data augmentation + modelo ligero + regularización.

2. **MCI como clase intermedia:** Es probable que MCI se confunda con CN y AD. Esto no es un fallo del modelo sino una limitación intrínseca: MCI es un continuo entre normalidad y demencia, y las diferencias estructurales son sutiles.

3. **Dependencia del split:** Sin cross-validation, un resultado fuerte podría no reproducirse con otro seed. Los 30 sujetos AD se reparten en 21/5/4 entre splits; la distribución de esas pocas muestras puede sesgar los resultados significativamente.

4. **Overfitting latente:** Sin augmentation, sin early stopping, sin weight decay, y con ratio parámetros/muestras de ~7000:1, el modelo probablemente memorizará el train set. Las curvas de train vs. val loss/accuracy serán el diagnóstico clave.

5. **Validación clínica:** Los mapas Grad-CAM (Sprint 5) serán críticos para validar que el modelo aprende features anatómicamente relevantes y no artefactos del preprocesamiento o del dataset.
