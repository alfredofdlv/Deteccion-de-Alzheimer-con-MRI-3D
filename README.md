# Deteccion Temprana de Alzheimer mediante Redes Neuronales Convolucionales 3D

**Trabajo de Fin de Grado**

## Descripcion

Sistema de **deteccion temprana de la enfermedad de Alzheimer** a partir de imagenes de Resonancia Magnetica (MRI) 3D, utilizando tecnicas de Deep Learning.

Se emplea un **ResNet-10 3D** (de MONAI) entrenado sobre los datasets [OASIS-1](https://www.oasis-brains.org/) y [OASIS-3](https://www.oasis-brains.org/) para clasificar sujetos en tres categorias:

| Clase | CDR  | Descripcion              |
| ----- | ---- | ------------------------ |
| CN    | 0    | Cognitivamente Normal    |
| MCI   | 0.5  | Deterioro Cognitivo Leve |
| AD    | >= 1 | Enfermedad de Alzheimer  |

La seleccion del mejor modelo en entrenamiento usa **clinical F2** (ponderado por clase; prioriza AD/MCI); tambien se registran **macro F2** y otras metricas. Ver `src/train.py` y `Docs/Context.md`.

### Stack Tecnologico

| Componente    | Tecnologia                  |
| ------------- | --------------------------- |
| Lenguaje      | Python 3.10+                |
| Framework DL  | PyTorch                     |
| Imagen Medica | MONAI, NiBabel              |
| Data Science  | Pandas, NumPy, Scikit-learn |
| Visualizacion | Matplotlib, Seaborn         |

## Estructura del Proyecto

```
tfg/
├── src/
│   ├── config.py           # Configuracion centralizada
│   ├── data_prepare.py     # ETL OASIS-1: extraccion, CSV maestro, splits
│   ├── data_utils.py       # Utilidades de carga de splits
│   ├── dataset.py          # MONAI Dataset, transforms, DataLoaders
│   ├── model.py            # AlzheimerResNet (ResNet-10 3D)
│   ├── train.py            # Training loop (macro F1, early stopping)
│   └── evaluate.py         # Evaluacion y metricas
├── scripts/                # Scripts auxiliares (descarga, exploracion)
├── notebooks/              # Jupyter notebooks de exploración
│   └── pipeline_oasis3.ipynb  # Pipeline completo OASIS-3
├── Docs/                   # Documentacion interna
│   ├── Context.md          # Fuente de verdad del proyecto
│   └── benchmark.md        # Tiempos de ejecucion medidos
├── outputs/                # Logs, pesos (.pth), graficas por experimento
├── data/                   # Datos (no versionados)
├── run_pipeline.py         # Pipeline CLI: train -> evaluate -> export
├── benchmark.py            # Medicion de tiempos del pipeline
├── prepare_oasis3_nifti_splits.py  # Splits NIfTI Linux desde data/raw/OASIS-3/
├── prepare_oasis3_splits.py        # Regenerar *_pt.csv desde .pt + clinica
├── preprocess_to_pt.py             # Preprocesamiento offline a tensores .pt
├── export_context.py       # Exportar codigo a Markdown
├── requirements.txt
├── DIARIO.md               # Historial de experimentos
```

## Setup

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd tfg
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Descargar y colocar los datos

> **IMPORTANTE**: Los datos medicos NO se incluyen en el repositorio por razones de privacidad y tamaño.

**OASIS-1:**

1. Solicita acceso al dataset en [OASIS Brains](https://www.oasis-brains.org/).
2. Descarga los archivos del dataset OASIS-1.
3. Coloca los `.tar.gz` (o carpetas extraidas) en `data/OASIS-1/raw/`.

**OASIS-3:**

1. Descarga T1w con `oasis-scripts` (p. ej. `download_oasis_scans.sh`) hacia `data/raw/OASIS-3/` (ver `Docs/DOWNLOAD_OASIS3_T1w.md`).
2. Genera splits NIfTI con rutas Linux: `python prepare_oasis3_nifti_splits.py`

### 4. Ejecutar el pipeline ETL

```bash
# OASIS-1: extrae imagenes, genera CSV maestro, crea splits
python -m src.data_prepare

# OASIS-3 (Linux): splits NIfTI desde data/raw/OASIS-3/ (opcional: notebook notebooks/pipeline_oasis3.ipynb para exploracion)
python prepare_oasis3_nifti_splits.py
```

### 5. Preprocesamiento offline (recomendado)

Convierte las imagenes NIfTI a tensores `.pt` para acelerar el entrenamiento:

```bash
python preprocess_to_pt.py --dataset oasis3
```

### 6. Verificar el entorno

```bash
python -m src.model     # Verifica que el modelo compila
```

## Uso

### Entrenamiento

```bash
python -m src.train --epochs 50 --run mi_experimento --patience 25 --dataset oasis3
```

### Evaluacion

```bash
python -m src.evaluate --run mi_experimento --split test --dataset oasis3
```

### Pipeline completo

```bash
python run_pipeline.py mi_experimento --dataset oasis3
```

### Sanity check (overfit one batch)

```bash
python -m src.train --overfit --dataset oasis3
```

### Benchmark de tiempos

```bash
python benchmark.py --dataset oasis3 --batches 15 --label "Mi config"
```

## Estado actual

- Datasets: OASIS-1 (235 sujetos) y OASIS-3 (2450 sesiones MRI)
- Modelo: AlzheimerResNet (ResNet-10 3D de MONAI, ~14.3M parametros)
- Metrica de seleccion (checkpoint): val clinical F2 (ver `src/train.py`)
- Datos preprocesados offline a tensores .pt
- Sprints 1-4 completados, Sprint 5 (Explainability) pendiente

Ver `Docs/Context.md` para el estado detallado y `DIARIO.md` para el historial de experimentos.

## Licencia

Proyecto academico (TFG). Uso con fines educativos y de investigacion.
