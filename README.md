# Detección Temprana de Alzheimer mediante Redes Neuronales Convolucionales 3D

**Trabajo de Fin de Grado**

## Descripción

Sistema de **detección temprana de la enfermedad de Alzheimer** a partir de imágenes de Resonancia Magnética (MRI) 3D, utilizando técnicas de Deep Learning.

Se emplea una **red neuronal convolucional 3D** (Simple3DCNN) entrenada sobre el dataset [OASIS-1](https://www.oasis-brains.org/) para clasificar sujetos en tres categorías:

| Clase | CDR | Descripción |
|-------|-----|-------------|
| CN | 0 | Cognitivamente Normal |
| MCI | 0.5 | Deterioro Cognitivo Leve |
| AD | >= 1 | Enfermedad de Alzheimer |

### Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Framework DL | PyTorch |
| Imagen Médica | MONAI, NiBabel |
| Data Science | Pandas, NumPy, Scikit-learn |
| Visualización | Matplotlib, Seaborn |

## Estructura del Proyecto

```
tfg/
├── src/
│   ├── config.py         # Configuración centralizada
│   ├── data_prepare.py   # ETL: extracción, CSV maestro, splits
│   ├── data_utils.py     # Utilidades de carga
│   ├── dataset.py        # MONAI Dataset y transforms
│   ├── model.py          # Simple3DCNN
│   ├── train.py          # Training loop
│   └── evaluate.py       # Evaluación y métricas
├── notebooks/            # Jupyter notebooks de exploración
├── Docs/                 # Documentación interna
│   ├── Context.md        # Fuente de verdad del proyecto
│   ├── Analisis_Proyecto.md
│   └── Roadmap.md
├── outputs/              # Logs, pesos (.pth), gráficas por experimento
├── data/                 # Datos (no versionados)
├── run_pipeline.py       # Pipeline CLI: train -> evaluate
├── requirements.txt
└── .cursor/rules/        # Reglas para agentes AI
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

### 3. Descargar y colocar los datos de OASIS-1

> **IMPORTANTE**: Los datos médicos NO se incluyen en el repositorio por razones de privacidad y tamaño.

1. Solicita acceso al dataset en [OASIS Brains](https://www.oasis-brains.org/).
2. Descarga los archivos del dataset **OASIS-1**.
3. Coloca los `.tar.gz` (o carpetas extraídas) en `data/OASIS-1/raw/`.

### 4. Ejecutar el pipeline ETL

```bash
python -m src.data_prepare
```

Esto extrae las imágenes `masked_gfc`, genera el CSV maestro y crea los splits train/val/test.

### 5. Verificar el entorno

```bash
python -m src.model     # Verifica que el modelo compila
```

## Uso

### Entrenamiento

```bash
python -m src.train --epochs 50 --run mi_experimento --patience 25
```

### Evaluación

```bash
python -m src.evaluate --run mi_experimento --split test
```

### Pipeline completo

```bash
python run_pipeline.py mi_experimento
```

### Sanity check (overfit one batch)

```bash
python -m src.train --overfit
```

## Estado actual

- Sprints 1-3 completados (datos, pipeline, modelo baseline)
- Sprint 4 en progreso (augmentation, métricas, experimentos de regularización)
- Sprint 5 pendiente (Grad-CAM / Explainability)
- Mejor test accuracy: ~56%, macro F1: ~0.43

Ver `Docs/Context.md` para el estado detallado y `DIARIO.md` para el historial de experimentos.

## Licencia

Proyecto académico (TFG). Uso con fines educativos y de investigación.
