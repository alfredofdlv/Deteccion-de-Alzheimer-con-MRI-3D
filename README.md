# Detección Temprana de Alzheimer mediante Redes Neuronales Convolucionales 3D

**Trabajo de Fin de Grado**

## Descripción

Este proyecto implementa un sistema de **detección temprana de la enfermedad de Alzheimer** a partir de imágenes de Resonancia Magnética (MRI) 3D, utilizando técnicas de Deep Learning.

Se emplean **redes neuronales convolucionales 3D** (3D CNNs) entrenadas sobre el dataset [OASIS-1](https://www.oasis-brains.org/) para clasificar sujetos como cognitivamente normales (CN) o con enfermedad de Alzheimer (AD).

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
├── data/
│   ├── raw/            # Archivos .nii.gz originales (OASIS-1)
│   ├── processed/      # Datos preprocesados / tensores
│   └── splits/         # CSVs de particiones train/val/test
├── notebooks/          # Jupyter notebooks de exploración
├── src/
│   ├── __init__.py
│   ├── config.py       # Configuración centralizada
│   └── data_utils.py   # Utilidades de carga de datos
├── outputs/            # Logs, pesos (.pth), métricas
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd tfg
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Descargar y colocar los datos de OASIS-1

> ⚠️ **IMPORTANTE**: Los datos médicos NO se incluyen en el repositorio por razones de privacidad y tamaño.

1. Solicita acceso al dataset en [OASIS Brains](https://www.oasis-brains.org/).
2. Descarga los archivos MRI del dataset **OASIS-1**.
3. Coloca los archivos `.nii.gz` en la carpeta `data/raw/`.

```
data/
└── raw/
    ├── OAS1_0001_MR1/
    │   └── OAS1_0001_MR1_mpr-1_anon.nii.gz
    ├── OAS1_0002_MR1/
    │   └── ...
    └── ...
```

### 4. Verificar la instalación

Abre el notebook `notebooks/00_environment_setup.ipynb` y ejecuta todas las celdas para comprobar que el entorno está correctamente configurado.

## Uso en Google Colab

Para ejecutar en Colab, monta tu Google Drive y ajusta las rutas en `src/config.py`:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Licencia

Este proyecto es parte de un Trabajo de Fin de Grado con fines académicos.

