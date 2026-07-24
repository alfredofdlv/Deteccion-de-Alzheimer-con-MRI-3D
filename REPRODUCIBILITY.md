# Guia de reproducibilidad

Esta guia describe como instalar el entorno, obtener datos, preprocesar imagenes y
reproducir los experimentos principales del TFG. El codigo asume **Linux** con GPU NVIDIA
(recomendado >=8 GB VRAM para DenseNet121 a 96³ con batch size 4).

Referencia detallada de flags y comandos: [`CLI.md`](CLI.md).

## 1. Requisitos

### Hardware

| Componente | Minimo | Recomendado (memoria TFG) |
|------------|--------|---------------------------|
| GPU | 8 GB VRAM | 12+ GB (TITAN Xp / RTX 3060+) |
| RAM | 16 GB | 32 GB (preprocesado MNI/N4) |
| Disco | 50 GB libres | 200+ GB (OASIS-3 + ADNI + `.pt`) |

### Software

- Python 3.10+
- CUDA compatible con PyTorch (opcional pero recomendado)
- `dcm2niix` (solo si conviertes DICOM ADNI a NIfTI)
- `oasis-scripts` (descarga OASIS-3 desde NITRC IR)

## 2. Instalacion

```bash
git clone https://github.com/alfredofdlv/Deteccion-de-Alzheimer-con-MRI-3D.git
cd Deteccion-de-Alzheimer-con-MRI-3D

python -m venv .venv
source .venv/bin/activate

# Instalar PyTorch segun tu CUDA: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

# Solo si vas a usar preprocesado MNI / MNI+N4 / MNI 128³
pip install -r requirements-optional.txt
```

### Verificacion sin datos

```bash
python -c "import torch, monai; from src.model import get_model; print('OK', torch.__version__)"
python -m src.model
python run_pipeline.py --help
python scripts/prepare_oasis3_nifti_splits.py --help
```

## 3. Datos (no incluidos en el repositorio)

### OASIS-3

1. Solicita acceso en [OASIS Brains / NITRC IR](https://www.oasis-brains.org/).
2. Descarga T1w con `oasis-scripts` hacia `data/raw/OASIS-3/`.
3. Coloca el CSV clinico en `data/oasis3_master_clinical.csv` (proporcionado por OASIS).
4. Genera splits NIfTI:

```bash
python scripts/prepare_oasis3_nifti_splits.py
```

Esto crea `data/splits/oasis3_{train,val,test}.csv` con rutas absolutas a los NIfTI.

**Layout esperado:** carpetas de sesion `OAS3XXXX_MR_dYYYY/` bajo `data/raw/OASIS-3/`,
con ficheros `*T1w*.nii.gz`. El script elige un unico T1w por sesion (prefiere `run-01`).

### ADNI (opcional, experimentos cross-dataset)

1. Solicita acceso en [ADNI](https://adni.loni.usc.edu/) y acepta el DUA.
2. Descarga DICOM segun manifest (generar con `scripts/build_adni_manifest.py`).
3. Convierte a NIfTI: `python scripts/convert_adni_dicom_to_nifti.py`
4. Prepara splits: `python scripts/prepare_adni_splits.py`

Los CSV clinicos ADNI **no** se distribuyen en este repositorio. Debes obtenerlos desde
el portal ADNI y colocarlos localmente segun las rutas en `src/config.py`.

### Union OASIS-3 + ADNI

Para el experimento principal de la memoria (cohorte unificada MNI+N4):

```bash
python scripts/build_oasis3_adni_union_global_splits.py --variant mni_n4
python scripts/build_oasis3_adni_mni_splits.py
```

Revisa los CSV resultantes en `data/splits/` antes de preprocesar.

## 4. Preprocesado offline

Convierte NIfTI a tensores `.pt` para acelerar el entrenamiento (~40x menos I/O).

| Variante | Descripcion | Dependencias extra |
|----------|-------------|-------------------|
| `cropped` | CropForeground + resize 96³ | ninguna |
| `mni` | Skull-strip + registro afín MNI152 + resize 96³ | `requirements-optional.txt` |
| `mni_n4` | `mni` + correccion N4 | `requirements-optional.txt` |
| `mni_128` | Como `mni` pero resize 128³ | `requirements-optional.txt` |

```bash
# Baseline OASIS-3
python scripts/preprocess_to_pt.py --dataset oasis3 --variant cropped

# MNI+N4 (experimento estrella de la memoria)
python scripts/preprocess_to_pt.py --dataset oasis3_adni --variant mni_n4 --workers 4

# QC visual antes de procesar todo el corpus MNI
python scripts/qc_mni_registration.py --n 10
```

Salida: `data/preprocessed/<dataset>/` y CSV `data/splits/<dataset>_*_pt.csv`.

## 5. Entrenamiento y evaluacion

### Pipeline completa

```bash
python run_pipeline.py <run-name> \
  --dataset oasis3_adni \
  --model densenet121 \
  --preprocess-variant mni_n4 \
  --no-export
```

Pasos ejecutados: `src.train` -> `src.evaluate` -> (opcional) export Markdown.

### Entrenamiento manual

```bash
python -m src.train \
  --run densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni \
  --model densenet121 \
  --preprocess-variant mni_n4 \
  --epochs 100 \
  --patience 15

python -m src.evaluate \
  --run densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni \
  --model densenet121 \
  --preprocess-variant mni_n4
```

### Smoke test (1 epoch, subset pequeno)

```bash
python -m src.train --overfit --dataset oasis3 --subset 32
python -m src.train --run smoke-test --dataset oasis3 --model densenet121 --epochs 1 --subset 64
```

## 6. Experimentos clave de la memoria

Comandos equivalentes a los runs documentados en el TFG. Los resultados exactos dependen
del hardware y del estado de los datos; las metricas de referencia provienen de
`outputs/<run>/training_summary.txt` en el entorno del autor.

### A. DenseNet121 MNI+N4, union OASIS-3 + ADNI

Modelo principal con preprocesado espacial MNI152 + N4 bias correction.

```bash
# Preprocesado (si no existe)
python scripts/preprocess_to_pt.py --dataset oasis3_adni --variant mni_n4 --workers 4

# Entrenamiento + evaluacion
python run_pipeline.py densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni \
  --model densenet121 \
  --preprocess-variant mni_n4 \
  --epochs 100 \
  --patience 15 \
  --no-export
```

Referencia (autor, 2026-07-02): val clinical F2 = 0,6081 en epoch 82 (best checkpoint).

### B. Ablacion preprocesado OASIS-3 (cropped vs MNI vs MNI+N4)

```bash
for variant in cropped mni mni_n4; do
  python scripts/preprocess_to_pt.py --dataset oasis3 --variant "$variant"
  python run_pipeline.py "densenet-oasis3-${variant}" \
    --dataset oasis3 --model densenet121 --preprocess-variant "$variant" --no-export
done
```

### C. Late fusion (imagen + covariables clinicas)

Requiere un run CNN previo como extractor de features:

```bash
python run_pipeline.py densenet-late-fusion \
  --dataset oasis3_adni \
  --model densenet121 \
  --preprocess-variant mni_n4 \
  --late-fusion \
  --run-source densenet-oasis3-adni-mni-n4 \
  --no-export
```

## 7. Metricas y configuracion

Toda la configuracion centralizada esta en `src/config.py` (`ProjectConfig`):

| Parametro | Valor por defecto | Uso |
|-----------|-------------------|-----|
| `RANDOM_SEED` | 42 | Splits e inicializacion |
| `IMAGE_SIZE` | (96, 96, 96) | Resolucion estandar |
| `BATCH_SIZE` | 4 | Ajustar segun VRAM |
| `LEARNING_RATE` | 1e-4 | Adam |
| `EARLY_STOP_METRIC` | `val_clinical_f2` | Early stopping y best checkpoint |
| `CLINICAL_F2_WEIGHTS` | CN=0.1, MCI=0.3, AD=0.6 | Metrica de seleccion |

**Clinical F2:** F-beta con beta=2 por clase, agregado con pesos clinicos (prioriza AD y MCI).
Tambien se registran macro F2, accuracy y loss en `outputs/<run>/history.csv`.

## 8. Artefactos generados

Cada run crea `outputs/<run-name>/`:

| Archivo | Contenido |
|---------|-----------|
| `best_model.pth` | Mejor checkpoint (por val clinical F2) |
| `history.csv` | Metricas por epoch |
| `training_summary.txt` | Resumen del entrenamiento |
| `pipeline.log` | Log completo si se uso `run_pipeline.py` |
| `confusion_matrix_test.png` | Matriz de confusion en test |

Los pesos (`.pth`) y logs **no** se versionan en git.

## 9. Que no esta incluido

| Recurso | Motivo |
|---------|--------|
| `data/` (NIfTI, DICOM, `.pt`) | Privacidad, tamano, DUA |
| `outputs/` (modelos entrenados) | Artefactos regenerables |
| `weights/` (SSL, y-Aware) | Descargar por separado; ver scripts en repo privado |
| CSV clinicos ADNI | Prohibido redistribuir (DUA ADNI) |
| Historial de experimentos (`DIARIO.md`) | Documentacion interna de desarrollo |

## 10. Publicacion del codigo

El repositorio se mantiene privado durante la evaluacion del TFG. Cuando se publique,
se usara un **snapshot limpio** (un unico commit) sin historial de desarrollo:

```bash
./scripts/build_public_release.sh
# Revisar rama public-release, luego:
# git push origin public-release:main --force   # mismo repo
# o crear un repositorio nuevo y push alli
```

## 11. Solucion de problemas

| Problema | Accion |
|----------|--------|
| CUDA OOM | Reduce `BATCH_SIZE` en `src/config.py` o usa `--batch-size 2` |
| MNI lento | Usa `--workers N` y `--gpus 0,1` en `preprocess_to_pt.py` |
| Rutas incorrectas en CSV | Regenera splits con `prepare_oasis3_nifti_splits.py` |
| Import error MONAI | Verifica version `monai>=1.3.0` compatible con tu PyTorch |

## 12. Contacto

Para acceso al repositorio privado o dudas sobre reproducibilidad, contacta con el autor
a traves de la Universidad de Oviedo (TFG 2025/2026).
