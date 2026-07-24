# Guia del CLI

Referencia de comandos de la version publica del repositorio. Los hiperparametros
por defecto viven en `src/config.py`. Para instalacion y datos, ver
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Indice de entrypoints

| Comando | Rol |
|---------|-----|
| `python run_pipeline.py <name> ...` | Orquesta train → evaluate → (late fusion / export) |
| `python -m src.train ...` | Entrenamiento |
| `python -m src.evaluate ...` | Evaluacion en test/val |
| `python -m src.late_fusion ...` | Late fusion CNN + XGBoost |
| `python -m src.threshold_tuning ...` | Optimizacion de umbrales en val |
| `python scripts/preprocess_to_pt.py ...` | NIfTI → tensores `.pt` |
| `python scripts/prepare_oasis3_nifti_splits.py` | Splits OASIS-3 (rutas NIfTI) |
| `python scripts/prepare_oasis3_splits.py` | Regenerar `*_pt.csv` desde `.pt` |
| `python scripts/prepare_adni_splits.py` | Splits ADNI |
| `python benchmark.py ...` | Benchmark de tiempos |

Todos los comandos se ejecutan desde la **raiz del repositorio**.

```bash
export CUDA_VISIBLE_DEVICES=0   # o el indice GPU que uses
```

---

## 1. Pipeline completa (`run_pipeline.py`)

Encadena entrenamiento y evaluacion. Escribe todo en `outputs/<name>/pipeline.log`.

```bash
python run_pipeline.py <name> [opciones]
```

### Argumentos principales

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `name` | (obligatorio) | Nombre del run → carpeta `outputs/<name>/` |
| `--dataset` | `oasis1` | `oasis1`, `oasis3`, `adni`, `adni_baseline_sc`, `oasis3_adni`, `oasis3_adni_baseline` |
| `--model` | `resnet10` | Ver modelos abajo |
| `--preprocess-variant` | `cropped` | `cropped`, `mni`, `mni_n4`, `mni_128` |
| `--epochs` | `config.py` | Maximo de epochs |
| `--patience` | `config.py` | Early stopping |
| `--batch-size` | auto | Override de batch size |
| `--aug` | `full` | `full`, `light`, `none` |
| `--dropout` | `config.py` | Dropout DenseNet |
| `--subset` | — | Limitar a N muestras por split (smoke) |
| `--no-export` | off | No generar Markdown de contexto |
| `--detach` | off | Modo resistente a caidas SSH |

### Modelos

`resnet10`, `simple3dcnn`, `densenet121`, `densenet121_coral`, `multimodal_densenet`,
`ultimate_fm`, `efficientnet25d`

### Loss y pesos

| Flag | Efecto |
|------|--------|
| `--ordinal` | CORAL + soft F2 (compatible con densenet121 / densenet121_coral) |
| `--focal` | Focal Loss (gamma=2); incompatible con `--ordinal` |
| `--clinical-weights` | Multiplicadores clinicos en class weights |
| `--pretrained` | MedicalNet (solo `resnet10`) |
| `--yaware-pretrained` | Encoder y-Aware BHB-10K (solo densenet121) |

### Labels y geometria

| Flag | Valores | Efecto |
|------|---------|--------|
| `--label-scheme` | `multiclass`, `binary_cn_imp`, `binary_mci_ad` | Esquema de etiquetas |
| `--roi` | `mtl` | Recorte ROI MTL online |
| `--spatial-size` | entero | Resize online a N³ |

### Post-procesado (via pipeline)

| Flag | Efecto |
|------|--------|
| `--tta` | TTA (flip LR) en evaluacion |
| `--tune-thresholds` | Optimiza umbrales en val tras evaluar |
| `--late-fusion` | Late fusion XGBoost tras evaluate |
| `--run-source` | Run extractor CNN para late fusion |
| `--late-fusion-only` | Solo late fusion (sin train/eval) |
| `--smote` | Sinteticos en late fusion (requiere `--late-fusion`) |
| `--inference-pipeline` | Ensemble + TTA + umbrales |

### Ejemplos

```bash
# DenseNet121 OASIS-3, preprocesado cropped
python run_pipeline.py densenet-oasis3 \
  --dataset oasis3 --model densenet121 --no-export

# Union OASIS+ADNI con MNI+N4 (experimento tipico de la memoria)
python run_pipeline.py densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni --model densenet121 \
  --preprocess-variant mni_n4 --epochs 100 --patience 15 --no-export

# Smoke rapido
python run_pipeline.py smoke --dataset oasis3 --model densenet121 \
  --epochs 1 --subset 32 --no-export

# Late fusion reutilizando un CNN ya entrenado
python run_pipeline.py densenet-lf \
  --dataset oasis3_adni --model densenet121 \
  --preprocess-variant mni_n4 \
  --late-fusion --late-fusion-only \
  --run-source densenet-oasis3-adni-mni-n4 --no-export
```

---

## 2. Entrenamiento (`python -m src.train`)

Misma familia de flags que `run_pipeline`, mas opciones de seleccion de modelo y loss.

```bash
python -m src.train --run <name> --dataset oasis3 --model densenet121 [opciones]
```

### Extras respecto a `run_pipeline`

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `--run` | `run_TIMESTAMP` | Carpeta en `outputs/` |
| `--overfit` | off | Sanity check: overfit one batch |
| `--expected-cost-loss` | off | CE + lambda × coste esperado |
| `--cost-lambda` | `0.5` | Peso del termino de coste |
| `--selection-metric` | `val_clinical_f2` | Metrica del best checkpoint |
| `--early-stop-metric` | = selection | Metrica solo para early stopping |
| `--seed` | `42` | Semilla (replicacion multi-semilla) |

Metricas de seleccion/early stop: `val_loss`, `val_accuracy`, `val_macro_f2`,
`val_clinical_f2`, `val_balanced_accuracy`, `val_qwk`, `val_expected_cost`,
`val_macro_mae` (y alias `val_quadratic_weighted_kappa`).

```bash
# Sanity check
python -m src.train --overfit --dataset oasis3 --subset 32

# Replicacion con semilla distinta
python -m src.train --run densenet-s7 --dataset oasis3_adni \
  --model densenet121 --preprocess-variant mni_n4 --seed 7
```

Salida tipica en `outputs/<run>/`: `best_model.pth`, `history.csv`,
`training_summary.txt`.

---

## 3. Evaluacion (`python -m src.evaluate`)

```bash
python -m src.evaluate --run <name> --dataset oasis3_adni --model densenet121
```

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `--run` | (obligatorio) | Carpeta en `outputs/` |
| `--split` | `test` | `test` o `val` |
| `--dataset` | `oasis1` | Igual que train |
| `--model` | auto | Si se omite, se lee del checkpoint |
| `--preprocess-variant` | del checkpoint | Debe coincidir con el entrenamiento |
| `--tta` | off | Promedia original + flip LR |
| `--mc-dropout` | off | Media de N pases con dropout activo |
| `--mc-samples` | `30` | Pases con `--mc-dropout` |
| `--no-thresholds` | off | Ignora `thresholds.json` si existe |
| `--source-filter` | — | En splits fusionados: `oasis3` o `adni` |
| `--label-scheme` | del checkpoint | `multiclass` / binarios |

```bash
python -m src.evaluate --run densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni --model densenet121 \
  --preprocess-variant mni_n4 --tta
```

---

## 4. Late fusion (`python -m src.late_fusion`)

Congela el CNN, extrae features y entrena XGBoost (+ covariables clinicas).

```bash
python -m src.late_fusion \
  --run densenet-lf \
  --run-source densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni
```

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `--run` | `densenet-cropped` | Carpeta de salida |
| `--run-source` | = `--run` | Checkpoint CNN extractor |
| `--preprocess-variant` | del checkpoint | Variante de `.pt` |
| `--batch-size` | `8` | Batch al extraer features |
| `--smote` | off | Sinteticos tabulares en CV |
| `--use-etiv` | off | 5ª covariable eTIV (requiere enrich previo) |
| `--source-filter` | — | Filtrar cohorte (p. ej. `adni`) |

Equivale a usarlo via `run_pipeline.py ... --late-fusion --late-fusion-only --run-source ...`.

---

## 5. Tuning de umbrales (`python -m src.threshold_tuning`)

Optimiza sesgos de logit en validacion y guarda `thresholds.json`.

```bash
python -m src.threshold_tuning \
  --run densenet-oasis3-adni-mni-n4 \
  --dataset oasis3_adni --tta
```

Opciones: `--bias-step`, `--output-run`, `--preprocess-variant`, `--subset`.

---

## 6. Preprocesado offline (`scripts/preprocess_to_pt.py`)

```bash
python scripts/preprocess_to_pt.py \
  --dataset oasis3 \
  --variant cropped
```

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `--dataset` | — | `oasis1`, `oasis3`, `adni` |
| `--variant` | `cropped` | `cropped`, `mni`, `mni_n4`, `mni_128` |
| `--spatial-size` | segun variant | Override del resize final |
| `--workers` | 1 | Procesos paralelos |
| `--gpus` | — | IDs GPU para HD-BET (`0,1`) |
| `--hdbet-device` | — | `cpu` o `cuda` |
| `--limit` | — | Solo primeras N imagenes (QC) |
| `--keep-intermediate` | off | Conservar NIfTI intermedios MNI |
| `--intermediate-local` | off | Intermedios en `/dev/shm` (NAS) |

`mni` / `mni_n4` / `mni_128` requieren `pip install -r requirements-optional.txt`.

```bash
python scripts/preprocess_to_pt.py --dataset oasis3 --variant mni_n4 --workers 4
python scripts/qc_mni_registration.py --n 10
```

---

## 7. Splits y datos

```bash
# OASIS-3: NIfTI → CSV de splits
python scripts/prepare_oasis3_nifti_splits.py
python scripts/prepare_oasis3_nifti_splits.py --window 180 --verbose

# Regenerar *_pt.csv si ya existen .pt
python scripts/prepare_oasis3_splits.py

# ADNI (tras DUA + descarga + dcm2niix)
python scripts/prepare_adni_splits.py
python scripts/convert_adni_dicom_to_nifti.py

# Union OASIS-3 + ADNI (tras tener ambos)
python scripts/build_oasis3_adni_union_global_splits.py --variant mni_n4
python scripts/build_oasis3_adni_mni_splits.py
```

---

## 8. Benchmark

```bash
python benchmark.py --dataset oasis3 --batches 15 --label "Mi config"
```

---

## 9. Flujo tipico de punta a punta

```bash
# 1. Splits
python scripts/prepare_oasis3_nifti_splits.py

# 2. Preprocesado
python scripts/preprocess_to_pt.py --dataset oasis3 --variant cropped

# 3. Entrenar + evaluar
python run_pipeline.py mi-experimento \
  --dataset oasis3 --model densenet121 --no-export

# 4. Re-evaluar con TTA (opcional)
python -m src.evaluate --run mi-experimento \
  --dataset oasis3 --model densenet121 --tta
```

Artefactos en `outputs/mi-experimento/`:

| Archivo | Contenido |
|---------|-----------|
| `best_model.pth` | Mejor checkpoint (por defecto: val clinical F2) |
| `history.csv` | Metricas por epoch |
| `training_summary.txt` | Resumen del entrenamiento |
| `pipeline.log` | Log si se uso `run_pipeline.py` |
| `confusion_matrix_test.png` | Matriz de confusion (tras evaluate) |

---

## 10. Variables de entorno utiles

| Variable | Uso |
|----------|-----|
| `CUDA_VISIBLE_DEVICES` | GPU visible (p. ej. `0` o `1`) |
| `RUN_PIPELINE_NON_INTERACTIVE=1` | Log directo a fichero (colas SSH) |
| `PREPROCESS_THREADS_PER_WORKER` | Hilos CPU por worker de preprocesado |

La configuracion central (`BATCH_SIZE`, `NUM_WORKERS`, `RANDOM_SEED`, metricas
clinicas, matriz de coste) se edita en `src/config.py`.
