# CLAUDE.md

## Proyecto
TFG: Detección de Alzheimer mediante CNNs 3D sobre imágenes MRI del dataset OASIS-3.
Pipeline: NIfTI/ANALYZE → preprocesado offline (.pt tensors) → entrenamiento ResNet-10 / Simple3DCNN.
Métrica principal: macro F1-score (clases desbalanceadas: CN, MCI, AD).

## Entorno de ejecución: Servidor Linux Ubuntu (compartido)
- **Datos en**: `/media/nas/aflorez/` (NAS compartida con otros usuarios)
- Estructura esperada:
  - `data/preprocessed/oasis3/*.pt` — tensores preprocesados (~3.4 MB cada uno)
  - `data/splits/oasis3_*_pt.csv` — índices train/val/test
  - `data/oasis3_master_clinical.csv` — metadata clínica

## REGLA CRÍTICA — Servidor compartido
**ANTES de cualquier ejecución que use GPU:**
```bash
nvidia-smi
```
El servidor es compartido con otros compañeros de universidad. Solo lanzar entrenamientos
si hay VRAM suficiente libre. No lanzar trabajos pesados sin comprobar esto primero.

## Problema de rutas al migrar desde Windows
Los CSV `data/splits/oasis3_*_pt.csv` contienen rutas absolutas Windows (`D:\clase\tfg\...`).
En Linux hay que regenerarlos:
```bash
python preprocess_to_pt.py --dataset oasis3
```
El script detecta que los `.pt` ya existen y solo reescribe los CSV con rutas Linux.

## Configuración (`src/config.py`)
Ajustar según VRAM disponible (ver `nvidia-smi`):
- `BATCH_SIZE`: 8–16 (valor Windows: 4)
- `NUM_WORKERS`: 8–12 (valor Windows: 4)

## Verificación inicial en el servidor
```bash
nvidia-smi                                                         # GPU disponible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python preprocess_to_pt.py --dataset oasis3                       # Regenerar CSV con rutas Linux
python run_pipeline.py --dataset oasis3 --model resnet10 --run-name test
```

## Flujo de trabajo típico
1. `nvidia-smi` — verificar GPU libre
2. Ajustar `BATCH_SIZE` / `NUM_WORKERS` en `src/config.py`
3. `python run_pipeline.py --dataset oasis3 --model resnet10 --run-name <nombre>`
4. Resultados en `outputs/<nombre>/`

## Estructura del proyecto
```
src/
  config.py         # Configuración centralizada (rutas, hiperparámetros)
  data_prepare.py   # ETL: extracción, CSV master, splits estratificados
  data_utils.py     # Carga de CSVs y rutas
  dataset.py        # MONAI Dataset + DataLoaders + transforms
  model.py          # ResNet-10, Simple3DCNN
  train.py          # Bucle de entrenamiento con early stopping
  evaluate.py       # Métricas, reporte, matriz de confusión
preprocess_to_pt.py # Preprocesado offline → .pt tensors
run_pipeline.py     # CLI principal para ejecutar experimentos
```
