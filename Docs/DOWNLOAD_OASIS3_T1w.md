# Descarga OASIS-3 (T1w) desde NITRC IR

## Lista de sesiones (`experiment_id`)

- Archivo generado (local, carpeta `data/` ignorada por git):  
  `data/oasis3_sessions_to_download.csv`
- Contiene una columna **`experiment_id`** con **2450** IDs únicos derivados de  
  `data/splits/oasis3_{train,val,test}_pt.csv` (formato `OAS30001_MR_d0129`).
- Regenerar tras cambiar splits:

```bash
cd /media/nas/aflorez/Deteccion-de-Alzheimer-con-MRI-3D
/home/aflorez/venv/bin/python scripts/build_oasis3_sessions_download_csv.py --validate
```

`--validate` cruza con `data/uo293619_3_10_2026_17_33_22.csv` (columnas `MR ID` y `Scans`); en la generación actual **todos** los IDs están en el inventario y **todos** tienen `T1w` en `Scans`.

## Comando de descarga (copiar y pegar en tu terminal)

**No guardes contraseñas en el repositorio.** El script de oasis-scripts pide la contraseña con `read -s`.

```bash
mkdir -p "/media/nas/aflorez/Deteccion-de-Alzheimer-con-MRI-3D/data/raw/OASIS-3"

bash "/home/aflorez/oasis-scripts/download_scans/download_oasis_scans.sh" \
  "/media/nas/aflorez/Deteccion-de-Alzheimer-con-MRI-3D/data/oasis3_sessions_to_download.csv" \
  "/media/nas/aflorez/Deteccion-de-Alzheimer-con-MRI-3D/data/raw/OASIS-3" \
  "YOUR_XNAT_USERNAME" \
  "T1w"
```

Sustituye `YOUR_XNAT_USERNAME` por tu usuario de NITRC IR (en minúsculas lo fuerza el propio script).

## Importante: layout tras la descarga vs. CSVs de NIfTI

`download_oasis_scans.sh` organiza bajo el directorio de destino carpetas por **`experiment_id`** (p. ej. `OAS30001_MR_d0129/T1w/...`).

Los CSV `data/splits/oasis3_{train,val,test}.csv` esperan rutas tipo **BIDS**:

`data/OASIS-3/sub-OAS30001/ses-d0129/anat/..._T1w.nii.gz`

Antes de ejecutar `preprocess_to_pt.py` debes **una** de estas opciones:

1. Reorganizar o enlazar los `.nii.gz` descargados al árbol BIDS bajo  
   `data/OASIS-3/` para que coincidan las rutas (o actualizar esos CSV con rutas Linux correctas).
2. Regenerar los CSV de NIfTI apuntando a las rutas reales bajo `data/raw/OASIS-3/`, manteniendo nombres de fichero coherentes con los `.pt` deseados.

## Después de tener los NIfTI + rutas correctas

1. Activar el entorno (`source .venv/bin/activate` o tu venv en `/home/aflorez/venv`).
2. Preprocesado offline (incluye **CropForeground** y resize a 96³ en `preprocess_to_pt.py`):

```bash
cd /media/nas/aflorez/Deteccion-de-Alzheimer-con-MRI-3D
python preprocess_to_pt.py --dataset oasis3
```

Eso reescribe `data/splits/oasis3_*_pt.csv` con rutas `.pt` bajo `data/preprocessed/oasis3/`.

3. El notebook `notebooks/pipeline_oasis3.ipynb` es útil para exploración; el flujo reproducible de splits + `.pt` está en `preprocess_to_pt.py` y, si hace falta rehacer splits desde cero a partir de `.pt`, en `prepare_oasis3_splits.py`.

## Script auxiliar

- [`scripts/build_oasis3_sessions_download_csv.py`](../scripts/build_oasis3_sessions_download_csv.py)
