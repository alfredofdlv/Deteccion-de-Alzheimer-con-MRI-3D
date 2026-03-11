import os
import pandas as pd

base_dir = r"d:\clase\tfg\data\OASIS-4\Non-Imaging Data"
output_file = r"d:\clase\tfg\resumen_non_imaging_csvs.txt"

print("Explorando CSVs en Non-Imaging Data...")

with open(output_file, "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk(base_dir):
        for file in sorted(files):
            if file.endswith(".csv"):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_dir)
                f.write(f"=== Archivo: {rel_path} ===\n")
                try:
                    df_head = pd.read_csv(filepath, nrows=0)
                    columnas = list(df_head.columns)
                    f.write(f"Total columnas: {len(columnas)}\n")
                    f.write(f"Primeras columnas: {columnas[:20]}\n")

                    df_sample = pd.read_csv(filepath, nrows=3)
                    f.write(f"Primeras 3 filas:\n{df_sample.to_string(index=False)}\n")
                except Exception as e:
                    f.write(f"Error leyendo el archivo: {e}\n")
                f.write("\n")

print(f"¡Listo! Archivo generado en: '{output_file}'")
