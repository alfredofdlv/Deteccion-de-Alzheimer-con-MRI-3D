"""
export_context.py — Exporta el código fuente y resultados a un único Markdown.

Genera un documento autocontenido con todo el código de src/ y opcionalmente
los resultados de una ejecución de outputs/, ideal para pasárselo a un chatbot
(ChatGPT, Claude, etc.) como contexto.

Uso:
    # Solo código fuente
    python export_context.py

    # Código + resultados de una ejecución concreta
    python export_context.py --run full_100ep

    # Código + resultados de varias ejecuciones
    python export_context.py --run full_100ep --run new-version-2-dataaug

    # Cambiar archivo de salida
    python export_context.py --run full_100ep -o mi_contexto.md
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SKIP_FILES = {"__pycache__"}
TEXT_EXTENSIONS = {".csv", ".txt", ".json", ".log", ".yaml", ".yml", ".toml"}
PY_EXTENSION = ".py"


def collect_py_files(src_dir: Path) -> list[Path]:
    """Recopila todos los .py de src/ ordenados lógicamente."""
    files = sorted(src_dir.glob("*.py"))
    priority = ["__init__.py", "config.py", "data_utils.py", "data_prepare.py",
                "dataset.py", "model.py", "train.py", "evaluate.py","losses.py"]
    ordered = []
    for name in priority:
        p = src_dir / name
        if p in files:
            ordered.append(p)
            files.remove(p)
    ordered.extend(files)
    return ordered


def build_source_section(py_files: list[Path]) -> str:
    """Genera la sección de código fuente."""
    lines = [
        "# Proyecto TFG — Detección Temprana de Alzheimer con 3D MRI",
        "",
        "## Código Fuente (`src/`)",
        "",
        f"El proyecto contiene {len(py_files)} módulos Python.",
        "",
    ]

    for py_file in py_files:
        rel = py_file.relative_to(PROJECT_ROOT)
        content = py_file.read_text(encoding="utf-8")
        n_lines = len(content.splitlines())

        lines.append(f"### `{rel}` ({n_lines} líneas)")
        lines.append("")
        lines.append("```python")
        lines.append(content.rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def build_run_section(run_name: str) -> str:
    """Genera la sección de resultados de una ejecución."""
    run_dir = OUTPUTS_DIR / run_name
    if not run_dir.exists():
        return f"\n## Resultados: `{run_name}`\n\n> ERROR: No se encontró `outputs/{run_name}/`\n"

    lines = [
        f"## Resultados: `{run_name}`",
        "",
        f"Directorio: `outputs/{run_name}/`",
        "",
    ]

    text_files = sorted(
        f for f in run_dir.iterdir()
        if f.is_file() and f.suffix in TEXT_EXTENSIONS
    )
    other_files = sorted(
        f for f in run_dir.iterdir()
        if f.is_file() and f.suffix not in TEXT_EXTENSIONS and f.suffix != PY_EXTENSION
    )

    for tf in text_files:
        content = tf.read_text(encoding="utf-8", errors="replace").rstrip()
        ext_label = tf.suffix.lstrip(".")

        lines.append(f"### `{tf.name}`")
        lines.append("")
        lines.append(f"```{ext_label}")
        lines.append(content)
        lines.append("```")
        lines.append("")

    if other_files:
        lines.append("### Otros archivos (no incluidos en texto)")
        lines.append("")
        for of in other_files:
            size_kb = of.stat().st_size / 1024
            lines.append(f"- `{of.name}` ({size_kb:.1f} KB)")
        lines.append("")

    return "\n".join(lines)


def list_available_runs() -> list[str]:
    """Lista las ejecuciones disponibles en outputs/."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir())


def main():
    parser = argparse.ArgumentParser(
        description="Exporta código fuente y resultados a un Markdown para chatbots"
    )
    parser.add_argument(
        "--run", action="append", default=None,
        help="Nombre(s) de ejecución en outputs/ (repetible). "
             "Usa --run ALL para incluir todas.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="context_export.md",
        help="Archivo de salida (default: context_export.md)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Lista las ejecuciones disponibles y sale",
    )
    args = parser.parse_args()

    if args.list:
        runs = list_available_runs()
        if runs:
            print("Ejecuciones disponibles en outputs/:")
            for r in runs:
                print(f"  - {r}")
        else:
            print("No hay ejecuciones en outputs/.")
        return

    py_files = collect_py_files(SRC_DIR)
    sections = [build_source_section(py_files)]

    run_names = args.run or []
    if run_names == ["ALL"]:
        run_names = list_available_runs()

    if run_names:
        sections.append("---\n")
        for run_name in run_names:
            sections.append(build_run_section(run_name))

    doc = "\n".join(sections)
    out_path = PROJECT_ROOT / args.output
    out_path.write_text(doc, encoding="utf-8")

    n_runs = len(run_names)
    print(f"Exportado: {out_path}")
    print(f"  - {len(py_files)} módulos de src/")
    print(f"  - {n_runs} ejecución(es) de outputs/")
    print(f"  - Tamaño: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
