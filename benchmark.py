"""
benchmark.py — Medir tiempos de cada fase del pipeline y generar guia de estimacion.

Mide:
  1. Tiempo de importacion de MONAI
  2. Tiempo de instanciacion de DataLoader (con N samples)
  3. Tiempo de carga + preprocesamiento de un batch (I/O + transforms)
  4. Tiempo de forward pass (GPU/CPU)
  5. Tiempo de forward + backward + optimizer step
  6. Estimacion de un epoch completo

Genera Docs/benchmark.md con los resultados.

Uso:
    python benchmark.py                          # benchmark con oasis3
    python benchmark.py --dataset oasis1         # benchmark con oasis1
    python benchmark.py --batches 20             # medir 20 batches (default: 10)
"""

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Benchmark del pipeline de entrenamiento")
    parser.add_argument("--dataset", type=str, default="oasis3", choices=["oasis1", "oasis3"])
    parser.add_argument("--batches", type=int, default=10, help="Numero de batches a medir")
    args = parser.parse_args()

    results = {}
    results["dataset"] = args.dataset
    results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["platform"] = platform.platform()
    results["processor"] = platform.processor()

    # 1. Import MONAI
    print("[1/5] Importando MONAI y dependencias...")
    t0 = time.time()
    import torch
    from src.config import cfg
    from src.data_utils import load_split
    from src.dataset import get_dataloader, get_transforms
    from src.model import Simple3DCNN
    t_import = time.time() - t0
    results["import_time"] = t_import
    print(f"  Importacion: {t_import:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results["device"] = str(device)
    if device.type == "cuda":
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"  Device: {device}" + (f" ({results.get('gpu_name', '')})" if device.type == "cuda" else ""))

    # 2. Contar samples
    df_train = load_split("train", dataset=args.dataset)
    df_val = load_split("val", dataset=args.dataset)
    df_test = load_split("test", dataset=args.dataset)
    results["train_samples"] = len(df_train)
    results["val_samples"] = len(df_val)
    results["test_samples"] = len(df_test)
    results["batch_size"] = cfg.BATCH_SIZE
    results["train_batches"] = (len(df_train) + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE
    results["val_batches"] = (len(df_val) + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE
    print(f"  Samples — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    print(f"  Batch size: {cfg.BATCH_SIZE}, Train batches/epoch: {results['train_batches']}, Val batches/epoch: {results['val_batches']}")

    # 3. DataLoader creation
    print("\n[2/5] Creando DataLoaders...")
    t0 = time.time()
    train_loader = get_dataloader("train", num_workers=0, dataset=args.dataset)
    val_loader = get_dataloader("val", num_workers=0, dataset=args.dataset)
    t_loader = time.time() - t0
    results["dataloader_creation_time"] = t_loader
    print(f"  DataLoader creation: {t_loader:.2f}s")

    # 4. Batch loading (I/O + transforms)
    n_measure = min(args.batches, len(train_loader))
    print(f"\n[3/5] Midiendo carga de {n_measure} batches (I/O + transforms MONAI)...")
    batch_load_times = []
    loader_iter = iter(train_loader)
    for i in range(n_measure):
        t0 = time.time()
        batch = next(loader_iter)
        t_batch = time.time() - t0
        batch_load_times.append(t_batch)
        print(f"  Batch {i+1}/{n_measure}: {t_batch:.2f}s | shape: {list(batch['image'].shape)}")

    avg_load = sum(batch_load_times) / len(batch_load_times)
    results["avg_batch_load_s"] = round(avg_load, 3)
    results["min_batch_load_s"] = round(min(batch_load_times), 3)
    results["max_batch_load_s"] = round(max(batch_load_times), 3)
    print(f"  Promedio: {avg_load:.3f}s | Min: {min(batch_load_times):.3f}s | Max: {max(batch_load_times):.3f}s")

    # 5. Forward pass
    print(f"\n[4/5] Midiendo forward pass ({n_measure} batches)...")
    model = Simple3DCNN().to(device)
    model.eval()

    forward_times = []
    loader_iter2 = iter(train_loader)
    with torch.no_grad():
        for i in range(n_measure):
            batch = next(loader_iter2)
            images = batch["image"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.time() - t0
            forward_times.append(t_fwd)
            print(f"  Batch {i+1}/{n_measure}: forward {t_fwd:.4f}s")

    avg_fwd = sum(forward_times) / len(forward_times)
    results["avg_forward_s"] = round(avg_fwd, 4)
    print(f"  Promedio forward: {avg_fwd:.4f}s")

    # 6. Full train step (forward + backward + optimizer)
    print(f"\n[5/5] Midiendo train step completo (fwd + bwd + optim, {n_measure} batches)...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    step_times = []
    loader_iter3 = iter(train_loader)
    for i in range(n_measure):
        batch = next(loader_iter3)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_step = time.time() - t0
        step_times.append(t_step)
        print(f"  Batch {i+1}/{n_measure}: step {t_step:.4f}s (loss: {loss.item():.4f})")

    avg_step = sum(step_times) / len(step_times)
    results["avg_train_step_s"] = round(avg_step, 4)
    print(f"  Promedio train step: {avg_step:.4f}s")

    # Estimaciones
    est_train_epoch = (avg_load + avg_step) * results["train_batches"]
    est_val_epoch = (avg_load + avg_fwd) * results["val_batches"]
    est_epoch_total = est_train_epoch + est_val_epoch
    results["est_train_epoch_s"] = round(est_train_epoch)
    results["est_val_epoch_s"] = round(est_val_epoch)
    results["est_epoch_total_s"] = round(est_epoch_total)
    results["est_epoch_total_min"] = round(est_epoch_total / 60, 1)

    print(f"\n{'='*60}")
    print(f"ESTIMACIONES POR EPOCH")
    print(f"{'='*60}")
    print(f"  Train:  {est_train_epoch:.0f}s ({est_train_epoch/60:.1f} min)")
    print(f"  Val:    {est_val_epoch:.0f}s ({est_val_epoch/60:.1f} min)")
    print(f"  Total:  {est_epoch_total:.0f}s ({est_epoch_total/60:.1f} min)")
    print(f"\n  50 epochs: ~{est_epoch_total*50/3600:.1f} horas")
    print(f"  100 epochs: ~{est_epoch_total*100/3600:.1f} horas")

    # Generar MD
    md = _generate_md(results)
    out_path = Path("Docs/benchmark.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n[OK] Guia guardada en: {out_path}")


def _generate_md(r: dict) -> str:
    lines = [
        f"# Benchmark de tiempos — {r['dataset'].upper()}",
        f"",
        f"> Generado: {r['date']}",
        f"",
        f"## Hardware",
        f"",
        f"| Componente | Valor |",
        f"|---|---|",
        f"| Plataforma | {r['platform']} |",
        f"| Procesador | {r['processor']} |",
        f"| Device | {r['device']} |",
    ]
    if "gpu_name" in r:
        lines.append(f"| GPU | {r['gpu_name']} |")
        lines.append(f"| VRAM | {r['gpu_vram_gb']} GB |")

    lines += [
        f"",
        f"## Dataset",
        f"",
        f"| Split | Samples | Batches (bs={r['batch_size']}) |",
        f"|---|---|---|",
        f"| Train | {r['train_samples']} | {r['train_batches']} |",
        f"| Val | {r['val_samples']} | {r['val_batches']} |",
        f"| Test | {r['test_samples']} | — |",
        f"",
        f"## Tiempos medidos",
        f"",
        f"| Fase | Tiempo |",
        f"|---|---|",
        f"| Import MONAI + deps | {r['import_time']:.2f}s |",
        f"| Crear DataLoaders | {r['dataloader_creation_time']:.2f}s |",
        f"| Carga batch (I/O + transforms) | {r['avg_batch_load_s']:.3f}s (min: {r['min_batch_load_s']}, max: {r['max_batch_load_s']}) |",
        f"| Forward pass | {r['avg_forward_s']:.4f}s |",
        f"| Train step (fwd+bwd+optim) | {r['avg_train_step_s']:.4f}s |",
        f"",
        f"## Estimacion por epoch",
        f"",
        f"| Fase | Tiempo |",
        f"|---|---|",
        f"| Train ({r['train_batches']} batches) | {r['est_train_epoch_s']}s ({r['est_train_epoch_s']/60:.1f} min) |",
        f"| Val ({r['val_batches']} batches) | {r['est_val_epoch_s']}s ({r['est_val_epoch_s']/60:.1f} min) |",
        f"| **Total por epoch** | **{r['est_epoch_total_s']}s ({r['est_epoch_total_min']} min)** |",
        f"",
        f"## Estimacion entrenamiento completo",
        f"",
        f"| Epochs | Tiempo estimado |",
        f"|---|---|",
        f"| 10 | {r['est_epoch_total_s']*10/60:.0f} min |",
        f"| 25 | {r['est_epoch_total_s']*25/3600:.1f} horas |",
        f"| 50 | {r['est_epoch_total_s']*50/3600:.1f} horas |",
        f"| 100 | {r['est_epoch_total_s']*100/3600:.1f} horas |",
        f"",
        f"---",
        f"*Nota: las estimaciones asumen tiempos constantes por batch. El primer epoch",
        f"puede ser mas lento por carga en cache del SO. Con `num_workers>0` los tiempos",
        f"de I/O pueden mejorar significativamente.*",
        f"",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
