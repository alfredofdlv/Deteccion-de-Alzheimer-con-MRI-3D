"""
benchmark.py — Medir tiempos de cada fase del pipeline y generar guia de estimacion.

Mide:
  1. Tiempo de importacion de MONAI
  2. Tiempo de instanciacion de DataLoader (con N samples)
  3. Tiempo de carga + preprocesamiento de un batch (I/O + transforms)
  4. Tiempo de forward pass (GPU/CPU)
  5. Tiempo de forward + backward + optimizer step
  6. Estimacion de un epoch completo

Appendea resultados a Docs/benchmark.md (preserva secciones anteriores).

Uso:
    python benchmark.py                                                  # baseline
    python benchmark.py --label "Preprocessed .pt + 4 workers"           # con label
    python benchmark.py --dataset oasis1                                 # otro dataset
    python benchmark.py --batches 20                                     # mas batches
    python benchmark.py --num-workers 4                                  # override workers
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
    parser.add_argument("--label", type=str, default=None,
                        help="Etiqueta para esta seccion del benchmark (ej. 'Baseline NIfTI')")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override de num_workers para el DataLoader")
    args = parser.parse_args()

    results = {}
    results["dataset"] = args.dataset
    results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["platform"] = platform.platform()
    results["processor"] = platform.processor()
    results["label"] = args.label or f"{args.dataset.upper()} benchmark"

    # 1. Import MONAI
    print("[1/5] Importando MONAI y dependencias...")
    t0 = time.time()
    import torch
    from src.config import cfg
    from src.data_utils import load_split
    from src.dataset import get_dataloader
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

    num_workers = args.num_workers if args.num_workers is not None else cfg.NUM_WORKERS
    results["num_workers"] = num_workers

    # 2. Contar samples y detectar tipo (.pt o NIfTI)
    df_train = load_split("train", dataset=args.dataset)
    df_val = load_split("val", dataset=args.dataset)
    df_test = load_split("test", dataset=args.dataset)
    is_pt = str(df_train.iloc[0]["image_path"]).endswith(".pt")
    results["data_format"] = ".pt (preprocessed)" if is_pt else "NIfTI (.nii.gz / .img)"
    results["train_samples"] = len(df_train)
    results["val_samples"] = len(df_val)
    results["test_samples"] = len(df_test)
    results["batch_size"] = cfg.BATCH_SIZE
    results["train_batches"] = (len(df_train) + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE
    results["val_batches"] = (len(df_val) + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE
    print(f"  Formato: {results['data_format']}")
    print(f"  Samples — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    print(f"  Batch size: {cfg.BATCH_SIZE}, num_workers: {num_workers}")
    print(f"  Train batches/epoch: {results['train_batches']}, Val batches/epoch: {results['val_batches']}")

    # 3. DataLoader creation
    print("\n[2/5] Creando DataLoaders...")
    t0 = time.time()
    train_loader = get_dataloader("train", num_workers=num_workers, dataset=args.dataset)
    val_loader = get_dataloader("val", num_workers=num_workers, dataset=args.dataset)
    t_loader = time.time() - t0
    results["dataloader_creation_time"] = t_loader
    print(f"  DataLoader creation: {t_loader:.2f}s")

    # 4. Batch loading (I/O + transforms)
    n_measure = min(args.batches, len(train_loader))
    print(f"\n[3/5] Midiendo carga de {n_measure} batches...")
    batch_load_times = []
    loader_iter = iter(train_loader)
    for i in range(n_measure):
        t0 = time.time()
        batch = next(loader_iter)
        t_batch = time.time() - t0
        batch_load_times.append(t_batch)
        print(f"  Batch {i+1}/{n_measure}: {t_batch:.3f}s | shape: {list(batch['image'].shape)}")

    avg_load = sum(batch_load_times) / len(batch_load_times)
    results["avg_batch_load_s"] = round(avg_load, 4)
    results["min_batch_load_s"] = round(min(batch_load_times), 4)
    results["max_batch_load_s"] = round(max(batch_load_times), 4)
    print(f"  Promedio: {avg_load:.4f}s | Min: {min(batch_load_times):.4f}s | Max: {max(batch_load_times):.4f}s")

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

    # Appendear seccion al MD
    section = _generate_section(results)
    out_path = Path("Docs/benchmark.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        new_content = existing.rstrip("\n") + "\n\n" + section
    else:
        header = "# Benchmark de tiempos\n\n"
        new_content = header + section

    out_path.write_text(new_content, encoding="utf-8")
    print(f"\n[OK] Seccion '{results['label']}' appendeada a: {out_path}")


def _generate_section(r: dict) -> str:
    lines = [
        f"---",
        f"",
        f"## {r['label']}",
        f"",
        f"> {r['date']}",
        f"",
        f"**Config**: {r['data_format']} | batch_size={r['batch_size']} | num_workers={r['num_workers']} | device={r['device']}",
    ]
    if "gpu_name" in r:
        lines.append(f"({r['gpu_name']}, {r['gpu_vram_gb']} GB VRAM)")
    lines.append("")

    lines += [
        f"| Fase | Tiempo |",
        f"|---|---|",
        f"| Carga batch | {r['avg_batch_load_s']:.4f}s (min: {r['min_batch_load_s']:.4f}, max: {r['max_batch_load_s']:.4f}) |",
        f"| Forward pass | {r['avg_forward_s']:.4f}s |",
        f"| Train step completo | {r['avg_train_step_s']:.4f}s |",
        f"",
        f"| Estimacion | Tiempo |",
        f"|---|---|",
        f"| Train/epoch ({r['train_batches']} batches) | {r['est_train_epoch_s']}s ({r['est_train_epoch_s']/60:.1f} min) |",
        f"| Val/epoch ({r['val_batches']} batches) | {r['est_val_epoch_s']}s ({r['est_val_epoch_s']/60:.1f} min) |",
        f"| **Total/epoch** | **{r['est_epoch_total_s']}s ({r['est_epoch_total_min']} min)** |",
        f"| 50 epochs | {r['est_epoch_total_s']*50/3600:.1f} horas |",
        f"| 100 epochs | {r['est_epoch_total_s']*100/3600:.1f} horas |",
        f"",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
