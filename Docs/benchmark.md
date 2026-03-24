# Benchmark de tiempos — OASIS3

> Generado: 2026-03-11 11:45:44

## Hardware

| Componente | Valor |
|---|---|
| Plataforma | Windows-10-10.0.19045-SP0 |
| Procesador | Intel64 Family 6 Model 158 Stepping 13, GenuineIntel |
| Device | cuda |
| GPU | NVIDIA GeForce RTX 2060 |
| VRAM | 6.4 GB |

## Dataset

| Split | Samples | Batches (bs=4) |
|---|---|---|
| Train | 1731 | 433 |
| Val | 369 | 93 |
| Test | 350 | — |

## Tiempos medidos

| Fase | Tiempo |
|---|---|
| Import MONAI + deps | 9.55s |
| Crear DataLoaders | 0.08s |
| Carga batch (I/O + transforms) | 1.458s (min: 1.343, max: 1.707) |
| Forward pass | 0.0745s |
| Train step (fwd+bwd+optim) | 0.1697s |

## Estimacion por epoch

| Fase | Tiempo |
|---|---|
| Train (433 batches) | 705s (11.8 min) |
| Val (93 batches) | 142s (2.4 min) |
| **Total por epoch** | **847s (14.1 min)** |

## Estimacion entrenamiento completo

| Epochs | Tiempo estimado |
|---|---|
| 10 | 141 min |
| 25 | 5.9 horas |
| 50 | 11.8 horas |
| 100 | 23.5 horas |

---
*Nota: las estimaciones asumen tiempos constantes por batch. El primer epoch
puede ser mas lento por carga en cache del SO. Con `num_workers>0` los tiempos
de I/O pueden mejorar significativamente.*

---

## Preprocessed .pt + 4 workers

> 2026-03-11 12:23:54

**Config**: .pt (preprocessed) | batch_size=4 | num_workers=4 | device=cuda
(NVIDIA GeForce RTX 2060, 6.4 GB VRAM)

| Fase | Tiempo |
|---|---|
| Carga batch | 0.4573s (min: 0.0, max: 3.8123) |
| Forward pass | 0.4633s |
| Train step completo | 0.5357s |

| Estimacion | Tiempo |
|---|---|
| Train/epoch (433 batches) | 430s (7.2 min) |
| Val/epoch (93 batches) | 86s (1.4 min) |
| **Total/epoch** | **516s (8.6 min)** |
| 50 epochs | 7.2 horas |
| 100 epochs | 14.3 horas |

---

## Preprocessed .pt + 4 workers (warmup excl.)

> 2026-03-11 12:30:14

**Config**: .pt (preprocessed) | batch_size=4 | num_workers=4 | device=cuda
(NVIDIA GeForce RTX 2060, 6.4 GB VRAM)

| Fase | Tiempo |
|---|---|
| Carga batch | 0.1773s (min: 0.0, max: 1.675) |
| Forward pass | 0.0468s |
| Train step completo | 0.1453s |

| Estimacion | Tiempo |
|---|---|
| Train/epoch (433 batches) | 140s (2.3 min) |
| Val/epoch (93 batches) | 21s (0.3 min) |
| **Total/epoch** | **160s (2.7 min)** |
| 50 epochs | 2.2 horas |
| 100 epochs | 4.4 horas |

---

## ResNet10 + .pt + 4 workers

> 2026-03-11 16:54:21

**Config**: .pt (preprocessed) | batch_size=4 | num_workers=4 | device=cuda
(NVIDIA GeForce RTX 2060, 6.4 GB VRAM)

| Fase | Tiempo |
|---|---|
| Carga batch | 0.0554s (min: 0.0, max: 0.2565) |
| Forward pass | 0.2676s |
| Train step completo | 9.4661s |

| Estimacion | Tiempo |
|---|---|
| Train/epoch (433 batches) | 4123s (68.7 min) |
| Val/epoch (93 batches) | 30s (0.5 min) |
| **Total/epoch** | **4153s (69.2 min)** |
| 50 epochs | 57.7 horas |
| 100 epochs | 115.4 horas |

---

## ResNet10 + .pt NAS — 4workers

> 2026-03-23 11:22:07

**Config**: model=resnet10 | .pt (preprocessed) | batch_size=4 | num_workers=4 | device=cuda
(NVIDIA TITAN Xp, 12.8 GB VRAM)

| Fase | Tiempo |
|---|---|
| Carga batch | 0.0516s (min: 0.0003, max: 0.4115) |
| Forward pass | 0.1199s |
| Train step completo | 0.3281s |

| Estimacion | Tiempo |
|---|---|
| Train/epoch (435 batches) | 165s (2.8 min) |
| Val/epoch (90 batches) | 15s (0.2 min) |
| **Total/epoch** | **181s (3.0 min)** |
| 50 epochs | 2.5 horas |
| 100 epochs | 5.0 horas |

---

## ResNet10 + .pt NAS — 8workers

> 2026-03-23 11:23:13

**Config**: model=resnet10 | .pt (preprocessed) | batch_size=4 | num_workers=8 | device=cuda
(NVIDIA TITAN Xp, 12.8 GB VRAM)

| Fase | Tiempo |
|---|---|
| Carga batch | 0.0493s (min: 0.0004, max: 0.5001) |
| Forward pass | 0.1268s |
| Train step completo | 0.3315s |

| Estimacion | Tiempo |
|---|---|
| Train/epoch (435 batches) | 166s (2.8 min) |
| Val/epoch (90 batches) | 16s (0.3 min) |
| **Total/epoch** | **181s (3.0 min)** |
| 50 epochs | 2.5 horas |
| 100 epochs | 5.0 horas |

---

## ResNet10 + persistent_workers + prefetch

> 2026-03-23 11:32:41

**Config**: model=resnet10 | .pt (preprocessed) | batch_size=4 | num_workers=4 | device=cuda
(NVIDIA TITAN Xp, 12.8 GB VRAM)

| Fase | Tiempo |
|---|---|
| Carga batch | 0.0469s (min: 0.0004, max: 0.3489) |
| Forward pass | 0.1124s |
| Train step completo | 0.3258s |

| Estimacion | Tiempo |
|---|---|
| Train/epoch (435 batches) | 162s (2.7 min) |
| Val/epoch (90 batches) | 14s (0.2 min) |
| **Total/epoch** | **176s (2.9 min)** |
| 50 epochs | 2.4 horas |
| 100 epochs | 4.9 horas |

---

## Analisis completo Ubuntu-Server — 2026-03-23

> Titan Xp × 2 | CUDA 12.5 | PyTorch 2.6.0+cu124 | NFS v3 NAS

### Hardware real

| | |
|---|---|
| GPU | NVIDIA Titan Xp (Pascal, 2017) |
| VRAM total | 12 GB |
| VRAM pico entrenamiento (bs=4) | 5.75 GB allocado / 7.22 GB reservado |
| VRAM libre durante train | ~5.5 GB |
| RAM servidor | 62 GB total, 40 GB disponible |
| NAS mount | NFS v3, rsize=8192, wsize=8192 bytes |

### Desglose GPU step (bs=4, steady-state)

| Fase | Tiempo | % |
|---|---|---|
| Forward pass | 107 ms | 33% |
| Backward + optimizer | 220 ms | 67% |
| **Total step** | **326 ms** | — |

El GPU es el cuello de botella. I/O promedio (47 ms) está **completamente oculto** por los 4 workers.

### Tests de optimizacion realizados

| Tecnica | Resultado | Aplicado |
|---|---|---|
| persistent_workers=True + prefetch_factor=2 | -3% overhead inter-epoch | SI |
| 8 workers vs 4 workers | sin diferencia (I/O ya oculto) | NO |
| AMP FP16 (autocast) | 0.82x — MAS LENTO | NO |
| batch_size=8 | OOM (GPU0 con otro proceso) | NO |
| torch.compile | 0.993x — sin efecto (MetaTensor graph breaks) | NO |

**Por que AMP es mas lento:** Titan Xp es arquitectura Pascal (2017), sin Tensor Cores.
AMP acelera solo en Volta+ (V100, A100, RTX 20xx+). Aqui solo añade overhead del GradScaler.

**Por que torch.compile no ayuda:** MONAI usa MetaTensor que causa graph breaks en dynamo.
El compilador no puede fusionar las operaciones correctamente.

**Por que 8 workers no ayuda:** con persistent_workers el dataset completo (5.9 GB)
cabe en la page cache del SO (40 GB libres). Desde epoch 2 en adelante, los .pt
se leen de RAM, no de NAS. Los 4 workers son suficientes para ocultar ese I/O.

### Nota sobre NFS rsize=8192

El mount NAS tiene rsize/wsize=8192 bytes (deberia ser 1 MB). Esto hace que leer
un .pt de 3.4 MB requiera 435 round-trips a la NAS en lugar de 3-4. Sin embargo,
el OS page cache de 40 GB absorbe practicamente todo el dataset tras el primer epoch,
por lo que no es un bloqueante real. Si se quisiera corregir: `sudo mount -o remount,
rsize=1048576,wsize=1048576 /media/nas` (requiere sudo en el servidor NAS).

### Estimacion final realista

| | Tiempo |
|---|---|
| Por epoch (train 435 + val 90 batches) | ~2.5 min |
| Epoch 1 (NAS frio, sin cache) | ~3.0 min |
| Epoch 2+ (dataset en page cache) | ~2.5 min |
| 100 epochs (sin early stop) | **~4.2 horas** |
| Con early stop (patience=25, ~60 ep tipico) | **~2.5 horas** |

