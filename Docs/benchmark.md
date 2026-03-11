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
