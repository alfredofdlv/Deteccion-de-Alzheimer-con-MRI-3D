"""
tab_diffusion.py — Tabular Diffusion Model (TabDDPM) para generacion de datos sinteticos.

Motivacion:
    SMOTE y SMOTE-ENN fallan en espacios de alta dimension (1028-D) por la maldicion de la
    dimensionalidad: los vecinos k-NN en 1028 dimensiones estan casi equidistantes, produciendo
    puntos sinteticos ruidosos y biologicamente imposibles. Un modelo de difusion condicional
    aprende la distribucion de los datos reales y genera muestras coherentes condicionadas
    en la clase (CN / MCI / AD).

Arquitectura:
    - MLPDenoiser: MLP 3 capas con embeddings de tiempo y clase (DDPM).
    - GaussianDiffusion: schedule lineal de varianza, T=100 pasos (rapido).
    - TabularDiffusionOversampler: wrapper imblearn.BaseOverSampler; encaja en ImbPipeline
      sin data leakage (el modelo se entrena solo con el fold de train dentro de CV).

Advertencias de uso:
    - DDPM se reentrena en cada fold del GridSearchCV -> usar n_jobs=1 en GridSearchCV.
    - Con T=100 y 600 epochs el coste es ~2-4 min por ajuste en GPU; ~15 min en CPU.
    - Las features categoricas (Sex=col 1025, APOE4=col 1027) se fuerzan a enteros validos
      tras el muestreo (snap al valor mas cercano en el rango observado).

Referencia: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from imblearn.over_sampling.base import BaseOverSampler


# ---------------------------------------------------------------------------
# Constantes del espacio de features
# ---------------------------------------------------------------------------

N_FEATURES: int = 1028        # 1024 CNN + Age, Sex, Educ, APOE4
N_CLASSES: int = 3             # CN=0, MCI=1, AD=2
IDX_SEX: int = 1025            # Binaria {0, 1}
IDX_APOE4: int = 1027          # Entera {0, 1, 2}


# ---------------------------------------------------------------------------
# MLPDenoiser — red que predice el ruido ε en cada paso de difusion
# ---------------------------------------------------------------------------

class MLPDenoiser(nn.Module):
    """
    MLP condicional que predice el ruido ε añadido en el paso t del proceso de difusion.

    Entradas:
        x_t  : (B, N_FEATURES) — muestra ruidosa en el paso t
        t    : (B,) int        — paso de difusion en [0, T-1]
        y    : (B,) int        — clase 0/1/2 (condicionamiento)

    Salida:
        eps_pred : (B, N_FEATURES) — ruido predicho
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_classes: int = N_CLASSES,
        hidden: int = 512,
        t_emb_dim: int = 64,
        c_emb_dim: int = 32,
        n_steps: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.t_emb = nn.Embedding(n_steps, t_emb_dim)
        self.c_emb = nn.Embedding(n_classes, c_emb_dim)

        in_dim = n_features + t_emb_dim + c_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_e = self.t_emb(t)                  # (B, t_emb_dim)
        c_e = self.c_emb(y)                  # (B, c_emb_dim)
        h = torch.cat([x_t, t_e, c_e], dim=-1)
        return self.net(h)                   # (B, n_features)


# ---------------------------------------------------------------------------
# GaussianDiffusion — schedule y operaciones forward/reverse
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    Proceso de difusion gaussiano con schedule lineal de varianza.

    Registra los buffers alpha_bar, alpha, beta para calculos forward/reverse
    de forma vectorizada sin recomputar en cada paso.
    """

    def __init__(self, n_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        self.n_steps = n_steps

        betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # Ratio previo: alpha_bar_{t-1}; para t=0 se usa 1 (sin ruido previo)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)

        # Varianza posterior q(x_{t-1}|x_t, x_0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_var", posterior_var)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Proceso forward: añade ruido a x0 en el paso t."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def training_loss(
        self,
        model: MLPDenoiser,
        x0: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """MSE entre ruido real y ruido predicho (objetivo eps-prediction)."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t, y)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        model: MLPDenoiser,
        n_samples: int,
        class_label: int,
        device: torch.device,
        n_features: int = N_FEATURES,
    ) -> torch.Tensor:
        """
        Proceso reverse: denoising desde ruido puro condicionado a class_label.

        Devuelve (n_samples, n_features) en CPU.
        """
        model.eval()
        y = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
        x = torch.randn(n_samples, n_features, device=device)

        for t_idx in reversed(range(self.n_steps)):
            t_tensor = torch.full((n_samples,), t_idx, dtype=torch.long, device=device)
            eps_pred = model(x, t_tensor, y)

            beta_t      = self.betas[t_idx]
            alpha_t     = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bar[t_idx]

            # Media predicha x_{t-1}
            coef = beta_t / (1.0 - alpha_bar_t).sqrt()
            mean = (x - coef * eps_pred) / alpha_t.sqrt()

            if t_idx > 0:
                var = self.posterior_var[t_idx]
                mean = mean + var.sqrt() * torch.randn_like(x)

            x = mean

        return x.cpu()


# ---------------------------------------------------------------------------
# TabularDiffusionOversampler — wrapper imblearn
# ---------------------------------------------------------------------------

class TabularDiffusionOversampler(BaseOverSampler):
    """
    Oversampler de clases minoritarias usando un modelo de difusion gaussiana tabular.

    Entrena un DDPM condicional (MLP) sobre el fold de train de cada split de CV
    y genera el numero exacto de muestras sinteticas necesario para igualar la clase
    mayoritaria (comportamiento 'auto' de BaseOverSampler).

    Las variables categoricas Sex (col 1025) y APOE4 (col 1027) se redondean al entero
    mas cercano y se clampan a su rango valido tras la generacion.

    Parametros
    ----------
    sampling_strategy : str | dict, default='auto'
        Estrategia de muestreo (mismo API que BaseOverSampler).
    random_state : int | None, default=None
        Semilla para reproducibilidad.
    n_steps : int, default=100
        Numero de pasos de difusion T.
    n_epochs : int, default=600
        Epochs de entrenamiento del DDPM por fold.
    batch_size : int, default=256
        Minibatch size para el entrenamiento del DDPM.
    lr : float, default=1e-3
        Learning rate de AdamW.
    weight_decay : float, default=1e-4
        Regularizacion L2.
    hidden : int, default=512
        Neuronas en capas ocultas del MLPDenoiser.
    t_emb_dim : int, default=64
        Dimension del embedding de tiempo.
    c_emb_dim : int, default=32
        Dimension del embedding de clase.
    dropout : float, default=0.0
        Dropout en el MLPDenoiser.
    device : str | None, default=None
        'cuda', 'cpu', o None (auto-detect).
    verbose : int, default=0
        0 = silencio, 1 = progreso cada 100 epochs.
    """

    def __init__(
        self,
        *,
        sampling_strategy: str = "auto",
        random_state: Optional[int] = None,
        n_steps: int = 100,
        n_epochs: int = 600,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden: int = 512,
        t_emb_dim: int = 64,
        c_emb_dim: int = 32,
        dropout: float = 0.0,
        device: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        # BaseOverSampler solo acepta sampling_strategy; random_state se guarda como atributo
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden = hidden
        self.t_emb_dim = t_emb_dim
        self.c_emb_dim = c_emb_dim
        self.dropout = dropout
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Metodo principal exigido por BaseOverSampler
    # ------------------------------------------------------------------

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Entrena el DDPM sobre (X, y) y genera muestras sinteticas de las clases
        minoritarias segun self.sampling_strategy_ (calculado por el padre).
        """
        rng = check_random_state(self.random_state)
        seed = int(rng.randint(0, 2**31 - 1))

        # Dispositivo de computo
        if self.device is not None:
            dev = torch.device(self.device)
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fijar semillas
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        n_features = X.shape[1]
        n_classes  = len(np.unique(y))

        # Tensores de entrenamiento
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y, dtype=torch.long,    device=dev)

        # Construir modelo y proceso de difusion
        model = MLPDenoiser(
            n_features=n_features,
            n_classes=N_CLASSES,
            hidden=self.hidden,
            t_emb_dim=self.t_emb_dim,
            c_emb_dim=self.c_emb_dim,
            n_steps=self.n_steps,
            dropout=self.dropout,
        ).to(dev)

        diffusion = GaussianDiffusion(n_steps=self.n_steps).to(dev)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Entrenamiento
        self._train(model, diffusion, X_t, y_t, optimizer, dev, seed)

        # Generacion de sinteticos por clase minoritaria
        X_syn_list: list[np.ndarray] = []
        y_syn_list: list[np.ndarray] = []

        for class_label, n_gen in self.sampling_strategy_.items():
            if n_gen <= 0:
                continue
            torch.manual_seed(seed + class_label)
            X_syn_c = diffusion.sample(
                model, n_gen, int(class_label), dev, n_features
            ).numpy()
            X_syn_c = self._postprocess(X_syn_c, X, y, int(class_label))
            X_syn_list.append(X_syn_c)
            y_syn_list.append(np.full(n_gen, class_label, dtype=y.dtype))

        if X_syn_list:
            X_res = np.vstack([X] + X_syn_list).astype(np.float32)
            y_res = np.concatenate([y] + y_syn_list)
        else:
            X_res = X.astype(np.float32)
            y_res = y.copy()

        return X_res, y_res

    # ------------------------------------------------------------------
    # Bucle de entrenamiento interno
    # ------------------------------------------------------------------

    def _train(
        self,
        model: MLPDenoiser,
        diffusion: GaussianDiffusion,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        seed: int,
    ) -> None:
        model.train()
        N = X.shape[0]
        rng_torch = torch.Generator(device=device)
        rng_torch.manual_seed(seed)

        for epoch in range(self.n_epochs):
            perm = torch.randperm(N, generator=rng_torch, device=device)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb  = X[idx]
                yb  = y[idx]

                optimizer.zero_grad()
                loss = diffusion.training_loss(model, xb, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if self.verbose >= 1 and (epoch + 1) % 100 == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"  [TabDDPM] epoch {epoch + 1}/{self.n_epochs} | loss={avg:.4f}")

    # ------------------------------------------------------------------
    # Postproceso de variables categoricas y rango
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        X_syn: np.ndarray,
        X_real: np.ndarray,
        y_real: np.ndarray,
        class_label: int,
    ) -> np.ndarray:
        """
        Fuerza Sex y APOE4 a valores enteros validos observados en los datos reales.
        Las features continuas se dejan tal cual (la red ya aprendio la escala).
        """
        # Sex (col 1025): {0, 1}
        sex_vals = np.unique(np.round(X_real[:, IDX_SEX]).astype(int))
        sex_vals = sex_vals[(sex_vals >= 0) & (sex_vals <= 1)]
        if len(sex_vals) == 0:
            sex_vals = np.array([0, 1])
        X_syn[:, IDX_SEX] = self._snap_to_set(X_syn[:, IDX_SEX], sex_vals)

        # APOE4 (col 1027): entero en rango observado
        apoe_vals = np.unique(np.round(X_real[:, IDX_APOE4]).astype(int))
        apoe_vals = apoe_vals[(apoe_vals >= 0) & (apoe_vals <= 2)]
        if len(apoe_vals) == 0:
            apoe_vals = np.array([0, 1, 2])
        X_syn[:, IDX_APOE4] = self._snap_to_set(X_syn[:, IDX_APOE4], apoe_vals)

        return X_syn

    @staticmethod
    def _snap_to_set(values: np.ndarray, valid_set: np.ndarray) -> np.ndarray:
        """Mapea cada valor al elemento mas cercano en valid_set."""
        dist = np.abs(values[:, None] - valid_set[None, :])
        return valid_set[dist.argmin(axis=1)].astype(np.float32)
