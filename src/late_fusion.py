"""
late_fusion.py — Late Fusion: DenseNet121 (congelado) como extractor + XGBoost.

Estrategia:
    1. Carga el best_model.pth del run extractor (--run-source, default: igual que --run).
    2. Congela el modelo y extrae vectores de features 1024-dim de train+val y test.
    3. Concatena los features CNN con 4 covariables clínicas → vector (1028,).
    4a. Sin --smote: XGBoost directo (1028-dim) con sample_weight="balanced" + GridSearchCV.
    4b. Con --smote: ImbPipeline(TabularDiffusionOversampler → XGBoost) en espacio 1028-dim
        dentro de GridSearchCV — DDPM condicional aprende la distribucion de cada clase y
        genera sinteticos biologicamente coherentes sin la maldicion de la dimensionalidad.
        Sex y APOE4 se fuerzan a valores enteros validos tras la generacion.
        IMPORTANTE: usar n_jobs=1 en GridSearchCV para compatibilidad con PyTorch multiprocessing.
    5. Evalua en test, imprime metricas (incl. Clinical F2) y guarda resultados en --run.

Uso:
    # Modo base (sin SMOTE)
    python -m src.late_fusion --run densenet-cropped --dataset oasis3

    # Modo TabDDPM + XGBoost (extractor en otro run)
    python -m src.late_fusion \\
        --run densenet-late-fusion-ddpm \\
        --run-source densenet-cropped \\
        --dataset oasis3 --smote
"""

from __future__ import annotations

import argparse
import signal
import time
import warnings
from datetime import datetime
from pathlib import Path

import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from imblearn.pipeline import Pipeline as ImbPipeline
from src.tab_diffusion import TabularDiffusionOversampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import cfg
from src.dataset import get_dataloader
from src.model import get_model

if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


CLASS_NAMES = ["CN", "MCI", "AD"]
CLINICAL_FEATURE_NAMES = ["Age", "Sex", "Educ", "APOE4"]
ETIV_FEATURE_NAME = "eTIV"


def clinical_feature_names(use_etiv: bool = False) -> list[str]:
    names = list(CLINICAL_FEATURE_NAMES)
    if use_etiv:
        names.append(ETIV_FEATURE_NAME)
    return names
DENSENET_CNN_DIM = cfg.DENSENET_EMBED_DIM
ULTIMATE_FM_CNN_DIM = 320


def _cnn_dim_for_model(model_name: str) -> int:
    if model_name == "ultimate_fm":
        return ULTIMATE_FM_CNN_DIM
    if model_name == "efficientnet25d":
        return cfg.EFFICIENTNET25D_EMBED_DIM
    return DENSENET_CNN_DIM


_CLASS_LABELS = (0, 1, 2)


def _f1_macro_scorer(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """f1_macro robusto: XGBoost multi:softprob puede devolver probs en predict."""
    y_pred = estimator.predict(X)
    if getattr(y_pred, "ndim", 1) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return float(
        f1_score(y, y_pred, average="macro", labels=_CLASS_LABELS, zero_division=0)
    )


# ---------------------------------------------------------------------------
# Feature extraction (sin cambios)
# ---------------------------------------------------------------------------

def _extract_cnn_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    model_name: str,
) -> np.ndarray:
    """Vector GAP por batch: (B, cnn_dim)."""
    if model_name == "ultimate_fm":
        feat = model.feature_extractor(images)
        feat = model.pool(feat)
        return feat.cpu().numpy()
    if model_name == "efficientnet25d":
        feat = model.extract_embedding(images)
        return feat.cpu().numpy()
    feat = model.net.features(images)
    feat = F.relu(feat, inplace=True)
    feat = F.adaptive_avg_pool3d(feat, 1)
    return feat.flatten(1).cpu().numpy()


def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    model_name: str = "densenet121",
    split_label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae features CNN (GAP) y las concatena con 4 covariables clinicas.

    Returns:
        X: (N, cnn_dim + 4) — [CNN_features | clinical]
        y: (N,)             — etiquetas
    """
    model.eval()
    X_list, y_list = [], []
    n_batches = len(loader)
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["image"].to(device)          # (B, 1, 96, 96, 96)
            clin   = batch["clinical"].cpu().numpy()    # (B, 4) normalizado
            labels = batch["label"].cpu().numpy()       # (B,)

            feat = _extract_cnn_batch(model, images, model_name)
            X_list.append(np.hstack([feat, clin]))
            y_list.append(labels)

            elapsed = time.time() - t0
            eta = (elapsed / i) * (n_batches - i) if i > 0 else 0
            print(
                f"\r  [{split_label}] [{i}/{n_batches}] "
                f"{i/n_batches:>6.1%} | {elapsed:.0f}s / ETA {eta:.0f}s",
                end="", flush=True,
            )

    print()
    return np.vstack(X_list), np.concatenate(y_list)


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feat_names: list[str],
    save_path: Path,
    top_n: int = 15,
    title_suffix: str = "",
) -> None:
    """Guarda un grafico de barras con las top_n features mas importantes."""
    top_n = min(top_n, len(importances))
    top_idx = np.argsort(importances)[-top_n:][::-1]

    labels = [feat_names[i] for i in top_idx]
    values = importances[top_idx]

    clinical_set = set(CLINICAL_FEATURE_NAMES)
    colors = ["#e74c3c" if lbl in clinical_set else "#3498db" for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    title = f"Top {top_n} Features — Late Fusion XGBoost"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title + "\n(rojo=clínico, azul=CNN/latente)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Feature importance guardada en: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_late_fusion(
    run_name: str,
    run_source: str | None = None,
    dataset: str = "oasis3",
    batch_size: int = 8,
    use_smote: bool = False,
    use_etiv: bool = False,
    grid_verbose: int = 5,
    preprocess_variant: str | None = None,
    subset: int | None = None,
    source_filter: str | None = None,
) -> None:
    """
    Pipeline completa de Late Fusion.

    Args:
        run_name:    Carpeta de salida en outputs/ (artefactos y reporte).
        run_source:  Run del que se carga best_model.pth. Si None, usa run_name.
        dataset:     'oasis3' o 'oasis1'.
        batch_size:  Batch size para extraccion de features (no afecta a XGBoost).
        use_smote:   Si True, TabDDPM + XGBoost dentro de CV (sin data leakage).
        grid_verbose: Nivel de verbose para GridSearchCV/joblib.
    """
    source = run_source or run_name
    source_dir = cfg.OUTPUTS_DIR / source
    model_path = source_dir / "best_model.pth"

    run_dir = cfg.OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro best_model.pth en: {model_path}\n"
            f"Asegurate de haber entrenado con --run {source}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Cargar modelo CNN congelado ---
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "densenet121")
    uses_ordinal = checkpoint.get("uses_ordinal", False)
    cnn_dim = _cnn_dim_for_model(model_name)
    clin_names = clinical_feature_names(use_etiv=use_etiv)
    fused_dim = cnn_dim + len(clin_names)
    mode_label = (
        f"TabDDPM (difusion tabular condicional) {fused_dim}-dim + XGBoost (dentro CV)"
        if use_smote
        else f"XGBoost directo {fused_dim}-dim"
    )

    print(f"\n{'=' * 60}")
    print(f"LATE FUSION — XGBoost sobre features CNN")
    print(f"{'=' * 60}")
    print(f"Extractor (source): {source}")
    print(f"Salida (run):       {run_name}")
    print(f"Dataset:            {dataset}")
    if source_filter:
        print(f"Source filter:      {source_filter}")
    print(f"Modo:               {mode_label}")
    print(f"Device:             {device}")
    print(f"Inicio:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    val_f2 = checkpoint.get("val_clinical_f2", "?")
    val_f2_str = f"{val_f2:.4f}" if isinstance(val_f2, (int, float)) else str(val_f2)
    print(f"\n[INFO] Modelo CNN: {model_name} | epoch {checkpoint['epoch']} "
          f"| val_clinical_f2={val_f2_str} | feature_dim={cnn_dim}")

    lora_stages = checkpoint.get("ultimate_fm_lora_stages")
    freeze_backbone = checkpoint.get("ultimate_fm_freeze_backbone")
    weights_path = checkpoint.get("weights_path")
    get_kwargs: dict = {"ordinal": uses_ordinal}
    if checkpoint.get("dropout_prob") is not None:
        get_kwargs["dropout_prob"] = checkpoint.get("dropout_prob")
    if checkpoint.get("yaware_pretrained"):
        get_kwargs["yaware_pretrained"] = checkpoint.get("yaware_pretrained")
    if checkpoint.get("yaware_weights_path"):
        get_kwargs["yaware_weights_path"] = checkpoint.get("yaware_weights_path")
    if model_name == "ultimate_fm":
        get_kwargs["weights_path"] = weights_path
        if lora_stages is not None:
            get_kwargs["lora_stages"] = lora_stages
        if freeze_backbone is not None:
            get_kwargs["freeze_backbone"] = freeze_backbone

    variant = preprocess_variant or checkpoint.get(
        "preprocess_variant", cfg.PREPROCESS_VARIANT_DEFAULT,
    )
    print(f"Preprocess variant:  {variant}")

    model = get_model(model_name, **get_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "ultimate_fm":
        if not hasattr(model, "feature_extractor") or not hasattr(model, "pool"):
            raise ValueError("ultimate_fm requiere feature_extractor y pool para Late Fusion.")
    elif not hasattr(model, "net") or not hasattr(model.net, "features"):
        if model_name != "efficientnet25d" or not hasattr(model, "extract_embedding"):
            raise ValueError(
                f"El modelo '{model_name}' no expone features para Late Fusion. "
                "Compatible: densenet121, efficientnet25d, multimodal_densenet, ultimate_fm."
            )

    # --- Extraer features ---
    print(f"\n[1/3] Extrayendo features CNN (train + val → XGBoost train)...")

    loader_kw = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        dataset=dataset,
        use_clinical=True,
        use_etiv=use_etiv,
        variant=variant,
        subset=subset,
        source_filter=source_filter,
    )
    train_loader = get_dataloader("train", **loader_kw)
    val_loader = get_dataloader("val", **loader_kw)
    test_loader = get_dataloader("test", **loader_kw)

    X_train_raw, y_train_raw = extract_features(
        model, train_loader, device, model_name, "train",
    )
    X_val_raw, y_val_raw = extract_features(
        model, val_loader, device, model_name, "val  ",
    )
    X_test, y_test = extract_features(
        model, test_loader, device, model_name, "test ",
    )

    # Combinar train + val para maximizar datos del XGBoost
    X_train = np.vstack([X_train_raw, X_val_raw])
    y_train = np.concatenate([y_train_raw, y_val_raw])

    print(f"\n  Train XGBoost: {X_train.shape}  (train={len(y_train_raw)}, val={len(y_val_raw)})")
    print(f"  Test  XGBoost: {X_test.shape}")
    print(f"  Distribucion train — CN={np.sum(y_train==0)}, MCI={np.sum(y_train==1)}, AD={np.sum(y_train==2)}")
    print(f"  Distribucion test  — CN={np.sum(y_test==0)},  MCI={np.sum(y_test==1)},  AD={np.sum(y_test==2)}")

    # --- GridSearchCV ---
    if use_smote:
        # TabDDPM: difusion tabular condicional — genera sinteticos de MCI/AD en 1028-dim
        tab_ddpm = TabularDiffusionOversampler(
            random_state=cfg.RANDOM_SEED,
            verbose=1,
        )
        xgb_base = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=cfg.RANDOM_SEED,
            verbosity=0,
        )
        pipeline = ImbPipeline([
            ("synthetic_generation", tab_ddpm),
            ("xgb", xgb_base),
        ])
        param_grid = {
            "xgb__max_depth":     [3, 5],
            "xgb__learning_rate": [0.01, 0.05],
            "xgb__n_estimators":  [100, 200],
        }
        # n_jobs=1: PyTorch dentro del ImbPipeline no es compatible con joblib multiprocessing
        grid_n_jobs = 1
        fit_kwargs: dict = {}  # TabDDPM gestiona el balanceo internamente por fold
    else:
        # Modo base: XGBoost directo sobre 1028-dim con sample_weight
        pipeline = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=cfg.RANDOM_SEED,
            verbosity=0,
        )
        param_grid = {
            "max_depth":     [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators":  [100, 200, 300],
        }
        fit_kwargs = {"sample_weight": compute_sample_weight("balanced", y_train)}
        # Evitar n_jobs=-1: muchos workers XGBoost + extracción GPU compiten por RAM/NAS.
        grid_n_jobs = min(4, max(1, (joblib.cpu_count() or 4) // 2))

    n_candidates = len(ParameterGrid(param_grid))
    n_splits = 3
    total_fits = n_candidates * n_splits
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_SEED
    )

    print(f"\n[2/3] GridSearchCV")
    print(f"      modo:         {mode_label}")
    print(f"      scoring=f1_macro | cv={n_splits}-fold stratified | n_jobs={grid_n_jobs}")
    print(f"      combinaciones: {n_candidates} | ajustes totales: {total_fits}")
    print(f"      muestras train: {X_train.shape[0]}")
    print(f"      verbose: {grid_verbose}")
    t_grid_start = time.perf_counter()
    print(f"      inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=_f1_macro_scorer,
        cv=cv,
        n_jobs=grid_n_jobs,
        verbose=max(0, grid_verbose),
        refit=True,
        error_score=0.0,
    )
    search.fit(X_train, y_train, **fit_kwargs)

    elapsed = time.perf_counter() - t_grid_start
    print(f"\n[GridSearch] fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[GridSearch] tiempo total: {elapsed / 60:.2f} min ({elapsed:.1f} s)")
    if total_fits > 0:
        print(f"[GridSearch] tiempo medio por ajuste: {elapsed / total_fits:.2f} s")

    best_estimator = search.best_estimator_
    print(f"\n  Mejores hiperparametros: {search.best_params_}")
    print(f"  Mejor CV f1_macro:       {search.best_score_:.4f}")

    # --- Evaluacion en test ---
    print(f"\n[3/3] Evaluacion en test set...")

    # ImbPipeline: en predict(test) no se re-muestrea; solo pasa al XGB entrenado
    y_pred = best_estimator.predict(X_test)

    acc = np.mean(y_pred == y_test)
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    f2_per_class = fbeta_score(
        y_test, y_pred, beta=2,
        labels=[0, 1, 2], average=None, zero_division=0,
    )
    macro_f2 = fbeta_score(
        y_test, y_pred, beta=2,
        labels=[0, 1, 2], average="macro", zero_division=0,
    )
    clinical_f2 = sum(
        cfg.CLINICAL_F2_WEIGHTS[c] * f2_per_class[c]
        for c in range(cfg.NUM_CLASSES)
    )

    f2_lines = [
        f"  {'Clase':<8}  {'F2 (beta=2)':>12}",
        f"  {'-' * 22}",
        *[f"  {CLASS_NAMES[c]:<8}  {f2_per_class[c]:>12.4f}" for c in range(3)],
        f"  {'-' * 22}",
        f"  {'Macro':<8}  {macro_f2:>12.4f}",
        f"  {'Clinical':<8}  {clinical_f2:>12.4f}  (pesos: {cfg.CLINICAL_F2_WEIGHTS})",
    ]

    smote_note = (
        "Balanceo: TabDDPM condicional (T=100, 600 epochs) solo en train por fold CV\n\n"
        if use_smote
        else ""
    )
    results_text = (
        f"{'=' * 60}\n"
        f"LATE FUSION — XGBoost sobre features {model_name}\n"
        f"Extractor (source): {source}\n"
        f"Salida (run):       {run_name}\n"
        f"Dataset:            {dataset}\n"
        f"Modo:               {mode_label}\n"
        + smote_note
        + f"Fecha:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + f"{'=' * 60}\n\n"
        + f"Mejores hiperparametros: {search.best_params_}\n"
        + f"CV f1_macro (train+val):  {search.best_score_:.4f}\n"
        + f"Tiempo GridSearch:        {elapsed / 60:.2f} min\n\n"
        + f"--- Evaluacion TEST SET ---\n"
        + f"Accuracy global: {acc:.2%}\n\n"
        + f"{report}\n"
        + f"--- F2-Score (beta=2) ---\n"
        + "\n".join(f2_lines)
        + "\n\n"
        + f"Confusion Matrix:\n{cm}\n"
    )

    print(f"\n{'=' * 60}")
    print(f"RESULTADOS — TEST SET (Late Fusion XGBoost)")
    print(f"{'=' * 60}")
    print(f"Accuracy global: {acc:.2%}\n")
    print(report)
    print("--- F2-Score (beta=2) ---")
    print("\n".join(f2_lines))
    print(f"\nConfusion Matrix:\n{cm}")

    # --- Guardar artefactos ---
    report_path = run_dir / "late_fusion_report.txt"
    report_path.write_text(results_text, encoding="utf-8")
    print(f"\n[OK] Reporte guardado en: {report_path}")

    pkl_path = run_dir / "xgboost_late_fusion.pkl"
    joblib.dump(best_estimator, pkl_path)
    print(f"[OK] Pipeline/modelo guardado en: {pkl_path}")

    # Importancias y nombres de features segun el modo
    if use_smote:
        xgb_step = best_estimator.named_steps["xgb"]
        importances = xgb_step.feature_importances_
        feat_names = [f"CNN_Feat_{i}" for i in range(cnn_dim)] + clin_names
        title_suffix = f"TabDDPM + XGBoost — espacio latente {fused_dim}-dim"
    else:
        importances = best_estimator.feature_importances_
        feat_names = [f"CNN_{i}" for i in range(cnn_dim)] + clin_names
        title_suffix = f"XGBoost directo — espacio {fused_dim}-dim"

    importance_path = run_dir / "xgboost_feature_importance.png"
    plot_feature_importance(importances, feat_names, importance_path, top_n=15, title_suffix=title_suffix)

    print(f"\n{'=' * 60}")
    print(f"Late Fusion completado.")
    print(f"  Clinical F2 (test): {clinical_f2:.4f}")
    print(f"  Artefactos en:      {run_dir}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Late Fusion: DenseNet121 (congelado) + XGBoost con GridSearchCV"
    )
    parser.add_argument(
        "--run", type=str, default="densenet-cropped",
        help="Nombre del run de salida en outputs/ (artefactos y reporte). "
             "Default: densenet-cropped",
    )
    parser.add_argument(
        "--run-source", type=str, default=None,
        help="Run del que se carga best_model.pth (extractor CNN). "
             "Default: igual que --run",
    )
    parser.add_argument(
        "--dataset", type=str, default="oasis3",
        choices=["oasis1", "oasis3", "adni", "adni_baseline_sc", "oasis3_adni", "oasis3_adni_baseline"],
        help="Dataset (default: oasis3)",
    )
    parser.add_argument(
        "--preprocess-variant",
        type=str,
        default=None,
        choices=list(cfg.PREPROCESS_VARIANTS.keys()),
        help="Variante de preprocesado (default: leer del checkpoint del extractor)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size para extraccion de features (default: 8)",
    )
    parser.add_argument(
        "--smote", action="store_true",
        help="TabDDPM (difusion tabular condicional) + XGBoost dentro de CV (sin data leakage). "
             "Genera sinteticos MCI/AD en 1028-dim; requiere n_jobs=1 en GridSearchCV.",
    )
    parser.add_argument(
        "--grid-verbose", type=int, default=5,
        metavar="N",
        help="Verbose de GridSearchCV/joblib (0=silencio, 5-10 recomendado). Default: 5",
    )
    parser.add_argument(
        "--use-etiv", action="store_true",
        help="Incluir eTIV_norm como 5ª covariable clinica (requiere enrich_splits_etiv.py)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limitar a N muestras por split (smoke test)",
    )
    parser.add_argument(
        "--source-filter", type=str, default=None,
        help="Filtrar cohorte por columna source (p. ej. adni para solo ADNI en oasis3_adni)",
    )
    args = parser.parse_args()

    run_late_fusion(
        run_name=args.run,
        run_source=args.run_source,
        dataset=args.dataset,
        batch_size=args.batch_size,
        use_smote=args.smote,
        use_etiv=args.use_etiv,
        grid_verbose=args.grid_verbose,
        preprocess_variant=args.preprocess_variant,
        subset=args.subset,
        source_filter=args.source_filter,
    )
