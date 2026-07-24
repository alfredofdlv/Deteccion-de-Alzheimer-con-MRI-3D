"""
late_fusion_logits.py — Logit-Level Late Fusion + Optimización de F2 Clínico

Estrategia:
    1. Carga el best_model.pth del extractor CNN.
    2. Extrae las PREDICCIONES FINALES (Probabilidades: 2 o 3 columnas) + 4 Clínicas.
    3. Entrena un XGBoost ligero y muy regularizado sobre esta tabla de 6-7 columnas.
    4. OPTIMIZACIÓN DE UMBRALES: Busca los pesos de probabilidad en el Validation Set 
       que maximizan exactamente la métrica Clinical F2 (60% AD, 30% MCI, 10% CN).
    5. Aplica esos pesos al Test Set y evalúa.
"""

import argparse
import time
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import cfg
from src.dataset import get_dataloader
from src.inference_utils import outputs_to_class_probs
from src.model import get_model

# ---------------------------------------------------------------------------
# Extracción a nivel de Logits (Probabilidades)
# ---------------------------------------------------------------------------
def extract_logits_and_clinical(model, loader, device, uses_ordinal):
    model.eval()
    X_list, y_list = [], []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            clin = batch["clinical"].cpu().numpy()
            labels = batch["label"].cpu().numpy()
            
            outputs = model(images)
            probs = outputs_to_class_probs(outputs, uses_ordinal).cpu().numpy()
                
            X_list.append(np.hstack([probs, clin]))
            y_list.append(labels)
            
    return np.vstack(X_list), np.concatenate(y_list)

# ---------------------------------------------------------------------------
# Optimizador de Umbrales para F2 Clínico
# ---------------------------------------------------------------------------
def compute_clinical_f2(y_true, y_pred):
    f2 = fbeta_score(y_true, y_pred, beta=2, average=None, labels=[0,1,2], zero_division=0)
    return sum(cfg.CLINICAL_F2_WEIGHTS.get(c, 0) * f2[c] for c in range(3))

def optimize_thresholds(y_val, y_val_proba):
    """
    Busca multiplicadores de probabilidad para sesgar la decisión del modelo
    hacia las clases minoritarias (MCI y AD) y maximizar el F2 Clínico.
    """
    print("\n[INFO] Buscando umbrales óptimos en el Validation Set...")
    best_f2 = -1
    best_weights = np.array([1.0, 1.0, 1.0])
    
    # Grid de multiplicadores: penalizamos predecir CN, bonificamos predecir AD/MCI
    w0_range = np.linspace(0.1, 1.0, 10)  # CN
    w1_range = np.linspace(0.5, 3.0, 15)  # MCI
    w2_range = np.linspace(1.0, 5.0, 20)  # AD
    
    for w0 in w0_range:
        for w1 in w1_range:
            for w2 in w2_range:
                weights = np.array([w0, w1, w2])
                # Multiplicamos las probabilidades por los pesos y sacamos la clase ganadora
                preds = np.argmax(y_val_proba * weights, axis=1)
                f2 = compute_clinical_f2(y_val, preds)
                
                if f2 > best_f2:
                    best_f2 = f2
                    best_weights = weights
                    
    print(f"  -> Mejores pesos de ajuste: CN={best_weights[0]:.2f}, MCI={best_weights[1]:.2f}, AD={best_weights[2]:.2f}")
    print(f"  -> F2 Clínico en Validación con estos pesos: {best_f2:.4f}")
    return best_weights

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(run_name, dataset):
    source_dir = cfg.OUTPUTS_DIR / run_name
    model_path = source_dir / "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar Modelo
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    uses_ordinal = checkpoint.get("uses_ordinal", False)
    model = get_model(checkpoint.get("model_name", "densenet121"), ordinal=uses_ordinal).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"=== LOGIT-LEVEL LATE FUSION ===")
    print(f"Extractor: {run_name} | Ordinal: {uses_ordinal}")
    
    # 2. Extraer Logits + Clínicos
    train_loader = get_dataloader("train", dataset=dataset, use_clinical=True, shuffle=False)
    val_loader = get_dataloader("val", dataset=dataset, use_clinical=True, shuffle=False)
    test_loader = get_dataloader("test", dataset=dataset, use_clinical=True, shuffle=False)
    
    X_train, y_train = extract_logits_and_clinical(model, train_loader, device, uses_ordinal)
    X_val, y_val = extract_logits_and_clinical(model, val_loader, device, uses_ordinal)
    X_test, y_test = extract_logits_and_clinical(model, test_loader, device, uses_ordinal)
    
    print(f"Shape Train: {X_train.shape} | Shape Test: {X_test.shape}")
    
    # 3. XGBoost Ligero y Regularizado
    xgb_base = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", 
        random_state=cfg.RANDOM_SEED, tree_method="hist"
    )
    
    param_grid = {
        "max_depth": [2, 3], # Árboles muy planos para evitar memorizar la CNN
        "learning_rate": [0.01, 0.05],
        "n_estimators": [50, 100],
        "colsample_bytree": [0.5, 0.8] # Forzar a mirar el APOE4
    }
    
    search = GridSearchCV(
        xgb_base, param_grid, scoring="f1_macro", cv=3, n_jobs=-1
    )
    
    sample_weights = compute_sample_weight("balanced", y_train)
    print("\n[INFO] Entrenando XGBoost...")
    search.fit(X_train, y_train, sample_weight=sample_weights)
    best_xgb = search.best_estimator_
    
    # 4. Optimización de umbrales en Validación
    y_val_proba = best_xgb.predict_proba(X_val)
    optimal_weights = optimize_thresholds(y_val, y_val_proba)
    
    # 5. Evaluación final en Test Set aplicando los umbrales
    print("\n=== RESULTADOS EN TEST SET ===")
    y_test_proba = best_xgb.predict_proba(X_test)
    y_pred = np.argmax(y_test_proba * optimal_weights, axis=1)
    
    acc = np.mean(y_pred == y_test)
    print(f"Accuracy: {acc:.2%}\n")
    print(classification_report(y_test, y_pred, target_names=["CN", "MCI", "AD"]))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    
    f2_per_class = fbeta_score(y_test, y_pred, beta=2, average=None, labels=[0,1,2])
    clinical_f2 = compute_clinical_f2(y_test, y_pred)
    
    print(f"\nF2 por clase: CN={f2_per_class[0]:.4f}, MCI={f2_per_class[1]:.4f}, AD={f2_per_class[2]:.4f}")
    print(f">>> LATE FUSION (LOGITS) CLINICAL F2: {clinical_f2:.4f} <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="oasis3")
    args = parser.parse_args()
    main(args.run, args.dataset)