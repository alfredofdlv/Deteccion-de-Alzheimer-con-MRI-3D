"""
Módulo de visualización para datos financieros y resultados de modelos.
Incluye funciones para comparación de APIs, correlaciones, predicciones
y visualizaciones de optimización con Optuna.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import optuna


# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_api_comparison(comparison_df: pd.DataFrame, metric: str = 'years_coverage') -> None:
    """
    Gráfico de barras comparando proveedores de APIs.
    
    Args:
        comparison_df: DataFrame con métricas de comparación de APIs
        metric: Métrica a visualizar ('years_coverage', 'n_observations', 'missing_adj_close')
    """
    # Calcular promedio por proveedor
    avg_by_provider = comparison_df.groupby('provider')[metric].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(avg_by_provider)), avg_by_provider.values)
    
    # Colorear barras
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    plt.xticks(range(len(avg_by_provider)), avg_by_provider.index, rotation=45)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Comparación de Proveedores de APIs - {metric.replace("_", " ").title()}')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Promedio de {metric} por proveedor:")
    print(avg_by_provider)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    subset_tickers: Optional[List[str]] = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    Heatmap de matriz de correlación.
    
    Args:
        corr_matrix: Matriz de correlación (DataFrame)
        subset_tickers: Lista opcional de tickers a visualizar (si None, usa todos)
        figsize: Tamaño de la figura
    """
    if subset_tickers is not None:
        # Filtrar solo los tickers especificados
        available_tickers = [t for t in subset_tickers if t in corr_matrix.columns]
        corr_subset = corr_matrix.loc[available_tickers, available_tickers]
    else:
        corr_subset = corr_matrix
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_subset,
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False
    )
    plt.title('Matriz de Correlación de Rendimientos Semanales', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de correlación
    corr_values = corr_subset.values[np.triu_indices_from(corr_subset.values, k=1)]
    print(f"\n📊 Estadísticas de correlación:")
    print(f"   Media: {corr_values.mean():.3f}")
    print(f"   Mediana: {np.median(corr_values):.3f}")
    print(f"   Std: {corr_values.std():.3f}")
    print(f"   Mín: {corr_values.min():.3f}")
    print(f"   Máx: {corr_values.max():.3f}")


def plot_price_history(
    df: pd.DataFrame,
    tickers: List[str],
    title: str = "Evolución de Precios Ajustados",
    normalize: bool = False
) -> None:
    """
    Gráfico temporal de precios para múltiples tickers.
    
    Args:
        df: DataFrame con fechas como índice y tickers como columnas
        tickers: Lista de tickers a visualizar
        title: Título del gráfico
        normalize: Si True, normaliza precios a base 100
    """
    plt.figure(figsize=(14, 7))
    
    for ticker in tickers:
        if ticker in df.columns:
            data = df[ticker].dropna()
            
            if normalize and len(data) > 0:
                data = (data / data.iloc[0]) * 100
            
            plt.plot(data.index, data.values, label=ticker, linewidth=2, alpha=0.8)
    
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio Ajustado' + (' (Base 100)' if normalize else ''), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Predicciones vs Valores Reales"
) -> None:
    """
    Gráfico comparativo de predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        dates: Fechas (opcional)
        title: Título del gráfico
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Eje X: fechas o índices
    x_axis = dates if dates is not None else np.arange(len(y_true))
    
    # Gráfico 1: Serie temporal
    axes[0].plot(x_axis, y_true, label='Real', color='#2c3e50', linewidth=2, alpha=0.8)
    axes[0].plot(x_axis, y_pred, label='Predicción', color='#e74c3c', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Fecha', fontsize=12)
    axes[0].set_ylabel('Retorno', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=30, color='#3498db')
    
    # Línea de identidad (predicción perfecta)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
    
    axes[1].set_xlabel('Valor Real', fontsize=12)
    axes[1].set_ylabel('Valor Predicho', fontsize=12)
    axes[1].set_title('Scatter Plot: Real vs Predicho', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calcular R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"📊 R² Score: {r2:.4f}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> None:
    """
    Análisis visual de residuos.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        dates: Fechas (opcional)
    """
    residuals = y_true - y_pred
    x_axis = dates if dates is not None else np.arange(len(y_true))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuos vs tiempo
    axes[0, 0].scatter(x_axis, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fecha')
    axes[0, 0].set_ylabel('Residuo')
    axes[0, 0].set_title('Residuos vs Tiempo')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuos vs predicciones
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Valor Predicho')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].set_title('Residuos vs Predicciones')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histograma de residuos
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuo')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Residuos')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Estadísticas de residuos:")
    print(f"   Media: {residuals.mean():.6f}")
    print(f"   Std: {residuals.std():.6f}")
    print(f"   Min: {residuals.min():.6f}")
    print(f"   Max: {residuals.max():.6f}")


def plot_optuna_history(study: optuna.Study) -> None:
    """
    Visualiza el historial de optimización de Optuna.
    
    Args:
        study: Objeto Study de Optuna
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Optimization history
    trials = study.trials
    values = [trial.value for trial in trials if trial.value is not None]
    best_values = [min(values[:i+1]) for i in range(len(values))]
    
    axes[0].plot(values, 'o-', alpha=0.6, label='Trial value')
    axes[0].plot(best_values, 'r-', linewidth=2, label='Best value')
    axes[0].set_xlabel('Trial', fontsize=12)
    axes[0].set_ylabel('Objective Value', fontsize=12)
    axes[0].set_title('Optimization History', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Parameter importances (si hay suficientes trials)
    if len(trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            axes[1].barh(params, importances)
            axes[1].set_xlabel('Importance', fontsize=12)
            axes[1].set_title('Parameter Importances', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='x')
        except:
            axes[1].text(0.5, 0.5, 'Not enough trials\nfor importance calculation',
                        ha='center', va='center', fontsize=12)
            axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Not enough trials\nfor importance calculation',
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Mejor valor encontrado: {study.best_value:.6f}")
    print(f"📊 Mejores hiperparámetros:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Curvas de Entrenamiento"
) -> None:
    """
    Visualiza curvas de pérdida durante el entrenamiento.
    
    Args:
        train_losses: Lista de pérdidas en entrenamiento por epoch
        val_losses: Lista de pérdidas en validación por epoch
        title: Título del gráfico
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    
    # Marcar el mejor epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Best Epoch ({best_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Mejor epoch: {best_epoch}")
    print(f"📊 Mejor val loss: {best_val_loss:.6f}")


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """
    Visualiza comparación de múltiples modelos.
    
    Args:
        results_df: DataFrame con columnas: model, MAE, RMSE, MAPE, accuracy
    """
    metrics = ['MAE', 'RMSE', 'MAPE']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        if metric in results_df.columns:
            bars = axes[idx].bar(results_df['model'], results_df[metric])
            
            # Colorear barras
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_title(f'Comparación: {metric}', fontsize=12, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar tabla
    print("\n📊 Tabla comparativa de modelos:")
    print(results_df.to_string(index=False))
    
    # Identificar mejor modelo por cada métrica
    print("\n🏆 Mejores modelos por métrica:")
    for metric in metrics:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmin()
            best_model = results_df.loc[best_idx, 'model']
            best_value = results_df.loc[best_idx, metric]
            print(f"   {metric}: {best_model} ({best_value:.6f})")

