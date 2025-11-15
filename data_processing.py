"""
Módulo para procesamiento y transformación de datos financieros.
Incluye funciones para resampling, construcción de panels, manejo de missing data
y preparación de datasets supervisados para modelado.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict


def resample_to_weekly(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """
    Convierte datos diarios a frecuencia semanal.
    
    Args:
        df: DataFrame con índice de fecha o columna 'date'
        freq: Frecuencia semanal ('W-FRI' = viernes)
    
    Returns:
        DataFrame resampleado a frecuencia semanal
    """
    df_copy = df.copy()
    
    # Asegurar que el índice sea datetime
    if 'date' in df_copy.columns:
        df_copy = df_copy.set_index('date')
    
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    
    # Resamplear: tomar último valor de la semana
    df_weekly = df_copy.resample(freq).last()
    
    # Eliminar filas con todos NaN
    df_weekly = df_weekly.dropna(how='all')
    
    return df_weekly


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte datos diarios a frecuencia mensual.
    
    Args:
        df: DataFrame con índice de fecha o columna 'date'
    
    Returns:
        DataFrame resampleado a frecuencia mensual
    """
    df_copy = df.copy()
    
    if 'date' in df_copy.columns:
        df_copy = df_copy.set_index('date')
    
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)
    
    # Resamplear: tomar último valor del mes
    df_monthly = df_copy.resample('M').last()
    
    df_monthly = df_monthly.dropna(how='all')
    
    return df_monthly


def build_panel_dataframe(ticker_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye un panel de datos ancho (fechas × tickers).
    
    Args:
        ticker_data_dict: Diccionario {ticker: DataFrame con columnas date y adj_close}
    
    Returns:
        DataFrame ancho con fechas como índice y tickers como columnas
    """
    panel_data = {}
    
    for ticker, df in ticker_data_dict.items():
        if df.empty:
            continue
        
        # Asegurar que tenemos date y adj_close
        if 'date' in df.columns and 'adj_close' in df.columns:
            temp_df = df[['date', 'adj_close']].copy()
            temp_df = temp_df.set_index('date')
            panel_data[ticker] = temp_df['adj_close']
    
    if not panel_data:
        print("⚠️ No hay datos para construir el panel")
        return pd.DataFrame()
    
    # Combinar todas las series
    panel = pd.DataFrame(panel_data)
    
    # Ordenar por fecha
    panel = panel.sort_index()
    
    print(f"✅ Panel construido: {panel.shape[0]} fechas × {panel.shape[1]} tickers")
    
    return panel


def handle_missing_data(
    df: pd.DataFrame,
    max_missing_pct: float = 0.5,
    ffill_limit: int = 5
) -> pd.DataFrame:
    """
    Gestiona valores faltantes en el panel de datos.
    
    Args:
        df: DataFrame con tickers como columnas
        max_missing_pct: Porcentaje máximo de missing permitido por ticker (0-1)
        ffill_limit: Número máximo de períodos consecutivos a rellenar con forward fill
    
    Returns:
        DataFrame con missing data gestionado
    """
    df_clean = df.copy()
    
    # 1. Eliminar tickers con demasiados missing
    missing_pct = df_clean.isna().sum() / len(df_clean)
    tickers_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
    
    if tickers_to_drop:
        print(f"🗑️ Eliminando {len(tickers_to_drop)} tickers con >{max_missing_pct*100}% missing")
        df_clean = df_clean.drop(columns=tickers_to_drop)
    
    # 2. Forward fill limitado para huecos pequeños
    df_clean = df_clean.ffill(limit=ffill_limit)
    
    # 3. Eliminar filas con cualquier NaN restante
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_after = len(df_clean)
    
    if rows_before > rows_after:
        print(f"🗑️ Eliminadas {rows_before - rows_after} filas con missing data")
    
    print(f"✅ Panel limpio: {df_clean.shape[0]} fechas × {df_clean.shape[1]} tickers")
    
    return df_clean


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rendimientos logarítmicos.
    
    Args:
        df: DataFrame con precios
    
    Returns:
        DataFrame con log returns
    """
    log_returns = np.log(df / df.shift(1))
    
    # Eliminar la primera fila (NaN)
    log_returns = log_returns.iloc[1:]
    
    return log_returns


def create_supervised_dataset(
    panel: pd.DataFrame,
    target_ticker: str,
    L: int,
    H: int
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Crea un dataset supervisado para predicción de series temporales.
    
    Args:
        panel: DataFrame con retornos (fechas × tickers)
        target_ticker: Ticker objetivo a predecir
        L: Longitud de la ventana de entrada (lookback)
        H: Horizonte de predicción (forecast horizon)
    
    Returns:
        Tupla (X, y, dates):
            - X: array de forma (n_samples, L, n_tickers) con ventanas de entrada
            - y: array de forma (n_samples,) con valores objetivo
            - dates: índice de fechas correspondiente a cada sample
    """
    if target_ticker not in panel.columns:
        raise ValueError(f"Ticker objetivo '{target_ticker}' no está en el panel")
    
    n_periods = len(panel)
    n_tickers = panel.shape[1]
    
    X_list = []
    y_list = []
    dates_list = []
    
    # Crear ventanas deslizantes
    for i in range(L, n_periods - H):
        # Ventana de entrada: últimas L observaciones de todos los tickers
        X_window = panel.iloc[i-L:i].values  # Shape: (L, n_tickers)
        
        # Valor objetivo: retorno del ticker objetivo H períodos adelante
        y_target = panel[target_ticker].iloc[i + H]
        
        # Fecha de la predicción
        date = panel.index[i]
        
        X_list.append(X_window)
        y_list.append(y_target)
        dates_list.append(date)
    
    X = np.array(X_list)  # Shape: (n_samples, L, n_tickers)
    y = np.array(y_list)  # Shape: (n_samples,)
    dates = pd.DatetimeIndex(dates_list)
    
    print(f"✅ Dataset supervisado creado:")
    print(f"   X: {X.shape} (samples, lookback, tickers)")
    print(f"   y: {y.shape} (samples,)")
    print(f"   Fechas: {dates[0]} a {dates[-1]}")
    
    return X, y, dates


def split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_pct: float = 0.8,
    val_pct: float = 0.1
) -> Tuple:
    """
    Divide dataset en train/val/test respetando el orden temporal.
    
    Args:
        X: Features
        y: Target
        dates: Fechas
        train_pct: Porcentaje para entrenamiento
        val_pct: Porcentaje para validación
    
    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test)
    """
    n_samples = len(X)
    
    train_size = int(n_samples * train_pct)
    val_size = int(n_samples * val_pct)
    
    # División temporal
    X_train = X[:train_size]
    y_train = y[:train_size]
    dates_train = dates[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    dates_val = dates[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    dates_test = dates[train_size + val_size:]
    
    print(f"✅ División temporal:")
    print(f"   Train: {len(X_train)} samples ({dates_train[0]} a {dates_train[-1]})")
    print(f"   Val:   {len(X_val)} samples ({dates_val[0]} a {dates_val[-1]})")
    print(f"   Test:  {len(X_test)} samples ({dates_test[0]} a {dates_test[-1]})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Normaliza features usando scaler ajustado solo en train.
    
    Args:
        X_train, X_val, X_test: Arrays de features
        method: 'standard' para StandardScaler o 'minmax' para MinMaxScaler
    
    Returns:
        Tupla (X_train_norm, X_val_norm, X_test_norm, scaler)
    """
    # Determinar forma original
    original_shape_train = X_train.shape
    original_shape_val = X_val.shape
    original_shape_test = X_test.shape
    
    # Reshape para ajustar scaler: (n_samples * L, n_tickers)
    if len(X_train.shape) == 3:  # (samples, L, tickers)
        n_samples_train, L, n_tickers = X_train.shape
        X_train_2d = X_train.reshape(-1, n_tickers)
        X_val_2d = X_val.reshape(-1, n_tickers)
        X_test_2d = X_test.reshape(-1, n_tickers)
    else:
        X_train_2d = X_train
        X_val_2d = X_val
        X_test_2d = X_test
    
    # Crear y ajustar scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Método '{method}' no reconocido. Use 'standard' o 'minmax'")
    
    scaler.fit(X_train_2d)
    
    # Transformar
    X_train_norm = scaler.transform(X_train_2d)
    X_val_norm = scaler.transform(X_val_2d)
    X_test_norm = scaler.transform(X_test_2d)
    
    # Reshape de vuelta a forma original
    if len(original_shape_train) == 3:
        X_train_norm = X_train_norm.reshape(original_shape_train)
        X_val_norm = X_val_norm.reshape(original_shape_val)
        X_test_norm = X_test_norm.reshape(original_shape_test)
    
    print(f"✅ Features normalizadas con {method} scaler")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


class StockDataset(Dataset):
    """
    Dataset de PyTorch para datos de acciones.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features de forma (n_samples, L, n_tickers)
            y: Targets de forma (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_pytorch_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea Datasets y DataLoaders de PyTorch.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Datos normalizados
        batch_size: Tamaño del batch
        shuffle_train: Si hacer shuffle del conjunto de entrenamiento
    
    Returns:
        Tupla (train_loader, val_loader, test_loader)
    """
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"✅ DataLoaders creados:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

