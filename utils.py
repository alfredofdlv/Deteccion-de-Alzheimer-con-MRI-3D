"""
Módulo de utilidades para descarga y procesamiento de datos financieros.
Incluye wrappers para APIs: yfinance, Alpha Vantage y Tiingo.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
from typing import Optional, List, Dict


def get_yf_history(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Descarga datos históricos de Yahoo Finance usando yfinance.
    
    Args:
        ticker: Símbolo del ticker (ej: 'AAPL')
        start: Fecha de inicio en formato 'YYYY-MM-DD'
        end: Fecha final en formato 'YYYY-MM-DD'
        interval: Intervalo de datos ('1d', '1wk', '1mo')
        auto_adjust: Si True, ajusta todos los precios OHLC
    
    Returns:
        DataFrame normalizado con columnas: date, open, high, low, close, adj_close, volume
    """
    try:
        # Descargar datos
        data = yf.download(
            ticker, 
            start=start, 
            end=end, 
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False
        )
        
        if data.empty:
            print(f"Advertencia: No se encontraron datos para {ticker}")
            return pd.DataFrame()
        
        # Normalizar nombres de columnas
        df = data.copy()
        df = df.reset_index()
        
        # Renombrar columnas al formato estándar
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Si auto_adjust=True, yfinance no incluye 'Adj Close', usar 'Close'
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close']
        
        # Asegurar que date sea datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Seleccionar columnas en orden estándar
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[col for col in standard_cols if col in df.columns]]
        
        return df
        
    except Exception as e:
        print(f"Error descargando {ticker} con yfinance: {e}")
        return pd.DataFrame()


def get_alphavantage_daily(ticker: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    """
    Descarga datos diarios ajustados de Alpha Vantage.
    
    Args:
        ticker: Símbolo del ticker
        api_key: Clave API de Alpha Vantage
        outputsize: 'compact' (últimos 100 días) o 'full' (histórico completo)
    
    Returns:
        DataFrame normalizado
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'apikey': api_key,
        'outputsize': outputsize,
        'datatype': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        # Verificar errores
        if 'Error Message' in data:
            print(f"Error API Alpha Vantage: {data['Error Message']}")
            return pd.DataFrame()
        
        if 'Note' in data:
            print(f"Advertencia: Limite de API alcanzado: {data['Note']}")
            return pd.DataFrame()
        
        # Parsear datos
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            print(f"Advertencia: No se encontraron datos de series temporales para {ticker}")
            return pd.DataFrame()
        
        time_series = data[time_series_key]
        
        # Convertir a DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.reset_index()
        
        # Mapeo de columnas Alpha Vantage
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend', 'split_coef']
        
        # Convertir a tipos numéricos
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Seleccionar columnas estándar
        df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        return df
        
    except Exception as e:
        print(f"Error descargando {ticker} de Alpha Vantage: {e}")
        return pd.DataFrame()


def get_alphavantage_weekly(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Descarga datos semanales ajustados de Alpha Vantage.
    
    Args:
        ticker: Símbolo del ticker
        api_key: Clave API de Alpha Vantage
    
    Returns:
        DataFrame normalizado
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
        'symbol': ticker,
        'apikey': api_key,
        'datatype': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'Error Message' in data:
            print(f"Error API: {data['Error Message']}")
            return pd.DataFrame()
        
        if 'Note' in data:
            print(f"Advertencia: Limite de API: {data['Note']}")
            return pd.DataFrame()
        
        time_series_key = 'Weekly Adjusted Time Series'
        if time_series_key not in data:
            print(f"Advertencia: No se encontraron datos semanales para {ticker}")
            return pd.DataFrame()
        
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.reset_index()
        
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend']
        
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        return df
        
    except Exception as e:
        print(f"Error descargando {ticker} semanal de Alpha Vantage: {e}")
        return pd.DataFrame()


def get_alphavantage_monthly(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Descarga datos mensuales ajustados de Alpha Vantage.
    
    Args:
        ticker: Símbolo del ticker
        api_key: Clave API de Alpha Vantage
    
    Returns:
        DataFrame normalizado
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
        'symbol': ticker,
        'apikey': api_key,
        'datatype': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'Error Message' in data:
            print(f"Error API: {data['Error Message']}")
            return pd.DataFrame()
        
        if 'Note' in data:
            print(f"Advertencia: Limite de API: {data['Note']}")
            return pd.DataFrame()
        
        time_series_key = 'Monthly Adjusted Time Series'
        if time_series_key not in data:
            print(f"Advertencia: No se encontraron datos mensuales para {ticker}")
            return pd.DataFrame()
        
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.reset_index()
        
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend']
        
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        return df
        
    except Exception as e:
        print(f"Error descargando {ticker} mensual de Alpha Vantage: {e}")
        return pd.DataFrame()


def get_tiingo_eod(
    ticker: str,
    start: str,
    end: str,
    api_key: str
) -> pd.DataFrame:
    """
    Descarga datos EOD (End of Day) de Tiingo.
    
    Args:
        ticker: Símbolo del ticker
        start: Fecha de inicio 'YYYY-MM-DD'
        end: Fecha final 'YYYY-MM-DD'
        api_key: Clave API de Tiingo
    
    Returns:
        DataFrame normalizado
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {api_key}'
    }
    params = {
        'startDate': start,
        'endDate': end
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"Error {response.status_code} para {ticker}: {response.text[:100]}")
            return pd.DataFrame()
        
        data = response.json()
        
        if not data:
            print(f"Advertencia: No se encontraron datos para {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Tiingo devuelve columnas con nombres específicos
        # Renombrar a formato estándar
        column_mapping = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adjClose': 'adj_close',
            'adjVolume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convertir date a datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Seleccionar columnas estándar
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[col for col in standard_cols if col in df.columns]]
        
        return df
        
    except Exception as e:
        print(f"Error descargando {ticker} de Tiingo: {e}")
        return pd.DataFrame()


def get_sp500_tickers_from_html(html_content: str) -> List[str]:
    """
    Extrae los tickers del S&P 500 desde contenido HTML de Wikipedia.
    Usa BeautifulSoup para parsing mas robusto.
    
    Args:
        html_content: Contenido HTML de la pagina de Wikipedia
    
    Returns:
        Lista de simbolos de tickers
    """
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Buscar la tabla con id="constituents"
        table = soup.find('table', {'id': 'constituents'})
        
        if not table:
            raise ValueError("No se encontro la tabla con id='constituents'")
        
        # Buscar todas las filas en tbody
        tbody = table.find('tbody')
        if not tbody:
            raise ValueError("No se encontro tbody en la tabla")
        
        rows = tbody.find_all('tr')
        
        tickers = []
        for row in rows:
            # La primera celda (<td>) contiene el ticker
            cells = row.find_all('td')
            if cells:
                # El ticker esta en un link dentro del primer <td>
                first_cell = cells[0]
                link = first_cell.find('a')
                if link:
                    ticker = link.text.strip()
                    if ticker and ticker != '':
                        tickers.append(ticker)
        
        return tickers
        
    except ImportError:
        raise ImportError("Se requiere BeautifulSoup4: pip install beautifulsoup4")
    except Exception as e:
        print(f"Error con metodo alternativo: {e}")
        return []


def get_sp500_tickers() -> List[str]:
    """
    Descarga la lista de tickers del S&P 500 desde Wikipedia o yfinance.
    
    Returns:
        Lista de simbolos de tickers
    """
    # Metodo 1: Intentar desde Wikipedia con headers
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # Headers para evitar bloqueo 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Descargar HTML con requests primero
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        html_content = response.text
        
        # Metodo 1A: Intentar con BeautifulSoup (mas robusto)
        print("Intentando extraer tickers con BeautifulSoup...")
        tickers = get_sp500_tickers_from_html(html_content)
        
        if tickers and len(tickers) > 0:
            # Limpiar tickers (algunos pueden tener caracteres especiales)
            tickers = [str(ticker).replace('.', '-').strip() for ticker in tickers]
            print(f"Se descargaron {len(tickers)} tickers del S&P 500 desde Wikipedia (BeautifulSoup)")
            return tickers
        
        print("BeautifulSoup no encontro tickers, intentando con pandas.read_html...")
        
        # Metodo 1B: Intentar parsear con lxml o html5lib
        try:
            tables = pd.read_html(html_content, flavor='lxml')
        except (ImportError, ValueError):
            try:
                tables = pd.read_html(html_content, flavor='html5lib')
            except ImportError:
                print("Advertencia: lxml y html5lib no disponibles, probando metodo alternativo...")
                raise ImportError("No parsers available")
        
        # Primera tabla contiene la lista
        sp500_table = tables[0]
        
        # Detectar automaticamente el nombre de la columna de simbolos
        symbol_column = None
        possible_names = ['Symbol', 'Ticker symbol', 'Ticker', 'symbol', 'ticker']
        
        # Intentar con nombres de columna primero
        for col_name in possible_names:
            if col_name in sp500_table.columns:
                symbol_column = col_name
                break
        
        # Si no encontramos por nombre, intentar por indice (columna 0 suele ser el simbolo)
        if symbol_column is None:
            # pandas a veces lee las columnas como indices numericos
            if 0 in sp500_table.columns or isinstance(sp500_table.columns[0], int):
                print(f"Usando columna por indice (columna 0 = simbolos)")
                tickers = sp500_table[0].tolist()
            else:
                print(f"No se encontro columna de simbolos. Columnas disponibles: {list(sp500_table.columns)}")
                raise ValueError("No se encontro columna de simbolos")
        else:
            tickers = sp500_table[symbol_column].tolist()
        
        # Limpiar tickers
        tickers = [str(ticker).replace('.', '-').strip() for ticker in tickers if pd.notna(ticker)]
        
        print(f"Se descargaron {len(tickers)} tickers del S&P 500 desde Wikipedia")
        return tickers
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("Wikipedia bloqueo la peticion (403). Usando metodo alternativo...")
        else:
            print(f"Error HTTP: {e}")
    except Exception as e:
        print(f"Error con Wikipedia: {e}")
    
    # Metodo 2: Usar lista hardcodeada como fallback
    print("Usando lista hardcodeada de tickers del S&P 500...")
    
    # Lista actualizada de los principales componentes del S&P 500
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH',
        'XOM', 'JNJ', 'JPM', 'WMT', 'LLY', 'MA', 'PG', 'AVGO', 'HD', 'MRK',
        'CVX', 'ABBV', 'COST', 'ADBE', 'PEP', 'CSCO', 'AMD', 'TMO', 'MCD', 'ACN',
        'NFLX', 'CRM', 'LIN', 'DIS', 'ABT', 'VZ', 'DHR', 'WFC', 'PM', 'NKE',
        'TXN', 'RTX', 'BMY', 'UPS', 'QCOM', 'AMGN', 'HON', 'AMAT', 'LOW', 'INTU',
        'SPGI', 'BKNG', 'DE', 'ADP', 'ELV', 'TJX', 'AXP', 'SBUX', 'GE', 'C',
        'ISRG', 'BLK', 'MDT', 'GILD', 'ADI', 'REGN', 'ZTS', 'CME', 'PANW', 'KLAC',
        'BA', 'CAT', 'GS', 'NOW', 'NEE', 'MMC', 'SO', 'PLD', 'SCHW', 'T',
        'SYK', 'IBN', 'MS', 'CB', 'MDLZ', 'BX', 'PGR', 'ETN', 'CI', 'DUK',
        'BSX', 'VRTX', 'EOG', 'ITW', 'EQIX', 'BDX', 'FI', 'CSX', 'HCA', 'MCK',
        'APD', 'NOC', 'CL', 'PNC', 'USB', 'AON', 'MAR', 'MO', 'TGT', 'ICE',
        'CMG', 'SLB', 'WM', 'PYPL', 'GM', 'NXPI', 'COP', 'PSX', 'ECL', 'SHW',
        'MPC', 'TDG', 'MCO', 'EMR', 'FDX', 'FCX', 'AJG', 'NSC', 'ROP', 'HUM',
        'GD', 'AFL', 'ADM', 'AZO', 'MMM', 'DLR', 'SPG', 'TT', 'ORLY', 'PSA',
        'WELL', 'TRV', 'JCI', 'AMP', 'PAYX', 'APH', 'O', 'SRE', 'ANET', 'PCAR',
        'LHX', 'KMI', 'TEL', 'CARR', 'ALL', 'EW', 'CCI', 'MSI', 'MSCI', 'RSG',
        'OTIS', 'DXCM', 'IDXX', 'RMD', 'A', 'IQV', 'CTAS', 'FAST', 'ROST',
        'MNST', 'VRSK', 'CTSH', 'ODFL', 'CPRT', 'KEYS', 'TROW', 'XYL', 'ROK', 'ANSS'
    ]
    
    print(f"Se cargaron {len(tickers)} tickers del S&P 500 desde lista hardcodeada")
    return tickers


def compare_api_coverage(
    tickers: List[str],
    providers: Dict[str, callable],
    start_date: Optional[str] = "1980-01-01"
) -> pd.DataFrame:
    """
    Compara la cobertura de datos entre diferentes proveedores de APIs.
    
    Args:
        tickers: Lista de tickers a comparar
        providers: Diccionario con nombre del proveedor y función de descarga
        start_date: Fecha de inicio para la descarga
    
    Returns:
        DataFrame con métricas comparativas
    """
    results = []
    
    for ticker in tickers:
        print(f"\nAnalizando {ticker}...")
        
        for provider_name, fetch_func in providers.items():
            print(f"  - Proveedor: {provider_name}")
            
            try:
                # Descargar datos
                df = fetch_func(ticker)
                
                if df.empty:
                    results.append({
                        'ticker': ticker,
                        'provider': provider_name,
                        'min_date': None,
                        'max_date': None,
                        'n_observations': 0,
                        'missing_adj_close': 0,
                        'years_coverage': 0
                    })
                    continue
                
                # Calcular métricas
                min_date = df['date'].min()
                max_date = df['date'].max()
                n_obs = len(df)
                missing = df['adj_close'].isna().sum()
                years = (max_date - min_date).days / 365.25
                
                results.append({
                    'ticker': ticker,
                    'provider': provider_name,
                    'min_date': min_date,
                    'max_date': max_date,
                    'n_observations': n_obs,
                    'missing_adj_close': missing,
                    'years_coverage': round(years, 2)
                })
                
                # Pausa para respetar rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"    Error: {e}")
                results.append({
                    'ticker': ticker,
                    'provider': provider_name,
                    'min_date': None,
                    'max_date': None,
                    'n_observations': 0,
                    'missing_adj_close': 0,
                    'years_coverage': 0
                })
    
    comparison_df = pd.DataFrame(results)
    return comparison_df

