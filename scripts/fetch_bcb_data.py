import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def fetch_economic_indicators(series_code, days=30):
    """
    Busca dados do BCB para um código de série específico
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"
    params = {
        'formato': 'json',
        'dataInicial': start_date.strftime('%d/%m/%Y'),
        'dataFinal': end_date.strftime('%d/%m/%Y')
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df['valor'] = pd.to_numeric(df['valor'])
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Erro ao buscar série {series_code}: {e}")
        return pd.DataFrame()

def fetch_all_indicators():
    """Busca todos os indicadores econômicos principais"""
    indicators = {
        'SELIC': 11,
        'IPCA': 433,
        'IPCA-15': 7478,
        'PIB': 4380,
        'Câmbio': 1,
        'Reservas': 13621
    }
    
    all_data = []
    for name, code in indicators.items():
        print(f"Coletando {name}...")
        data = fetch_economic_indicators(code)
        if not data.empty:
            data['indicador'] = name
            all_data.append(data)
        time.sleep(1)  # Delay para não sobrecarregar a API
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()