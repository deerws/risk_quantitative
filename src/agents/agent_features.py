# src/agents/agent_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import ray
from .ray_orchestrator import AgentBase

@ray.remote
class AgentFeatures(AgentBase):
    """Feature engineering temporal avançado"""
    
    def __init__(self):
        super().__init__("AgentFeatures")
        self.feature_cache = {}
    
    def create_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Cria features de volatilidade rolling"""
        features = pd.DataFrame(index=returns.index)
        
        # Rolling volatility em diferentes janelas
        windows = [5, 10, 20, 60]
        for window in windows:
            features[f'vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Volatilidade exponencial (EWMA)
        for span in [10, 30]:
            features[f'vol_ewma_{span}'] = returns.ewm(span=span).std() * np.sqrt(252)
        
        return features
    
    def create_correlation_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Cria features de correlação rolling"""
        features = pd.DataFrame(index=returns.index)
        
        if len(returns.columns) > 1:
            # Correlação média rolling
            features['corr_mean_20d'] = returns.rolling(20).corr(returns.mean(axis=1)).mean(axis=1)
            
            # Máxima correlação pairwise
            def max_correlation(x):
                corr_matrix = x.corr()
                np.fill_diagonal(corr_matrix.values, -np.inf)
                return corr_matrix.max().max()
            
            features['corr_max_20d'] = returns.rolling(20).apply(max_correlation, raw=False)
        
        return features
    
    def create_drawdown_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Cria features de drawdown"""
        features = pd.DataFrame(index=prices.index)
        
        cumulative_returns = (1 + prices.pct_change().fillna(0)).cumprod()
        
        for asset in cumulative_returns.columns:
            rolling_max = cumulative_returns[asset].expanding().max()
            drawdown = (cumulative_returns[asset] - rolling_max) / rolling_max
            
            features[f'drawdown_{asset}'] = drawdown
            features[f'max_drawdown_30d_{asset}'] = drawdown.rolling(30).min()
            features[f'drawdown_duration_{asset}'] = self.calculate_drawdown_duration(drawdown)
        
        return features
    
    def calculate_drawdown_duration(self, drawdown: pd.Series) -> pd.Series:
        """Calcula duração do drawdown em dias"""
        duration = pd.Series(0, index=drawdown.index)
        current_duration = 0
        
        for i in range(1, len(drawdown)):
            if drawdown.iloc[i] < 0:
                current_duration += 1
            else:
                current_duration = 0
            duration.iloc[i] = current_duration
        
        return duration
    
    def create_momentum_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Cria features de momentum"""
        features = pd.DataFrame(index=prices.index)
        
        # Retornos em diferentes períodos
        for period in [1, 5, 10, 20, 60]:
            features[f'momentum_{period}d'] = prices / prices.shift(period) - 1
        
        # SMA ratios
        for short, long in [(10, 20), (20, 50)]:
            features[f'sma_ratio_{short}_{long}'] = (
                prices.rolling(short).mean() / prices.rolling(long).mean() - 1
            )
        
        return features
    
    def process_data(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Processa dados e cria todas as features"""
        alerts = []
        
        try:
            returns = market_data.pct_change().fillna(0)
            
            # Criar todas as features
            vol_features = self.create_volatility_features(returns)
            corr_features = self.create_correlation_features(returns)
            dd_features = self.create_drawdown_features(market_data)
            mom_features = self.create_momentum_features(market_data)
            
            # Combinar todas as features
            all_features = pd.concat([
                vol_features, corr_features, dd_features, mom_features
            ], axis=1)
            
            # Cache features para outros agentes
            self.feature_cache = all_features
            
            # Verificar features extremas
            for col in all_features.columns:
                if all_features[col].dtype in [np.float64, np.float32]:
                    z_scores = (all_features[col] - all_features[col].mean()) / all_features[col].std()
                    extreme_count = (abs(z_scores) > 3).sum()
                    
                    if extreme_count > 0:
                        alerts.append(self.generate_alert(
                            f"Feature {col} com {extreme_count} valores extremos",
                            "medium",
                            {
                                'feature': col,
                                'extreme_count': int(extreme_count),
                                'max_z_score': float(z_scores.max()),
                                'min_z_score': float(z_scores.min())
                            }
                        ))
            
            alerts.append(self.generate_alert(
                f"Feature engineering completo: {len(all_features.columns)} features criadas",
                "low",
                {'total_features': len(all_features.columns), 'shape': all_features.shape}
            ))
            
        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentFeatures: {str(e)}",
                "critical",
                {'error': str(e)}
            ))
        
        return alerts
    
    def get_features(self) -> pd.DataFrame:
        """Retorna features calculadas"""
        return self.feature_cache