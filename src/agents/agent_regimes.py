# src/agents/agent_regimes.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import ray
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from .ray_orchestrator import AgentBase

@ray.remote
class AgentRegimes(AgentBase):
    """Detecta regimes de mercado usando clustering"""
    
    def __init__(self, n_regimes: int = 3):
        super().__init__("AgentRegimes")
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.kmeans = None
        self.current_regime = None
        self.regime_history = []
    
    def find_optimal_clusters(self, features: pd.DataFrame, max_k: int = 5) -> int:
        """Encontra número ótimo de clusters usando método do cotovelo"""
        if len(features) < max_k:
            return min(3, len(features))
        
        inertias = []
        silhouette_scores = []
        
        feature_array = self.scaler.fit_transform(features.dropna())
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_array)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(feature_array, labels))
            else:
                silhouette_scores.append(0)
        
        # Método simples: escolher k com melhor silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2  # +2 porque começamos de k=2
        return optimal_k
    
    def detect_market_regimes(self, features: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Detecta regimes de mercado"""
        features_clean = features.dropna()
        
        if len(features_clean) < 10:
            return np.array([]), {}
        
        # Encontrar número ótimo de clusters
        optimal_k = self.find_optimal_clusters(features_clean)
        
        # Aplicar K-Means
        feature_array = self.scaler.fit_transform(features_clean)
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        regimes = self.kmeans.fit_predict(feature_array)
        
        # Analisar características de cada regime
        regime_features = {}
        for regime in range(optimal_k):
            regime_mask = regimes == regime
            if not regime_mask.any():
                continue
                
            regime_data = features_clean.iloc[regime_mask]
            
            # Calcar estatísticas do regime
            regime_features[regime] = {
                'volatility_mean': float(regime_data.filter(like='vol').mean().mean()),
                'drawdown_mean': float(regime_data.filter(like='drawdown').mean().mean()),
                'count': int(regime_mask.sum()),
                'proportion': float(regime_mask.mean()),
                'last_occurrence': features_clean.index[regime_mask][-1] if regime_mask.any() else None,
                'is_stress_regime': self.is_stress_regime(regime_data)
            }
        
        # Identificar regime atual
        if len(regimes) > 0:
            self.current_regime = int(regimes[-1])
            self.regime_history.append({
                'timestamp': features_clean.index[-1],
                'regime': self.current_regime,
                'features': regime_features.get(self.current_regime, {})
            })
        
        return regimes, regime_features
    
    def is_stress_regime(self, regime_data: pd.DataFrame) -> bool:
        """Identifica se é um regime de stress"""
        vol_threshold = 0.25  # 25% de volatilidade
        dd_threshold = -0.10  # -10% drawdown
        
        avg_vol = regime_data.filter(like='vol').mean().mean()
        avg_dd = regime_data.filter(like='drawdown').mean().mean()
        
        return avg_vol > vol_threshold or avg_dd < dd_threshold
    
    def process_data(self, features_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processa features e detecta regimes"""
        alerts = []
        
        try:
            features = features_data.get('features')
            if features is None or features.empty:
                alerts.append(self.generate_alert(
                    "Features vazias para análise de regimes",
                    "low",
                    {}
                ))
                return alerts
            
            regimes, regime_features = self.detect_market_regimes(features)
            
            if len(regimes) == 0:
                alerts.append(self.generate_alert(
                    "Dados insuficientes para clustering",
                    "low",
                    {}
                ))
                return alerts
            
            # Alertas baseados no regime atual
            current_regime_info = regime_features.get(self.current_regime, {})
            
            if current_regime_info.get('is_stress_regime', False):
                alerts.append(self.generate_alert(
                    f"Regime de STRESS detectado (Regime {self.current_regime})",
                    "high",
                    {
                        'regime': self.current_regime,
                        'volatility': current_regime_info.get('volatility_mean', 0),
                        'drawdown': current_regime_info.get('drawdown_mean', 0),
                        'regime_features': regime_features
                    }
                ))
            
            # Detectar mudança de regime
            if len(self.regime_history) > 1:
                last_regime = self.regime_history[-2]['regime']
                if last_regime != self.current_regime:
                    alerts.append(self.generate_alert(
                        f"Mudança de regime: {last_regime} → {self.current_regime}",
                        "medium",
                        {
                            'previous_regime': last_regime,
                            'current_regime': self.current_regime,
                            'regime_features': regime_features
                        }
                    ))
            
            # Estatísticas gerais
            stress_regimes = [r for r, info in regime_features.items() if info.get('is_stress_regime')]
            if stress_regimes:
                alerts.append(self.generate_alert(
                    f"Detectados {len(stress_regimes)} regimes de stress",
                    "medium" if len(stress_regimes) > 1 else "low",
                    {
                        'stress_regimes': stress_regimes,
                        'total_regimes': len(regime_features),
                        'current_regime': self.current_regime
                    }
                ))
                
        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentRegimes: {str(e)}",
                "critical",
                {'error': str(e)}
            ))
        
        return alerts
    
    def get_current_regime(self) -> int:
        """Retorna regime atual"""
        return self.current_regime