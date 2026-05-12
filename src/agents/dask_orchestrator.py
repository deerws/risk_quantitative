import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
import sys
import os

try:
    from dask import delayed, compute
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    def delayed(f): return f
    def compute(*args, **kwargs): return args

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ✅ CORREÇÃO: Importar AgentBase do arquivo separado
try:
    from .agent_base import AgentBase
except ImportError:
    # Fallback para desenvolvimento
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from agents.agent_base import AgentBase

warnings.filterwarnings('ignore')

class AgentMarket(AgentBase):
    """Monitora condições gerais de mercado - CORRIGIDO"""
    
    def __init__(self):
        super().__init__("AgentMarket")
        self.volatility_threshold = 0.30
        self.correlation_threshold = 0.8
        
    def detect_regime_changes(self, returns: pd.DataFrame, window: int = 60) -> Dict[str, Any]:
        """Detecta mudanças de regime usando rolling statistics - CORRIGIDO"""
        try:
            if returns.empty:
                return {}
                
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            
            # ✅ CORREÇÃO: Verificar se há dados suficientes
            if len(rolling_vol) < window:
                return {}
            
            # ✅ CORREÇÃO: Usar mean(axis=1) corretamente para portfolio returns
            market_returns = returns.mean(axis=1)
            
            # ✅ CORREÇÃO: Calcular correlação de forma segura
            rolling_corr_list = []
            for col in returns.columns:
                try:
                    corr_series = returns[col].rolling(window).corr(market_returns)
                    rolling_corr_list.append(corr_series)
                except:
                    continue
            
            if rolling_corr_list:
                avg_corr = pd.concat(rolling_corr_list, axis=1).mean(axis=1)
            else:
                avg_corr = pd.Series(0, index=returns.index)
            
            # ✅ CORREÇÃO: Calcular z-scores de forma segura
            vol_zscore = (rolling_vol - rolling_vol.mean()) / (rolling_vol.std() + 1e-8)
            
            # ✅ CORREÇÃO: Aplicar mean(axis=1) apenas em DataFrame, não Series
            if isinstance(vol_zscore, pd.DataFrame):
                vol_zscore_mean = vol_zscore.mean(axis=1)
            else:
                vol_zscore_mean = vol_zscore
                
            corr_zscore = (avg_corr - avg_corr.mean()) / (avg_corr.std() + 1e-8)
            
            # ✅ CORREÇÃO: Converter para numérico antes de comparar
            high_vol_periods = (vol_zscore_mean > 2).sum() if not vol_zscore_mean.empty else 0
            low_vol_periods = (vol_zscore_mean < -2).sum() if not vol_zscore_mean.empty else 0
            high_corr_periods = (corr_zscore > 2).sum() if not corr_zscore.empty else 0
            
            # ✅ CORREÇÃO: Calcular mudanças de regime
            vol_diff = vol_zscore_mean.diff().abs() if not vol_zscore_mean.empty else pd.Series()
            corr_diff = corr_zscore.diff().abs() if not corr_zscore.empty else pd.Series()
            
            vol_regime_change = (vol_diff > 1).sum() if not vol_diff.empty else 0
            corr_regime_change = (corr_diff > 1).sum() if not corr_diff.empty else 0
            
            regime_changes = {
                'high_vol_periods': int(high_vol_periods),
                'low_vol_periods': int(low_vol_periods), 
                'high_corr_periods': int(high_corr_periods),
                'vol_regime_change': int(vol_regime_change),
                'corr_regime_change': int(corr_regime_change)
            }
            
            return regime_changes
            
        except Exception as e:
            print(f"Erro em detect_regime_changes: {e}")
            return {}
    def process_data(self, market_data: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Processa dados de mercado - CORRIGIDO"""
        alerts = []
        
        try:
            returns = market_data.pct_change().dropna()
            
            if returns.empty:
                alerts.append(self.generate_alert(
                    "Dados de retornos vazios",
                    "low",
                    {}
                ))
                return alerts
            
            # ✅ CORREÇÃO: Análise de volatilidade com verificação de empty
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            
            if not rolling_vol.empty:
                current_vol = rolling_vol.iloc[-1]  # ✅ Pega apenas o último valor
                
                # ✅ CORREÇÃO: Converter para série antes da comparação
                if isinstance(current_vol, pd.Series):
                    high_vol_assets = current_vol[current_vol > self.volatility_threshold]
                    
                    for asset, vol in high_vol_assets.items():
                        alerts.append(self.generate_alert(
                            f"Alta volatilidade em {asset}: {vol:.1%}",
                            "high" if vol > 0.4 else "medium",
                            {'asset': asset, 'volatility': float(vol), 'threshold': self.volatility_threshold}
                        ))
            
            # ✅ CORREÇÃO: Análise de correlações com try/except
            try:
                corr_matrix = returns.corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > self.correlation_threshold:
                            high_corr_pairs.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j],
                                float(corr_val)
                            ))
                
                if high_corr_pairs:
                    alerts.append(self.generate_alert(
                        f"Alta correlação detectada entre {len(high_corr_pairs)} pares",
                        "medium",
                        {'correlation_pairs': high_corr_pairs}
                    ))
            except Exception as corr_error:
                alerts.append(self.generate_alert(
                    f"Erro na análise de correlação: {str(corr_error)}",
                    "medium",
                    {'error': str(corr_error)}
                ))
            
            # ✅ CORREÇÃO: Detecção de mudanças de regime com verificação
            regime_analysis = self.detect_regime_changes(returns)
            if regime_analysis and regime_analysis.get('vol_regime_change', 0) > 5:
                alerts.append(self.generate_alert(
                    "Múltiplas mudanças de regime de volatilidade detectadas",
                    "medium",
                    regime_analysis
                ))
                
        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro na análise de mercado: {str(e)}",
                "critical",
                {'error': str(e)}
            ))
        
        return alerts

class AgentClustering(AgentBase):
    """Agente para clustering de ativos usando ML - CORRIGIDO"""
    
    def __init__(self, n_clusters: int = 3):  # ✅ Reduzido para ser mais estável
        super().__init__("AgentClustering")
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.current_clusters = None
        
    def prepare_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para clustering - CORRIGIDO"""
        try:
            features = pd.DataFrame(index=returns.index)
            
            # ✅ CORREÇÃO: Features básicas e seguras
            portfolio_returns = returns.mean(axis=1)
            
            # Features de retorno
            features['mean_return'] = portfolio_returns
            features['volatility'] = portfolio_returns.rolling(20).std() * np.sqrt(252)
            features['sharpe'] = features['mean_return'] / (features['volatility'] + 1e-8)
            
            # ✅ CORREÇÃO: Features de correlação simplificadas
            try:
                if len(returns.columns) > 1:
                    # Correlação média com o portfólio
                    rolling_corr = returns.rolling(20).apply(
                        lambda x: x.corrwith(portfolio_returns).mean() if len(x) > 1 else 0,
                        raw=False
                    )
                    features['avg_correlation'] = rolling_corr
                else:
                    features['avg_correlation'] = 1.0
            except:
                features['avg_correlation'] = 1.0
            
            # ✅ CORREÇÃO: Features de drawdown simplificadas
            try:
                cumulative = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                features['current_drawdown'] = drawdown
                features['max_drawdown_30d'] = drawdown.rolling(30).min()
            except:
                features['current_drawdown'] = 0
                features['max_drawdown_30d'] = 0
            
            # ✅ CORREÇÃO: Features de momentum CORRIGIDAS
            try:
                for window in [5, 10, 20]:
                    # ✅ CORREÇÃO CRÍTICA: Calcular momentum para o portfólio, não para cada ativo
                    momentum = cumulative / cumulative.shift(window) - 1
                    features[f'momentum_{window}'] = momentum  # ✅ Agora é uma Series, não DataFrame
            except Exception as momentum_error:
                print(f"Erro no momentum: {momentum_error}")
                for window in [5, 10, 20]:
                    features[f'momentum_{window}'] = 0
            
            return features.dropna()
            
        except Exception as e:
            print(f"Erro crítico em prepare_features: {e}")
            # Retornar DataFrame vazio em caso de erro
            return pd.DataFrame()
    
    def perform_clustering(self, features: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Executa clustering nos ativos - CORRIGIDO"""
        try:
            if len(features) < 10 or len(features.columns) == 0:
                return np.array([]), pd.DataFrame()
            
            # ✅ CORREÇÃO: Verificar e tratar NaN/inf
            features_clean = features.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(features_clean) < 10:
                return np.array([]), pd.DataFrame()
            
            # Normalizar features
            features_scaled = self.scaler.fit_transform(features_clean)
            
            # Redução de dimensionalidade com PCA
            features_pca = self.pca.fit_transform(features_scaled)
            
            # Clustering
            clusters = self.kmeans.fit_predict(features_pca)
            
            # Analisar clusters
            cluster_analysis = pd.DataFrame({
                'timestamp': features_clean.index,
                'cluster': clusters
            })
            
            return clusters, cluster_analysis
            
        except Exception as e:
            print(f"Erro no clustering: {e}")
            return np.array([]), pd.DataFrame()
    
    def analyze_cluster_characteristics(self, features: pd.DataFrame, clusters: np.ndarray) -> Dict[str, Any]:
        """Analisa características de cada cluster - CORRIGIDO"""
        cluster_stats = {}
        
        if len(clusters) == 0:
            return cluster_stats
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = features.iloc[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_stats[cluster_id] = {
                    'size': len(cluster_data),
                    'avg_volatility': float(cluster_data['volatility'].mean()) if 'volatility' in cluster_data else 0,
                    'avg_return': float(cluster_data['mean_return'].mean()) if 'mean_return' in cluster_data else 0,
                    'avg_correlation': float(cluster_data['avg_correlation'].mean()) if 'avg_correlation' in cluster_data else 0
                }
        
        return cluster_stats
    
    def process_data(self, market_data: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Processa clustering de ativos - CORRIGIDO"""
        alerts = []
        
        try:
            returns = market_data.pct_change().dropna()
            
            if len(returns) < 30 or len(returns.columns) < 2:
                alerts.append(self.generate_alert(
                    "Dados insuficientes para clustering",
                    "low",
                    {'data_points': len(returns), 'assets': len(returns.columns)}
                ))
                return alerts
            
            # Preparar features
            features = self.prepare_features(returns)
            
            if features.empty:
                alerts.append(self.generate_alert(
                    "Features vazias para clustering",
                    "low",
                    {}
                ))
                return alerts
            
            # Executar clustering
            clusters, cluster_analysis = self.perform_clustering(features)
            
            if len(clusters) == 0:
                alerts.append(self.generate_alert(
                    "Clustering não produziu resultados",
                    "low",
                    {}
                ))
                return alerts
            
            # Analisar clusters
            cluster_stats = self.analyze_cluster_characteristics(features, clusters)
            
            if not cluster_stats:
                alerts.append(self.generate_alert(
                    "Nenhum cluster significativo encontrado",
                    "low",
                    {}
                ))
                return alerts
            
            # Gerar alertas baseados em clusters
            for cluster_id, stats in cluster_stats.items():
                if stats['avg_volatility'] > 0.35:
                    alerts.append(self.generate_alert(
                        f"Cluster {cluster_id} - Alta volatilidade: {stats['avg_volatility']:.1%}",
                        "medium",
                        {
                            'cluster_id': cluster_id,
                            'volatility': stats['avg_volatility'],
                            'size': stats['size']
                        }
                    ))
            
            alerts.append(self.generate_alert(
                f"Clustering concluído: {len(cluster_stats)} clusters identificados",
                "low",
                {
                    'total_clusters': len(cluster_stats),
                    'cluster_distribution': {k: v['size'] for k, v in cluster_stats.items()}
                }
            ))
                
        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentClustering: {str(e)}",
                "critical",
                {'error': str(e)}
            ))
        
        return alerts

class AgentML(AgentBase):
    """Agente para modelos de Machine Learning - CORRIGIDO"""
    
    def __init__(self):
        super().__init__("AgentML")
        self.models = {}
        self.feature_importance = {}
        
    def create_features(self, returns: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Cria features para modelos preditivos"""
        try:
            features = pd.DataFrame(index=returns.index)
            
            # Features de retorno
            portfolio_returns = returns.mean(axis=1)
            features['returns_lag1'] = portfolio_returns.shift(1)
            features['returns_lag5'] = portfolio_returns.shift(5)
            features['volatility'] = portfolio_returns.rolling(lookback).std()
            
            # Features técnicas
            prices = (1 + returns).cumprod()
            sma_short = prices.rolling(10).mean().mean(axis=1)
            sma_long = prices.rolling(30).mean().mean(axis=1)
            features['sma_ratio'] = sma_short / sma_long
            
            # Features de mercado
            features['market_volatility'] = returns.std(axis=1).rolling(lookback).mean()
            
            # Target: retorno futuro (5 dias)
            features['target'] = portfolio_returns.shift(-5)
            
            return features.dropna()
        except Exception as e:
            print(f"Erro criando features: {e}")
            return pd.DataFrame()
    
    def detect_anomalies(self, returns: pd.DataFrame) -> pd.Series:
        """Detecta anomalias usando Isolation Forest"""
        try:
            features = self.create_features(returns).iloc[:, :-1]  # Excluir target
            
            if len(features) < 50:
                return pd.Series([False] * len(features), index=features.index)
            
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(features)
            
            return pd.Series(anomalies == -1, index=features.index)
        except Exception as e:
            print(f"Erro detectando anomalias: {e}")
            return pd.Series([False] * len(returns), index=returns.index)
    
    def predict_volatility_regime(self, returns: pd.DataFrame) -> pd.Series:
        """Prevê regimes de volatilidade"""
        try:
            features = self.create_features(returns).iloc[:, :-1]
            volatility = returns.std(axis=1).rolling(10).mean()
            
            # Criar target binário: alta vs baixa volatilidade
            vol_threshold = volatility.quantile(0.7)
            target = (volatility > vol_threshold).astype(int)
            
            if len(features) < 100 or target.nunique() < 2:
                return pd.Series([0] * len(features), index=features.index)
            
            # Alinhar features e target
            common_idx = features.index.intersection(target.index)
            features_aligned = features.loc[common_idx]
            target_aligned = target.loc[common_idx]
            
            if len(features_aligned) < 50:
                return pd.Series([0] * len(features), index=features.index)
            
            # Treinar modelo
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(features_aligned.iloc[:-50], target_aligned.iloc[:-50])
            
            # Prever
            predictions = clf.predict(features_aligned.iloc[-50:])
            
            # Guardar importância das features
            self.feature_importance['volatility_regime'] = dict(
                zip(features.columns, clf.feature_importances_)
            )
            
            result_series = pd.Series([0] * len(features), index=features.index)
            result_series.iloc[-50:] = predictions
            return result_series
            
        except Exception as e:
            print(f"Erro prevendo regimes: {e}")
            return pd.Series([0] * len(returns), index=returns.index)
    
    def process_data(self, market_data: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executa análises de ML - CORRIGIDO: removido @delayed"""
        alerts = []
        
        try:
            returns = market_data.pct_change().dropna()
            
            if len(returns) < 100:
                alerts.append(self.generate_alert(
                    "Dados insuficientes para análise de ML",
                    "low",
                    {'data_points': len(returns)}
                ))
                return alerts
            
            # Detecção de anomalias
            anomalies = self.detect_anomalies(returns)
            recent_anomalies = anomalies.tail(10)
            
            if recent_anomalies.any():
                anomaly_count = recent_anomalies.sum()
                alerts.append(self.generate_alert(
                    f"Anomalias detectadas: {anomaly_count} períodos recentes",
                    "high",
                    {
                        'anomaly_count': int(anomaly_count),
                        'total_anomalies': int(anomalies.sum()),
                        'anomaly_rate': float(anomalies.mean())
                    }
                ))
            
            # Previsão de regimes de volatilidade
            vol_predictions = self.predict_volatility_regime(returns)
            high_vol_periods = vol_predictions.sum()
            
            if high_vol_periods > len(vol_predictions) * 0.6:  # Mais de 60% alta vol
                alerts.append(self.generate_alert(
                    f"Período de alta volatilidade previsto: {high_vol_periods}/{len(vol_predictions)} períodos",
                    "medium",
                    {
                        'high_vol_periods': int(high_vol_periods),
                        'total_periods': len(vol_predictions)
                    }
                ))
            
            # Análise de correlação dinâmica
            rolling_corr = returns.rolling(30).corr(returns.mean(axis=1)).mean(axis=1)
            corr_volatility = rolling_corr.std()
            
            if corr_volatility > 0.3:
                alerts.append(self.generate_alert(
                    f"Alta instabilidade nas correlações: {corr_volatility:.3f}",
                    "medium",
                    {
                        'correlation_volatility': float(corr_volatility),
                        'current_correlation': float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else 0
                    }
                ))
            
            alerts.append(self.generate_alert(
                "Análise de ML concluída",
                "low",
                {
                    'total_anomalies_detected': int(anomalies.sum()),
                    'anomaly_rate': float(anomalies.mean())
                }
            ))
                
        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentML: {str(e)}",
                "critical",
                {'error': str(e)}
            ))
        
        return alerts

class AgentSimulation(AgentBase):
    """Agente para simulações avançadas de Monte Carlo e análise de cenários - CORRIGIDO"""
    
    def __init__(self, n_simulations: int = 1000, time_horizon: int = 252):
        super().__init__("AgentSimulation")
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.initial_investment = 10000
        self.simulation_results = {}
    
    def monte_carlo_baseline(self, returns: pd.Series, portfolio_value: float = 10000, 
                           confidence_level: float = 0.95, days: int = 252, 
                           num_simulations: int = 1000) -> Dict[str, Any]:
        """Calcula Value at Risk usando Monte Carlo clássico - CORRIGIDO"""
        try:
            # ✅ CORREÇÃO: Converter Series para numpy array
            returns_array = returns.values
            
            # Estatísticas dos retornos
            mu = np.mean(returns_array)
            sigma = np.std(returns_array)
            
            # Simular retornos futuros
            simulated_returns = np.random.normal(mu, sigma, (days, num_simulations))
            
            # ✅ CORREÇÃO: Calcular valores do portfólio sem usar axis em Series
            portfolio_values = np.zeros((days, num_simulations))
            for i in range(num_simulations):
                cumulative_returns = np.cumsum(simulated_returns[:, i])
                portfolio_values[:, i] = portfolio_value * (1 + cumulative_returns)
            
            # Calcular P&L
            final_values = portfolio_values[-1, :] if days > 1 else portfolio_values[0, :]
            portfolio_changes = final_values - portfolio_value
            
            # Calcular VaR e CVaR
            var = np.percentile(portfolio_changes, (1 - confidence_level) * 100)
            cvar = portfolio_changes[portfolio_changes <= var].mean()
            
            # Calcular probabilidade de prejuízo
            prob_loss = (final_values < portfolio_value).mean()
            
            return {
                'var': float(var),
                'cvar': float(cvar),
                'confidence_level': confidence_level,
                'time_horizon': days,
                'simulations': num_simulations,
                'portfolio_value': portfolio_value,
                'probability_loss': float(prob_loss),
                'expected_return': float(np.mean(final_values) - portfolio_value),
                'final_values': final_values,
                'simulation_paths': portfolio_values,
                'method': 'Monte Carlo Clássico'
            }
            
        except Exception as e:
            print(f"Erro no Monte Carlo VaR: {e}")
            return {}

    def historical_bootstrapping(self, returns: pd.Series, portfolio_value: float = 10000,
                               confidence_level: float = 0.95, days: int = 252,
                               num_simulations: int = 1000) -> Dict[str, Any]:
        """Simulação histórica via bootstrapping - CORRIGIDO"""
        try:
            # ✅ CORREÇÃO: Converter Series para numpy array
            historical_returns = returns.values
            
            if len(historical_returns) == 0:
                return {}
            
            # Bootstrapping
            simulated_paths = np.zeros((days, num_simulations))
            
            for i in range(num_simulations):
                # Amostrar com reposição
                bootstrap_sample = np.random.choice(historical_returns, size=days, replace=True)
                
                # ✅ CORREÇÃO: Calcular caminho de preços sem usar axis
                price_path = np.zeros(days)
                current_value = portfolio_value
                for j in range(days):
                    current_value = current_value * (1 + bootstrap_sample[j])
                    price_path[j] = current_value
                
                simulated_paths[:, i] = price_path
            
            final_values = simulated_paths[-1, :] if days > 0 else simulated_paths[0, :]
            
            # Calcular métricas de risco
            portfolio_changes = final_values - portfolio_value
            var = np.percentile(portfolio_changes, (1 - confidence_level) * 100)
            cvar = portfolio_changes[portfolio_changes <= var].mean()
            prob_loss = (final_values < portfolio_value).mean()
            
            return {
                'var': float(var),
                'cvar': float(cvar),
                'confidence_level': confidence_level,
                'time_horizon': days,
                'portfolio_value': portfolio_value,
                'probability_loss': float(prob_loss),
                'expected_return': float(np.mean(final_values) - portfolio_value),
                'final_values': final_values,
                'simulation_paths': simulated_paths,
                'method': 'Bootstrapping Histórico'
            }
            
        except Exception as e:
            print(f"Erro na simulação histórica: {e}")
            return {}

    def merton_jump_diffusion(self, returns: pd.Series, portfolio_value: float = 10000,
                            days: int = 252, num_simulations: int = 1000) -> Dict[str, Any]:
        """Modelo de Merton com saltos para eventos extremos - CORRIGIDO"""
        try:
            # ✅ CORREÇÃO: Usar array numpy diretamente
            returns_array = returns.values
            
            # Estimar parâmetros
            mu = np.mean(returns_array) * 252
            sigma = np.std(returns_array) * np.sqrt(252)
            
            # Parâmetros de saltos (simplificado)
            lambda_jump = 0.05
            mu_jump = -0.10
            sigma_jump = 0.05
            
            dt = 1/252
            n_steps = days
            
            paths = np.zeros((n_steps + 1, num_simulations))
            paths[0] = portfolio_value
            
            for i in range(num_simulations):
                current_price = portfolio_value
                for t in range(1, n_steps + 1):
                    # Componente Browniano
                    Z = np.random.normal(0, 1)
                    
                    # Componente de saltos
                    N = np.random.poisson(lambda_jump * dt)
                    jumps = np.random.normal(mu_jump, sigma_jump, N).sum()
                    
                    # Atualizar preço
                    current_price = current_price * np.exp(
                        (mu - 0.5 * sigma**2) * dt + 
                        sigma * np.sqrt(dt) * Z + 
                        jumps
                    )
                    paths[t, i] = current_price
            
            final_values = paths[-1, :]
            portfolio_changes = final_values - portfolio_value
            
            # Calcular métricas
            var_95 = np.percentile(portfolio_changes, 5)
            cvar_95 = portfolio_changes[portfolio_changes <= var_95].mean()
            prob_loss = (final_values < portfolio_value).mean()
            
            return {
                'var': float(var_95),
                'cvar': float(cvar_95),
                'portfolio_value': portfolio_value,
                'probability_loss': float(prob_loss),
                'expected_return': float(np.mean(final_values) - portfolio_value),
                'final_values': final_values,
                'simulation_paths': paths,
                'method': 'Merton Jump Diffusion'
            }
            
        except Exception as e:
            print(f"Erro no modelo Merton: {e}")
            # Fallback para Monte Carlo
            return self.monte_carlo_baseline(returns, portfolio_value, 0.95, days, num_simulations)
    
    def garch_simulation(self, returns: pd.Series, portfolio_value: float = 10000,
                        days: int = 252, num_simulations: int = 1000) -> Dict[str, Any]:
        """Simulação GARCH para volatilidade estocástica - CORRIGIDO"""
        try:
            # ✅ CORREÇÃO: Fallback simplificado usando Monte Carlo
            return self.monte_carlo_baseline(returns, portfolio_value, 0.95, days, num_simulations)
            
        except Exception as e:
            print(f"Erro na simulação GARCH: {e}")
            return self.monte_carlo_baseline(returns, portfolio_value, 0.95, days, num_simulations)

    def run_simulation(self, method: str, portfolio_returns: pd.Series, portfolio_config: Dict[str, Any]) -> Dict[str, Any]:
        """Executa simulação baseada no método selecionado - CORRIGIDO"""
        portfolio_value = portfolio_config.get('value', 10000)
        confidence_level = portfolio_config.get('confidence_level', 0.95)
        time_horizon = portfolio_config.get('time_horizon', 252)
        num_simulations = portfolio_config.get('num_simulations', 1000)
        
        if method == "Monte Carlo Clássico":
            return self.monte_carlo_baseline(
                portfolio_returns, portfolio_value, confidence_level, time_horizon, num_simulations
            )
        elif method == "Bootstrapping":
            return self.historical_bootstrapping(
                portfolio_returns, portfolio_value, confidence_level, time_horizon, num_simulations
            )
        elif method == "Merton Jump Diffusion":
            return self.merton_jump_diffusion(
                portfolio_returns, portfolio_value, time_horizon, num_simulations
            )
        elif method == "GARCH":
            return self.garch_simulation(
                portfolio_returns, portfolio_value, time_horizon, num_simulations
            )
        else:
            return self.monte_carlo_baseline(
                portfolio_returns, portfolio_value, confidence_level, time_horizon, num_simulations
            )

    def monte_carlo_simulation(self, returns: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executa simulação de Monte Carlo para o portfólio - CORRIGIDO"""
        try:
            # Se há configuração de portfólio, usar pesos específicos, senão igualmente ponderado
            if portfolio_config and 'weights' in portfolio_config:
                weights = portfolio_config['weights']
                # ✅ CORREÇÃO: Garantir que os pesos estão normalizados e alinhados com os ativos
                weight_series = pd.Series(weights)
                aligned_weights = weight_series.reindex(returns.columns, fill_value=0)
                weighted_returns = returns.mul(aligned_weights, axis=1)
                portfolio_returns = weighted_returns.sum(axis=1)
            else:
                # ✅ CORREÇÃO: Usar mean(axis=1) apenas em DataFrame, não em Series
                portfolio_returns = returns.mean(axis=1)

            # ✅ CORREÇÃO: Converter Series para array numpy
            portfolio_returns_array = portfolio_returns.values

            # Calcular parâmetros para a simulação
            mean_returns = np.mean(portfolio_returns_array)
            std_returns = np.std(portfolio_returns_array)
            
            # Simulação de Monte Carlo
            simulated_paths = np.zeros((self.time_horizon, self.n_simulations))
            for i in range(self.n_simulations):
                # Gerar caminhos usando geometria browniana
                shocks = np.random.normal(0, 1, self.time_horizon)
                returns_path = mean_returns + std_returns * shocks
                # ✅ CORREÇÃO: Usar cumprod corretamente
                cumulative_returns = np.cumprod(1 + returns_path)
                price_path = self.initial_investment * cumulative_returns
                simulated_paths[:, i] = price_path

            # Calcular métricas de risco
            final_prices = simulated_paths[-1, :]
            var_95 = np.percentile(final_prices, 5)
            cvar_95 = final_prices[final_prices <= var_95].mean()
            median_final_price = np.median(final_prices)

            # Calcular probabilidade de prejuízo (preço final < inicial)
            prob_loss = (final_prices < self.initial_investment).mean()

            return {
                'simulated_paths': simulated_paths,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'median_final_price': median_final_price,
                'prob_loss': prob_loss,
                'mean_final_price': final_prices.mean(),
                'std_final_price': final_prices.std(),
                'n_simulations': self.n_simulations,
                'time_horizon': self.time_horizon
            }

        except Exception as e:
            print(f"Erro na simulação de Monte Carlo: {e}")
            return {}

    def process_data(self, market_data: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Processa dados e executa simulações - CORRIGIDO"""
        alerts = []

        try:
            returns = market_data.pct_change().dropna()

            if returns.empty:
                alerts.append(self.generate_alert(
                    "Dados de retornos vazios para simulação",
                    "low",
                    {}
                ))
                return alerts

            # Executar simulação de Monte Carlo
            simulation_results = self.monte_carlo_simulation(returns, portfolio_config)

            if not simulation_results:
                alerts.append(self.generate_alert(
                    "Simulação de Monte Carlo falhou",
                    "medium",
                    {}
                ))
                return alerts

            # Gerar alertas baseados nos resultados da simulação
            if simulation_results['prob_loss'] > 0.2:
                alerts.append(self.generate_alert(
                    f"Alta probabilidade de prejuízo: {simulation_results['prob_loss']:.1%}",
                    "high",
                    simulation_results
                ))

            if simulation_results['var_95'] < self.initial_investment * 0.9:  # Perda > 10%
                loss_percentage = (self.initial_investment - simulation_results['var_95']) / self.initial_investment
                alerts.append(self.generate_alert(
                    f"VaR 95% indica possível perda de {loss_percentage:.1%}",
                    "medium",
                    simulation_results
                ))

            # Alerta para alta volatilidade nas simulações
            if simulation_results['std_final_price'] > self.initial_investment * 0.3:
                alerts.append(self.generate_alert(
                    f"Alta incerteza nas simulações: desvio padrão de {simulation_results['std_final_price']:.0f}",
                    "medium",
                    simulation_results
                ))

            alerts.append(self.generate_alert(
                f"Simulação de Monte Carlo concluída: {self.n_simulations} simulações",
                "low",
                simulation_results
            ))

        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentSimulation: {str(e)}",
                "critical",
                {'error': str(e)}
            ))

        return alerts
            
class AgentAlert(AgentBase):
    """Centraliza e prioriza alertas - CORRIGIDO"""
    
    def __init__(self):
        super().__init__("AgentAlert")
        self.alert_history = []
        self.severity_weights = {'critical': 10, 'high': 6, 'medium': 3, 'low': 1}
    
    def calculate_alert_score(self, alert: Dict[str, Any]) -> float:
        base_score = self.severity_weights.get(alert['severity'], 1)
        
        if 'data' in alert:
            # Bonus por múltiplos ativos envolvidos
            if 'assets' in alert['data'] and len(alert['data']['assets']) > 3:
                base_score *= 1.3
            
            # Bonus por alta volatilidade
            if 'volatility' in alert['data'] and alert['data']['volatility'] > 0.4:
                base_score *= 1.5
        
        return base_score
    
    def process_alerts(self, alerts_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Processa alertas - CORRIGIDO: removido @delayed"""
        all_alerts = []
        for alerts in alerts_list:
            if alerts is not None:  # ✅ Proteção contra None
                all_alerts.extend(alerts)
        
        for alert in all_alerts:
            alert['score'] = self.calculate_alert_score(alert)
        
        sorted_alerts = sorted(all_alerts, key=lambda x: x['score'], reverse=True)
        self.alert_history.extend(sorted_alerts)
        
        # Manter apenas últimos 1000 alertas no histórico
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        return sorted_alerts[:15]

class AgentLSTM(AgentBase):
    """Previsão temporal via MLP com janela deslizante (proxy LSTM sem TF).
    Aprende padrões sequenciais nos retornos e projeta os próximos `horizon` dias."""

    def __init__(self, window: int = 20, horizon: int = 5):
        super().__init__("AgentLSTM")
        self.window   = window
        self.horizon  = horizon
        self.lstm_results: Dict[str, Any] = {}

    def _make_sequences(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.window):
            X.append(series[i : i + self.window])
            y.append(series[i + self.window])
        return np.array(X), np.array(y)

    def _fit_predict_asset(self, ret: pd.Series) -> Dict[str, Any]:
        from sklearn.neural_network import MLPRegressor

        r = ret.dropna().values
        if len(r) < self.window + 30:
            return {}

        scaler = StandardScaler()
        r_sc   = scaler.fit_transform(r.reshape(-1, 1)).flatten()

        X, y   = self._make_sequences(r_sc)
        split  = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        model.fit(X_tr, y_tr)

        y_pred_sc = model.predict(X_te)
        mae = float(np.mean(np.abs(y_pred_sc - y_te)))

        # Forecast horizon dias para frente
        window_cur = r_sc[-self.window :].copy()
        forecast_sc = []
        for _ in range(self.horizon):
            nxt = float(model.predict(window_cur.reshape(1, -1))[0])
            forecast_sc.append(nxt)
            window_cur = np.append(window_cur[1:], nxt)

        forecast = scaler.inverse_transform(
            np.array(forecast_sc).reshape(-1, 1)
        ).flatten().tolist()
        y_pred = scaler.inverse_transform(
            y_pred_sc.reshape(-1, 1)
        ).flatten().tolist()
        actual = r[-len(y_te) :].tolist()

        trend_avg = float(np.mean(forecast))
        if trend_avg > 0.002:
            trend = "alta"
        elif trend_avg < -0.002:
            trend = "baixa"
        else:
            trend = "lateral"

        return {
            "forecast":   forecast,
            "trend":      trend,
            "trend_avg":  trend_avg,
            "mae":        mae,
            "actual":     actual,
            "predicted":  y_pred,
            "test_size":  len(y_te),
        }

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        try:
            # Aceita preços (base-100) ou retornos diretamente
            if market_data.max().max() > 10:
                returns = market_data.pct_change().dropna()
            else:
                returns = market_data.dropna()

            self.lstm_results = {}
            for asset in returns.columns:
                res = self._fit_predict_asset(returns[asset])
                if not res:
                    continue
                self.lstm_results[asset] = res

                if res["trend"] == "baixa" and res["trend_avg"] < -0.003:
                    alerts.append(self.generate_alert(
                        f"LSTM prevê baixa para {asset}: {res['trend_avg']:.2%}/dia "
                        f"nos próximos {self.horizon} dias",
                        "high",
                        {"asset": asset, "trend": res["trend"], "trend_avg": res["trend_avg"],
                         "forecast": res["forecast"]},
                    ))
                elif res["trend"] == "alta" and res["trend_avg"] > 0.003:
                    alerts.append(self.generate_alert(
                        f"LSTM prevê alta para {asset}: {res['trend_avg']:.2%}/dia "
                        f"nos próximos {self.horizon} dias",
                        "low",
                        {"asset": asset, "trend": res["trend"], "trend_avg": res["trend_avg"],
                         "forecast": res["forecast"]},
                    ))

            n = len(self.lstm_results)
            if n == 0:
                alerts.append(self.generate_alert(
                    "LSTM: dados insuficientes (mínimo 50 dias por ativo)", "medium", {}
                ))
            else:
                alerts.append(self.generate_alert(
                    f"LSTM: {n} ativo(s) analisados — horizonte {self.horizon} dias",
                    "low",
                    {"assets_analyzed": list(self.lstm_results.keys())},
                ))
        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentLSTM: {str(e)}", "critical", {"error": str(e)}
            ))
        return alerts


class AgentAutoencoder(AgentBase):
    """Detecção de anomalias via Autoencoder MLP com gargalo (bottleneck).
    Treina para reconstruir retornos normais; alto erro de reconstrução = anomalia."""

    def __init__(self):
        super().__init__("AgentAutoencoder")
        self.autoencoder_results: Dict[str, Any] = {}

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from sklearn.neural_network import MLPRegressor

        alerts = []
        try:
            if market_data.max().max() > 10:
                returns = market_data.pct_change().dropna()
            else:
                returns = market_data.dropna()

            returns = returns.dropna()
            n_samples, n_assets = returns.shape

            if n_samples < 30:
                alerts.append(self.generate_alert(
                    "Autoencoder: mínimo 30 observações necessárias", "medium", {}
                ))
                return alerts

            if n_assets < 2:
                alerts.append(self.generate_alert(
                    "Autoencoder: mínimo 2 ativos para análise multivariada", "medium", {}
                ))
                return alerts

            # Escalar
            scaler  = StandardScaler()
            X       = scaler.fit_transform(returns.values)

            split   = int(n_samples * 0.8)
            X_train = X[:split]

            # Bottleneck = max(2, n_assets//2)
            bottleneck = max(2, n_assets // 2)

            model = MLPRegressor(
                hidden_layer_sizes=(bottleneck,),
                activation="tanh",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            model.fit(X_train, X_train)   # target = input → aprende compressão

            X_recon = model.predict(X)
            errors  = np.mean((X - X_recon) ** 2, axis=1)   # MSE por passo de tempo

            train_err  = errors[:split]
            threshold  = float(np.mean(train_err) + 2.5 * np.std(train_err))
            is_anomaly = errors > threshold

            # Contribuição por ativo (MSE individual)
            errors_by_asset = {
                col: float(np.mean((X[:, i] - X_recon[:, i]) ** 2))
                for i, col in enumerate(returns.columns)
            }
            mean_asset_err = np.mean(list(errors_by_asset.values()))
            high_err_assets = [
                a for a, e in errors_by_asset.items() if e > 2 * mean_asset_err
            ]

            recent_anomalies = int(is_anomaly[-5:].sum())
            n_anomalies      = int(is_anomaly.sum())

            self.autoencoder_results = {
                "errors":          errors.tolist(),
                "dates":           [str(d) for d in returns.index],
                "threshold":       threshold,
                "is_anomaly":      is_anomaly.tolist(),
                "n_anomalies":     n_anomalies,
                "recent_anomalies": recent_anomalies,
                "errors_by_asset": errors_by_asset,
                "contamination":   float(n_anomalies / n_samples),
                "assets":          returns.columns.tolist(),
            }

            if recent_anomalies >= 3:
                alerts.append(self.generate_alert(
                    f"Autoencoder: {recent_anomalies} anomalia(s) nos últimos 5 dias — padrão incomum",
                    "high",
                    self.autoencoder_results,
                ))
            elif recent_anomalies > 0:
                alerts.append(self.generate_alert(
                    f"Autoencoder: {recent_anomalies} anomalia(s) recente(s) detectada(s)",
                    "medium",
                    self.autoencoder_results,
                ))

            if high_err_assets:
                alerts.append(self.generate_alert(
                    f"Autoencoder: padrão atípico em {', '.join(high_err_assets)}",
                    "medium",
                    {"assets": high_err_assets, "errors": errors_by_asset},
                ))

            alerts.append(self.generate_alert(
                f"Autoencoder: {n_anomalies}/{n_samples} anomalia(s) — "
                f"taxa {n_anomalies/n_samples:.1%} | bottleneck={bottleneck}d",
                "low",
                self.autoencoder_results,
            ))

        except Exception as e:
            alerts.append(self.generate_alert(
                f"Erro no AgentAutoencoder: {str(e)}", "critical", {"error": str(e)}
            ))
        return alerts

class AgentFundamental(AgentBase):
    """Análise de múltiplos fundamentais (P/E, P/B, EV/EBITDA, DCF) via yfinance."""

    def __init__(self):
        super().__init__("AgentFundamental")
        self.fundamental_results: Dict[str, Any] = {}

    def _get_row(self, df: pd.DataFrame, keywords: List[str]) -> Optional[float]:
        """Busca valor numa linha do DF pelo nome (tentativa exata, depois parcial)."""
        if df is None or df.empty:
            return None
        kws = [k.lower() for k in keywords]
        # exact match first
        for idx in df.index:
            if str(idx).lower() == " ".join(kws):
                vals = df.loc[idx].dropna().values
                return float(vals[0]) if len(vals) > 0 else None
        # partial match — all keywords must appear
        for idx in df.index:
            idx_s = str(idx).lower()
            if all(k in idx_s for k in kws):
                vals = df.loc[idx].dropna().values
                return float(vals[0]) if len(vals) > 0 else None
        # fallback — any keyword
        for k in kws:
            for idx in df.index:
                if k in str(idx).lower():
                    vals = df.loc[idx].dropna().values
                    return float(vals[0]) if len(vals) > 0 else None
        return None

    def _fetch(self, ticker: str) -> Optional[Dict]:
        result: Dict[str, Any] = {
            "info": {}, "fin": pd.DataFrame(), "bs": pd.DataFrame(), "cf": pd.DataFrame(), "cvm": None
        }
        # CVM Dados Abertos — fonte primária para ativos B3
        if ticker.endswith(".SA"):
            try:
                from src.etl.cvm_fundamentals import get_fundamental_summary
                cvm = get_fundamental_summary(ticker)
                if cvm:
                    result["cvm"] = cvm
            except Exception:
                pass
        # yfinance — market data e múltiplos de valuation
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            result["info"] = info
            if info.get("quoteType") != "CRYPTOCURRENCY":
                result["fin"] = self._safe(t.financials)
                result["bs"]  = self._safe(t.balance_sheet)
                result["cf"]  = self._safe(t.cashflow)
        except Exception:
            pass
        return result if (result["info"] or result["cvm"]) else None

    def _safe(self, obj) -> pd.DataFrame:
        try:
            return obj if isinstance(obj, pd.DataFrame) and not obj.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _compute(self, ticker: str, data: Dict) -> Dict:
        info, fin, bs, cf, cvm = (
            data["info"], data["fin"], data["bs"], data["cf"], data.get("cvm")
        )
        g = self._get_row

        res: Dict[str, Any] = {
            "ticker":   ticker,
            "name":     info.get("shortName") or info.get("longName", ticker),
            "sector":   info.get("sector", "N/D"),
            "industry": info.get("industry", "N/D"),
            "data_source": "CVM+yfinance" if cvm else "yfinance",
            # valuation (market data — yfinance é a fonte correta)
            "pe_ratio":         info.get("trailingPE") or info.get("forwardPE"),
            "pb_ratio":         info.get("priceToBook"),
            "ev_ebitda":        info.get("enterpriseToEbitda"),
            "market_cap":       info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "current_price":    info.get("currentPrice") or info.get("regularMarketPrice"),
            "dividend_yield":   info.get("dividendYield"),
        }

        if cvm:
            # Dados contábeis via CVM — cobertura total B3
            res["revenue_growth_yoy"] = cvm.get("crescimento_receita_yoy")
            res["revenue_cagr_3y"]    = cvm.get("cagr_receita_3a")
            res["ebitda"]             = cvm.get("ebitda")
            res["ebitda_margin"]      = cvm.get("margem_ebitda")
            res["ebitda_margin_trend"]= cvm.get("margem_ebitda_trend")
            res["fcf"]                = cvm.get("fcf")
            res["net_margin"]         = cvm.get("margem_liquida")
            res["roe"]                = cvm.get("roe")
            res["roa"]                = cvm.get("roa")
        else:
            # Fallback yfinance para não-B3 ou quando CVM falha
            rev_row = None
            if fin is not None and not fin.empty:
                for idx in fin.index:
                    if "revenue" in str(idx).lower():
                        vals = fin.loc[idx].dropna().values
                        if len(vals) >= 2:
                            rev_row = vals
                        break
            if rev_row is not None and rev_row[1] != 0:
                res["revenue_growth_yoy"] = (rev_row[0] - rev_row[1]) / abs(rev_row[1])
            else:
                res["revenue_growth_yoy"] = None
            res["revenue_cagr_3y"] = None

            ebitda = g(fin, ["EBITDA"])
            if ebitda is None:
                ebit = g(fin, ["EBIT"]) or g(fin, ["Operating Income"])
                da   = g(cf, ["Depreciation"]) or g(fin, ["Reconciled Depreciation"])
                if ebit is not None and da is not None:
                    ebitda = ebit + abs(da)
            res["ebitda"] = ebitda

            revenue = g(fin, ["Total Revenue"]) or g(fin, ["Revenue"])
            res["ebitda_margin"] = (ebitda / revenue) if (ebitda and revenue and revenue != 0) else None
            res["ebitda_margin_trend"] = None

            fcf = g(cf, ["Free Cash Flow"])
            if fcf is None:
                ocf   = g(cf, ["Operating Cash Flow"]) or g(cf, ["Total Cash From Operating Activities"])
                capex = g(cf, ["Capital Expenditures"]) or g(cf, ["Capital Expenditure"])
                if ocf is not None and capex is not None:
                    fcf = ocf - abs(capex)
            res["fcf"] = fcf
            res["net_margin"] = None
            res["roe"] = res["roa"] = None

        # DCF (Gordon Growth Model) — WACC 12% BR, g terminal 4%
        WACC, g_term = 0.12, 0.04
        mc  = res["market_cap"]
        fcf = res["fcf"]
        if fcf and fcf > 0 and mc and mc > 0:
            dcf_val = fcf * (1 + g_term) / (WACC - g_term)
            res["dcf_value"]  = dcf_val
            res["dcf_upside"] = (dcf_val - mc) / mc
        else:
            res["dcf_value"]  = None
            res["dcf_upside"] = None

        return res

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.fundamental_results = {}

        skip = ("^", "BRL", "=X", "USDBRL")
        tickers = [c for c in market_data.columns if not any(c.startswith(p) for p in skip)]

        for ticker in tickers:
            try:
                raw = self._fetch(ticker)
                if not raw or not raw["info"]:
                    continue
                r = self._compute(ticker, raw)
                self.fundamental_results[ticker] = r

                pe       = r.get("pe_ratio")
                ev_ebitda = r.get("ev_ebitda")
                dcf_up   = r.get("dcf_upside")
                rev_g    = r.get("revenue_growth_yoy")

                if pe and pe > 40:
                    alerts.append(self.generate_alert(
                        f"{ticker}: P/E elevado ({pe:.1f}x) — possível sobrevalorização",
                        "medium", {"ticker": ticker, "pe_ratio": pe}))
                if ev_ebitda and ev_ebitda > 15:
                    alerts.append(self.generate_alert(
                        f"{ticker}: EV/EBITDA elevado ({ev_ebitda:.1f}x)",
                        "medium", {"ticker": ticker, "ev_ebitda": ev_ebitda}))
                if dcf_up is not None and dcf_up < -0.30:
                    alerts.append(self.generate_alert(
                        f"{ticker}: DCF sugere sobrevalorização de {dcf_up:.1%}",
                        "high", {"ticker": ticker, "dcf_upside": dcf_up}))
                if rev_g is not None and rev_g < -0.10:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Queda de receita YoY de {rev_g:.1%}",
                        "high", {"ticker": ticker, "revenue_growth_yoy": rev_g}))
            except Exception as e:
                alerts.append(self.generate_alert(
                    f"Erro em AgentFundamental para {ticker}: {e}", "low",
                    {"ticker": ticker, "error": str(e)}))

        n = len(self.fundamental_results)
        alerts.append(self.generate_alert(
            f"AgentFundamental: {n} ativo(s) analisado(s)" if n else
            "AgentFundamental: sem dados — cobertura limitada para B3 small/mid caps",
            "low" if n else "medium",
            {"tickers": list(self.fundamental_results.keys())}))
        return alerts


class AgentCredit(AgentBase):
    """Score proprietário de crédito: Dívida Liq./EBITDA, ICR, Liquidez, FCF Yield."""

    def __init__(self):
        super().__init__("AgentCredit")
        self.credit_results: Dict[str, Any] = {}

    def _get_row(self, df: pd.DataFrame, keywords: List[str]) -> Optional[float]:
        if df is None or df.empty:
            return None
        kws = [k.lower() for k in keywords]
        for idx in df.index:
            idx_s = str(idx).lower()
            if all(k in idx_s for k in kws):
                vals = df.loc[idx].dropna().values
                return float(vals[0]) if len(vals) > 0 else None
        for k in kws:
            for idx in df.index:
                if k in str(idx).lower():
                    vals = df.loc[idx].dropna().values
                    return float(vals[0]) if len(vals) > 0 else None
        return None

    def _safe(self, obj) -> pd.DataFrame:
        try:
            return obj if isinstance(obj, pd.DataFrame) and not obj.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _compute_credit(self, ticker: str) -> Optional[Dict]:
        try:
            info: Dict[str, Any] = {}
            mc: Optional[float] = None

            # yfinance — market cap e informações de setor
            try:
                import yfinance as yf
                t    = yf.Ticker(ticker)
                info = t.info or {}
                mc   = info.get("marketCap")
            except Exception:
                pass

            res: Dict[str, Any] = {
                "ticker": ticker,
                "name":   info.get("shortName", ticker),
                "sector": info.get("sector", "N/D"),
                "data_source": "yfinance",
            }

            # CVM — fonte primária para ativos B3
            if ticker.endswith(".SA"):
                try:
                    from src.etl.cvm_fundamentals import get_fundamental_summary
                    cvm = get_fundamental_summary(ticker)
                    if cvm:
                        res["data_source"]     = "CVM+yfinance"
                        res["net_debt"]        = cvm.get("divida_liquida")
                        res["net_debt_ebitda"] = cvm.get("net_debt_ebitda")
                        res["interest_coverage"]= cvm.get("interest_coverage")
                        res["current_ratio"]   = cvm.get("current_ratio")
                        res["fcf"]             = cvm.get("fcf")
                        res["fcf_yield"]       = cvm.get("fcf_yield") if mc else None
                        # score e rating calculados abaixo com os valores já preenchidos
                        self._fill_credit_score(res)
                        return res
                except Exception:
                    pass

            # Fallback yfinance (não-B3 ou falha CVM)
            try:
                import yfinance as yf
                t    = yf.Ticker(ticker)
                bs   = self._safe(t.balance_sheet)
                cf   = self._safe(t.cashflow)
                fin  = self._safe(t.financials)
            except Exception:
                return res
            g = self._get_row

            # Dívida total e caixa
            total_debt = g(bs, ["Total Debt"]) or (
                (g(bs, ["Long Term Debt"]) or 0) + (g(bs, ["Short Long Term Debt"]) or 0) or None)
            cash = g(bs, ["Cash And Cash Equivalents"]) or g(bs, ["Cash"])

            # EBITDA
            ebitda = g(fin, ["EBITDA"])
            if ebitda is None:
                ebit = g(fin, ["EBIT"]) or g(fin, ["Operating Income"])
                da   = g(cf, ["Depreciation"])
                if ebit is not None and da is not None:
                    ebitda = ebit + abs(da)

            # Dívida Líquida / EBITDA
            if total_debt is not None and cash is not None and ebitda and ebitda != 0:
                net_debt = total_debt - cash
                res["net_debt"]        = net_debt
                res["net_debt_ebitda"] = net_debt / ebitda
            else:
                res["net_debt"]        = None
                res["net_debt_ebitda"] = None

            # ICR: EBIT / Despesa Financeira
            ebit     = g(fin, ["EBIT"]) or g(fin, ["Operating Income"])
            int_exp  = g(fin, ["Interest Expense"])
            if ebit is not None and int_exp and int_exp != 0:
                res["interest_coverage"] = ebit / abs(int_exp)
            else:
                res["interest_coverage"] = None

            # Liquidez Corrente
            cur_assets = g(bs, ["Current Assets"]) or g(bs, ["Total Current Assets"])
            cur_liab   = g(bs, ["Current Liabilities"]) or g(bs, ["Total Current Liabilities"])
            res["current_ratio"] = (cur_assets / cur_liab) if (cur_assets and cur_liab and cur_liab != 0) else None

            # FCF Yield
            ocf   = g(cf, ["Operating Cash Flow"]) or g(cf, ["Total Cash From Operating Activities"])
            capex = g(cf, ["Capital Expenditures"]) or g(cf, ["Capital Expenditure"])
            if ocf and capex is not None and mc and mc > 0:
                fcf = ocf - abs(capex)
                res["fcf"]       = fcf
                res["fcf_yield"] = fcf / mc
            else:
                res["fcf"]       = None
                res["fcf_yield"] = None

            self._fill_credit_score(res)
            return res

        except Exception as e:
            return {"ticker": ticker, "error": str(e), "credit_score": None, "credit_rating": "N/D"}

    def _fill_credit_score(self, res: Dict[str, Any]) -> None:
        """Calcula credit_score e credit_rating in-place a partir dos campos de res."""
        score = 50
        details: Dict[str, int] = {}

        nd = res.get("net_debt_ebitda")
        if nd is not None:
            s = 25 if nd < 1 else 15 if nd < 2 else 5 if nd < 3 else 0 if nd < 4 else -20
            score += s; details["net_debt_ebitda"] = s

        icr = res.get("interest_coverage")
        if icr is not None:
            s = 20 if icr > 5 else 10 if icr > 3 else 0 if icr > 1.5 else -20
            score += s; details["interest_coverage"] = s

        cr = res.get("current_ratio")
        if cr is not None:
            s = 10 if cr > 2 else 5 if cr > 1.5 else 0 if cr > 1 else -15
            score += s; details["current_ratio"] = s

        fy = res.get("fcf_yield")
        if fy is not None:
            s = 10 if fy > 0.08 else 5 if fy > 0.04 else 0 if fy > 0 else -15
            score += s; details["fcf_yield"] = s

        score = max(0, min(100, score))
        res["credit_score"]  = score
        res["score_details"] = details
        res["credit_rating"] = (
            "BAIXO RISCO"    if score >= 70 else
            "RISCO MODERADO" if score >= 50 else
            "RISCO ELEVADO"  if score >= 30 else
            "ALTO RISCO"
        )

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.credit_results = {}

        skip = ("^", "BRL", "=X", "USDBRL")
        tickers = [c for c in market_data.columns if not any(c.startswith(p) for p in skip)]

        for ticker in tickers:
            try:
                r = self._compute_credit(ticker)
                if not r:
                    continue
                self.credit_results[ticker] = r

                score     = r.get("credit_score")
                nd_ebitda = r.get("net_debt_ebitda")
                icr       = r.get("interest_coverage")
                cr        = r.get("current_ratio")

                if score is not None:
                    if score < 30:
                        alerts.append(self.generate_alert(
                            f"{ticker}: ALTO RISCO de crédito — score {score:.0f}/100",
                            "critical", r))
                    elif score < 50:
                        alerts.append(self.generate_alert(
                            f"{ticker}: Risco elevado de crédito — score {score:.0f}/100",
                            "high", r))

                if nd_ebitda is not None and nd_ebitda > 4:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Alavancagem crítica — Dív. Líq./EBITDA = {nd_ebitda:.1f}x",
                        "critical", {"ticker": ticker, "net_debt_ebitda": nd_ebitda}))
                if icr is not None and icr < 1.5:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Cobertura de juros insuficiente — ICR = {icr:.1f}x",
                        "high", {"ticker": ticker, "interest_coverage": icr}))
                if cr is not None and cr < 1.0:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Liquidez corrente abaixo de 1,0 ({cr:.2f}x)",
                        "high", {"ticker": ticker, "current_ratio": cr}))

            except Exception as e:
                alerts.append(self.generate_alert(
                    f"Erro em AgentCredit para {ticker}: {e}", "low",
                    {"ticker": ticker, "error": str(e)}))

        scored = [v for v in self.credit_results.values() if v.get("credit_score") is not None]
        if scored:
            avg = np.mean([v["credit_score"] for v in scored])
            sev = "high" if avg < 40 else "medium" if avg < 60 else "low"
            alerts.append(self.generate_alert(
                f"AgentCredit: {len(scored)} ativo(s), score médio {avg:.0f}/100",
                sev, {"avg_credit_score": avg, "tickers": list(self.credit_results.keys())}))
        else:
            alerts.append(self.generate_alert(
                "AgentCredit: sem dados de balanço disponíveis via yfinance",
                "medium", {}))
        return alerts


class AgentDividend(AgentBase):
    """DY histórico, payout ratio, consistência e crescimento de dividendos via yfinance."""

    def __init__(self):
        super().__init__("AgentDividend")
        self.dividend_results: Dict[str, Any] = {}

    def _safe(self, obj) -> pd.DataFrame:
        try:
            return obj if isinstance(obj, pd.DataFrame) and not obj.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.dividend_results = {}

        skip = ("^", "BRL", "=X", "USDBRL")
        tickers = [c for c in market_data.columns if not any(c.startswith(p) for p in skip)]

        for ticker in tickers:
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                info = t.info or {}
                divs = self._safe(t.dividends)

                res: Dict[str, Any] = {
                    "ticker":        ticker,
                    "name":          info.get("shortName", ticker),
                    "sector":        info.get("sector", "N/D"),
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "payout_ratio":  info.get("payoutRatio"),
                }

                if isinstance(divs, pd.Series) and len(divs) > 0:
                    divs.index = pd.to_datetime(divs.index).tz_localize(None)
                    now = pd.Timestamp.now()

                    # Trailing 12 meses
                    t12 = divs[divs.index >= now - pd.DateOffset(years=1)].sum()
                    price = res["current_price"]
                    res["trailing_dividends_1y"] = float(t12)
                    res["dy_trailing"] = float(t12 / price) if (price and price > 0) else None

                    # Dividendos anuais
                    divs_annual = divs.resample("YE").sum()
                    res["div_history_annual"] = divs_annual.to_dict()

                    # CAGR 5 anos
                    years = min(5, len(divs_annual) - 1)
                    if years >= 1 and divs_annual.iloc[-years - 1] > 0:
                        res["dividend_cagr_5y"] = float(
                            (divs_annual.iloc[-1] / divs_annual.iloc[-years - 1]) ** (1 / years) - 1)
                    else:
                        res["dividend_cagr_5y"] = None

                    # Consistência: % dos últimos 5 anos com pagamento
                    d5y = divs[divs.index >= now - pd.DateOffset(years=5)]
                    years_paid = len(d5y.resample("YE").sum().loc[lambda s: s > 0])
                    total_y = max(1, min(5, len(divs_annual)))
                    res["consistency_score"] = years_paid / total_y
                    res["years_paying"] = years_paid
                else:
                    res.update({
                        "trailing_dividends_1y": 0.0, "dy_trailing": 0.0,
                        "dividend_cagr_5y": None, "consistency_score": 0.0,
                        "years_paying": 0, "div_history_annual": {},
                    })

                self.dividend_results[ticker] = res

                dy   = res.get("dy_trailing") or 0.0
                cagr = res.get("dividend_cagr_5y")
                cons = res.get("consistency_score", 0.0)

                if dy > 0.10:
                    alerts.append(self.generate_alert(
                        f"{ticker}: DY muito elevado ({dy:.1%}) — possível dividend trap",
                        "medium", {"ticker": ticker, "dy": dy}))
                if 0 < dy and cons < 0.5:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Histórico de dividendos inconsistente ({cons:.0%} dos anos)",
                        "high", {"ticker": ticker, "consistency_score": cons}))
                if cagr is not None and cagr < -0.10:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Crescimento negativo de dividendos 5Y ({cagr:.1%}/ano)",
                        "medium", {"ticker": ticker, "dividend_cagr_5y": cagr}))

            except Exception as e:
                alerts.append(self.generate_alert(
                    f"Erro em AgentDividend para {ticker}: {e}", "low",
                    {"ticker": ticker, "error": str(e)}))

        n = len(self.dividend_results)
        alerts.append(self.generate_alert(
            f"AgentDividend: {n} ativo(s) analisado(s)", "low",
            {"tickers": list(self.dividend_results.keys())}))
        return alerts


class AgentPeerComparison(AgentBase):
    """Comparação de múltiplos entre ativos do mesmo setor (ranking relativo + z-score)."""

    def __init__(self):
        super().__init__("AgentPeerComparison")
        self.peer_results: Dict[str, Any] = {}

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.peer_results = {}

        skip = ("^", "BRL", "=X", "USDBRL")
        tickers = [c for c in market_data.columns if not any(c.startswith(p) for p in skip)]

        # Coletar setor e múltiplos
        ticker_info: Dict[str, Dict] = {}
        for ticker in tickers:
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info or {}
                ticker_info[ticker] = {
                    "name":      info.get("shortName", ticker),
                    "sector":    info.get("sector") or "Sem setor",
                    "pe":        info.get("trailingPE") or info.get("forwardPE"),
                    "pb":        info.get("priceToBook"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "dy":        info.get("dividendYield"),
                    "mkt_cap":   info.get("marketCap"),
                }
            except Exception:
                ticker_info[ticker] = {"name": ticker, "sector": "Sem setor"}

        # Agrupar por setor
        sectors: Dict[str, List[str]] = {}
        for tk, d in ticker_info.items():
            sectors.setdefault(d["sector"], []).append(tk)

        comparison: Dict[str, Any] = {}
        METRICS = ["pe", "pb", "ev_ebitda", "dy"]

        for sector, members in sectors.items():
            if len(members) < 2:
                # Setor com apenas 1 ativo — sem comparação possível
                comparison[sector] = {"members": members, "solo": True}
                continue

            comp: Dict[str, Any] = {"members": members, "solo": False}
            for metric in METRICS:
                vals = {tk: ticker_info[tk][metric] for tk in members
                        if ticker_info[tk].get(metric) is not None}
                if len(vals) < 2:
                    continue

                arr = np.array(list(vals.values()))
                mean, std = float(np.mean(arr)), float(np.std(arr))

                rankings: Dict[str, Any] = {}
                for tk, v in vals.items():
                    z = (v - mean) / std if std > 0 else 0.0
                    pct = float(np.mean(arr <= v))
                    rankings[tk] = {"value": float(v), "zscore": float(z), "pct_rank": pct}

                    if abs(z) > 2.0:
                        direction = "acima" if z > 0 else "abaixo"
                        lbl = {"pe": "P/E", "pb": "P/B", "ev_ebitda": "EV/EBITDA", "dy": "DY"}.get(metric, metric)
                        alerts.append(self.generate_alert(
                            f"{tk} [{sector}]: {lbl} {direction} da média setorial "
                            f"({v:.2f} vs média {mean:.2f}, z={z:.1f})",
                            "medium",
                            {"ticker": tk, "metric": metric, "value": v, "zscore": z, "sector": sector}))

                comp[metric] = {"rankings": rankings, "sector_mean": mean, "sector_std": std}
            comparison[sector] = comp

        self.peer_results = {
            "ticker_info": ticker_info,
            "sectors": sectors,
            "comparison": comparison,
        }
        n_sectors = len([s for s, d in comparison.items() if not d.get("solo")])
        alerts.append(self.generate_alert(
            f"AgentPeerComparison: {len(sectors)} setor(es), {n_sectors} com comparação ativa",
            "low", {"sectors": list(sectors.keys())}))
        return alerts


class AgentMacroSensitivity(AgentBase):
    """Beta rolling a CDI, IPCA e câmbio via regressão OLS sobre retornos históricos."""

    BCB_SERIES = {"CDI": 12, "CAMBIO": 1, "IPCA": 433}

    def __init__(self):
        super().__init__("AgentMacroSensitivity")
        self.macro_sensitivity_results: Dict[str, Any] = {}

    def _fetch_bcb(self, code: int, n: int = 756) -> Optional[pd.Series]:
        try:
            import requests
            url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}"
                   f"/dados/ultimos/{n}?formato=json")
            data = requests.get(url, timeout=10).json()
            s = pd.Series({
                pd.to_datetime(d["data"], dayfirst=True): float(d["valor"].replace(",", "."))
                for d in data
            }).sort_index()
            return s
        except Exception:
            return None

    def _ols_beta(self, y: np.ndarray, x: np.ndarray) -> float:
        if len(y) < 5 or np.std(x) == 0:
            return 0.0
        return float(np.cov(y, x)[0, 1] / np.var(x))

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.macro_sensitivity_results = {}

        # 1. Obter séries macro (portfolio_config > BCB)
        macro: Dict[str, pd.Series] = {}
        pc_macro = (portfolio_config or {}).get("macro_df")
        if isinstance(pc_macro, pd.DataFrame) and not pc_macro.empty:
            for col in pc_macro.columns:
                macro[col] = pc_macro[col].dropna()

        # Buscar o que faltar diretamente no BCB
        for name, code in self.BCB_SERIES.items():
            if name not in macro:
                s = self._fetch_bcb(code)
                if s is not None:
                    macro[name] = s

        if not macro:
            alerts.append(self.generate_alert(
                "AgentMacroSensitivity: sem dados macro (BCB indisponível)", "medium", {}))
            return alerts

        # 2. Retornos dos ativos
        skip = ("^", "BRL", "=X", "USDBRL")
        asset_cols = [c for c in market_data.columns if not any(c.startswith(p) for p in skip)]
        asset_ret = market_data[asset_cols].pct_change().dropna()

        WINDOW = 63
        results: Dict[str, Any] = {"factors": list(macro.keys()), "assets": {}}

        for factor, series in macro.items():
            # CDI/IPCA são taxas — pct_change; câmbio já é preço — pct_change também
            factor_ret = series.pct_change().dropna()
            factor_ret.index = pd.to_datetime(factor_ret.index).tz_localize(None)

            common = asset_ret.index.intersection(factor_ret.index)
            if len(common) < WINDOW:
                continue

            ar = asset_ret.loc[common]
            fr = factor_ret.loc[common]

            for ticker in ar.columns:
                results["assets"].setdefault(ticker, {})
                y = ar[ticker].values
                x = fr.values

                beta_full    = self._ols_beta(y, x)
                beta_rolling = self._ols_beta(y[-WINDOW:], x[-WINDOW:]) if len(y) >= WINDOW else beta_full
                r2 = float(np.corrcoef(y, x)[0, 1] ** 2) if (np.std(y) > 0 and np.std(x) > 0) else 0.0

                results["assets"][ticker][factor] = {
                    "beta_full":       round(beta_full, 4),
                    "beta_rolling_63d": round(beta_rolling, 4),
                    "r2":              round(r2, 4),
                }

                if factor == "CAMBIO" and abs(beta_rolling) > 0.5:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Alta exposição cambial — beta CÂMBIO {beta_rolling:.2f}",
                        "medium", {"ticker": ticker, "beta_cambio": beta_rolling}))
                if factor == "CDI" and beta_rolling < -0.8:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Alta sensibilidade negativa a juros — beta CDI {beta_rolling:.2f}",
                        "medium", {"ticker": ticker, "beta_cdi": beta_rolling}))

        self.macro_sensitivity_results = results
        alerts.append(self.generate_alert(
            f"AgentMacroSensitivity: {len(results['assets'])} ativo(s), fatores {list(macro.keys())}",
            "low", {"n_assets": len(results["assets"]), "factors": list(macro.keys())}))
        return alerts


class AgentScenario(AgentBase):
    """Stress test: SELIC+300bps, IPCA+4%, câmbio+20% — impacto estimado no portfólio."""

    SCENARIOS: Dict[str, Dict[str, float]] = {
        "SELIC +300bps":  {"CDI": 0.03,  "CAMBIO": 0.00, "IPCA": 0.00},
        "IPCA +4% a.a.":  {"CDI": 0.00,  "CAMBIO": 0.00, "IPCA": 0.04},
        "Câmbio +20%":    {"CDI": 0.00,  "CAMBIO": 0.20, "IPCA": 0.00},
        "Combinado":      {"CDI": 0.03,  "CAMBIO": 0.20, "IPCA": 0.04},
    }

    def __init__(self):
        super().__init__("AgentScenario")
        self.scenario_results: Dict[str, Any] = {}

    def _estimate_betas(self, returns: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Estima betas a CDI, CÂMBIO e IPCA com proxies históricas."""
        betas: Dict[str, Dict[str, float]] = {}
        # Proxy de câmbio: coluna FX no market_data (se existir)
        fx_col = next((c for c in market_data.columns
                       if any(k in c.upper() for k in ("USD", "BRL", "CAMBIO", "FX"))), None)

        for ticker in returns.columns:
            betas[ticker] = {}
            ann_vol = float(returns[ticker].std() * np.sqrt(252))

            # Beta CÂMBIO via correlação histórica com coluna FX
            if fx_col and fx_col in market_data.columns:
                fx_ret = market_data[fx_col].pct_change().dropna()
                common = returns[ticker].dropna().index.intersection(fx_ret.index)
                if len(common) > 60:
                    y = returns[ticker].loc[common].values
                    x = fx_ret.loc[common].values
                    betas[ticker]["CAMBIO"] = float(np.cov(y, x)[0, 1] / np.var(x)) if np.var(x) > 0 else 0.0
                else:
                    betas[ticker]["CAMBIO"] = 0.0
            else:
                betas[ticker]["CAMBIO"] = 0.0

            # Beta CDI: proxy negativo para ativos de baixa vol (mais "bond-like")
            # Alta vol → equity-like → menor sensibilidade a taxa
            betas[ticker]["CDI"] = float(-0.5 * max(0.0, 1.0 - ann_vol / 0.30))

            # Beta IPCA: simplificado (0 — sem dados suficientes para regressão precisa)
            betas[ticker]["IPCA"] = 0.0

        return betas

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.scenario_results = {}

        skip = ("^", "BRL", "=X", "USDBRL")
        asset_cols = [c for c in market_data.columns if not any(c.startswith(p) for p in skip)]
        returns = market_data[asset_cols].pct_change().dropna()

        if len(returns) < 30 or not asset_cols:
            alerts.append(self.generate_alert(
                "AgentScenario: dados insuficientes para stress test", "low", {}))
            return alerts

        # Pesos (equiponderado se não informado)
        weights_cfg = (portfolio_config or {}).get("weights") or {}
        w = {tk: weights_cfg.get(tk, 1.0 / len(asset_cols)) for tk in asset_cols}
        w_sum = sum(w.values())
        weights = {k: v / w_sum for k, v in w.items()}

        portfolio_value = float((portfolio_config or {}).get("value", 100_000))

        # Estimar betas
        asset_betas = self._estimate_betas(returns, market_data)

        # Calcular impacto por cenário
        scenarios_out: Dict[str, Any] = {}
        for name, shocks in self.SCENARIOS.items():
            asset_impacts: Dict[str, float] = {}
            for tk in asset_cols:
                impact = sum(asset_betas[tk].get(f, 0.0) * shock
                             for f, shock in shocks.items())
                asset_impacts[tk] = float(impact)

            port_impact = sum(weights[tk] * asset_impacts[tk] for tk in asset_cols)
            port_brl    = portfolio_value * port_impact
            severity    = ("SEVERO"   if port_impact < -0.10 else
                           "MODERADO" if port_impact < -0.05 else
                           "LEVE"     if port_impact < 0     else "POSITIVO")

            scenarios_out[name] = {
                "shocks":              shocks,
                "asset_impacts":       asset_impacts,
                "portfolio_impact_pct": round(port_impact, 6),
                "portfolio_impact_brl": round(port_brl, 2),
                "severity":            severity,
            }

            sev_map = {"SEVERO": "critical", "MODERADO": "high", "LEVE": "medium", "POSITIVO": "low"}
            if port_impact < -0.05:
                alerts.append(self.generate_alert(
                    f"Stress '{name}': impacto {port_impact:.1%} no portfólio — {severity}",
                    sev_map[severity],
                    {"scenario": name, "impact_pct": port_impact, "impact_brl": port_brl}))

        self.scenario_results = {
            "scenarios": scenarios_out,
            "weights": weights,
            "asset_betas": asset_betas,
            "portfolio_value": portfolio_value,
        }

        worst_name, worst = min(scenarios_out.items(),
                                key=lambda x: x[1]["portfolio_impact_pct"])
        alerts.append(self.generate_alert(
            f"AgentScenario concluído — pior cenário: '{worst_name}' → "
            f"{worst['portfolio_impact_pct']:.1%} ({worst['severity']})",
            "medium" if worst["portfolio_impact_pct"] < -0.05 else "low",
            {"worst_scenario": worst_name, **worst}))
        return alerts


class AgentCVM(AgentBase):
    """Análise de séries temporais contábeis via CVM Dados Abertos.

    Cobre 100% das empresas de capital aberto B3 (incluindo small/mid caps).
    Detecta tendências: CAGR de receita, deterioração de margem, alavancagem crescente.
    Requer ativos com sufixo .SA — ignora silenciosamente os demais.
    """

    def __init__(self):
        super().__init__("AgentCVM")
        self.cvm_results: Dict[str, Any] = {}

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.cvm_results = {}

        b3_tickers = [c for c in market_data.columns if c.endswith(".SA")]
        if not b3_tickers:
            alerts.append(self.generate_alert(
                "AgentCVM: nenhum ativo B3 (.SA) no portfólio — agente ignorado", "low", {}))
            return alerts

        try:
            from src.etl.cvm_fundamentals import get_fundamental_summary
        except ImportError:
            alerts.append(self.generate_alert(
                "AgentCVM: módulo cvm_fundamentals não encontrado", "medium", {}))
            return alerts

        for ticker in b3_tickers:
            try:
                summary = get_fundamental_summary(ticker, years=3)
                if not summary:
                    continue
                self.cvm_results[ticker] = summary

                cagr = summary.get("cagr_receita_3a")
                if cagr is not None and cagr < -0.05:
                    alerts.append(self.generate_alert(
                        f"{ticker}: CAGR receita 3a = {cagr:.1%} — tendência de queda",
                        "high", {"ticker": ticker, "cagr_receita_3a": cagr}))

                trend = summary.get("margem_ebitda_trend")
                if trend is not None and trend < -0.05:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Margem EBITDA deteriorando {trend:+.1%} nos últimos 3 anos",
                        "high", {"ticker": ticker, "margem_ebitda_trend": trend}))

                nd_ebitda = summary.get("net_debt_ebitda")
                if nd_ebitda is not None and nd_ebitda > 3.5:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Alavancagem elevada — Dív. Líq./EBITDA = {nd_ebitda:.1f}x (CVM)",
                        "high", {"ticker": ticker, "net_debt_ebitda": nd_ebitda}))

                icr = summary.get("interest_coverage")
                if icr is not None and icr < 2.0:
                    alerts.append(self.generate_alert(
                        f"{ticker}: Cobertura de juros baixa — ICR = {icr:.1f}x (CVM)",
                        "medium", {"ticker": ticker, "interest_coverage": icr}))

            except Exception as e:
                alerts.append(self.generate_alert(
                    f"AgentCVM: erro em {ticker}: {e}", "low",
                    {"ticker": ticker, "error": str(e)}))

        n = len(self.cvm_results)
        alerts.append(self.generate_alert(
            f"AgentCVM: {n} ativo(s) B3 analisado(s) via CVM Dados Abertos" if n else
            "AgentCVM: nenhum dado CVM disponível para os ativos do portfólio",
            "low", {"tickers": list(self.cvm_results.keys())}))
        return alerts


class AgentScreener(AgentBase):
    """Ranking composto de ativos: fundamental + crédito + dividendos.

    Combina resultados de AgentFundamental, AgentCredit e AgentDividend em um
    score único (0–100) e gera recomendação COMPRA / NEUTRO / EVITAR.

    Deve ser executado APÓS os três agentes acima — recebe os resultados
    via portfolio_config['fundamental_results'], ['credit_results'] e ['dividend_results'].
    """

    def __init__(self):
        super().__init__("AgentScreener")
        self.screener_results: Dict[str, Any] = {}

    def process_data(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alerts = []
        self.screener_results = {}

        cfg            = portfolio_config or {}
        fund_results   = cfg.get("fundamental_results", {})
        credit_results = cfg.get("credit_results", {})
        div_results    = cfg.get("dividend_results", {})

        all_tickers = set(fund_results) | set(credit_results) | set(div_results)
        if not all_tickers:
            alerts.append(self.generate_alert(
                "AgentScreener: sem dados — execute AgentFundamental, AgentCredit e AgentDividend primeiro",
                "medium", {}))
            return alerts

        scores = []
        for ticker in sorted(all_tickers):
            fund   = fund_results.get(ticker, {})
            credit = credit_results.get(ticker, {})
            div    = div_results.get(ticker, {})

            weighted_score = 0.0
            total_weight   = 0.0
            components: Dict[str, float] = {}

            # ── Fundamental (40%) ────────────────────────────────────────────
            if fund:
                fs = 50.0
                pe = fund.get("pe_ratio")
                ev = fund.get("ev_ebitda")
                rg = fund.get("revenue_growth_yoy")
                em = fund.get("ebitda_margin")
                du = fund.get("dcf_upside")

                if pe  is not None: fs += 15 if pe < 12  else 8 if pe < 20  else 0 if pe < 30  else -10 if pe < 50 else -20
                if ev  is not None: fs += 15 if ev < 6   else 8 if ev < 10  else 0 if ev < 15  else -10
                if rg  is not None: fs += 10 if rg > .15 else 5 if rg > .05 else 0 if rg >= 0  else -10
                if em  is not None: fs += 10 if em > .30 else 5 if em > .20 else 0 if em > .10 else -10
                if du  is not None: fs += 15 if du > .30 else 8 if du > .10 else 0 if du >= 0  else -10

                fs = max(0.0, min(100.0, fs))
                components["fundamental"] = round(fs, 1)
                weighted_score += fs * 0.40
                total_weight   += 0.40

            # ── Crédito (40%) ────────────────────────────────────────────────
            cs = credit.get("credit_score")
            if cs is not None:
                components["credit"] = float(cs)
                weighted_score += cs * 0.40
                total_weight   += 0.40

            # ── Dividendos (20%) ─────────────────────────────────────────────
            if div:
                ds = 50.0
                dy   = div.get("dividend_yield_avg")
                cagr = div.get("dividend_cagr_5y")
                cons = div.get("consistency_score")  # 0–1

                if dy   is not None: ds += 20 if dy   > .08 else 10 if dy   > .05 else 5 if dy > .03 else 0
                if cagr is not None: ds += 15 if cagr > .10 else 8  if cagr > .05 else 0 if cagr >= 0 else -10
                if cons is not None: ds += cons * 15

                ds = max(0.0, min(100.0, ds))
                components["dividend"] = round(ds, 1)
                weighted_score += ds * 0.20
                total_weight   += 0.20

            if total_weight == 0:
                continue

            composite = round(weighted_score / total_weight, 1)
            recommendation = "COMPRA" if composite >= 65 else "NEUTRO" if composite >= 45 else "EVITAR"

            entry = {
                "ticker":              ticker,
                "name":                fund.get("name") or credit.get("name", ticker),
                "sector":              fund.get("sector") or credit.get("sector", "N/D"),
                "composite_score":     composite,
                "recommendation":      recommendation,
                "score_components":    components,
                "pe_ratio":            fund.get("pe_ratio"),
                "ev_ebitda":           fund.get("ev_ebitda"),
                "ebitda_margin":       fund.get("ebitda_margin"),
                "revenue_growth_yoy":  fund.get("revenue_growth_yoy"),
                "dcf_upside":          fund.get("dcf_upside"),
                "credit_score":        cs,
                "credit_rating":       credit.get("credit_rating", "N/D"),
                "dividend_yield":      div.get("dividend_yield_avg"),
                "dividend_cagr_5y":    div.get("dividend_cagr_5y"),
            }
            scores.append(entry)

        scores.sort(key=lambda x: x["composite_score"], reverse=True)
        self.screener_results = {s["ticker"]: s for s in scores}

        top   = [s for s in scores if s["recommendation"] == "COMPRA"]
        avoid = [s for s in scores if s["recommendation"] == "EVITAR"]

        if top:
            top_str = ", ".join(f"{s['ticker']} ({s['composite_score']:.0f})" for s in top[:3])
            alerts.append(self.generate_alert(
                f"AgentScreener — Top picks: {top_str}",
                "low", {"top_picks": [s["ticker"] for s in top]}))

        if avoid:
            av_str = ", ".join(f"{s['ticker']} ({s['composite_score']:.0f})" for s in avoid[:3])
            alerts.append(self.generate_alert(
                f"AgentScreener — Evitar: {av_str}",
                "high", {"avoid_list": [s["ticker"] for s in avoid]}))

        alerts.append(self.generate_alert(
            f"AgentScreener: {len(scores)} ativo(s) ranqueados",
            "low", {"total": len(scores), "ranking": [s["ticker"] for s in scores]}))
        return alerts


class DaskMultiAgentOrchestrator:
    """Orquestrador principal usando Dask - CORRIGIDO"""
    
    def __init__(self, use_dask: bool = False, use_tensorflow: bool = False):
        self.use_dask = use_dask
        self.use_tensorflow = use_tensorflow
        self.client = None

        # Inicializar todos os agentes
        self.agent_market = AgentMarket()
        self.agent_clustering = AgentClustering()
        self.agent_ml = AgentML()
        self.agent_simulation = AgentSimulation()
        self.agent_alert = AgentAlert()
        self.agent_lstm = AgentLSTM()
        self.agent_autoencoder = AgentAutoencoder()
        self.agent_fundamental = AgentFundamental()
        self.agent_credit      = AgentCredit()
        self.agent_dividend    = AgentDividend()
        self.agent_peer        = AgentPeerComparison()
        self.agent_macro       = AgentMacroSensitivity()
        self.agent_scenario    = AgentScenario()
        self.agent_cvm         = AgentCVM()
        self.agent_screener    = AgentScreener()

        if use_dask:
            try:
                from dask.distributed import Client

                self.client = Client(
                    n_workers=2,
                    threads_per_worker=2,
                    processes=False,
                    memory_limit='2GB',
                    silence_logs=30
                )
                print("✅ Dask cluster inicializado")
            except Exception as e:
                print(f"⚠️ Dask falhou: {e}, usando modo sequencial")
                self.use_dask = False
            
    def run_analysis(self, market_data: pd.DataFrame,
                     portfolio_config: Optional[Dict[str, Any]] = None,
                     enabled_agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Executa análise coordenada. enabled_agents=None roda todos."""
        print(f"🔍 Executando análise multiagente (agentes: {enabled_agents or 'todos'})...")
        if self.use_dask and self.client:
            return self._run_dask_parallel(market_data, portfolio_config)
        return self._run_sequential(market_data, portfolio_config, enabled_agents)
    
    def _run_dask_parallel(self, market_data: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None):
        """Executa agentes em paralelo usando Dask - CORRIGIDO"""
        try:
            # ✅ CORREÇÃO: Usar delayed explicitamente
            market_task = delayed(self.agent_market.process_data)(market_data)
            clustering_task = delayed(self.agent_clustering.process_data)(market_data)
            ml_task = delayed(self.agent_ml.process_data)(market_data)
            simulation_task = delayed(self.agent_simulation.process_data)(market_data, portfolio_config)  # ✅ NOVO
            
            # Adicionar agentes TensorFlow se disponíveis
            tasks = [market_task, clustering_task, ml_task, simulation_task]  # ✅ ATUALIZADO: inclui simulation
            task_names = ["Market", "Clustering", "ML", "Simulation"]  # ✅ ATUALIZADO
            
            if hasattr(self, 'agent_lstm'):
                lstm_task = delayed(self.agent_lstm.process_data)(market_data)
                tasks.append(lstm_task)
                task_names.append("LSTM")
            
            if hasattr(self, 'agent_autoencoder'):
                autoencoder_task = delayed(self.agent_autoencoder.process_data)(market_data)
                tasks.append(autoencoder_task)
                task_names.append("Autoencoder")
            
            print(f"   📊 Tasks Dask criadas: {task_names}")
            
            # Computar em paralelo
            alerts_results = compute(*tasks, scheduler=self.client)
            print(f"   ✅ Tasks Dask concluídas")
            
            # Processar alertas
            alerts_task = delayed(self.agent_alert.process_alerts)(alerts_results)
            prioritized_alerts = compute(alerts_task, scheduler=self.client)[0]
            
            return prioritized_alerts
            
        except Exception as e:
            print(f"⚠️ Dask paralelo falhou: {e}, usando sequencial")
            return self._run_sequential(market_data, portfolio_config)
    
    def _run_sequential(self, market_data: pd.DataFrame,
                         portfolio_config: Optional[Dict[str, Any]] = None,
                         enabled_agents: Optional[List[str]] = None):
        """Executa agentes sequencialmente. enabled_agents=None roda todos."""
        # None = todos habilitados; lista vazia = nenhum
        def _enabled(name: str) -> bool:
            return enabled_agents is None or name in enabled_agents

        alerts_results = []
        try:
            if _enabled("AgentMarket"):
                print("   🔄 AgentMarket...")
                alerts_results.append(self.agent_market.process_data(market_data))

            if _enabled("AgentClustering"):
                print("   🔄 AgentClustering...")
                alerts_results.append(self.agent_clustering.process_data(market_data))

            if _enabled("AgentML"):
                print("   🔄 AgentML...")
                alerts_results.append(self.agent_ml.process_data(market_data))

            if _enabled("AgentSimulation"):
                print("   🔄 AgentSimulation...")
                alerts_results.append(self.agent_simulation.process_data(market_data, portfolio_config))

            if _enabled("AgentLSTM"):
                print("   🔄 AgentLSTM...")
                alerts_results.append(self.agent_lstm.process_data(market_data))

            if _enabled("AgentAutoencoder"):
                print("   🔄 AgentAutoencoder...")
                alerts_results.append(self.agent_autoencoder.process_data(market_data))

            if _enabled("AgentFundamental"):
                print("   🔄 AgentFundamental...")
                alerts_results.append(self.agent_fundamental.process_data(market_data, portfolio_config))

            if _enabled("AgentCredit"):
                print("   🔄 AgentCredit...")
                alerts_results.append(self.agent_credit.process_data(market_data, portfolio_config))

            if _enabled("AgentDividend"):
                print("   🔄 AgentDividend...")
                alerts_results.append(self.agent_dividend.process_data(market_data, portfolio_config))

            if _enabled("AgentPeerComparison"):
                print("   🔄 AgentPeerComparison...")
                alerts_results.append(self.agent_peer.process_data(market_data, portfolio_config))

            if _enabled("AgentMacroSensitivity"):
                print("   🔄 AgentMacroSensitivity...")
                alerts_results.append(self.agent_macro.process_data(market_data, portfolio_config))

            if _enabled("AgentScenario"):
                print("   🔄 AgentScenario...")
                alerts_results.append(self.agent_scenario.process_data(market_data, portfolio_config))

            if _enabled("AgentCVM"):
                print("   🔄 AgentCVM...")
                alerts_results.append(self.agent_cvm.process_data(market_data, portfolio_config))

            if _enabled("AgentScreener"):
                print("   🔄 AgentScreener...")
                screener_cfg = {
                    **(portfolio_config or {}),
                    "fundamental_results": self.agent_fundamental.fundamental_results,
                    "credit_results":      self.agent_credit.credit_results,
                    "dividend_results":    self.agent_dividend.dividend_results,
                }
                alerts_results.append(self.agent_screener.process_data(market_data, screener_cfg))

            print(f"   ✅ {len(alerts_results)} agente(s) executados")
            
        except Exception as e:
            print(f"   ❌ Erro na execução sequencial: {e}")
            # Adicionar alerta de erro
            error_alert = self.agent_market.generate_alert(
                f"Erro na execução dos agentes: {str(e)}", 
                "critical", 
                {'error': str(e)}
            )
            alerts_results = [[error_alert]]
        
        # Processar alertas
        prioritized_alerts = self.agent_alert.process_alerts(alerts_results)
        return prioritized_alerts
    
    def generate_report(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera relatório consolidado - ATUALIZADA"""
        severity_counts = {}
        asset_alerts = {}
        cluster_alerts = []
        ml_alerts = []
        simulation_alerts = []
        
        # ✅ COLETAR TODOS OS ALERTAS
        all_alerts = []
        
        for alert in alerts:
            all_alerts.append(alert)
            sev = alert['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            # Agrupar por tipo de análise
            if 'AgentClustering' in alert['agent_id']:
                cluster_alerts.append(alert)
            elif any(agent in alert['agent_id'] for agent in ['AgentML', 'AgentLSTM', 'AgentAutoencoder']):
                ml_alerts.append(alert)
            elif 'AgentSimulation' in alert['agent_id']:
                simulation_alerts.append(alert)
            
            if 'data' in alert and 'asset' in alert['data']:
                asset = alert['data']['asset']
                if asset not in asset_alerts:
                    asset_alerts[asset] = []
                asset_alerts[asset].append(alert)
        
        critical_count = len([a for a in alerts if a['severity'] in ['critical', 'high']])
        summary = f"{critical_count} alertas de alta prioridade" if critical_count > 0 else "Nenhum alerta crítico"
        
        return {
            'total_alerts': len(alerts),
            'severity_breakdown': severity_counts,
            'asset_alerts': asset_alerts,
            'cluster_alerts': cluster_alerts,
            'ml_alerts': ml_alerts,
            'simulation_alerts': simulation_alerts,
            'critical_alerts': [a for a in alerts if a['severity'] in ['critical', 'high']],
            'all_alerts': all_alerts,  # ✅ NOVO: TODOS OS ALERTAS
            'timestamp': datetime.now(),
            'summary': summary,
            'orchestrator': 'Dask' if self.use_dask else 'Sequencial'
        }    
    
    def __del__(self):
        """Cleanup do cliente Dask"""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except:
                pass