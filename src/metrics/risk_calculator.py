import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

class RiskCalculator:
    def __init__(self, returns_df, risk_free_rate=0.1175):  # SELIC atual ~11.75%
        """
        Inicializa o calculador de risco para dados macroeconômicos
        
        Parameters:
        returns_df: DataFrame com retornos dos ativos macro
        risk_free_rate: Taxa livre de risco anualizada
        """
        self.returns = returns_df
        self.prices = (1 + returns_df).cumprod() * 100  # Preços normalizados
        self.risk_free_rate = risk_free_rate / 100
        self.n_days = len(returns_df)
        
        # Carregar categorias se disponível
        try:
            with open('data/processed/portfolio_categories.json', 'r') as f:
                self.categories = json.load(f)
        except:
            self.categories = None
    
    def basic_risk_metrics(self):
        """Calcula métricas básicas de risco e retorno para cada ativo"""
        print("📊 CALCULANDO MÉTRICAS BÁSICAS DE RISCO...")
        
        metrics_list = []
        
        for asset in self.returns.columns:
            returns = self.returns[asset].dropna()
            
            if len(returns) < 10:  # Mínimo de dados
                continue
                
            # Métricas de retorno
            total_return = (1 + returns).prod() - 1
            annual_return = returns.mean() * 252
            cagr = (1 + total_return) ** (252/len(returns)) - 1
            
            # Métricas de risco
            annual_volatility = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
            
            # Ratios
            sharpe = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            sortino = (annual_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Risk metrics
            var_95 = self.var_historical(returns)
            cvar_95 = self.cvar_historical(returns)
            max_dd = self.max_drawdown(returns)
            ulcer_index = self.ulcer_index(returns)
            
            # Estatísticas
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            metrics_list.append({
                'Ativo': asset,
                'Retorno_Total': total_return,
                'Retorno_Anual': annual_return,
                'CAGR': cagr,
                'Volatilidade_Anual': annual_volatility,
                'Volatilidade_Baixa': downside_volatility,
                'Sharpe': sharpe,
                'Sortino': sortino,
                'VaR_95%': var_95,
                'CVaR_95%': cvar_95,
                'Max_Drawdown': max_dd,
                'Ulcer_Index': ulcer_index,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Ratio_Sharpe_Sortino': sharpe/sortino if sortino != 0 else 0
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df.set_index('Ativo')
    
    def portfolio_analysis(self, weights=None):
        """Análise de portfólio com pesos específicos"""
        print("🎯 ANALISANDO PORTFÓLIO...")
        
        if weights is None:
            # Pesos igualmente distribuídos
            weights = np.array([1/len(self.returns.columns)] * len(self.returns.columns))
        else:
            weights = np.array(weights)
        
        # Garantir que pesos somem 1
        weights = weights / weights.sum()
        
        # Retorno do portfólio
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Métricas do portfólio
        portfolio_stats = self._calculate_portfolio_stats(portfolio_returns, weights)
        
        return portfolio_stats, portfolio_returns
    
    def _calculate_portfolio_stats(self, portfolio_returns, weights):
        """Calcula estatísticas detalhadas do portfólio"""
        # Retorno e risco
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Ratios
        sharpe = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Risk metrics
        var_95 = self.var_historical(portfolio_returns)
        cvar_95 = self.cvar_historical(portfolio_returns)
        max_dd = self.max_drawdown(portfolio_returns)
        
        # Decomposição do risco
        risk_decomp = self.risk_decomposition(weights)
        
        # Diversificação
        diversification_ratio = self.diversification_ratio(weights)
        
        stats = {
            'Retorno_Total': total_return,
            'Retorno_Anual': annual_return,
            'Volatilidade_Anual': annual_volatility,
            'Sharpe_Ratio': sharpe,
            'VaR_95%': var_95,
            'CVaR_95%': cvar_95,
            'Max_Drawdown': max_dd,
            'Diversification_Ratio': diversification_ratio,
            'Risk_Decomposition': risk_decomp,
            'Weights': dict(zip(self.returns.columns, weights))
        }
        
        return stats
    
    def risk_decomposition(self, weights):
        """Decomposição do risco por ativo"""
        cov_matrix = self.returns.cov() * 252
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Contribuição marginal ao risco
        marginal_risk = (cov_matrix @ weights) / portfolio_volatility
        risk_contribution = weights * marginal_risk
        percent_contribution = risk_contribution / portfolio_volatility
        
        decomposition = pd.DataFrame({
            'Peso': weights,
            'Contribuição_Risco_Absoluta': risk_contribution,
            'Contribuição_Risco_Percentual': percent_contribution
        }, index=self.returns.columns)
        
        return decomposition
    
    def diversification_ratio(self, weights):
        """Calcula ratio de diversificação"""
        individual_volatilities = self.returns.std() * np.sqrt(252)
        weighted_vol = (weights * individual_volatilities).sum()
        portfolio_vol = np.sqrt(weights.T @ (self.returns.cov() * 252) @ weights)
        
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1
    
    def correlation_analysis(self):
        """Análise completa de correlações"""
        print("🔗 ANALISANDO CORRELAÇÕES...")
        
        corr_matrix = self.returns.corr()
        
        # Análise de clusters de correlação
        correlation_insights = {
            'matrix': corr_matrix,
            'high_correlation_pairs': self._find_high_correlations(corr_matrix),
            'correlation_statistics': self._correlation_stats(corr_matrix)
        }
        
        return correlation_insights
    
    def _find_high_correlations(self, corr_matrix, threshold=0.7):
        """Encontra pares com alta correlação"""
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'Ativo_1': corr_matrix.columns[i],
                        'Ativo_2': corr_matrix.columns[j],
                        'Correlação': corr_matrix.iloc[i, j]
                    })
        return high_corr_pairs
    
    def _correlation_stats(self, corr_matrix):
        """Estatísticas das correlações"""
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        return {
            'Correlação_Média': corr_values.mean(),
            'Correlação_Máxima': corr_values.max(),
            'Correlação_Mínima': corr_values.min(),
            'Perc_Positivas': (corr_values > 0).mean(),
            'Perc_Alta_Correlação': (abs(corr_values) > 0.7).mean()
        }
    
    # Métricas de risco individuais
    def var_historical(self, returns, alpha=0.05):
        """Value at Risk histórico"""
        return np.percentile(returns, alpha * 100)
    
    def cvar_historical(self, returns, alpha=0.05):
        """Conditional Value at Risk (Expected Shortfall)"""
        var = self.var_historical(returns, alpha)
        return returns[returns <= var].mean()
    
    def max_drawdown(self, returns):
        """Calcula o drawdown máximo"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def ulcer_index(self, returns, window=14):
        """Índice de Úlcera - mede profundidade e duração de drawdowns"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return np.sqrt((drawdown ** 2).mean())
    
    def rolling_risk_metrics(self, window=63):
        """Métricas de risco móveis (3 meses)"""
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(252)
        rolling_var = self.returns.rolling(window).apply(
            lambda x: np.percentile(x, 5) if len(x) == window else np.nan
        )
        
        return {
            'Rolling_Volatility': rolling_vol,
            'Rolling_VaR': rolling_var
        }
    
    def generate_comprehensive_report(self):
        """Gera relatório completo de risco"""
        print("📋 GERANDO RELATÓRIO COMPLETO DE RISCO...")
        print("=" * 70)
        
        report = {}
        
        # 1. Métricas básicas por ativo
        print("\n1. 📈 MÉTRICAS DE RISCO POR ATIVO:")
        basic_metrics = self.basic_risk_metrics()
        report['basic_metrics'] = basic_metrics
        print(basic_metrics.round(4))
        
        # 2. Análise de portfólio equal-weight
        print("\n2. 🎯 PORTFÓLIO EQUAL-WEIGHT:")
        portfolio_stats, portfolio_returns = self.portfolio_analysis()
        report['portfolio_stats'] = portfolio_stats
        
        for key, value in portfolio_stats.items():
            if key not in ['Risk_Decomposition', 'Weights']:
                if isinstance(value, float):
                    print(f"   {key:25}: {value:8.4%}")
                else:
                    print(f"   {key:25}: {value}")
        
        # 3. Decomposição do risco
        print("\n3. 📊 DECOMPOSIÇÃO DO RISCO:")
        risk_decomp = portfolio_stats['Risk_Decomposition']
        print(risk_decomp.round(4))
        
        # 4. Análise de correlações
        print("\n4. 🔗 ANÁLISE DE CORRELAÇÕES:")
        corr_analysis = self.correlation_analysis()
        report['correlation_analysis'] = corr_analysis
        
        print("   📈 Estatísticas de correlação:")
        for key, value in corr_analysis['correlation_statistics'].items():
            print(f"     {key:25}: {value:8.4f}")
        
        if corr_analysis['high_correlation_pairs']:
            print("\n   🔥 Pares com alta correlação (>0.7):")
            for pair in corr_analysis['high_correlation_pairs'][:5]:  # Mostrar apenas os 5 primeiros
                print(f"     • {pair['Ativo_1']:15} ↔ {pair['Ativo_2']:15}: {pair['Correlação']:.3f}")
        
        # 5. Métricas móveis
        print("\n5. 📅 MÉTRICAS DE RISCO MÓVEIS (3 meses):")
        rolling_metrics = self.rolling_risk_metrics()
        report['rolling_metrics'] = rolling_metrics
        print("   ✅ Volatilidade e VaR móveis calculados")
        
        print("\n" + "=" * 70)
        print("🎯 RELATÓRIO DE RISCO CONCLUÍDO!")
        
        return report

# Funções auxiliares
def load_macro_data():
    """Carrega os dados macroeconômicos"""
    try:
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        print(f"✅ Dados carregados: {returns.shape}")
        print(f"📊 Ativos: {list(returns.columns)}")
        return returns
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None

def main():
    """Função principal"""
    print("🚀 CALCULADOR DE RISCO - DADOS MACROECONÔMICOS")
    print("=" * 70)
    
    # Carregar dados
    returns = load_macro_data()
    
    if returns is not None and not returns.empty:
        # Inicializar calculador
        risk_calc = RiskCalculator(returns)
        
        # Gerar relatório completo
        report = risk_calc.generate_comprehensive_report()
        
        # Salvar relatório
        try:
            report_df = report['basic_metrics']
            report_df.to_csv('data/processed/risk_report.csv')
            report_df.to_parquet('data/processed/risk_report.parquet')
            print("💾 Relatório salvo em data/processed/risk_report.*")
        except Exception as e:
            print(f"⚠️  Erro ao salvar relatório: {e}")
        
    else:
        print("❌ Não foi possível carregar os dados para análise")

if __name__ == "__main__":
    main()