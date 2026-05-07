# src/agents/agent_simulation.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
import sys
import os

# ✅ CORREÇÃO: Mesma importação do AgentBase
try:
    from .agent_base import AgentBase
except ImportError:
    # Fallback para desenvolvimento
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from agents.agent_base import AgentBase

warnings.filterwarnings('ignore')

class AgentSimulation(AgentBase):
    """Agente para simulações de Monte Carlo e análise de cenários - CORRIGIDO"""

    def __init__(self, n_simulations: int = 1000, time_horizon: int = 252):
        super().__init__("AgentSimulation")
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.initial_investment = 10000
        self.simulation_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Métodos de simulação individuais (chamados por run_simulation)
    # ------------------------------------------------------------------

    def monte_carlo_baseline(self, returns: pd.Series, portfolio_value: float = 10000,
                             confidence_level: float = 0.95, days: int = 252,
                             num_simulations: int = 1000) -> Dict[str, Any]:
        """Monte Carlo clássico com distribuição normal."""
        try:
            r = np.asarray(returns)
            mu, sigma = r.mean(), r.std()
            shocks = np.random.normal(mu, sigma, (days, num_simulations))
            paths = portfolio_value * np.cumprod(1 + shocks, axis=0)
            final = paths[-1]
            changes = final - portfolio_value
            var = np.percentile(changes, (1 - confidence_level) * 100)
            tail = changes[changes <= var]
            cvar = float(tail.mean()) if len(tail) > 0 else float(var)
            return {
                'var': float(var), 'cvar': cvar,
                'confidence_level': confidence_level, 'time_horizon': days,
                'simulations': num_simulations, 'portfolio_value': portfolio_value,
                'probability_loss': float((final < portfolio_value).mean()),
                'expected_return': float(final.mean() - portfolio_value),
                'final_values': final, 'simulation_paths': paths,
                'method': 'Monte Carlo Clássico'
            }
        except Exception as e:
            print(f"Erro em monte_carlo_baseline: {e}")
            return {}

    def historical_bootstrapping(self, returns: pd.Series, portfolio_value: float = 10000,
                                 confidence_level: float = 0.95, days: int = 252,
                                 num_simulations: int = 1000) -> Dict[str, Any]:
        """Bootstrapping histórico — reamostragem com reposição."""
        try:
            r = np.asarray(returns)
            if len(r) == 0:
                return {}
            idx = np.random.randint(0, len(r), size=(days, num_simulations))
            sampled = r[idx]
            paths = portfolio_value * np.cumprod(1 + sampled, axis=0)
            final = paths[-1]
            changes = final - portfolio_value
            var = np.percentile(changes, (1 - confidence_level) * 100)
            tail = changes[changes <= var]
            cvar = float(tail.mean()) if len(tail) > 0 else float(var)
            return {
                'var': float(var), 'cvar': cvar,
                'confidence_level': confidence_level, 'time_horizon': days,
                'simulations': num_simulations, 'portfolio_value': portfolio_value,
                'probability_loss': float((final < portfolio_value).mean()),
                'expected_return': float(final.mean() - portfolio_value),
                'final_values': final, 'simulation_paths': paths,
                'method': 'Bootstrapping'
            }
        except Exception as e:
            print(f"Erro em historical_bootstrapping: {e}")
            return {}

    def merton_jump_diffusion(self, returns: pd.Series, portfolio_value: float = 10000,
                              days: int = 252, num_simulations: int = 1000) -> Dict[str, Any]:
        """Merton Jump Diffusion — eventos extremos com saltos."""
        try:
            r = np.asarray(returns)
            mu = r.mean() * 252
            sigma = r.std() * np.sqrt(252)
            lambda_j, mu_j, sigma_j = 0.05, -0.10, 0.05
            dt = 1 / 252
            Z = np.random.normal(0, 1, (days, num_simulations))
            N = np.random.poisson(lambda_j * dt, (days, num_simulations))
            J = np.random.normal(mu_j, sigma_j, (days, num_simulations)) * N
            log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J
            paths = portfolio_value * np.cumprod(np.exp(log_ret), axis=0)
            final = paths[-1]
            changes = final - portfolio_value
            var = np.percentile(changes, 5)
            tail = changes[changes <= var]
            cvar = float(tail.mean()) if len(tail) > 0 else float(var)
            return {
                'var': float(var), 'cvar': cvar,
                'confidence_level': 0.95, 'time_horizon': days,
                'simulations': num_simulations, 'portfolio_value': portfolio_value,
                'probability_loss': float((final < portfolio_value).mean()),
                'expected_return': float(final.mean() - portfolio_value),
                'final_values': final, 'simulation_paths': paths,
                'method': 'Merton Jump Diffusion'
            }
        except Exception as e:
            print(f"Erro em merton_jump_diffusion: {e}")
            return self.monte_carlo_baseline(returns, portfolio_value, 0.95, days, num_simulations)

    def garch_simulation(self, returns: pd.Series, portfolio_value: float = 10000,
                         days: int = 252, num_simulations: int = 1000) -> Dict[str, Any]:
        """GARCH(1,1) simplificado — volatilidade variante no tempo."""
        try:
            r = np.asarray(returns)
            omega = r.var() * 0.05
            alpha, beta = 0.10, 0.85
            sigma2_0 = r.var()
            paths = np.zeros((days, num_simulations))
            for s in range(num_simulations):
                sigma2 = sigma2_0
                value = portfolio_value
                for t in range(days):
                    eps = np.random.normal(0, np.sqrt(sigma2))
                    value *= (1 + eps)
                    paths[t, s] = value
                    sigma2 = omega + alpha * eps**2 + beta * sigma2
            final = paths[-1]
            changes = final - portfolio_value
            var = np.percentile(changes, 5)
            tail = changes[changes <= var]
            cvar = float(tail.mean()) if len(tail) > 0 else float(var)
            return {
                'var': float(var), 'cvar': cvar,
                'confidence_level': 0.95, 'time_horizon': days,
                'simulations': num_simulations, 'portfolio_value': portfolio_value,
                'probability_loss': float((final < portfolio_value).mean()),
                'expected_return': float(final.mean() - portfolio_value),
                'final_values': final, 'simulation_paths': paths,
                'method': 'GARCH'
            }
        except Exception as e:
            print(f"Erro em garch_simulation: {e}")
            return self.monte_carlo_baseline(returns, portfolio_value, 0.95, days, num_simulations)

    def run_simulation(self, method: str, portfolio_returns: pd.Series,
                       portfolio_config: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatcher: executa o método de simulação solicitado."""
        pv = portfolio_config.get('value', 10000)
        cl = portfolio_config.get('confidence_level', 0.95)
        th = portfolio_config.get('time_horizon', 252)
        ns = portfolio_config.get('num_simulations', 1000)

        dispatch = {
            'Monte Carlo Clássico': lambda: self.monte_carlo_baseline(portfolio_returns, pv, cl, th, ns),
            'Bootstrapping':        lambda: self.historical_bootstrapping(portfolio_returns, pv, cl, th, ns),
            'Merton Jump Diffusion':lambda: self.merton_jump_diffusion(portfolio_returns, pv, th, ns),
            'GARCH':                lambda: self.garch_simulation(portfolio_returns, pv, th, ns),
        }
        fn = dispatch.get(method, dispatch['Monte Carlo Clássico'])
        result = fn()
        if result:
            self.simulation_results[method] = result
        return result

    # ------------------------------------------------------------------

    def monte_carlo_simulation(self, returns: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executa simulação de Monte Carlo para o portfólio - CORRIGIDO"""
        try:
            # Se há configuração de portfólio, usar pesos específicos, senão igualmente ponderado
            if portfolio_config and 'weights' in portfolio_config:
                weights = portfolio_config['weights']
                # Garantir que os pesos estão normalizados e alinhados com os ativos
                weighted_returns = returns * pd.Series(weights).reindex(returns.columns, fill_value=0)
                portfolio_returns = weighted_returns.sum(axis=1)
            else:
                portfolio_returns = returns.mean(axis=1)

            # Calcular parâmetros para a simulação
            mean_returns = portfolio_returns.mean()
            std_returns = portfolio_returns.std()
            
            # Simulação de Monte Carlo (vetorizada)
            shocks = np.random.normal(0, 1, (self.time_horizon, self.n_simulations))
            returns_path = mean_returns + std_returns * shocks
            simulated_paths = self.initial_investment * np.cumprod(1 + returns_path, axis=0)

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

            self.simulation_results['Monte Carlo Clássico'] = simulation_results

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