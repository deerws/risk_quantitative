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
            
            # Simulação de Monte Carlo
            simulated_paths = np.zeros((self.time_horizon, self.n_simulations))
            for i in range(self.n_simulations):
                # Gerar caminhos usando geometria browniana
                shocks = np.random.normal(0, 1, self.time_horizon)
                returns_path = mean_returns + std_returns * shocks
                price_path = self.initial_investment * (1 + returns_path).cumprod()
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