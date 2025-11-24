# src/agents/agent_base.py
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

class AgentBase:
    """Classe base para todos os agentes do sistema multiagente"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.alert_thresholds = {}
        self.performance_metrics = {}
    
    def process_data(self, data: pd.DataFrame, portfolio_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Método principal de processamento - deve ser implementado por cada agente"""
        raise NotImplementedError(f"Método process_data não implementado no agente {self.agent_id}")
    
    def generate_alert(self, message: str, severity: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera um alerta padronizado para o sistema"""
        return {
            'agent_id': self.agent_id,
            'timestamp': datetime.now(),
            'message': message,
            'severity': severity,
            'data': data,
            'alert_id': f"{self.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }