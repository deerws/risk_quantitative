# src/agents/__init__.py
"""
Pacote de agentes para o Sistema Multiagente Quantum Risk Analytics
"""

from .agent_base import AgentBase
from .dask_orchestrator import (
    AgentMarket, 
    AgentClustering, 
    AgentML, 
    AgentAlert,
    AgentLSTM,
    AgentAutoencoder,
    DaskMultiAgentOrchestrator
)
from .agent_simulation import AgentSimulation

__all__ = [
    'AgentBase',
    'AgentMarket',
    'AgentClustering', 
    'AgentML',
    'AgentAlert', 
    'AgentLSTM',
    'AgentAutoencoder',
    'AgentSimulation',
    'DaskMultiAgentOrchestrator'
]