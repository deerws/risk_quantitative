#!/usr/bin/env python3
"""
Script para executar o pipeline completo de análise de risco
"""

import sys
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_pipeline():
    """Executa o pipeline completo"""
    logger.info("🚀 INICIANDO PIPELINE DE ANÁLISE DE RISCO")
    
    try:
        # 1. Coleta de dados
        logger.info("📥 Etapa 1: Coleta de dados do BCB")
        from src.etl.data_collector_bcb import main as collect_data
        collect_data()
        
        # 2. Métricas de risco
        logger.info("🧮 Etapa 2: Cálculo de métricas de risco")
        from src.metrics.risk_calculator import main as calculate_risk
        calculate_risk()
        
        # 3. Visualizações
        logger.info("📊 Etapa 3: Geração de visualizações")
        from src.visualization.risk_plots import main as generate_plots
        generate_plots()
        
        # 4. Simulações
        logger.info("🎲 Etapa 4: Simulações de Monte Carlo")
        from src.simulation.monte_carlo import main as run_simulations
        run_simulations()
        
        # 5. Simulações avançadas (opcional)
        logger.info("🔬 Etapa 5: Simulações avançadas")
        from src.simulation.advanced_simulators import demo_advanced_simulators
        demo_advanced_simulators()
        
        logger.info("✅ PIPELINE EXECUTADO COM SUCESSO!")
        
    except Exception as e:
        logger.error(f"❌ Erro no pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()