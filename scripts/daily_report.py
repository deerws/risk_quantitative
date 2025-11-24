#!/usr/bin/env python3
"""
Gera relatório diário automático
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyReporter:
    def __init__(self):
        self.today = datetime.now().date()
        self.report_dir = "reports/daily"
        os.makedirs(self.report_dir, exist_ok=True)
        
    def load_data(self):
        """Carrega dados mais recentes"""
        try:
            returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
            prices = pd.read_parquet('data/processed/macro_portfolio_prices.parquet')
            return returns, prices
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return None, None
    
    def generate_report(self):
        """Gera relatório diário"""
        returns, prices = self.load_data()
        if returns is None:
            return False
            
        # Filtra últimos 30 dias
        recent_returns = returns.last('30D')
        recent_prices = prices.last('30D')
        
        # Cria relatório visual
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Relatório Diário de Risco - {self.today.strftime("%d/%m/%Y")}')
        
        # 1. Evolução recente
        recent_prices.plot(ax=axes[0,0], title='Evolução dos Preços (30 dias)')
        axes[0,0].set_ylabel('Preço (Base 100)')
        
        # 2. Volatilidade recente
        vol_30d = recent_returns.std() * np.sqrt(252)
        vol_30d.plot(kind='bar', ax=axes[0,1], title='Volatilidade 30D (Anualizada)', color='orange')
        axes[0,1].set_ylabel('Volatilidade')
        
        # 3. Retornos do dia
        if not recent_returns.empty:
            daily_returns = recent_returns.iloc[-1] if len(recent_returns) > 1 else recent_returns.iloc[0]
            daily_returns.plot(kind='bar', ax=axes[1,0], title='Retornos do Dia', color='green')
            axes[1,0].set_ylabel('Retorno')
        
        # 4. Correlações recentes
        sns.heatmap(recent_returns.corr(), annot=True, ax=axes[1,1], cmap='coolwarm', center=0)
        axes[1,1].set_title('Correlações (30 dias)')
        
        plt.tight_layout()
        
        # Salvar relatório
        report_path = f'{self.report_dir}/report_{self.today.strftime("%Y%m%d")}.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Relatório gerado: {report_path}")
        return True
    
    def send_email_report(self, to_emails):
        """Envia relatório por email (opcional)"""
        # Implementação básica - adaptar com suas credenciais SMTP
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f'Relatório Diário de Risco - {self.today.strftime("%d/%m/%Y")}'
            msg['From'] = 'risk@empresa.com'
            msg['To'] = ', '.join(to_emails)
            
            # Corpo do email
            body = f"""
            <h2>Relatório Diário de Análise de Risco</h2>
            <p>Data: {self.today.strftime("%d/%m/%Y")}</p>
            <p>Pipeline executado com sucesso.</p>
            <p>Acesse o dashboard: http://localhost:8501</p>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # Aqui você adicionaria o anexo e enviaria o email
            # smtp_server.send_message(msg)
            
            logger.info("Email preparado para envio")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao preparar email: {e}")
            return False

def main():
    reporter = DailyReporter()
    success = reporter.generate_report()
    
    if success:
        # Lista de emails para notificação
        emails = ["analista@empresa.com"]
        reporter.send_email_report(emails)

if __name__ == "__main__":
    main()