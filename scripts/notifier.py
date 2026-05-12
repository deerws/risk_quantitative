"""
Notifier — envia alertas por email (prioridade) e Telegram (fallback/paralelo).

Configuração via variáveis de ambiente (.env):
  EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO, SMTP_SERVER, SMTP_PORT
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID  (opcionais)

Para Gmail use App Password (não a senha real):
  Conta Google > Segurança > Verificação em 2 etapas > Senhas de app
"""

import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.email_to = os.getenv("EMAIL_TO", self.email_user)
        self.telegram_token = os.getenv("TELEGRAM_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    def send_email(
        self,
        subject: str,
        body_html: str,
        attachment_path: Optional[str] = None,
    ) -> bool:
        if not self.email_user or not self.email_password:
            logger.warning("Email não configurado — defina EMAIL_USER e EMAIL_PASSWORD no .env")
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = self.email_user
        msg["To"] = self.email_to
        msg["Subject"] = subject

        msg.attach(MIMEText(body_html, "html", "utf-8"))

        if attachment_path and Path(attachment_path).exists():
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{Path(attachment_path).name}"',
            )
            msg.attach(part)

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.sendmail(self.email_user, self.email_to, msg.as_string())
            logger.info(f"Email enviado: {subject}")
            return True
        except smtplib.SMTPAuthenticationError:
            logger.error("Falha de autenticação SMTP — verifique EMAIL_USER e EMAIL_PASSWORD")
        except smtplib.SMTPException as e:
            logger.error(f"Erro SMTP: {e}")
        except Exception as e:
            logger.error(f"Erro ao enviar email: {e}")
        return False

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------

    def send_telegram(self, message: str) -> bool:
        if not self.telegram_token or not self.telegram_chat_id:
            logger.debug("Telegram não configurado — ignorando")
            return False

        try:
            import requests  # noqa: PLC0415

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": self.telegram_chat_id,
                    "text": message[:4096],
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if resp.ok:
                logger.info("Telegram: mensagem enviada")
                return True
            logger.warning(f"Telegram erro {resp.status_code}: {resp.text[:200]}")
        except ImportError:
            logger.warning("requests não instalado — Telegram indisponível")
        except Exception as e:
            logger.error(f"Erro ao enviar Telegram: {e}")
        return False

    # ------------------------------------------------------------------
    # Interface unificada
    # ------------------------------------------------------------------

    def notify(
        self,
        subject: str,
        body_html: str,
        attachment_path: Optional[str] = None,
        telegram_text: Optional[str] = None,
    ) -> None:
        """Envia por email (prioritário) e Telegram em paralelo."""
        self.send_email(subject, body_html, attachment_path)
        tg_msg = telegram_text or _strip_html(f"<b>{subject}</b>\n\n{body_html}")
        self.send_telegram(tg_msg[:4096])

    def notify_alert(self, ticker: str, alert_type: str, detail: str) -> None:
        subject = f"[Quantum Risk] Alerta {alert_type} — {ticker}"
        body = _alert_html(ticker, alert_type, detail)
        tg = f"⚠️ <b>{alert_type}</b> — {ticker}\n{detail}"
        self.notify(subject, body, telegram_text=tg)

    def notify_daily_report(self, report_path: Optional[str] = None) -> None:
        from datetime import date

        subject = f"[Quantum Risk] Relatório EOD — {date.today():%d/%m/%Y}"
        body = _report_html(date.today())
        self.notify(subject, body, attachment_path=report_path)


# ------------------------------------------------------------------
# Helpers de formatação
# ------------------------------------------------------------------

def _strip_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text)


def _alert_html(ticker: str, alert_type: str, detail: str) -> str:
    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#0e1117;color:#e0e0e0;padding:24px">
      <h2 style="color:#ff4b4b">⚠️ Alerta de Risco — {ticker}</h2>
      <table style="border-collapse:collapse;width:100%;max-width:600px">
        <tr><td style="padding:8px;color:#aaa">Tipo</td>
            <td style="padding:8px;font-weight:bold">{alert_type}</td></tr>
        <tr><td style="padding:8px;color:#aaa">Ativo</td>
            <td style="padding:8px">{ticker}</td></tr>
        <tr><td style="padding:8px;color:#aaa">Detalhe</td>
            <td style="padding:8px">{detail}</td></tr>
      </table>
      <p style="color:#666;font-size:12px;margin-top:32px">Quantum Risk Analytics</p>
    </body></html>
    """


def _report_html(report_date) -> str:
    return f"""
    <html><body style="font-family:Arial,sans-serif;background:#0e1117;color:#e0e0e0;padding:24px">
      <h2 style="color:#00d4ff">📊 Relatório Diário — {report_date:%d/%m/%Y}</h2>
      <p>Segue em anexo o relatório consolidado de fim de pregão.</p>
      <p>Acesse o dashboard para análise interativa:
         <a href="http://localhost:8501" style="color:#00d4ff">localhost:8501</a></p>
      <p style="color:#666;font-size:12px;margin-top:32px">Quantum Risk Analytics</p>
    </body></html>
    """
