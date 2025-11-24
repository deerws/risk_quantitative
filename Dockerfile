FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    cron \
    curl \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar SEU requirements.txt atual
COPY requirements.txt .

# Instalar suas dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/ 
COPY data/ ./data/
COPY *.py ./ 

# Copiar configuração do cron
COPY cronjobs/cron.config /etc/cron.d/economic-cron

# Dar permissão e instalar cron de forma mais segura
RUN chmod 0644 /etc/cron.d/economic-cron

# Verificar sintaxe do cron antes de instalar
RUN crontab -l || true  # Verifica se crontab funciona
RUN cat /etc/cron.d/economic-cron  # Mostra o conteúdo para debug

# Instalar o cron job
RUN crontab /etc/cron.d/economic-cron

# Criar log file
RUN touch /var/log/cron.log

# Comando para rodar cron em foreground
CMD ["cron", "-f"]