VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
STREAM  := $(VENV)/bin/streamlit

.PHONY: install dev pipeline report test clean build up down logs

# ── Ambiente ──────────────────────────────────────────────────────────────────
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Desenvolvimento ───────────────────────────────────────────────────────────
dev:
	$(STREAM) run dashboard/app.py

# ── Scripts ───────────────────────────────────────────────────────────────────
pipeline:
	$(PYTHON) scripts/run_pipeline.py

report:
	$(PYTHON) scripts/daily_report.py

# ── Testes ────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) scripts/test_multiagent.py
	$(PYTHON) scripts/test_quant_agents.py

# ── Docker ────────────────────────────────────────────────────────────────────
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# ── Limpeza ───────────────────────────────────────────────────────────────────
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -f reports/figures/*.png
	rm -f data/processed/*.parquet
