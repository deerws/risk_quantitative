.PHONY: build up down logs clean pipeline report

# Docker
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# Pipeline
pipeline:
	python scripts/run_pipeline.py

report:
	python scripts/daily_report.py

# Desenvolvimento
dev:
	streamlit run dashboard/app.py

# Limpeza
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf data/processed/*.parquet
	rm -rf reports/figures/*.png

# Airflow
airflow-init:
	docker-compose up airflow-init

airflow-up:
	docker-compose up -d airflow