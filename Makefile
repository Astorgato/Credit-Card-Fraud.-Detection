# Makefile for Credit Card Fraud Detection project

.PHONY: install test lint format clean docker-build docker-run help

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run code quality checks"
	@echo "  format       Format code with black"
	@echo "  clean        Clean temporary files"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  notebook     Start Jupyter notebook"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort jupyter

test:
	pytest tests/ -v --cov=src/

lint:
	flake8 src/ tests/ --max-line-length=88
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

docker-build:
	docker build -t fraud-detection:latest .

docker-run:
	docker run --rm -it fraud-detection:latest

notebook:
	jupyter notebook notebooks/

# Run the main fraud detection analysis
run:
	python src/fraud_detection_models.py
