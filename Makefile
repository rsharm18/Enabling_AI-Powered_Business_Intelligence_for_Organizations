.PHONY: help setup web csv process stop kill-port start-db stop-db clean

help:
	@echo "AI-Powered Business Intelligence Platform - Make Commands"
	@echo ""
	@echo "Application Modes:"
	@echo "  make web        - Start chat interface (port 7860)"
	@echo "  make debug      - Start chat interface in debug mode with live code refresh"
	@echo "  make csv        - Run CSV analysis mode"
	@echo "  make process    - Run document processing mode"
	@echo ""
	@echo "Service Management:"
	@echo "  make stop       - Stop application (kill port 7860)"
	@echo "  make kill-port  - Kill process on port 7860"
	@echo "  make start-db   - Start PostgreSQL database (Docker)"
	@echo "  make stop-db    - Stop PostgreSQL database (Docker)"
	@echo ""
	@echo "Setup:"
	@echo "  make setup     - Create virtual environment and install dependencies"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Clean up Python cache files"

web:
	@echo "Starting chat interface..."
	@if lsof -ti:7860 >/dev/null 2>&1; then \
		echo "Port 7860 is in use, killing process..."; \
		kill -9 $$(lsof -ti:7860) 2>/dev/null || true; \
	fi
	@PYTHONPATH=/mnt/m/github_project/Python/Enabling_AI-Powered_Business_Intelligence_for_Organizations .venv/bin/python app/main.py --mode chat

debug:
	@echo "Starting chat interface in lightweight debug mode..."
	@if lsof -ti:7860 >/dev/null 2>&1; then \
		echo "Port 7860 is in use, killing process..."; \
		kill -9 $$(lsof -ti:7860) 2>/dev/null || true; \
	fi
	@PYTHONPATH=/mnt/m/github_project/Python/Enabling_AI-Powered_Business_Intelligence_for_Organizations .venv/bin/python app/main.py --mode chat --debug-light

csv:
	@echo "Starting CSV analysis mode..."
	@PYTHONPATH=/mnt/m/github_project/Python/Enabling_AI-Powered_Business_Intelligence_for_Organizations .venv/bin/python app/main.py --mode csv

process:
	@echo "Starting document processing mode..."
	@PYTHONPATH=/mnt/m/github_project/Python/Enabling_AI-Powered_Business_Intelligence_for_Organizations .venv/bin/python app/main.py --mode process

stop:
	@echo "Stopping application..."
	@if lsof -ti:7860 >/dev/null 2>&1; then \
		kill -9 $$(lsof -ti:7860) 2>/dev/null || true; \
		echo "Application stopped"; \
	else \
		echo "No application running on port 7860"; \
	fi

kill-port:
	@echo "Killing process on port 7860..."
	@if lsof -ti:7860 >/dev/null 2>&1; then \
		kill -9 $$(lsof -ti:7860) 2>/dev/null || true; \
		echo "Port 7860 freed"; \
	else \
		echo "No process found on port 7860"; \
	fi

start-db:
	@echo "Starting PostgreSQL database..."
	@docker-compose up -d postgres

stop-db:
	@echo "Stopping PostgreSQL database..."
	@docker-compose stop postgres

setup:
	@echo "Setting up development environment..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv .venv; \
	else \
		echo "Virtual environment already exists"; \
	fi
	@echo "Installing dependencies..."
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install uv
	@.venv/bin/uv pip install -r requirements.txt
	@echo "Setup complete! Activate with: source .venv/bin/activate"

clean:
	@echo "Cleaning Python cache files..."
	@find . -name "*.pyc" -delete 2>/dev/null || echo "No .pyc files found"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || echo "No __pycache__ directories found"
	@echo "Cleanup complete"
