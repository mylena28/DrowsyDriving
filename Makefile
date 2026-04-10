# Variáveis
PROJECT_NAME = aitireddriver
DOCKER_COMPOSE = docker compose

# Alvo padrão (help)
.PHONY: help
help:
	@echo "Comandos disponíveis:"
	@echo "  make build    - Constrói a imagem Docker"
	@echo "  make up       - Sobe o container (modo attach)"
	@echo "  make up-d     - Sobe o container em background (detached)"
	@echo "  make down     - Para e remove o container"
	@echo "  make logs     - Mostra os logs do container"
	@echo "  make shell    - Abre um shell interativo dentro do container"
	@echo "  make clean    - Remove __pycache__ e arquivos temporários"
	@echo "  make clean-docker - Remove containers, imagens e volumes não usados"
	@echo "  make all      - build + up"

# Docker targets
.PHONY: build
build:
	$(DOCKER_COMPOSE) build

.PHONY: up
up:
	$(DOCKER_COMPOSE) up

.PHONY: up-d
up-d:
	$(DOCKER_COMPOSE) up -d

.PHONY: down
down:
	$(DOCKER_COMPOSE) down

.PHONY: logs
logs:
	$(DOCKER_COMPOSE) logs -f

.PHONY: shell
shell:
	$(DOCKER_COMPOSE) run --rm monitor bash

# Python clean (__pycache__, .pyc, etc.)
.PHONY: clean
clean:
	@echo "Removendo pastas __pycache__ e arquivos .pyc..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Limpeza concluída."

# Docker system prune (cuidado: remove recursos não utilizados)
.PHONY: clean-docker
clean-docker:
	docker system prune -f
	docker volume prune -f

.PHONY: clean-project
clean-project:
	docker compose down --volumes --rmi local

# All: build + up
.PHONY: all
all: build up
