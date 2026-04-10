#!/bin/bash

# ============================================
# Script de sincronização para o Raspberry Pi
# ============================================

# Configurações
PI_USER="raspberrypi5"
PI_IP="192.168.3.177"
PI_PATH="~/TesteMylena"
LOCAL_PATH="/home/mylena/Documentos/EngMec/FAPEG/codigo/Seguranca/DrowsyDriving/"

# Cores para output (opcional)
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Iniciando sincronização com o Raspberry Pi...${NC}"
echo "Origem: $LOCAL_PATH"
echo "Destino: $PI_USER@$PI_IP:$PI_PATH"
echo "-------------------------------------------"

# Executa o rsync
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='*.log' \
    "$LOCAL_PATH" "$PI_USER@$PI_IP:$PI_PATH"

# Verifica o retorno
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Sincronização concluída com sucesso!${NC}"
else
    echo -e "${RED}❌ Erro durante a sincronização.${NC}"
fi
