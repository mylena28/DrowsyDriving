
#!/bin/bash
# docker-entrypoint.sh

set -e

echo "=== Configurando ambiente para Picamera2 no Pi 5 ==="

# Verifica dispositivos disponíveis
echo "Dispositivos de mídia disponíveis:"
ls -la /dev/media* 2>/dev/null || echo "Nenhum dispositivo /dev/media encontrado"

echo "Dispositivos DRI:"
ls -la /dev/dri/ 2>/dev/null || echo "Nenhum dispositivo DRI encontrado"

# Verifica se a câmera é detectada pela libcamera
echo "Testando detecção da câmera..."
if command -v libcamera-hello &> /dev/null; then
    libcamera-hello --list-cameras || echo "Nenhuma câmera detectada pela libcamera"
fi

# Cria diretórios necessários
mkdir -p /tmp/libcamera /app/logs

# Exporta variáveis para o Picamera2
export LIBCAMERA_LOG_LEVELS=WARNING
export PYTHONUNBUFFERED=1

echo "=== Ambiente configurado. Iniciando aplicação ==="

# Executa o comando principal
exec "$@"
