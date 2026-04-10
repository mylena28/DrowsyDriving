FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instala dependências básicas e repositório RPi
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg curl ca-certificates && \
    curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key | \
    gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian/ bookworm main" \
    > /etc/apt/sources.list.d/raspi.list

# Instala OpenCV e NumPy do SISTEMA (Otimizados para ARM se no RPi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-picamera2 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala Ultralytics SEM sobrescrever o NumPy do sistema
# Isso evita o erro de "binary incompatibility"
RUN pip3 install --no-cache-dir --break-system-packages "ultralytics>=8.0.0"

COPY . .

# Permissões de usuário
RUN groupadd -f -g 44 video && groupadd -f -g 993 render && \
    useradd -m -s /bin/bash -G video,render salte

# Baixa os modelos como root para garantir que fiquem na imagem
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"

# Ajusta as permissões para o usuário 'salte' poder ler/escrever na pasta /app
RUN chown -R salte:salte /app


RUN mkdir -p /tmp/Ultralytics && chown -R salte:salte /tmp/Ultralytics
USER salte


CMD ["python3", "main.py"]
