FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# Adiciona repositório da Raspberry Pi
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg curl ca-certificates \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] \
        http://archive.raspberrypi.com/debian/ bookworm main" \
        > /etc/apt/sources.list.d/raspi.list \
    && apt-get update

# Instala dependências do sistema
RUN apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-numpy \
        python3-picamera2 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala pacotes Python adicionais (sem picamera2, que já veio do apt)
RUN pip3 install --no-cache-dir --break-system-packages \
        opencv-python-headless \
        ultralytics

WORKDIR /app
COPY . .

# Cria usuário não-root (opcional, mas recomendado)
RUN groupadd -f -g 44 video && \
    groupadd -f -g 993 render && \
    useradd -m -s /bin/bash -G video,render salte
USER salte

CMD ["python3", "-u", "main.py"]
