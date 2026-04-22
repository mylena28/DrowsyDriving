# =============================================================
# STAGE 1 — builder
# Exports .pt → .onnx only. Torch + CUDA libs never reach runtime.
# =============================================================
FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-numpy python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU-only wheel — explicitly excludes all CUDA/nvidia packages.
# The +cpu suffix is what prevents pip from pulling ~2 GB of nvidia libs.
RUN pip3 install --no-cache-dir --break-system-packages \
    "torch==2.11.0+cpu" \
    "torchvision==0.26.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# ultralytics sees python3-opencv from apt and skips pulling its own cv2 wheel
RUN pip3 install --no-cache-dir --break-system-packages \
    "ultralytics>=8.0.0" \
    "onnx>=1.12.0,<2.0.0" \
    onnxslim \
    onnxruntime

COPY export_models.py .
COPY *.pt ./

RUN python3 export_models.py

# =============================================================
# STAGE 2 — runtime
# No torch at all — inference runs entirely via onnxruntime.
# pip numpy is never installed here, so picamera2 works fine.
# =============================================================
FROM debian:bookworm-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base packages — work on both amd64 and arm64
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-numpy python3-opencv \
    libglib2.0-0 libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# RPi-only packages: installed only on arm64 (Raspberry Pi 5)
# On amd64 (laptop) these packages don't exist, so this block is skipped entirely.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
    apt-get update && \
    apt-get install -y --no-install-recommends gnupg curl ca-certificates && \
    curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key | \
        gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] \
        http://archive.raspberrypi.com/debian/ bookworm main" \
        > /etc/apt/sources.list.d/raspi.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-picamera2 libcamera-ipa rpicam-apps-lite && \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# No torch, no ultralytics — inference runs entirely via onnxruntime.
# The thin YOLOOnnx wrapper in detectors/yolo_onnx.py replaces the
# ultralytics YOLO API, so numpy stays as the system apt package and
# picamera2 has no conflicts.
RUN pip3 install --no-cache-dir --break-system-packages \
    onnxruntime

COPY --from=builder /app/yolov8n.onnx .
COPY --from=builder /app/yolov8n-pose.onnx .

COPY . .

RUN groupadd -f -g 44 video && groupadd -f -g 993 render && \
    useradd -m -s /bin/bash -G video,render salte && \
    mkdir -p /tmp/Ultralytics && \
    chown -R salte:salte /app /tmp/Ultralytics

USER salte
CMD ["python3", "main.py"]
