FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Отключаем интерактивные диалоги при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/comfyui

COPY src /app/src
COPY handler.py /app/handler.py
COPY models_list.txt /app/models_list.txt

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    curl

RUN while IFS="|" read -r url bucket filename; do \
    if [ -n "$url" ] && [ -n "$filename" ]; then \
        mkdir -p "${HF_HOME}/$bucket" \
        echo "Downloading $filename to ${HF_HOME}/$bucket/$filename..." && \
        curl -L -o "${HF_HOME}/$bucket/$filename" "$url"; \
    fi; \
  done < /app/models_list.txt

# Устанавливаем базовые утилиты и добавляем репозиторий для свежего Python 3.11
RUN apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-distutils \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Ставим pip именно для Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

WORKDIR /app

# Обновляем инструменты сборки для Python 3.11
RUN python3.13 -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

# Устанавливаем ваши зависимости в изолированную папку через Python 3.11
RUN python3.13 -m pip install --no-cache-dir --target=/app/site-packages -r /app/requirements.txt

# Ставим runpod глобально
RUN python3.13 -m pip install --no-cache-dir runpod

# Важно: запускаем именно через python3.13
CMD ["python3.13", "-u", "/app/handler.py"]
