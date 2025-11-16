FROM python:3.11

ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

WORKDIR $AIRFLOW_HOME

# 🔥 Instalar dependencias necesarias para OpenCV en Debian trixie
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copiar todo el proyecto
COPY . $AIRFLOW_HOME

# Instalar Airflow
RUN pip install "apache-airflow==2.8.1" --constraint \
    "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.11.txt"

# Instalar PyTorch CPU (más liviano y rápido)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar otras dependencias Python (OpenCV, etc)
RUN pip install --no-cache-dir numpy opencv-python pillow

# 💥 Quitar DAGs de ejemplo de Airflow
RUN rm -rf /usr/local/lib/python3.11/site-packages/airflow/example_dags

EXPOSE 8080

CMD ["airflow", "standalone"]
