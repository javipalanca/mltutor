FROM mcr.microsoft.com/devcontainers/python:3.9

# Evitar prompts durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurar entorno Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Actualizar pip y herramientas de desarrollo
RUN pip install --upgrade pip setuptools wheel

# Instalar Streamlit
RUN pip install streamlit

# Configurar el workspace
WORKDIR /workspace

# Configurar Streamlit para permitir conexiones externas
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml

# Configurar puerto para Streamlit
EXPOSE 8501
