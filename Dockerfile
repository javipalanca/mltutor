FROM python:3.10-slim

WORKDIR /app

# Copia los archivos necesarios
COPY requirements.txt .
COPY app_streamlit_v2.py .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Puerto para Streamlit
EXPOSE 8501

# Variable de entorno para decir a Streamlit que se ejecute en modo servidor
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app_streamlit_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
