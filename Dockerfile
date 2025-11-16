FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Copia los archivos necesarios
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY ./mltutor /app/mltutor
COPY ./.streamlit /app/.streamlit

# Puerto para Streamlit
EXPOSE 8501

# Variable de entorno para decir a Streamlit que se ejecute en modo servidor
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=true
# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "mltutor/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
