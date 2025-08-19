FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Copia los archivos necesarios
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY ./mltutor /app/mltutor

# Puerto para Streamlit
EXPOSE 8502

# Variable de entorno para decir a Streamlit que se ejecute en modo servidor
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "mltutor/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
