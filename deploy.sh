# Construir la imagen de Docker
docker build -t mltutor .

# Ejecutar el contenedor
docker run -p 8501:8501 mltutor
