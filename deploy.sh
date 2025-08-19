# Construir la imagen de Docker
docker build -t mltutor .

# Ejecutar el contenedor
docker run -p 8502:8502 mltutor
