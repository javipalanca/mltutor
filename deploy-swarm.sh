#!/bin/bash

# Script para desplegar MLTutor en Docker Swarm
# Permite escalar la aplicación para soportar múltiples usuarios simultáneos

set -e

echo "🚀 Desplegando MLTutor en Docker Swarm"

# Verificar si Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker no está corriendo"
    exit 1
fi

# Inicializar Swarm si no está inicializado
if ! docker info | grep -q "Swarm: active"; then
    echo "📦 Inicializando Docker Swarm..."
    docker swarm init
else
    echo "✅ Docker Swarm ya está activo"
fi

# Construir la imagen
echo "🔨 Construyendo imagen Docker..."
docker build -t mltutor:latest .

# Desplegar el stack
echo "🚀 Desplegando stack mltutor..."
docker stack deploy -c docker-compose.swarm.yml mltutor

echo ""
echo "✅ Despliegue completado!"
echo ""
echo "📊 Comandos útiles:"
echo "  - Ver servicios:        docker stack services mltutor"
echo "  - Ver contenedores:     docker stack ps mltutor"
echo "  - Ver logs:             docker service logs -f mltutor_mltutor"
echo "  - Escalar servicio:     docker service scale mltutor_mltutor=5"
echo "  - Actualizar servicio:  docker service update mltutor_mltutor"
echo "  - Eliminar stack:       docker stack rm mltutor"
echo ""
echo "🌐 Accede a la aplicación en: http://localhost"
echo ""
echo "⏳ Esperando a que los servicios estén listos..."
sleep 5

# Mostrar estado de los servicios
docker stack services mltutor

