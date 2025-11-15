#!/bin/bash

# Script para monitorear el estado del cluster Docker Swarm

set -e

echo "📊 Monitor de Docker Swarm - MLTutor"
echo "======================================"
echo ""

# Verificar que Swarm está activo
if ! docker info | grep -q "Swarm: active"; then
    echo "❌ Error: Docker Swarm no está activo"
    exit 1
fi

# Mostrar información del nodo
echo "🖥️  Información del nodo:"
docker node ls
echo ""

# Mostrar servicios
echo "🔧 Servicios del stack mltutor:"
docker stack services mltutor
echo ""

# Mostrar contenedores
echo "📦 Contenedores ejecutándose:"
docker stack ps mltutor --no-trunc
echo ""

# Mostrar estadísticas de recursos
echo "📈 Uso de recursos:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $(docker ps -q --filter "label=com.docker.stack.namespace=mltutor")
echo ""

echo "💡 Para ver logs en tiempo real:"
echo "   docker service logs -f mltutor_mltutor"
echo ""
echo "💡 Para ver logs de nginx:"
echo "   docker service logs -f mltutor_nginx"

