#!/bin/bash

# Script para escalar el servicio MLTutor en Docker Swarm

set -e

# Número de réplicas (por defecto 3)
REPLICAS=${1:-3}

echo "⚖️  Escalando MLTutor a $REPLICAS réplicas..."

# Verificar que el stack está desplegado
if ! docker stack services mltutor > /dev/null 2>&1; then
    echo "❌ Error: El stack 'mltutor' no está desplegado"
    echo "💡 Ejecuta primero: ./deploy-swarm.sh"
    exit 1
fi

# Escalar el servicio
docker service scale mltutor_mltutor=$REPLICAS

echo ""
echo "✅ Servicio escalado a $REPLICAS réplicas"
echo ""
echo "📊 Estado actual:"
docker service ps mltutor_mltutor

echo ""
echo "💡 Consejos:"
echo "  - Para cargas bajas:     ./scale-swarm.sh 2"
echo "  - Para cargas medias:    ./scale-swarm.sh 3"
echo "  - Para cargas altas:     ./scale-swarm.sh 5"
echo "  - Para cargas muy altas: ./scale-swarm.sh 10"

