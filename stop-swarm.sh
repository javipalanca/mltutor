#!/bin/bash

# Script para detener y limpiar Docker Swarm

set -e

echo "🛑 Deteniendo MLTutor en Docker Swarm"
echo ""

# Eliminar el stack
if docker stack services mltutor > /dev/null 2>&1; then
    echo "🗑️  Eliminando stack mltutor..."
    docker stack rm mltutor
    echo "⏳ Esperando a que los servicios se detengan..."
    sleep 10
    echo "✅ Stack eliminado"
else
    echo "ℹ️  El stack 'mltutor' no está desplegado"
fi

echo ""
read -p "¿Deseas abandonar el modo Swarm? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚪 Abandonando Docker Swarm..."
    docker swarm leave --force
    echo "✅ Swarm desactivado"
else
    echo "ℹ️  Swarm sigue activo. Puedes volver a desplegar con ./deploy-swarm.sh"
fi

echo ""
echo "✅ Limpieza completada"

