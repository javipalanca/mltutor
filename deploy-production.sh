#!/bin/bash

# Script de despliegue simplificado para MLTutor
# Usa escalado vertical (1 réplica con más recursos) en lugar de múltiples réplicas

set -e

echo "🚀 Desplegando MLTutor (Escalado Vertical - Producción)"
echo "=========================================================="
echo ""

# Verificar si Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker no está corriendo"
    exit 1
fi

# Preguntar método de despliegue
echo "Selecciona el método de despliegue:"
echo "  1) Docker Compose (RECOMENDADO)"
echo "  2) Docker Swarm"
echo ""
read -p "Opción [1]: " option
option=${option:-1}

echo ""

# Construir la imagen
echo "🔨 Construyendo imagen Docker..."
docker build -t mltutor:latest -f Dockerfile .

echo ""

if [ "$option" = "1" ]; then
    # Despliegue con Docker Compose
    echo "📦 Desplegando con Docker Compose..."
    
    # Detener contenedores previos si existen
    docker-compose -f docker-compose.production.yml down 2>/dev/null || true
    
    # Iniciar servicio
    docker-compose -f docker-compose.production.yml up -d
    
    echo ""
    echo "✅ Despliegue completado con Docker Compose!"
    echo ""
    echo "📊 Comandos útiles:"
    echo "  - Ver logs:             docker-compose -f docker-compose.production.yml logs -f"
    echo "  - Detener:              docker-compose -f docker-compose.production.yml down"
    echo "  - Reiniciar:            docker-compose -f docker-compose.production.yml restart"
    echo "  - Ver estado:           docker-compose -f docker-compose.production.yml ps"
    
else
    # Despliegue con Docker Swarm
    echo "📦 Desplegando con Docker Swarm..."
    
    # Inicializar Swarm si no está inicializado
    if ! docker info | grep -q "Swarm: active"; then
        echo "🔧 Inicializando Docker Swarm..."
        docker swarm init
    else
        echo "✅ Docker Swarm ya está activo"
    fi
    
    # Detener stack previo si existe
    docker stack rm mltutor 2>/dev/null || true
    echo "⏳ Esperando a que se eliminen servicios previos..."
    sleep 10
    
    # Desplegar el stack
    docker stack deploy -c docker-compose.production.yml mltutor
    
    echo ""
    echo "✅ Despliegue completado con Docker Swarm!"
    echo ""
    echo "📊 Comandos útiles:"
    echo "  - Ver servicios:        docker stack services mltutor"
    echo "  - Ver logs:             docker service logs -f mltutor_mltutor"
    echo "  - Detener:              docker stack rm mltutor"
    echo "  - Ver contenedores:     docker stack ps mltutor"
fi

echo ""
echo "🌐 Accede a la aplicación en: http://localhost:8501"
echo ""
echo "⏳ Esperando a que el servicio esté listo..."
sleep 5

# Verificar estado
if [ "$option" = "1" ]; then
    docker-compose -f docker-compose.production.yml ps
else
    docker stack services mltutor
fi

echo ""
echo "💡 Nota: Esta configuración usa escalado VERTICAL (1 réplica con más recursos)"
echo "   Esto es lo recomendado por Streamlit para producción."
echo "   Soporta 50-100 usuarios simultáneos con la configuración actual."
