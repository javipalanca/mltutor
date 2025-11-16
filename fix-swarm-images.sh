#!/bin/bash

# Script para aplicar los cambios de las imágenes dinámicas en Swarm
# Autor: Sistema de corrección automática
# Fecha: $(date)

set -e

echo "🔧 Aplicando correcciones para imágenes dinámicas en Docker Swarm..."
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "docker-compose.swarm.yml" ]; then
    print_error "No se encontró docker-compose.swarm.yml"
    print_error "Por favor, ejecuta este script desde el directorio raíz del proyecto"
    exit 1
fi

print_status "Verificando Docker Swarm..."
if ! docker info | grep -q "Swarm: active"; then
    print_error "Docker Swarm no está activo"
    print_error "Inicializa Swarm con: docker swarm init"
    exit 1
fi

# Paso 1: Detener el stack actual
print_status "Paso 1: Deteniendo stack actual..."
if docker stack ls | grep -q "mltutor"; then
    docker stack rm mltutor
    print_status "Stack detenido. Esperando limpieza..."
    sleep 15
else
    print_warning "No hay stack activo para detener"
fi

# Paso 2: Crear directorio .streamlit si no existe
print_status "Paso 2: Verificando configuración de Streamlit..."
if [ ! -d ".streamlit" ]; then
    print_warning "Directorio .streamlit no encontrado. Debería haberse creado automáticamente."
fi

# Paso 3: Reconstruir la imagen
print_status "Paso 3: Reconstruyendo imagen Docker..."
docker build -t mltutor:latest -f Dockerfile.swarm .

if [ $? -eq 0 ]; then
    print_status "Imagen reconstruida exitosamente"
else
    print_error "Error al reconstruir la imagen"
    exit 1
fi

# Paso 4: Desplegar el nuevo stack
print_status "Paso 4: Desplegando nuevo stack..."
docker stack deploy -c docker-compose.swarm.yml mltutor

if [ $? -eq 0 ]; then
    print_status "Stack desplegado exitosamente"
else
    print_error "Error al desplegar el stack"
    exit 1
fi

# Paso 5: Esperar a que los servicios estén listos
print_status "Paso 5: Esperando a que los servicios estén listos..."
echo ""
sleep 5

# Mostrar estado de los servicios
echo "📊 Estado de los servicios:"
docker service ls

echo ""
print_status "Esperando a que todos los contenedores estén saludables (30s)..."
sleep 30

echo ""
echo "🔍 Verificando servicios..."
docker service ls | grep mltutor

echo ""
echo "📝 Logs recientes de mltutor:"
docker service logs mltutor_mltutor --tail 20

echo ""
echo "📝 Logs recientes de nginx:"
docker service logs mltutor_nginx --tail 10

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
print_status "¡Despliegue completado!"
echo ""
echo "🌐 La aplicación debería estar disponible en: http://localhost:8502"
echo ""
echo "Comandos útiles:"
echo "  • Ver logs de mltutor:  docker service logs mltutor_mltutor -f"
echo "  • Ver logs de nginx:    docker service logs mltutor_nginx -f"
echo "  • Ver estado:           docker service ls"
echo "  • Escalar réplicas:     ./scale-swarm.sh <número>"
echo "  • Monitorear:           ./monitor-swarm.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════"
