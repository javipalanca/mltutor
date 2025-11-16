#!/bin/bash

# Script para forzar la reconstrucción y actualización completa del stack

set -e

echo "🔄 RECONSTRUCCIÓN FORZADA DE MLTUTOR"
echo "===================================="
echo ""

# Verificar que Swarm está activo
if ! docker info | grep -q "Swarm: active"; then
    echo "❌ Error: Docker Swarm no está activo"
    echo "💡 Ejecuta primero: docker swarm init"
    exit 1
fi

# Paso 1: Eliminar stack
echo "1️⃣  Eliminando stack mltutor..."
docker stack rm mltutor 2>/dev/null || echo "   (No había stack previo)"

echo "⏳ Esperando limpieza de recursos (20 segundos)..."
sleep 20

# Paso 2: Limpiar imágenes antiguas
echo ""
echo "2️⃣  Limpiando imágenes antiguas de mltutor..."
docker images | grep mltutor | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || echo "   (No había imágenes previas)"

# Paso 3: Limpiar build cache
echo ""
echo "3️⃣  Limpiando caché de build..."
docker builder prune -f

# Paso 4: Construir imagen desde cero
echo ""
echo "4️⃣  Construyendo imagen desde cero (sin caché)..."
docker build --no-cache -t mltutor:latest .

# Paso 5: Verificar que la imagen se construyó correctamente
echo ""
echo "5️⃣  Verificando imagen construida..."
docker images mltutor:latest

# Paso 6: Redesplegar stack
echo ""
echo "6️⃣  Desplegando stack con imagen nueva..."
docker stack deploy -c docker-compose.swarm.yml mltutor

echo ""
echo "✅ Despliegue completado!"
echo ""
echo "⏳ Esperando a que los servicios inicien (15 segundos)..."
sleep 15

# Paso 7: Verificar estado
echo ""
echo "📊 Estado de los servicios:"
docker stack services mltutor

echo ""
echo "📦 Contenedores en ejecución:"
docker stack ps mltutor --no-trunc | head -10

echo ""
echo "🔍 Últimas líneas de logs:"
docker service logs --tail 20 mltutor_mltutor

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Accede a la aplicación en: http://localhost:8502"
echo ""
echo "💡 Comandos útiles:"
echo "   - Ver logs:      docker service logs -f mltutor_mltutor"
echo "   - Diagnosticar:  ./diagnose-swarm.sh"
echo "   - Escalar:       ./scale-swarm.sh 5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

