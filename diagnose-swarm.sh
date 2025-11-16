#!/bin/bash

# Script de diagnóstico para Docker Swarm
# Verifica el estado de los servicios y ayuda a diagnosticar problemas

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🔍 Diagnóstico de Docker Swarm - MLTutor"
echo "========================================"
echo ""

# 1. Verificar que Swarm está activo
echo -e "${BLUE}[1/8]${NC} Verificando Docker Swarm..."
if docker info | grep -q "Swarm: active"; then
    echo -e "${GREEN}✓${NC} Swarm está activo"
else
    echo -e "${RED}✗${NC} Swarm NO está activo"
    exit 1
fi
echo ""

# 2. Listar servicios
echo -e "${BLUE}[2/8]${NC} Estado de los servicios:"
docker service ls
echo ""

# 3. Verificar réplicas de mltutor
echo -e "${BLUE}[3/8]${NC} Verificando réplicas de mltutor..."
REPLICAS=$(docker service ls --filter name=mltutor_mltutor --format "{{.Replicas}}")
echo "Réplicas: $REPLICAS"

if echo "$REPLICAS" | grep -q "3/3"; then
    echo -e "${GREEN}✓${NC} Todas las réplicas están activas"
else
    echo -e "${YELLOW}⚠${NC} No todas las réplicas están activas aún"
fi
echo ""

# 4. Verificar tareas
echo -e "${BLUE}[4/8]${NC} Estado de las tareas:"
docker service ps mltutor_mltutor --no-trunc
echo ""

# 5. Verificar nginx
echo -e "${BLUE}[5/8]${NC} Estado de nginx:"
docker service ps mltutor_nginx --no-trunc
echo ""

# 6. Probar resolución DNS
echo -e "${BLUE}[6/8]${NC} Probando resolución DNS de servicios dentro del stack..."
NGINX_CONTAINER=$(docker ps --filter name=mltutor_nginx --format "{{.ID}}" | head -n1)

if [ -n "$NGINX_CONTAINER" ]; then
    echo "Contenedor nginx: $NGINX_CONTAINER"
    echo "• Resolviendo tasks.mltutor_mltutor:"
    docker exec $NGINX_CONTAINER nslookup tasks.mltutor_mltutor 2>/dev/null || echo -e "${YELLOW}⚠${NC} No se pudo resolver tasks.mltutor_mltutor"
    echo "• Resolviendo mltutor_mltutor:"
    docker exec $NGINX_CONTAINER nslookup mltutor_mltutor 2>/dev/null || echo -e "${YELLOW}⚠${NC} No se pudo resolver mltutor_mltutor"
else
    echo -e "${RED}✗${NC} No se encontró contenedor de nginx"
fi
echo ""

# 7. Verificar logs recientes
echo -e "${BLUE}[7/8]${NC} Logs recientes de mltutor (últimas 10 líneas):"
docker service logs mltutor_mltutor --tail 10 2>&1 | grep -v "You can now view your Streamlit app" || true
echo ""

echo -e "${BLUE}[8/8]${NC} Logs recientes de nginx (últimas 10 líneas):"
docker service logs mltutor_nginx --tail 10 2>&1
echo ""

# 8. Intentar conectar desde nginx a mltutor
echo -e "${BLUE}[BONUS]${NC} Intentando conectar desde nginx a mltutor..."
if [ -n "$NGINX_CONTAINER" ]; then
    echo "Probando conexión a tasks.mltutor_mltutor:8501..."
    docker exec $NGINX_CONTAINER wget -qO- --timeout=5 http://tasks.mltutor_mltutor:8501/healthz 2>/dev/null && \
        echo -e "${GREEN}✓${NC} Conexión exitosa por tasks.mltutor_mltutor!" || \
        echo -e "${RED}✗${NC} No se pudo conectar a tasks.mltutor_mltutor"
    echo "Probando conexión a mltutor_mltutor:8501 (VIP)..."
    docker exec $NGINX_CONTAINER wget -qO- --timeout=5 http://mltutor_mltutor:8501/healthz 2>/dev/null && \
        echo -e "${GREEN}✓${NC} Conexión exitosa por VIP!" || \
        echo -e "${RED}✗${NC} No se pudo conectar por VIP"
fi
echo ""

# Resumen
echo "========================================"
echo "📊 Resumen del diagnóstico"
echo "========================================"
echo ""
echo "Comandos útiles:"
echo "  • Ver logs en tiempo real:"
echo "    docker service logs mltutor_mltutor -f"
echo "    docker service logs mltutor_nginx -f"
echo ""
echo "  • Reiniciar un servicio:"
echo "    docker service update --force mltutor_mltutor"
echo "    docker service update --force mltutor_nginx"
echo ""
echo "  • Escalar réplicas:"
echo "    docker service scale mltutor_mltutor=5"
echo ""
echo "  • Ver estado detallado:"
echo "    docker service inspect mltutor_mltutor --pretty"
echo ""
