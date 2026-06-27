# 🚨 IMPORTANTE: Problemas con Docker Swarm

## El Problema

**Docker Swarm con múltiples réplicas NO funciona correctamente con esta aplicación** debido a limitaciones arquitectónicas de Streamlit:

- ❌ Estado de sesión se pierde entre réplicas
- ❌ Recursos dinámicos (imágenes/gráficos) no se comparten
- ❌ WebSockets requieren conexión al mismo contenedor
- ❌ Sticky sessions insuficientes con proxy Apache

## La Solución ✅

**Usa ESCALADO VERTICAL** (1 réplica con más recursos) en lugar de múltiples réplicas:

```bash
# Despliegue recomendado (simple y estable)
./deploy-production.sh
```

O manualmente:
```bash
docker-compose -f docker-compose.production.yml up -d
```

## Documentación

📖 Lee estos documentos en orden:

1. **`SOLUCION_SWARM.md`** - Explicación detallada del problema
2. **`COMPARATIVA_DESPLIEGUE.md`** - Comparación de opciones
3. **`PLAN_MIGRACION.md`** - Guía paso a paso para migrar

## Capacidad

Con escalado vertical (4 CPUs, 4GB RAM):
- ✅ **50-100 usuarios simultáneos** sin problemas
- ⚠️ Si necesitas más, aumenta recursos o lee la documentación

---
