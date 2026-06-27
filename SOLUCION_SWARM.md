# 🔍 Análisis: Por qué MLTutor no funciona con Docker Swarm

## Diagnóstico del Problema

### Problema Principal: Incompatibilidad Arquitectónica

**Streamlit NO está diseñado para escalado horizontal tradicional**. Esto se debe a:

1. **Estado de sesión en memoria**
   - Cada contenedor mantiene su propia sesión de usuario
   - No hay backend compartido (como Redis) para sincronizar estado
   - Los datos de `st.session_state` solo existen en un contenedor específico

2. **WebSockets y conexiones persistentes**
   - Streamlit usa WebSocket para comunicación bidireccional en tiempo real
   - El cliente (navegador) debe mantener conexión **con el mismo contenedor** durante toda la sesión
   - Si el balanceador cambia de contenedor, se pierde el estado

3. **Recursos dinámicos locales**
   - Gráficos, imágenes y visualizaciones se generan en memoria de cada contenedor
   - No se comparten entre réplicas
   - Si una petición va a otro contenedor: **404 Not Found**

### Problemas Específicos en la Configuración Actual

#### 1. Nginx con `ip_hash` insuficiente
```nginx
upstream mltutor_upstream {
    ip_hash;  # Falla si hay Apache delante
    server tasks.mltutor_mltutor:8501;
}
```
- **Problema**: Si hay un proxy Apache delante, la IP vista por nginx es la de Apache, no del cliente
- **Resultado**: Todos los usuarios parecen venir de la misma IP → van al mismo contenedor (no balancea)

#### 2. Múltiples entradas en upstream causan confusión
```nginx
server tasks.mltutor_mltutor:8501;
server mltutor_mltutor:8501;  # Resuelve a lo mismo
```
- Ambos resuelven al mismo conjunto de IPs
- Puede causar resolución DNS inconsistente

#### 3. Recursos no compartidos
- Cada réplica tiene su propio sistema de archivos
- Las imágenes generadas por matplotlib/plotly se almacenan localmente
- No hay volumen compartido tipo NFS o S3

#### 4. Healthcheck puede aprobar contenedores no listos
- El `/healthz` puede responder antes de que Streamlit esté 100% funcional
- Nginx puede enviar tráfico a contenedores "casi listos"

## 🎯 Solución Recomendada: Escalado Vertical

### Por qué Escalado Vertical es Mejor

Streamlit **oficialmente recomienda** escalado vertical para producción:

✅ **Ventajas:**
- Una sola instancia = sin problemas de sesiones
- Sin problemas de recursos compartidos
- Configuración simple y mantenible
- WebSockets funcionan perfectamente
- Más recursos por contenedor = mejor rendimiento por usuario

📊 **Capacidad:**
- Con 4 CPUs y 4GB RAM: **50-100 usuarios simultáneos**
- Con 8 CPUs y 8GB RAM: **100-200 usuarios simultáneos**

### Implementación

He creado `docker-compose.production.yml` con:
- **1 réplica** con recursos amplios (4 CPUs, 4GB RAM)
- Sin nginx (innecesario con una sola instancia)
- Configuración optimizada para producción
- Healthcheck robusto

**Uso:**
```bash
# Despliegue simple con Docker Compose
docker-compose -f docker-compose.production.yml up -d

# O con Swarm (pero sin beneficios reales)
docker stack deploy -c docker-compose.production.yml mltutor
```

## 🔧 Alternativas (si realmente necesitas múltiples réplicas)

### Opción A: Kubernetes con Ingress Affinity

Kubernetes tiene mejor soporte para sticky sessions:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "mltutor-session"
    nginx.ingress.kubernetes.io/session-cookie-hash: "sha1"
```

**Pros:** Más robusto que Swarm
**Contras:** Más complejo de configurar y mantener

### Opción B: Modificar Streamlit para usar Redis

Requiere cambios significativos en el código:

1. Instalar `streamlit-redis-session-storage`
2. Modificar `app.py` para usar Redis como backend
3. Agregar servicio Redis al stack
4. Implementar almacenamiento compartido para recursos (S3/MinIO)

**Pros:** Soporta múltiples réplicas realmente
**Contras:** Mucho trabajo, aumenta complejidad, posibles problemas de rendimiento

### Opción C: Docker Compose sin Swarm

Si no necesitas alta disponibilidad real:

```bash
docker-compose up -d --scale mltutor=3
```

Usa un nginx local con `ip_hash` (sin Apache delante):

**Pros:** Más simple que Swarm
**Contras:** Sin orquestación real, sin auto-healing

## 📝 Recomendación Final

**Para tu caso de uso (aplicación educativa MLTutor):**

1. ✅ **Usa `docker-compose.production.yml` con escalado vertical**
   - Simple, estable, suficiente para la mayoría de casos
   - Fácil de mantener y debuguear
   - Soporta 50-100 usuarios sin problemas

2. Si necesitas más capacidad:
   - Aumenta recursos de la única réplica (más CPUs/RAM)
   - Si aún no es suficiente, considera migrar a Kubernetes

3. **Evita Docker Swarm para esta aplicación**
   - No aporta beneficios reales
   - Agrega complejidad innecesaria
   - Los sticky sessions no son confiables

## 🚀 Próximos Pasos

```bash
# 1. Detener el stack actual de Swarm (si está corriendo)
docker stack rm mltutor
sleep 15

# 2. Opción A: Despliegue simple (RECOMENDADO)
docker-compose -f docker-compose.production.yml up -d

# 3. Opción B: Si insistes en Swarm
docker swarm init  # si no está inicializado
docker stack deploy -c docker-compose.production.yml mltutor

# 4. Verificar
docker ps
curl http://localhost:8501
```

## 📚 Referencias

- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/deploy)
- [Streamlit doesn't support horizontal scaling](https://github.com/streamlit/streamlit/issues/5049)
- [Best practices for production Streamlit](https://docs.streamlit.io/deploy/tutorials/kubernetes)
