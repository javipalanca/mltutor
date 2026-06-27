# Correcciones para Imágenes Dinámicas en Docker Swarm

## Problema Identificado

Las imágenes dinámicas generadas por Streamlit no se cargaban correctamente cuando se usaba Docker Swarm con múltiples réplicas. Esto se debía a que:

1. **Sticky Sessions**: Las peticiones para recursos estáticos (`/_stcore/`, `/media/`) podían ir a contenedores diferentes
2. **Falta de configuración específica**: Nginx no tenía rutas específicas para los recursos de Streamlit
3. **Inconsistencias en puertos**: Había discrepancias entre los puertos configurados

## Cambios Realizados

### 1. nginx.conf
- ✅ Añadidas rutas específicas para `/_stcore/` (recursos estáticos de Streamlit)
- ✅ Añadida ruta específica para `/media/` (imágenes dinámicas)
- ✅ Configurado sistema de caché para recursos estáticos
- ✅ Mantenido `ip_hash` para sticky sessions
- ✅ Añadido `client_max_body_size 200M` para uploads grandes
- ✅ Configurados timeouts largos para todas las rutas

### 2. docker-compose.swarm.yml
- ✅ Unificado puerto a 8501 en todos los servicios
- ✅ Añadidas variables de entorno adicionales:
  - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
  - `STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=true`
  - `STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false`

### 3. Dockerfile y Dockerfile.swarm
- ✅ Unificado puerto a 8501
- ✅ Copiado directorio `.streamlit` con configuración
- ✅ Añadidas variables de entorno consistentes
- ✅ Corregido comando de inicio en Dockerfile.swarm

### 4. .streamlit/config.toml (NUEVO)
- ✅ Creado archivo de configuración centralizado
- ✅ Configuración optimizada para producción detrás de proxy
- ✅ Habilitada compresión WebSocket

## Cómo Desplegar

### 1. Reconstruir la imagen
```bash
# Para desarrollo
docker build -t mltutor:latest -f Dockerfile .

# Para Swarm (recomendado)
docker build -t mltutor:latest -f Dockerfile.swarm .
```

### 2. Redesplegar el stack
```bash
# Detener el stack actual
docker stack rm mltutor

# Esperar a que se eliminen completamente los servicios
sleep 15

# Desplegar el nuevo stack
docker stack deploy -c docker-compose.swarm.yml mltutor

# IMPORTANTE: Esperar a que todos los contenedores pasen el healthcheck
# Esto puede tomar 30-60 segundos
sleep 30
```

### 3. Diagnosticar problemas (NUEVO)
```bash
# Ejecutar script de diagnóstico
chmod +x diagnose-swarm.sh
./diagnose-swarm.sh
```

### 3. Verificar el despliegue
```bash
# Ver servicios
docker service ls

# Ver logs
docker service logs mltutor_mltutor -f

# Ver estado de nginx
docker service logs mltutor_nginx -f
```

### 4. Probar la aplicación
```bash
# La aplicación estará disponible en:
curl http://localhost:8502

# O abrir en el navegador:
# http://localhost:8502
```

## Verificación de Funcionamiento

1. **Imágenes estáticas**: Deben cargarse desde `/_stcore/` con caché
2. **Imágenes dinámicas**: Deben cargarse correctamente desde `/media/`
3. **WebSocket**: La conexión debe mantenerse estable
4. **Sticky Sessions**: Todas las peticiones de un cliente deben ir al mismo contenedor

## Si hay un proxy Apache delante (TLS/host público)

Para que WebSocket y Streamlit funcionen correctamente a través de Apache → Nginx → Streamlit:

1. Habilita módulos necesarios en Apache:

```
a2enmod proxy proxy_http proxy_wstunnel headers rewrite
```

2. Configura el VirtualHost (ejemplo HTTPS) y reenvía WebSocket y HTTP a Nginx (puerto 8502 en el host):

```
<VirtualHost *:443>
  ServerName mltutor.gti-ia.upv.es

  # TLS config ...

  RequestHeader set X-Forwarded-Proto "https"
  RequestHeader set X-Forwarded-Port "443"

  # WebSocket de Streamlit
  ProxyPass "/_stcore/stream"  "ws://127.0.0.1:8502/_stcore/stream"
  ProxyPassReverse "/_stcore/stream"  "ws://127.0.0.1:8502/_stcore/stream"

  # Resto de recursos (HTTP)
  ProxyPass "/"  "http://127.0.0.1:8502/"
  ProxyPassReverse "/"  "http://127.0.0.1:8502/"

  # Tamaño de subida
  LimitRequestBody 209715200
</VirtualHost>
```

3. Reinicia Apache tras aplicar cambios.

Notas:
- La ruta `/._stcore/stream` es la del WebSocket principal de Streamlit.
- Si publicas en un subpath (por ejemplo `/mltutor`), añade `server.baseUrlPath = "mltutor"` en `.streamlit/config.toml` y ajusta las rutas de ProxyPass.

## Troubleshooting

### "no live upstreams" en logs de nginx
Este error indica que nginx no puede conectar con los contenedores de mltutor. Posibles causas:

1. **Los contenedores aún no están listos**: Espera 30-60 segundos después del despliegue
2. **Healthcheck fallando**: Verifica con `./diagnose-swarm.sh`
3. **Problema de DNS**: Verifica que nginx puede resolver `tasks.mltutor`

```bash
# Ejecutar diagnóstico completo
./diagnose-swarm.sh

# Ver estado de healthchecks
docker service ps mltutor_mltutor

# Forzar actualización de nginx para que reintente
docker service update --force mltutor_nginx
```

### Las imágenes siguen sin cargar
```bash
# Verificar logs de nginx
docker service logs mltutor_nginx --tail 100

# Verificar que el cache funciona
curl -I http://localhost:8502/_stcore/static/css/main.css

# Debería mostrar: X-Cache-Status: HIT o MISS
```

### WebSocket no funciona
```bash
# Verificar headers en navegador (F12 > Network > WS)
# Debe mostrar: Upgrade: websocket
```

### Problemas de balanceo
```bash
# Verificar que ip_hash está activo
docker exec $(docker ps -q -f name=mltutor_nginx) cat /etc/nginx/nginx.conf | grep ip_hash

# Verificar upstream zone
docker exec $(docker ps -q -f name=mltutor_nginx) cat /etc/nginx/nginx.conf | grep zone
```

## Scripts Útiles

Los scripts existentes funcionan sin cambios:
- `./deploy-swarm.sh` - Despliega el stack
- `./scale-swarm.sh N` - Escala a N réplicas
- `./monitor-swarm.sh` - Monitorea el estado
- `./rebuild-swarm.sh` - Reconstruye y redespliega
- `./stop-swarm.sh` - Detiene el stack

## Notas Adicionales

- **Puerto 8501**: Puerto interno de los contenedores Streamlit
- **Puerto 8502**: Puerto expuesto por nginx al host
- **Sticky Sessions**: Garantizado por `ip_hash` en nginx
- **Caché**: Solo para recursos estáticos de `/_stcore/`
- **Media**: Sin caché, siempre se sirve dinámicamente
