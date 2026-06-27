# Plan de Migración: De Swarm a Producción Simple

## 🎯 Objetivo

Migrar de la configuración actual problemática con Docker Swarm (múltiples réplicas) a una configuración estable con escalado vertical (1 réplica con más recursos).

## 📋 Checklist Pre-Migración

- [ ] Backup de datos importantes en `/app/data`
- [ ] Documentar configuración actual de Apache (si existe)
- [ ] Verificar que Docker está actualizado
- [ ] Leer `SOLUCION_SWARM.md` y `COMPARATIVA_DESPLIEGUE.md`

## 🔄 Pasos de Migración

### Paso 1: Detener Configuración Actual

```bash
# Si tienes un stack de Swarm corriendo
docker stack rm mltutor

# Esperar a que se eliminen completamente
sleep 15

# Verificar que no quedan servicios
docker service ls

# Verificar que no quedan contenedores
docker ps -a | grep mltutor
```

### Paso 2: Backup de Datos (si es necesario)

```bash
# Si tienes datos importantes en volúmenes
docker run --rm -v mltutor_mltutor-data:/data -v $(pwd)/backup:/backup \
  alpine tar czf /backup/mltutor-data-backup-$(date +%Y%m%d).tar.gz -C /data .

# Verificar backup
ls -lh backup/
```

### Paso 3: Limpiar Configuración Anterior

```bash
# Eliminar redes antiguas (opcional)
docker network rm mltutor_mltutor-network 2>/dev/null || true

# NO eliminar volumen si tiene datos importantes
# docker volume rm mltutor_mltutor-data

# Limpiar imágenes antiguas (opcional)
docker image prune -f
```

### Paso 4: Desplegar Nueva Configuración

```bash
# Opción A: Con el script automatizado (RECOMENDADO)
./deploy-production.sh

# O manualmente:
# Opción B: Con Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Opción C: Con Swarm (si prefieres mantener Swarm)
docker swarm init  # si no está inicializado
docker stack deploy -c docker-compose.production.yml mltutor
```

### Paso 5: Verificar Despliegue

```bash
# Verificar que el contenedor está corriendo
docker ps

# Debería mostrar:
# CONTAINER ID   IMAGE           STATUS         PORTS                    NAMES
# xxxxx          mltutor:latest  Up X seconds   0.0.0.0:8501->8501/tcp   ...

# Ver logs
docker-compose -f docker-compose.production.yml logs -f
# O si usaste Swarm:
# docker service logs -f mltutor_mltutor

# Esperar a ver:
# "You can now view your Streamlit app in your browser"
```

### Paso 6: Probar la Aplicación

```bash
# Prueba básica con curl
curl -I http://localhost:8501

# Debería devolver: HTTP/1.1 200 OK

# Abrir en navegador
open http://localhost:8501  # macOS
# o visita manualmente: http://localhost:8501
```

### Paso 7: Configurar Apache (si aplica)

Si tienes Apache como proxy frontal, actualiza la configuración:

```apache
# /etc/apache2/sites-available/mltutor.conf

<VirtualHost *:443>
    ServerName mltutor.tudominio.com
    
    # ... configuración SSL ...
    
    # Headers para proxy correcto
    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Forwarded-Port "443"
    
    # WebSocket para Streamlit
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule ^/?(.*) "ws://127.0.0.1:8501/$1" [P,L]
    
    # Tráfico HTTP normal
    ProxyPass / http://127.0.0.1:8501/
    ProxyPassReverse / http://127.0.0.1:8501/
    
    # Timeouts
    ProxyTimeout 300
    
    # Tamaño de subida
    LimitRequestBody 209715200
</VirtualHost>
```

```bash
# Activar módulos necesarios
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite headers

# Verificar configuración
sudo apache2ctl configtest

# Recargar Apache
sudo systemctl reload apache2
```

### Paso 8: Monitoreo Post-Migración

```bash
# Crear script de monitoreo simple
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "Contenedores:"
    docker ps --filter name=mltutor --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "Uso de recursos:"
    docker stats --no-stream --filter name=mltutor
    echo ""
    echo "Estado HTTP:"
    curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8501
    echo ""
    sleep 30
done
EOF

chmod +x monitor.sh

# Ejecutar en otra terminal
./monitor.sh
```

## 🔧 Troubleshooting

### Problema: El contenedor no inicia

```bash
# Ver logs detallados
docker-compose -f docker-compose.production.yml logs

# Verificar configuración
docker-compose -f docker-compose.production.yml config

# Verificar imagen
docker images | grep mltutor

# Reconstruir imagen
docker build -t mltutor:latest -f Dockerfile .
```

### Problema: "Address already in use" (puerto 8501)

```bash
# Verificar qué está usando el puerto
lsof -i :8501
# o en Linux:
# sudo netstat -tlnp | grep 8501

# Detener el proceso conflictivo
docker stop $(docker ps -q --filter "publish=8501")

# O cambiar el puerto en docker-compose.production.yml
# ports:
#   - "8502:8501"  # Usar 8502 externamente
```

### Problema: Apache no puede conectar a localhost:8501

```bash
# Verificar que el contenedor acepta conexiones
docker exec $(docker ps -q --filter name=mltutor) netstat -tlnp | grep 8501

# Verificar firewall
sudo ufw status
# Asegurar que 8501 está permitido desde localhost

# Probar conexión desde host
curl -v http://localhost:8501/healthz
```

### Problema: WebSocket no funciona

```bash
# Verificar headers en navegador (F12 > Network > WS)
# Debe mostrar: Status 101 Switching Protocols

# Verificar configuración de Apache
sudo apache2ctl -M | grep -E "proxy|websocket|rewrite"

# Debe mostrar:
#  proxy_module
#  proxy_http_module
#  proxy_wstunnel_module
#  rewrite_module
```

## 📊 Verificación Final

### Checklist Post-Migración

- [ ] Contenedor corriendo (`docker ps`)
- [ ] Healthcheck pasa (`curl http://localhost:8501/healthz`)
- [ ] Aplicación carga en navegador
- [ ] Puedes navegar entre páginas
- [ ] Puedes entrenar un modelo
- [ ] Las visualizaciones se muestran correctamente
- [ ] No hay errores en logs
- [ ] Apache conecta correctamente (si aplica)
- [ ] WebSocket funciona (en DevTools > Network)
- [ ] Rendimiento aceptable (< 2s carga página)

### Métricas a Monitorear

```bash
# CPU y Memoria
docker stats mltutor_mltutor_1

# Logs de errores
docker logs mltutor_mltutor_1 2>&1 | grep -i error

# Peticiones exitosas vs fallidas (en Apache)
tail -f /var/log/apache2/access.log | grep mltutor
```

## 🎉 Migración Completa

Si todos los checks pasan, ¡felicidades! Has migrado exitosamente a una configuración estable.

### Mantener la Configuración

```bash
# Agregar al crontab para auto-reinicio si cae
crontab -e

# Agregar:
*/5 * * * * cd /path/to/mltutor && docker-compose -f docker-compose.production.yml up -d

# O usar systemd para gestion como servicio
```

### Backup Automático

```bash
# Script de backup diario
cat > /etc/cron.daily/mltutor-backup << 'EOF'
#!/bin/bash
BACKUP_DIR=/var/backups/mltutor
mkdir -p $BACKUP_DIR
docker run --rm -v mltutor_mltutor-data:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/data-$(date +%Y%m%d).tar.gz -C /data .
# Mantener solo últimos 7 días
find $BACKUP_DIR -name "data-*.tar.gz" -mtime +7 -delete
EOF

chmod +x /etc/cron.daily/mltutor-backup
```

## 📚 Siguientes Pasos

1. **Monitorear durante 1 semana** para asegurar estabilidad
2. **Documentar incidencias** si las hay
3. **Ajustar recursos** si es necesario:
   ```yaml
   # En docker-compose.production.yml
   resources:
     limits:
       cpus: '6.0'  # Aumentar si hace falta
       memory: 6G
   ```
4. **Evaluar rendimiento** con usuarios reales
5. **Si necesitas más capacidad**: leer `COMPARATIVA_DESPLIEGUE.md`

## 🆘 Soporte

Si encuentras problemas:

1. Revisa logs: `docker-compose -f docker-compose.production.yml logs`
2. Consulta `SOLUCION_SWARM.md` para entender el problema original
3. Verifica que no haya procesos conflictivos en el puerto 8501
4. Asegúrate de que Docker tiene suficientes recursos asignados

## 📞 Contacto

Para problemas específicos de la configuración, abre un issue en el repositorio con:
- Logs completos del contenedor
- Configuración de Apache (si aplica)
- Salida de `docker ps` y `docker stats`
- Descripción detallada del problema
