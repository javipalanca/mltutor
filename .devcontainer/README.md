# MLTutor DevContainer para Streamlit

Este directorio contiene la configuración necesaria para ejecutar MLTutor en un entorno de desarrollo containerizado con VS Code.

## Requisitos previos

- [Docker](https://www.docker.com/products/docker-desktop) instalado y funcionando
- [Visual Studio Code](https://code.visualstudio.com/) con la extensión [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Cómo usar

1. Abre VS Code en el directorio raíz del proyecto
2. VS Code debería detectar la configuración del devcontainer y mostrar un mensaje para reabrir el proyecto en un contenedor. Haz clic en "Reopen in Container"
3. Alternativamente, puedes:
   - Pulsar F1 o Ctrl+Shift+P
   - Escribir "Remote-Containers: Reopen in Container"
   - Presionar Enter

## Ejecutar la aplicación Streamlit

Una vez dentro del contenedor, puedes ejecutar la aplicación Streamlit con el siguiente comando:

```bash
streamlit run streamlit_app.py
```

La aplicación estará disponible en http://localhost:8501

## Características del DevContainer

- Python 3.9 con todas las dependencias del proyecto
- Extensiones de VS Code para desarrollo Python preinstaladas
- Configuración optimizada para Streamlit
- Puerto 8501 expuesto para acceder a la aplicación
- Volumen montado para sincronización de archivos en tiempo real

## Personalización

- Puedes añadir más extensiones o configuraciones en el archivo `devcontainer.json`
- Para añadir más dependencias, modifica el archivo `Dockerfile` o `requirements.txt`
