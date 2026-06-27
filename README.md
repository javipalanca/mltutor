# MLTutor - Plataforma de Aprendizaje de Machine Learning

Esta aplicación permite visualizar y comprender algoritmos de machine learning de forma interactiva. Actualmente incluye:

- 🌲 Árboles de Decisión (disponible)
- 📊 Regresión Logística (próximamente)
- 🔍 K-Nearest Neighbors (próximamente)
- 🧠 Redes Neuronales (próximamente)

## Características

- Interfaz interactiva para experimentar con diferentes parámetros
- Navegación mejorada con separación entre página de inicio y algoritmos
- Visualizaciones avanzadas de modelos:
  - **Árbol interactivo completo**: Con explicaciones al pasar el cursor
  - **Árbol paso a paso**: Construcción animada del árbol
  - **Árbol explicativo detallado**: Guía educativa completa
- Métricas de evaluación y explicaciones
- Exportación de modelos en diferentes formatos
- Datasets de ejemplo incluidos

## Ejecutables para estudiantes

Hay ejecutables autocontenidos para Windows, macOS (Apple Silicon) y Linux
que no requieren instalar Python ni dependencias: basta con descargarlos
desde la página de *Releases* del repositorio, descomprimir y ejecutar.
Consulta [BUILD_EXECUTABLES.md](BUILD_EXECUTABLES.md) para los detalles de
generación y las instrucciones de uso por plataforma.

## Requisitos

- Python 3.7 o superior
- Dependencias listadas en requirements.txt

## Instalación

1. Clona o descarga este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Para iniciar la aplicación, ejecuta:

```bash
./run_mltutor.sh
```

o directamente:

```bash
streamlit run app_refactored.py
```

## Estructura del Proyecto

- `app_refactored.py`: Versión actualizada con navegación mejorada
- `streamlit_app.py`: Versión original
- `ui.py`: Componentes de la interfaz de usuario
- `dataset_manager.py`: Gestión y procesamiento de conjuntos de datos
- `model_training.py`: Entrenamiento de modelos de ML
- `model_evaluation.py`: Evaluación y métricas de rendimiento
- `tree_visualizer.py` y `tree_visualization.py`: Visualizaciones para árboles de decisión
- `utils.py`: Funciones auxiliares
- `data/`: Conjunto de datos de ejemplo

## Guía de uso

1. Al abrir la aplicación, verás la página de inicio con información general
2. Selecciona un algoritmo desde el menú lateral o usando los botones de la página principal
3. Configura los parámetros del modelo
4. Explora los datos, entrena el modelo y analiza las visualizaciones

## Novedades en la Versión Actual

- Separación entre página de inicio y páginas específicas de algoritmos
- Interfaz mejorada con tarjetas informativas
- Sistema de navegación optimizado
- Contenido educativo expandido

## Despliegue con Docker

Para construir la imagen:

```bash
docker build -t mltutor .
```

Para ejecutar el contenedor:

```bash
docker run -p 8501:8501 mltutor
```

Luego, accede a la aplicación en tu navegador: http://localhost:8501

## Desarrollado por

Javier Palanca, Universitat Politècnica de València, 2025