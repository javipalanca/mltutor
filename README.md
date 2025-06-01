# MLTutor - Plataforma de Aprendizaje de Machine Learning

Esta aplicación permite visualizar y comprender algoritmos de machine learning de forma interactiva. Actualmente incluye:

- Árboles de Decisión (disponible)
- Regresión Logística (próximamente)
- K-Nearest Neighbors (próximamente)
- Redes Neuronales (próximamente)

## Características

- Interfaz interactiva para experimentar con diferentes parámetros
- Visualizaciones avanzadas de modelos:
  - **Árbol interactivo completo**: Con explicaciones al pasar el cursor
  - **Árbol paso a paso**: Construcción animada del árbol
  - **Árbol explicativo detallado**: Guía educativa completa
- Métricas de evaluación y explicaciones
- Exportación de modelos en diferentes formatos
- Datasets de ejemplo incluidos

Para más detalles sobre las visualizaciones, consulta [README_visualizaciones.md](README_visualizaciones.md).

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
streamlit run app_streamlit_v2.py
```

## Guía de uso

1. Selecciona el algoritmo que quieres explorar en la barra lateral
2. Selecciona un dataset de ejemplo
3. Configura los parámetros del modelo (específicos para cada algoritmo)
4. Haz clic en "Entrenar" y explora las visualizaciones en las diferentes pestañas

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