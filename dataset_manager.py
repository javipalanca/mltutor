"""
Este módulo contiene funciones para la carga y gestión de datos en la aplicación MLTutor.
Incluye funciones para cargar conjuntos de datos integrados, cargar archivos CSV y procesar datos.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split


def load_builtin_dataset(dataset_name):
    """
    Carga un conjunto de datos integrado según el nombre.

    Parameters:
    -----------
    dataset_name : str
        Nombre del conjunto de datos ('Iris', 'Vino', 'Cáncer')

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases
    dataset_info : str
        Información sobre el conjunto de datos
    """
    if "Iris" in dataset_name:
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="species")
        class_names = data.target_names.tolist()
        dataset_info = "Dataset Iris: 150 muestras, 4 características, 3 clases de flores"
        task_type = "Clasificación"

    elif "Vino" in dataset_name:
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="wine_type")
        class_names = data.target_names.tolist()
        dataset_info = "Dataset Vino: 178 muestras, 13 características, 3 tipos de vino"
        task_type = "Clasificación"

    elif "Cáncer" in dataset_name:
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="malignant")
        class_names = data.target_names.tolist()
        dataset_info = "Dataset Cáncer: 569 muestras, 30 características, diagnóstico binario"
        task_type = "Clasificación"

    else:
        raise ValueError(f"Conjunto de datos '{dataset_name}' no reconocido")

    # Asegurarse de que feature_names sea una lista
    feature_names = data.feature_names.tolist() if hasattr(
        data.feature_names, 'tolist') else list(data.feature_names)

    return X, y, feature_names, class_names, dataset_info, task_type


def load_dataset_from_file(file_path, target_column=None, task_type="auto"):
    """
    Carga un conjunto de datos desde un archivo CSV.

    Parameters:
    -----------
    file_path : str
        Ruta al archivo CSV
    target_column : str, optional
        Nombre de la columna objetivo. Si es None, se usa la última columna.
    task_type : str, default="auto"
        Tipo de tarea ('Clasificación', 'Regresión', 'auto')

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    feature_names : list
        Nombres de las características
    class_names : list or None
        Nombres de las clases (solo para clasificación)
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea ('Clasificación' o 'Regresión')
    """
    # Cargar el dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")

    # Determinar la columna objetivo
    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        raise ValueError(
            f"Columna objetivo '{target_column}' no encontrada en el dataset")

    # Separar características y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Obtener nombres de características
    feature_names = X.columns.tolist()

    # Determinar tipo de tarea si es automático
    if task_type == "auto":
        # Si y tiene pocos valores únicos (menos del 5% del total), asumimos clasificación
        if len(y.unique()) < max(5, len(y) * 0.05):
            task_type = "Clasificación"
        else:
            task_type = "Regresión"

    # Para clasificación, determinar nombres de clases
    if task_type == "Clasificación":
        class_values = sorted(y.unique())
        # Intentar usar etiquetas si están disponibles
        if hasattr(y, 'cat') and hasattr(y.cat, 'categories'):
            class_names = y.cat.categories.tolist()
        else:
            # Si son números enteros, usar nombres genéricos
            if all(isinstance(val, (int, np.integer)) for val in class_values):
                class_names = [f"Clase {i}" for i in class_values]
            else:
                class_names = [str(val) for val in class_values]
    else:
        class_names = None

    # Generar información del dataset
    n_samples, n_features = X.shape
    if task_type == "Clasificación":
        n_classes = len(class_names)
        dataset_info = f"Dataset personalizado: {n_samples} muestras, {n_features} características, {n_classes} clases"
    else:
        dataset_info = f"Dataset personalizado: {n_samples} muestras, {n_features} características, regresión"

    return X, y, feature_names, class_names, dataset_info, task_type


def list_sample_datasets():
    """
    Lista los conjuntos de datos de ejemplo disponibles en el directorio 'data/sample_datasets'.

    Returns:
    --------
    list
        Lista de rutas a los archivos CSV de ejemplo
    """
    sample_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data", "sample_datasets")
    if not os.path.exists(sample_dir):
        return []

    return [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith('.csv')
    ]


def preprocess_data(X, y, test_size=0.3, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : Series o array
        Variable objetivo
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Conjuntos de entrenamiento y prueba
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_data(dataset_option):
    """
    Carga un conjunto de datos según la opción seleccionada.

    Parameters:
    -----------
    dataset_option : str
        Nombre del conjunto de datos a cargar

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    feature_names : list
        Nombres de las características
    class_names : list o None
        Nombres de las clases (para clasificación)
    dataset_info : str
        Información sobre el conjunto de datos
    """
    # Conjuntos de datos integrados
    builtin_datasets = [
        "Iris (clasificación de flores)",
        "Vino (clasificación de vinos)",
        "Cáncer de mama (diagnóstico)"
    ]

    if dataset_option in builtin_datasets:
        return load_builtin_dataset(dataset_option)

    # Comprobar si es un conjunto de datos personalizado
    if dataset_option.endswith('.csv'):
        file_path = dataset_option
        if not os.path.exists(file_path):
            # Comprobar si está en el directorio de muestras
            sample_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "data", "sample_datasets")
            sample_path = os.path.join(sample_dir, os.path.basename(file_path))
            if os.path.exists(sample_path):
                file_path = sample_path

        return load_dataset_from_file(file_path)

    # Si llegamos aquí, el conjunto de datos no se reconoce
    raise ValueError(f"Conjunto de datos '{dataset_option}' no reconocido")


def create_dataset_selector():
    """
    Crea un selector de conjuntos de datos para la interfaz de usuario.

    Returns:
    --------
    str
        Opción de conjunto de datos seleccionada
    """
    # Conjuntos de datos integrados
    builtin_options = [
        "Iris (clasificación de flores)",
        "Vino (clasificación de vinos)",
        "Cáncer de mama (diagnóstico)"
    ]

    # Conjuntos de datos de ejemplo
    sample_files = list_sample_datasets()
    sample_options = [os.path.basename(f) for f in sample_files]

    # Opción de cargar archivo personalizado
    has_upload = st.checkbox("Cargar mi propio conjunto de datos", value=False)

    if has_upload:
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV", type=["csv"])
        if uploaded_file is not None:
            # Guardar el archivo temporalmente
            temp_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "data", "temp", uploaded_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Mostrar vista previa
            df = pd.read_csv(temp_path)
            st.write("Vista previa de los datos:")
            st.dataframe(df.head())

            # Configuración adicional
            target_col = st.selectbox(
                "Columna objetivo:", df.columns.tolist(), index=len(df.columns)-1)
            task_type = st.radio("Tipo de tarea:", [
                                 "auto", "Clasificación", "Regresión"])

            # Devolver ruta al archivo temporal y configuración
            return temp_path, target_col, task_type

        # Si no hay archivo cargado, mostrar los conjuntos integrados y de ejemplo
        dataset_option = st.selectbox(
            "Dataset de ejemplo:",
            builtin_options + sample_options
        )
    else:
        # Si no se selecciona cargar archivo, mostrar solo los conjuntos integrados y de ejemplo
        dataset_option = st.selectbox(
            "Dataset de ejemplo:",
            builtin_options + sample_options
        )

    return dataset_option


def preprocess_data(X, y, test_size=0.3, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Conjuntos de entrenamiento y prueba
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
