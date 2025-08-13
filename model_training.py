"""
Este módulo contiene funciones para entrenar modelos de árboles de decisión, modelos lineales y K-Nearest Neighbors.
Incluye funciones para crear, entrenar y evaluar modelos de clasificación y regresión.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import streamlit as st
from dataset_manager import preprocess_data
from model_evaluation import evaluate_classification_model, evaluate_regression_model


def train_decision_tree_classifier(X, y, max_depth, min_samples_split, criterion, test_size=0.3, random_state=42):
    """
    Crea y entrena un modelo de árbol de decisión para clasificación.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio para medir la calidad de una división
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y, test_size=test_size, random_state=random_state)

    # Crear y entrenar el modelo
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    evaluation = evaluate_classification_model(y_test, y_pred, list(set(y)))

    # Devolver resultados
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "evaluation": evaluation
    }


def train_decision_tree_regressor(X, y, max_depth, min_samples_split, criterion, test_size=0.3, random_state=42):
    """
    Crea y entrena un modelo de árbol de decisión para regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio para medir la calidad de una división
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y, test_size=test_size, random_state=random_state)

    # Crear y entrenar el modelo
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    evaluation = evaluate_regression_model(y_test, y_pred)

    # Devolver resultados
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "evaluation": evaluation
    }


def train_decision_tree(X, y, tree_type, max_depth, min_samples_split, criterion, test_size=0.3, random_state=42):
    """
    Crea y entrena un modelo de árbol de decisión para clasificación o regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    tree_type : str
        Tipo de árbol ('Clasificación' o 'Regresión')
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio para medir la calidad de una división
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    if tree_type == "Clasificación":
        return train_decision_tree_classifier(
            X, y, max_depth, min_samples_split, criterion, test_size, random_state)
    else:
        return train_decision_tree_regressor(
            X, y, max_depth, min_samples_split, criterion, test_size, random_state)


def predict_sample(model, X_new):
    """
    Realiza una predicción para una nueva muestra.

    Parameters:
    -----------
    model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo entrenado
    X_new : array
        Características de la muestra a predecir

    Returns:
    --------
    prediction : array
        Predicción del modelo
    proba : array, optional
        Probabilidades de cada clase (solo para clasificación)
    """
    # Realizar predicción
    prediction = model.predict([X_new])

    # Si es un clasificador y soporta probabilidades, calcular probabilidades
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([X_new])[0]
        return prediction, proba

    return prediction, None


def train_linear_regression(X, y, test_size=0.3, random_state=42):
    """
    Crea y entrena un modelo de regresión lineal.

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
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y, test_size=test_size, random_state=random_state)

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    test_results = evaluate_regression_model(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_results": test_results
    }


def train_logistic_regression(X, y, max_iter=1000, test_size=0.3, random_state=42):
    """
    Crea y entrena un modelo de regresión logística.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    max_iter : int, default=1000
        Número máximo de iteraciones para el algoritmo de optimización
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y, test_size=test_size, random_state=random_state)

    # Crear y entrenar el modelo
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    test_results = evaluate_classification_model(y_test, y_pred, None)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_results": test_results
    }


def train_linear_model(X, y, model_type="Linear", max_iter=1000, test_size=0.3, random_state=42):
    """
    Crea y entrena un modelo lineal (regresión lineal o logística).

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    model_type : str, default="Linear"
        Tipo de modelo ("Linear" para regresión lineal, "Logistic" para regresión logística)
    max_iter : int, default=1000
        Número máximo de iteraciones (solo para regresión logística)
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    if model_type == "Linear":
        return train_linear_regression(X, y, test_size, random_state)
    else:  # Logistic
        return train_logistic_regression(X, y, max_iter, test_size, random_state)


def train_knn_classifier(X, y, n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    Crea y entrena un modelo de K-Nearest Neighbors para clasificación.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    n_neighbors : int, default=5
        Número de vecinos cercanos a considerar
    weights : str, default='uniform'
        Función de peso ('uniform', 'distance')
    metric : str, default='minkowski'
        Métrica de distancia ('minkowski', 'euclidean', 'manhattan')

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """

    # Crear y entrenar el modelo
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    model.fit(X, y)

    return model


def train_knn_regressor(X, y, n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    Crea y entrena un modelo de K-Nearest Neighbors para regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    n_neighbors : int, default=5
        Número de vecinos cercanos a considerar
    weights : str, default='uniform'
        Función de peso ('uniform', 'distance')
    metric : str, default='minkowski'
        Métrica de distancia ('minkowski', 'euclidean', 'manhattan')

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """

    # Crear y entrenar el modelo
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    model.fit(X, y)

    # Devolver resultados
    return model


def train_knn_model(X, y, task_type="Clasificación", n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    Crea y entrena un modelo de K-Nearest Neighbors para clasificación o regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    task_type : str, default="Clasificación"
        Tipo de tarea ('Clasificación' o 'Regresión')
    n_neighbors : int, default=5
        Número de vecinos cercanos a considerar
    weights : str, default='uniform'
        Función de peso ('uniform', 'distance')
    metric : str, default='minkowski'
        Métrica de distancia ('minkowski', 'euclidean', 'manhattan')
    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    if task_type == "Clasificación":
        return train_knn_classifier(
            X, y, n_neighbors, weights, metric)
    else:
        return train_knn_regressor(
            X, y, n_neighbors, weights, metric)
