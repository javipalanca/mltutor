"""
Este módulo contiene funciones para entrenar modelos de árboles de decisión.
Incluye funciones para crear, entrenar y evaluar modelos de clasificación y regresión.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
