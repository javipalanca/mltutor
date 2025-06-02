"""
Este módulo contiene funciones para visualizar árboles de decisión.
Integra diferentes tipos de visualizaciones disponibles en el proyecto.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree, export_text

# Funciones para visualizaciones estándar


def create_static_tree_visualization(tree_model, feature_names, class_names=None, figsize=(10, 6)):
    """
    Crea una visualización estática de un árbol de decisión.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    figsize : tuple, default=(10, 6)
        Tamaño de la figura en pulgadas

    Returns:
    --------
    fig : Figure
        Figura de matplotlib con la visualización del árbol
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        proportion=True,
        impurity=True
    )
    plt.tight_layout()
    return fig


def get_tree_text(tree_model, feature_names):
    """
    Obtiene una representación textual del árbol de decisión.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características

    Returns:
    --------
    str
        Representación textual del árbol
    """
    return export_text(
        tree_model,
        feature_names=feature_names,
        show_weights=True
    )


def render_visualization(viz_type, tree_model, feature_names, class_names=None,
                         X=None, y=None, fps=1.0, use_improved=True, show_math=True):
    """
    Renderiza una visualización del árbol de decisión en Streamlit.

    Parameters:
    -----------
    viz_type : str
        Tipo de visualización ('standard', 'text', 'animated', 'interactive', 'explanatory', 'boundary')
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    X : array, opcional
        Datos de características (requerido para 'boundary')
    y : array, opcional
        Etiquetas (requerido para 'boundary')
    fps : float, default=1.0
        Cuadros por segundo para la animación
    use_improved : bool, default=True
        Si se debe usar la visualización mejorada (v2)
    show_math : bool, default=True
        Si se deben mostrar las fórmulas matemáticas
    """
    if viz_type == 'standard':
        # Visualización estándar con matplotlib
        fig = create_static_tree_visualization(
            tree_model, feature_names, class_names)
        st.pyplot(fig)

    elif viz_type == 'text':
        # Visualización como texto
        tree_text = get_tree_text(tree_model, feature_names)
        st.text(tree_text)

        # Mostrar visualización estándar como fallback
        fig = create_static_tree_visualization(
            tree_model, feature_names, class_names)
        st.pyplot(fig)
