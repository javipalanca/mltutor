"""
Este módulo contiene funciones para visualizar árboles de decisión en diferentes formatos.
Incluye funciones para renderizar diferentes tipos de visualizaciones en la aplicación MLTutor.
"""

import streamlit as st
import numpy as np
from sklearn.tree import plot_tree, export_text, DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt

# Verificar disponibilidad de módulos de visualización


def check_visualization_modules():
    """
    Verifica qué módulos de visualización están disponibles.
    Nota: Se ha simplificado para mantener solo visualizaciones estáticas.

    Returns:
    --------
    dict
        Diccionario con la disponibilidad de cada módulo
    """
    # Eliminamos las verificaciones para módulos dinámicos
    # Solo mantenemos el diccionario para compatibilidad
    availability = {
        "animated": False,
        "interactive": False,
        "interactive_improved": False,
        "explanatory": False
    }

    return availability

# Funciones para visualización estándar


def create_tree_visualization(tree_model, feature_names, class_names=None, figsize=(12, 8)):
    """
    Crea una visualización estática del árbol de decisión con elementos explicativos.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    figsize : tuple, default=(12, 8)
        Tamaño de la figura

    Returns:
    --------
    fig : Figure
        Figura de matplotlib con la visualización
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Añadir título informativo
    is_classifier = isinstance(tree_model, DecisionTreeClassifier)
    max_depth = tree_model.get_params()['max_depth']
    criterion = tree_model.get_params()['criterion']
    title = f"Árbol de {'Clasificación' if is_classifier else 'Regresión'} (Profundidad: {max_depth}, Criterio: {criterion})"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Asegurarse de que feature_names y class_names sean listas
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    elif not isinstance(feature_names, list):
        feature_names = list(feature_names)

    if class_names is not None:
        if hasattr(class_names, 'tolist'):
            class_names = class_names.tolist()
        elif not isinstance(class_names, list):
            class_names = list(class_names)

    # Crear la visualización del árbol con parámetros para mayor claridad
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        proportion=True,
        impurity=True,
        precision=2,  # Mostrar valores decimales con precisión 2
        fontsize=10   # Tamaño de fuente para mejor legibilidad
    )

    # Añadir leyenda para los colores en clasificación
    if is_classifier and class_names:
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors

        # Generar colores para la leyenda
        cmap = plt.cm.Blues if len(class_names) <= 2 else plt.cm.viridis
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(class_names))]

        # Crear parches para la leyenda
        patches = [mpatches.Patch(color=colors[i], label=class_names[i])
                   for i in range(len(class_names))]

        # Añadir la leyenda
        ax.legend(handles=patches, title="Clases", loc='upper right',
                  bbox_to_anchor=(1.1, 1), frameon=True, fontsize=9)

    plt.tight_layout()
    return fig


def get_tree_text(tree_model, feature_names):
    """
    Genera una representación textual del árbol.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión
    feature_names : list
        Nombres de las características

    Returns:
    --------
    str
        Representación textual del árbol
    """
    # Asegurarse de que feature_names sea una lista
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    elif not isinstance(feature_names, list):
        feature_names = list(feature_names)

    return export_text(
        tree_model,
        feature_names=feature_names,
        show_weights=True
    )

# Función para renderizar las visualizaciones en Streamlit


def render_tree_visualization(viz_type, tree_model, feature_names, class_names=None,
                              X=None, y=None, fps=1.0, use_improved=True, show_math=True):
    """
    Renderiza una visualización del árbol en Streamlit.
    Nota: Se ha simplificado para mantener solo visualizaciones estáticas.

    Parameters:
    -----------
    viz_type : str
        Tipo de visualización ('standard', 'text')
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    X : array, opcional
        No utilizado en visualizaciones estáticas
    y : array, opcional
        No utilizado en visualizaciones estáticas
    fps : float, default=1.0
        No utilizado en visualizaciones estáticas
    use_improved : bool, default=True
        No utilizado en visualizaciones estáticas
    show_math : bool, default=True
        No utilizado en visualizaciones estáticas
    """
    # Verificar disponibilidad de módulos
    availability = check_visualization_modules()

    if viz_type == 'standard':
        # Visualización estándar
        fig = create_tree_visualization(tree_model, feature_names, class_names)
        st.pyplot(fig)
        return fig

    elif viz_type == 'text':
        # Visualización textual
        tree_text = get_tree_text(tree_model, feature_names)
        st.text(tree_text)
        return tree_text

    else:
        # Visualización no válida o no disponible
        st.warning(
            f"La visualización '{viz_type}' no está disponible. Solo se admiten 'standard' y 'text'.")
        # Fallback a visualización estándar
        fig = create_tree_visualization(tree_model, feature_names, class_names)
        st.pyplot(fig)
        return fig
