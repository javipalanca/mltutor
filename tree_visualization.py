"""
Este módulo contiene funciones para visualizar árboles de decisión.
Integra diferentes tipos de visualizaciones disponibles en el proyecto.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree, export_text

# Funciones para visualizaciones estándar


def create_static_tree_visualization(tree_model, feature_names, class_names=None, figsize=(12, 8)):
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
    figsize : tuple, default=(12, 8)
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


# Función wrapper para visualizaciones avanzadas
def create_animated_tree(tree_model, feature_names, class_names=None, fps=1.0):
    """
    Crea una visualización animada del árbol de decisión.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    fps : float, default=1.0
        Cuadros por segundo para la animación

    Returns:
    --------
    str
        HTML con la animación del árbol
    """
    # Importar dinámicamente para evitar errores
    try:
        from tree_animation_enhanced import create_tree_animation
        return create_tree_animation(
            tree_model,
            feature_names,
            class_names,
            fps=fps
        )
    except ImportError:
        return None


def create_static_frames(tree_model, feature_names, class_names=None):
    """
    Crea frames estáticos para la visualización del árbol.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)

    Returns:
    --------
    str
        HTML con los frames estáticos
    """
    try:
        from tree_animation_enhanced import get_static_frames_html
        return get_static_frames_html(
            tree_model,
            feature_names,
            class_names
        )
    except ImportError:
        return None


def create_interactive_visualization(tree_model, feature_names, class_names=None, use_improved=True, show_math=True):
    """
    Crea una visualización interactiva del árbol de decisión.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)
    use_improved : bool, default=True
        Si se debe usar la visualización mejorada (v2)
    show_math : bool, default=True
        Si se deben mostrar las fórmulas matemáticas

    Returns:
    --------
    str
        HTML con la visualización interactiva
    """
    try:
        if use_improved:
            from interactive_tree_v2 import create_interactive_tree
            return create_interactive_tree(
                tree_model,
                feature_names,
                class_names,
                show_math=show_math
            )
        else:
            from interactive_tree import create_interactive_tree
            return create_interactive_tree(
                tree_model,
                feature_names,
                class_names
            )
    except ImportError:
        return None


def create_decision_boundary(tree_model, X, y, feature_names, class_names=None, use_improved=True):
    """
    Crea una visualización del límite de decisión para un modelo de clasificación.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier
        Modelo de árbol de decisión entrenado
    X : array
        Datos de características
    y : array
        Etiquetas
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases
    use_improved : bool, default=True
        Si se debe usar la visualización mejorada (v2)

    Returns:
    --------
    str
        HTML con la visualización del límite de decisión
    """
    try:
        if use_improved:
            from interactive_tree_v2 import create_decision_boundary_plot
            return create_decision_boundary_plot(
                tree_model,
                X,
                y,
                feature_names,
                class_names
            )
        else:
            from interactive_tree import create_decision_boundary_plot
            return create_decision_boundary_plot(
                tree_model,
                X,
                y,
                feature_names,
                class_names
            )
    except ImportError:
        return None


def create_explanatory_visualization(tree_model, feature_names, class_names=None):
    """
    Crea una visualización explicativa del árbol de decisión.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    feature_names : list
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases (para clasificación)

    Returns:
    --------
    str
        HTML con la visualización explicativa
    """
    try:
        from explanatory_tree import create_explanatory_tree_visualization
        return create_explanatory_tree_visualization(
            tree_model,
            feature_names,
            class_names
        )
    except ImportError:
        return None


def check_visualization_availability():
    """
    Verifica qué visualizaciones están disponibles en el sistema.

    Returns:
    --------
    dict
        Diccionario con las disponibilidades de cada visualización
    """
    availability = {
        "animated": False,
        "interactive": False,
        "interactive_improved": False,
        "explanatory": False,
        "mpld3_compatible": False
    }

    # Verificar animación
    try:
        import tree_animation_enhanced
        availability["animated"] = True
    except ImportError:
        pass

    # Verificar interactiva básica
    try:
        import interactive_tree
        availability["interactive"] = True
    except ImportError:
        pass

    # Verificar interactiva mejorada
    try:
        import interactive_tree_v2
        availability["interactive_improved"] = True

        # Verificar compatibilidad mpld3
        try:
            from interactive_tree_v2 import check_mpld3_compatibility
            availability["mpld3_compatible"] = check_mpld3_compatibility()
        except (ImportError, AttributeError):
            pass
    except ImportError:
        pass

    # Verificar explicativa
    try:
        import explanatory_tree
        availability["explanatory"] = True
    except ImportError:
        pass

    return availability


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
    # Verificar disponibilidades
    availability = check_visualization_availability()

    if viz_type == 'standard':
        # Visualización estándar con matplotlib
        fig = create_static_tree_visualization(
            tree_model, feature_names, class_names)
        st.pyplot(fig)

    elif viz_type == 'text':
        # Visualización como texto
        tree_text = get_tree_text(tree_model, feature_names)
        st.text(tree_text)

    elif viz_type == 'animated' and availability["animated"]:
        # Visualización animada
        animation_html = create_animated_tree(
            tree_model, feature_names, class_names, fps)
        if animation_html:
            st.components.v1.html(animation_html, height=600)
        else:
            st.error("No se pudo crear la visualización animada.")

    elif viz_type == 'interactive':
        # Visualización interactiva
        if use_improved and availability["interactive_improved"]:
            interactive_html = create_interactive_visualization(
                tree_model, feature_names, class_names,
                use_improved=True, show_math=show_math
            )
        elif availability["interactive"]:
            interactive_html = create_interactive_visualization(
                tree_model, feature_names, class_names,
                use_improved=False, show_math=show_math
            )
        else:
            interactive_html = None

        if interactive_html:
            st.components.v1.html(interactive_html, height=600)
        else:
            st.error("No se pudo crear la visualización interactiva.")

    elif viz_type == 'explanatory' and availability["explanatory"]:
        # Visualización explicativa
        explanatory_html = create_explanatory_visualization(
            tree_model, feature_names, class_names)
        if explanatory_html:
            st.components.v1.html(explanatory_html, height=800)
        else:
            st.error("No se pudo crear la visualización explicativa.")

    elif viz_type == 'boundary':
        # Visualización del límite de decisión (requiere X e y)
        if X is None or y is None:
            st.error("Se requieren datos X e y para mostrar el límite de decisión.")
            return

        if X.shape[1] != 2:
            st.warning(
                "La visualización del límite de decisión solo está disponible para 2 características.")
            return

        boundary_html = create_decision_boundary(
            tree_model, X, y, feature_names, class_names, use_improved=use_improved
        )

        if boundary_html:
            st.components.v1.html(boundary_html, height=600)
        else:
            st.error("No se pudo crear la visualización del límite de decisión.")
    else:
        st.warning(
            f"La visualización '{viz_type}' no está disponible o no es válida.")

        # Mostrar visualización estándar como fallback
        fig = create_static_tree_visualization(
            tree_model, feature_names, class_names)
        st.pyplot(fig)
