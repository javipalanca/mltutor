import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import streamlit as st
from utils import get_image_download_link, show_code_with_download


def plot_decision_boundary(model, X, y, ax=None, feature_names=None, class_names=None, show_code=True):
    """
    Plot the decision boundary of a tree model for 2D data.

    Parameters:
    -----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        The trained tree model
    X : array-like of shape (n_samples, 2)
        The feature data used for training (only first 2 dimensions are used)
    y : array-like of shape (n_samples,)
        The target values
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    feature_names : list, optional
        Names of the features (only first 2 are used)
    class_names : list, optional
        Names of the classes
    show_code : bool, optional
        Whether to show the code that generates this plot

    Returns:
    --------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes with the plot
    """
    # Ensure X is a numpy array and has exactly 2 features
    if isinstance(X, pd.DataFrame):
        if len(X.columns) != 2:
            raise ValueError(
                "For DataFrame input, X must have exactly 2 columns")
        X_plot = X.values
        # If feature_names is not provided, use column names
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        # For numpy arrays
        if X.shape[1] != 2:
            raise ValueError(
                "X must have exactly 2 features for decision boundary visualization")
        X_plot = X

    # Ensure y is a numpy array
    if hasattr(y, 'values'):
        y = y.values

    # Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Get unique classes
    classes = np.unique(y)
    n_classes = len(classes)

    # Train a new model using only the selected 2 features
    # This is necessary because the original model might have been trained with more features
    try:
        if isinstance(model, DecisionTreeClassifier):
            # For classification
            new_model = DecisionTreeClassifier(
                max_depth=getattr(model, 'max_depth', None),
                min_samples_split=getattr(model, 'min_samples_split', 2),
                criterion=getattr(model, 'criterion', 'gini'),
                random_state=42
            )
        else:
            # For regression
            new_model = DecisionTreeRegressor(
                max_depth=getattr(model, 'max_depth', None),
                min_samples_split=getattr(model, 'min_samples_split', 2),
                criterion=getattr(model, 'criterion', 'squared_error'),
                random_state=42
            )

        # Train the new model with only the 2 selected features
        new_model.fit(X_plot, y)

        # Create the decision boundary using the new model
        Z = new_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        if n_classes == 2:  # Binary classification
            contour = ax.contourf(xx, yy, Z, alpha=0.3, levels=50, cmap='RdBu')
        else:  # Multi-class classification or regression
            contour = ax.contourf(xx, yy, Z, alpha=0.3,
                                  levels=50, cmap='viridis')

        # Create scatter plot of training data
        if isinstance(model, DecisionTreeClassifier):  # Classification
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y,
                                 cmap='tab10' if n_classes <= 10 else 'viridis',
                                 edgecolor='k', s=50, alpha=0.8)
        else:  # Regression
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y,
                                 cmap='viridis', edgecolor='k', s=50, alpha=0.8)

        # Set axis labels
        if feature_names is not None and len(feature_names) >= 2:
            ax.set_xlabel(feature_names[0], fontsize=12)
            ax.set_ylabel(feature_names[1], fontsize=12)
        else:
            ax.set_xlabel('Característica 1', fontsize=12)
            ax.set_ylabel('Característica 2', fontsize=12)

        # Add legend for classification
        if class_names is not None and isinstance(model, DecisionTreeClassifier) and n_classes <= 10:
            # Create legend patches
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
            patches = [mpatches.Patch(color=colors[i], label=class_names[i])
                       for i in range(min(len(class_names), n_classes))]
            ax.legend(handles=patches, loc='best', title='Clases')

        # Set title
        ax.set_title('Frontera de Decisión del Árbol',
                     fontsize=14, fontweight='bold')

        # Set axis limits
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.grid(True, alpha=0.3)

    except Exception as e:
        # If something goes wrong, create a simple scatter plot
        ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap='viridis',
                   edgecolor='k', s=50, alpha=0.8)
        ax.set_title(f'Error en frontera de decisión: {str(e)}')
        if feature_names is not None and len(feature_names) >= 2:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])

    # Add explanation text
    explanation_text = """
    📊 Esta visualización muestra:
    • Áreas coloreadas: regiones de decisión para cada clase
    • Puntos: datos de entrenamiento
    • Fronteras: límites donde el modelo cambia de predicción
    """
    ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
                                               facecolor='white', alpha=0.8), fontsize=9)

    # Show code if requested
    if show_code and hasattr(st, 'expander'):
        with st.expander("📝 Ver código para generar esta visualización"):
            code = '''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def plot_decision_boundary(model, X, y, feature_names=None):
    """Visualiza la frontera de decisión de un árbol de decisión."""
    
    # Crear malla de puntos
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Entrenar modelo solo con 2 características
    model_2d = DecisionTreeClassifier(
        max_depth=model.max_depth,
        random_state=42
    )
    model_2d.fit(X, y)
    
    # Predecir en la malla
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', edgecolor='k')
    
    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    
    ax.set_title('Frontera de Decisión')
    plt.show()
'''
            st.code(code, language='python')

    return fig, ax


def create_boundary_animation_frames(models, X, y, feature_names=None, class_names=None):
    """
    Create a list of figures showing decision boundaries for a sequence of models.

    Parameters:
    -----------
    models : list of trained tree models
        List of models with increasing complexity
    X : array-like of shape (n_samples, 2)
        The feature data used for training (only first 2 dimensions are used)
    y : array-like of shape (n_samples,)
        The target values
    feature_names : list, optional
        Names of the features (only first 2 are used)
    class_names : list, optional
        Names of the classes

    Returns:
    --------
    figs : list of matplotlib.figure.Figure
        List of figures with decision boundaries
    """
    figs = []

    for i, model in enumerate(models):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_decision_boundary(model, X, y, ax, feature_names, class_names)
        ax.set_title(f'Decision Boundary - Depth: {i+1}')
        figs.append(fig)

    return figs
