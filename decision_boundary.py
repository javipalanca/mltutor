import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd


def plot_decision_boundary(model, X, y, ax=None, feature_names=None, class_names=None):
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

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot
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

    # Get unique classes
    classes = np.unique(y)
    n_classes = len(classes)

    # Create colormap
    if n_classes <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.viridis

    # Train a new model using only the selected 2 features
    # This is necessary because the original model might have been trained with more features
    if isinstance(model, DecisionTreeClassifier):
        # For classification
        new_model = DecisionTreeClassifier(
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            criterion=model.criterion,
            random_state=42
        )
    else:
        # For regression
        new_model = DecisionTreeRegressor(
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            criterion=model.criterion,
            random_state=42
        )

    # Train the new model with only the 2 selected features
    new_model.fit(X_plot, y)

    # Create the decision boundary using the new model
    if hasattr(new_model, 'predict_proba'):
        if n_classes == 2:  # Binary classification
            Z = new_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:  # Multi-class classification
            Z = new_model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = new_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape Z to match the mesh grid
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    if n_classes == 2:  # Binary classification
        contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    else:  # Multi-class classification or regression
        contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # Create scatter plot of training data
    if n_classes <= 10:  # Classification
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y,
                             cmap=cmap, edgecolor='k', s=20)
    else:  # Regression
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y,
                             cmap=plt.cm.viridis, edgecolor='k', s=20)

    # Set axis labels
    if feature_names is not None and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    # Add legend
    if class_names is not None and n_classes <= 10:
        legend_labels = [class_names[i] for i in range(n_classes)]
        patches = [mpatches.Patch(color=cmap(i / n_classes), label=legend_labels[i])
                   for i in range(n_classes)]
        ax.legend(handles=patches, loc='lower right')

    # Set title
    ax.set_title('Decision Boundary')

    # Set axis limits
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    return ax


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
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_decision_boundary(model, X, y, ax, feature_names, class_names)
        ax.set_title(f'Decision Boundary - Depth: {i+1}')
        figs.append(fig)

    return figs
