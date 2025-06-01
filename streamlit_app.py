import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import train_decision_tree
from tree_visualization import visualize_tree
from model_evaluation import evaluate_model
from decision_boundary import plot_decision_boundary

# Configuración de la página
st.set_page_config(
    page_title="MLTutor",
    page_icon="🌲",
    layout="wide"
)

# Título principal
st.title("MLTutor - Visualizador de Árboles de Decisión")
st.markdown("---")

# Barra lateral para cargar datos y configurar parámetros
with st.sidebar:
    st.header("Configuración")
    
    # Selección de dataset
    dataset_option = st.selectbox(
        "Selecciona un dataset:",
        ["Iris", "Breast Cancer", "Wine"]
    )
    
    # Cargar el dataset seleccionado
    if dataset_option == "Iris":
        data = datasets.load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_option == "Breast Cancer":
        data = datasets.load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:  # Wine
        data = datasets.load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    
    # Parámetros del árbol de decisión
    st.subheader("Parámetros del Árbol")
    max_depth = st.slider("Profundidad máxima", 1, 10, 3)
    min_samples_split = st.slider("Mínimo de muestras para dividir", 2, 20, 2)
    min_samples_leaf = st.slider("Mínimo de muestras en hoja", 1, 20, 1)
    
    # Botón para entrenar el modelo
    train_button = st.button("Entrenar Modelo")

# Visualización principal
if 'df' in locals():
    # Mostrar información del dataset
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos del dataset")
        st.dataframe(df.head())
        
    with col2:
        st.subheader("Estadísticas del dataset")
        st.dataframe(df.describe())
    
    # Visualización de características
    st.subheader("Visualización de características")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if dataset_option != "Breast Cancer":  # Demasiadas características para BC
        features = df.drop('target', axis=1).columns[:5]  # Primeras 5 características
        sns.boxplot(data=df[features])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        features = df.drop('target', axis=1).columns[:5]
        sns.boxplot(data=df[features])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Entrenamiento y visualización del modelo
    if train_button:
        with st.spinner('Entrenando modelo...'):
            # Seleccionar todas las características excepto la target
            X = df.drop('target', axis=1).values
            y = df['target'].values
            
            # Entrenar el modelo con los parámetros seleccionados
            model, X_train, X_test, y_train, y_test = train_decision_tree(
                X, y, 
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
            
            # Evaluar el modelo
            accuracy, report = evaluate_model(model, X_test, y_test)
            
            # Mostrar resultados
            st.subheader("Resultados del Modelo")
            st.metric("Precisión del modelo", f"{accuracy:.2%}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Reporte de Clasificación")
                st.text(report)
                
            with col4:
                st.subheader("Visualización del Árbol")
                # Visualizar el árbol (usando solo 2 características para simplificar)
                if X.shape[1] > 2:
                    feature_idx = [0, 1]  # Usar las primeras 2 características
                    tree_img = visualize_tree(model, feature_names=data.feature_names[feature_idx])
                else:
                    tree_img = visualize_tree(model, feature_names=data.feature_names)
                
                st.image(tree_img, use_column_width=True)
            
            # Visualizar frontera de decisión si es posible (solo para 2D)
            if X.shape[1] >= 2:
                st.subheader("Frontera de Decisión")
                # Usar las primeras 2 características para la visualización
                boundary_img = plot_decision_boundary(
                    model, 
                    X_train[:, :2], 
                    y_train,
                    X_test[:, :2],
                    y_test,
                    feature_names=data.feature_names[:2]
                )
                st.image(boundary_img, use_column_width=True)

# Pie de página
st.markdown("---")
st.caption("MLTutor - Herramienta para visualización y aprendizaje de Árboles de Decisión")
