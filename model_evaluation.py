"""
Este módulo contiene funciones para evaluar modelos de árboles de decisión.
Incluye funciones para calcular métricas y visualizar resultados de clasificación y regresión.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, mean_squared_error, confusion_matrix,
    classification_report, precision_recall_fscore_support,
    r2_score, mean_absolute_error

)

from utils import get_image_download_link, show_code_with_download


def evaluate_classification_model(y_test, y_pred, class_names):
    """
    Evalúa un modelo de clasificación y devuelve las métricas principales.

    Parameters:
    -----------
    y_test : array
        Valores reales
    y_pred : array
        Valores predichos
    class_names : list
        Nombres de las clases

    Returns:
    --------
    dict
        Diccionario con las métricas del modelo
    """
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Extraer precision y recall promedio ponderado para compatibilidad
    precision = report.get('weighted avg', {}).get('precision', 0.0)
    recall = report.get('weighted avg', {}).get('recall', 0.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


def evaluate_regression_model(y_test, y_pred):
    """
    Evalúa un modelo de regresión y devuelve las métricas principales.

    Parameters:
    -----------
    y_test : array
        Valores reales
    y_pred : array
        Valores predichos

    Returns:
    --------
    dict
        Diccionario con las métricas del modelo
    """
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # np.mean(np.abs(y_test - y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # 1 - np.sum((y_test - y_pred)**2) / \
    # np.sum((y_test - np.mean(y_test))**2)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred
    }


def show_detailed_evaluation(y_test, y_pred, class_names, tree_type):
    """
    Muestra una evaluación detallada del modelo con gráficos y explicaciones.

    Parameters:
    -----------
    y_test : array
        Valores reales
    y_pred : array
        Valores predichos
    class_names : list
        Nombres de las clases
    tree_type : str
        Tipo de árbol ('Clasificación' o 'Regresión')
    """
    if tree_type == "Clasificación":
        # Calcular métricas
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Métricas globales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Exactitud (Accuracy)", f"{report['accuracy']:.4f}",
                      help="Proporción total de predicciones correctas")
        with col2:
            st.metric("Precisión media", f"{report['weighted avg']['precision']:.4f}",
                      help="Media ponderada de la precisión de cada clase")
        with col3:
            st.metric("Exhaustividad media", f"{report['weighted avg']['recall']:.4f}",
                      help="Media ponderada de la exhaustividad de cada clase")
        with col4:
            st.metric("F1-Score medio", f"{report['weighted avg']['f1-score']:.4f}",
                      help="Media armónica de precisión y exhaustividad")

        # Métricas por clase
        st.markdown("### Métricas por clase")

        # Excluir filas avg y accuracy del dataframe
        report_by_class = report_df.drop(
            ['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))  # Reducir tamaño
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=class_names, yticklabels=class_names)
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')
        ax_cm.set_title('Matriz de Confusión')

        col1, col2 = st.columns([1, 1])
        with col1:
            # Mostrar con tamaño reducido
            col_inner1, col_inner2, col_inner3 = st.columns([1, 3, 1])
            with col_inner2:
                st.pyplot(fig_cm, use_container_width=True)

            st.markdown(get_image_download_link(
                fig_cm, "matriz_confusion", "📥 Descargar matriz de confusión"), unsafe_allow_html=True)

            # Mostrar código para generar la matriz de confusión
            code_cm = """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusión')

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
            show_code_with_download(
                code_cm, "Código para generar la matriz de confusión", "matriz_confusion.py")

        with col2:
            st.dataframe(report_by_class.style.format({
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1-score': '{:.4f}',
                'support': '{:.0f}'
            }))

            with st.expander("📊 Explicación de métricas"):
                st.markdown("""
                - **Precisión**: De todas las muestras que se predijeron como clase X, ¿qué porcentaje eran realmente X?
                - **Exhaustividad (Recall)**: De todas las muestras que realmente son clase X, ¿qué porcentaje se predijo correctamente?
                - **F1-Score**: Media armónica de precisión y exhaustividad. Útil cuando las clases están desbalanceadas.
                - **Support**: Número de muestras de cada clase en el conjunto de prueba.
                """)

        # Visualización avanzada - Predicciones correctas e incorrectas
        st.markdown("### Visualización de Predicciones")

        # Crear dataframe con resultados
        results_df = pd.DataFrame({
            'Real': [class_names[x] for x in y_test],
            'Predicción': [class_names[x] for x in y_pred],
            'Correcto': y_test == y_pred
        })

        # Mostrar algunas muestras
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("#### Muestras correctamente clasificadas")
            correct_samples = results_df[results_df['Correcto']].head(10)
            st.dataframe(correct_samples, height=300)

        with col_viz2:
            st.markdown("#### Muestras incorrectamente clasificadas")
            incorrect_samples = results_df[~results_df['Correcto']].head(10)
            if len(incorrect_samples) > 0:
                st.dataframe(incorrect_samples, height=300)
            else:
                st.info("¡Todas las muestras fueron clasificadas correctamente!")

        # Gráfico de precisión por clase
        fig_prec, ax_prec = plt.subplots(figsize=(8, 4))  # Reducir altura
        prec_by_class = {
            class_name: report[class_name]['precision'] for class_name in class_names}
        sns.barplot(x=list(prec_by_class.keys()), y=list(
            prec_by_class.values()), ax=ax_prec)
        ax_prec.set_ylim(0, 1)
        ax_prec.set_title('Precisión por clase')
        ax_prec.set_ylabel('Precisión')
        ax_prec.set_xlabel('Clase')

        # Mostrar con tamaño reducido
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.pyplot(fig_prec, use_container_width=True)

        st.markdown(get_image_download_link(
            fig_prec, "precision_por_clase", "📥 Descargar gráfico de precisión"), unsafe_allow_html=True)

        # Mostrar código para generar el gráfico de precisión por clase
        code_prec = """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd

# Obtener el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

# Extraer precisión por clase
prec_by_class = {
    class_name: report[class_name]['precision'] for class_name in class_names
}

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=list(prec_by_class.keys()), y=list(prec_by_class.values()), ax=ax)
ax.set_ylim(0, 1)
ax.set_title('Precisión por clase')
ax.set_ylabel('Precisión')
ax.set_xlabel('Clase')

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
        show_code_with_download(
            code_prec, "Código para generar el gráfico de precisión", "precision_por_clase.py")
    else:
        # Métricas para regresión
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Error Cuadrático Medio (MSE)", f"{mse:.4f}",
                      help="Promedio de los errores al cuadrado. Penaliza más los errores grandes.")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}",
                      help="Raíz cuadrada del MSE. En las mismas unidades que la variable objetivo.")
        with col3:
            st.metric("Error Absoluto Medio (MAE)", f"{mae:.4f}",
                      help="Promedio de los errores absolutos. Menos sensible a valores atípicos.")
        with col4:
            st.metric("R² Score", f"{r2:.4f}",
                      help="Proporción de la varianza explicada por el modelo. 1 es predicción perfecta.")

        # Gráfico de predicciones vs valores reales
        fig, ax = plt.subplots(figsize=(6, 5))  # Reducir tamaño
        scatter = ax.scatter(y_test, y_pred, alpha=0.5,
                             c=np.abs(y_test - y_pred), cmap='viridis')
        ax.plot([y_test.min(), y_test.max()], [
                y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Valores reales')
        ax.set_ylabel('Predicciones')
        ax.set_title('Predicciones vs Valores reales')
        plt.colorbar(scatter, ax=ax, label='Error absoluto')

        # Mostrar con tamaño reducido
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.pyplot(fig, use_container_width=True)

        st.markdown(get_image_download_link(
            fig, "predicciones_vs_reales", "📥 Descargar gráfico"), unsafe_allow_html=True)

        # Mostrar el código para generar el gráfico de predicciones vs valores reales
        code_pred = """
import matplotlib.pyplot as plt
import numpy as np

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(y_test, y_pred, alpha=0.5,
                    c=np.abs(y_test - y_pred), cmap='viridis')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Valores reales')
ax.set_ylabel('Predicciones')
ax.set_title('Predicciones vs Valores reales')
plt.colorbar(scatter, ax=ax, label='Error absoluto')

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
        show_code_with_download(
            code_pred, "Código para generar este gráfico", "predicciones_vs_reales.py")

        # Distribución de errores
        fig_err, ax_err = plt.subplots(figsize=(6, 4))  # Reducir tamaño
        errors = y_test - y_pred
        sns.histplot(errors, kde=True, ax=ax_err)
        ax_err.axvline(x=0, color='r', linestyle='--')
        ax_err.set_title('Distribución de errores')
        ax_err.set_xlabel('Error (Real - Predicción)')

        # Mostrar con tamaño reducido
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.pyplot(fig_err, use_container_width=True)

        st.markdown(get_image_download_link(
            fig_err, "distribucion_errores", "📥 Descargar gráfico"), unsafe_allow_html=True)

        # Mostrar el código para generar el gráfico de distribución de errores
        code_err = """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calcular los errores
errors = y_test - y_pred

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(errors, kde=True, ax=ax)
ax.axvline(x=0, color='r', linestyle='--')
ax.set_title('Distribución de errores')
ax.set_xlabel('Error (Real - Predicción)')

# Para mostrar en Streamlit
# st.pyplot(fig)

# Para uso normal en Python/Jupyter
# plt.tight_layout()
# plt.show()
"""
        show_code_with_download(
            code_err, "Código para generar este gráfico", "distribucion_errores.py")

        with st.expander("📊 Explicación de métricas de regresión"):
            st.markdown("""
            - **MSE (Error Cuadrático Medio)**: Promedio de los errores al cuadrado. Penaliza más los errores grandes.
            - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del MSE. Está en las mismas unidades que la variable objetivo.
            - **MAE (Error Absoluto Medio)**: Promedio de los valores absolutos de los errores. Menos sensible a valores atípicos que MSE.
            - **R² Score**: Indica qué proporción de la varianza en la variable dependiente es predecible. 1 es predicción perfecta, 0 significa que el modelo no es mejor que predecir la media.
            """)


def show_prediction_path(tree_model, X_new, feature_names, class_names=None):
    """
    Muestra el camino de decisión para una predicción específica.

    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo entrenado
    X_new : array
        Ejemplo para predecir
    feature_names : list
        Nombres de las características
    class_names : list, optional
        Nombres de las clases (solo para clasificación)
    """
    # Convertir a numpy array si no lo es ya
    X_new = np.asarray(X_new)

    # Asegurarse de que X_new tenga el formato correcto (2D array con un solo ejemplo)
    if X_new.ndim > 2:
        # Si tiene más de 2 dimensiones, aplanamos a 2D
        X_new = X_new.reshape(1, -1)
    elif X_new.ndim == 1:
        # Si es un vector 1D, lo convertimos a 2D
        X_new = X_new.reshape(1, -1)

    # Obtener información del árbol
    feature_idx = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold

    try:
        # Construir el camino de decisión
        node_indicator = tree_model.decision_path(X_new)
        leaf_id = tree_model.apply(X_new)

        # Obtener los nodos en el camino
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        path_explanation = []
        for node_id in node_index:
            # Detener si es un nodo hoja
            if leaf_id[0] == node_id:
                continue

            # Obtener la característica y el umbral de la decisión
            feature_id = feature_idx[node_id]
            feature_name = feature_names[feature_id]
            threshold_value = threshold[node_id]

            # Comprobar si la muestra va por la izquierda o derecha
            # Usamos X_new[0, feature_id] porque X_new es un array 2D con un solo ejemplo
            if X_new[0, feature_id] <= threshold_value:
                path_explanation.append(
                    f"- {feature_name} = {X_new[0, feature_id]:.4f} ≤ {threshold_value:.4f} ✅")
            else:
                path_explanation.append(
                    f"- {feature_name} = {X_new[0, feature_id]:.4f} > {threshold_value:.4f} ✅")

        # Mostrar el camino de decisión
        st.markdown("\n".join(path_explanation))

        # Mostrar el código que genera este camino de decisión
        code_path = """
import numpy as np

def mostrar_camino_decision(tree_model, X_nuevo, feature_names, class_names=None):
    \"\"\"
    Muestra el camino de decisión para un ejemplo específico.
    
    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    X_nuevo : array
        Ejemplo para predecir (debe ser un solo ejemplo)
    feature_names : list
        Nombres de las características
    class_names : list, optional
        Nombres de las clases (solo para clasificación)
    \"\"\"
    # Asegurar que X_nuevo sea un array numpy 2D con una sola fila
    X_nuevo = np.asarray(X_nuevo).reshape(1, -1)
    
    # Obtener información del árbol
    feature_idx = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    
    # Construir el camino de decisión
    node_indicator = tree_model.decision_path(X_nuevo)
    leaf_id = tree_model.apply(X_nuevo)
    
    # Obtener los nodos en el camino
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    # Mostrar el camino paso a paso
    print("Camino de decisión:")
    for node_id in node_index:
        # Detener si es un nodo hoja
        if leaf_id[0] == node_id:
            continue
            
        # Obtener la característica y el umbral de la decisión
        feature_id = feature_idx[node_id]
        feature_name = feature_names[feature_id]
        threshold_value = threshold[node_id]
        
        # Comprobar si la muestra va por la izquierda o derecha
        if X_nuevo[0, feature_id] <= threshold_value:
            print(f"- {feature_name} = {X_nuevo[0, feature_id]:.4f} ≤ {threshold_value:.4f}")
        else:
            print(f"- {feature_name} = {X_nuevo[0, feature_id]:.4f} > {threshold_value:.4f}")
    
    # Mostrar la predicción final
    prediccion = tree_model.predict(X_nuevo)[0]
    if hasattr(tree_model, 'classes_') and class_names:
        print(f"Predicción final: {class_names[prediccion]}")
    else:
        print(f"Predicción final: {prediccion:.4f}")

# Ejemplo de uso:
# mostrar_camino_decision(tree_model, X_nuevo, feature_names, class_names)
"""
        show_code_with_download(
            code_path, "Código para generar el camino de decisión", "camino_decision.py")
    except Exception as e:
        st.error(f"Error al mostrar el camino de decisión: {str(e)}")
        st.info(
            "Intenta reformatear los datos de entrada o verificar que el modelo sea compatible.")
