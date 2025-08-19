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
from algorithms.code_examples import (
    CONFUSION_MATRIX_CODE,
    PRECISION_CODE,
    PRED_VS_REAL_CODE,
    ERROR_DISTRIBUTION_CODE,
    DECISION_PATH_CODE,
)


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


def show_detailed_evaluation(y_test, y_pred, class_names, model_type):
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
    model_type : str
        Tipo de modelo ('Clasificación' o 'Regresión')
    """
    if model_type == "Clasificación":
        # Preparar etiquetas y nombres de clase de forma robusta
        if class_names is None:
            labels = sorted(np.unique(y_test))
            target_names = [str(l) for l in labels]
        else:
            # Si class_names está presente, asumir que las clases son 0..n-1
            try:
                labels = list(range(len(class_names)))
                target_names = [str(n) for n in class_names]
            except Exception:
                labels = sorted(np.unique(y_test))
                target_names = [str(l) for l in labels]

        # Calcular métricas usando labels y target_names
        report = classification_report(
            y_test, y_pred, labels=labels, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Métricas globales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            acc_value = report.get('accuracy', 0.0)
            st.metric("Exactitud (Accuracy)", f"{acc_value:.4f}",
                      help="Proporción total de predicciones correctas")
            # Indicador de calidad
            if acc_value >= 0.9:
                st.success("Excelente", icon="🌟")
            elif acc_value >= 0.8:
                st.info("Muy bueno", icon="👍")
            elif acc_value >= 0.7:
                st.warning("Bueno", icon="⚠️")
            else:
                st.warning("Necesita mejora", icon="❌")
        with col2:
            st.metric("Precisión media", f"{report.get('weighted avg', {}).get('precision', 0.0):.4f}",
                      help="Media ponderada de la precisión de cada clase")
        with col3:
            st.metric("Exhaustividad media (Recall)", f"{report.get('weighted avg', {}).get('recall', 0.0):.4f}",
                      help="Media ponderada de la exhaustividad de cada clase")
        with col4:
            st.metric("F1-Score medio", f"{report.get('weighted avg', {}).get('f1-score', 0.0):.4f}",
                      help="Media armónica de precisión y exhaustividad")
            f1_score = report.get('weighted avg', {}).get('f1-score', 0.0)
            if f1_score >= 0.8:
                st.success("🌟 Excelente balance")
            elif f1_score >= 0.7:
                st.info("👍 Buen balance")
            else:
                st.warning("⚠️ Balance mejorable")

        # Métricas por clase
        st.markdown("### ⚖️ Matriz de Confusión")

        # Excluir filas avg y accuracy del dataframe
        report_by_class = report_df.drop(
            ['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))  # Reducir tamaño
        xticks = target_names if target_names is not None else None
        yticks = target_names if target_names is not None else None
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=xticks, yticklabels=yticks)
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
            code_cm = CONFUSION_MATRIX_CODE
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
                 **Accuracy (Exactitud):**
                    - Porcentaje de predicciones correctas del total
                    - Rango: 0 a 1 (0% a 100%)
                    - **Interpretación:** Valores más altos = mejor modelo
                    - **Cuidado:** Puede ser engañosa con clases desbalanceadas
                    
                    **Precision (Precisión):**
                    - De todas las predicciones positivas, cuántas fueron correctas
                    - Fórmula: VP / (VP + FP)
                    - Importante cuando los falsos positivos son costosos
                    - **Interpretación:** Valores más altos = mejor modelo
                    
                    **Recall (Sensibilidad o Exhaustividad):**
                    - De todos los casos positivos reales, cuántos detectó el modelo
                    - Fórmula: VP / (VP + FN)
                    - Importante cuando los falsos negativos son costosos
                    - **Interpretación:** Valores más altos = mejor modelo
                    
                    **F1-Score:**
                    - Media armónica entre precisión y recall
                    - Fórmula: 2 × (Precisión × Recall) / (Precisión + Recall)
                    - Útil cuando necesitas balance entre precisión y recall
                    - **Interpretación:** Valores más altos = mejor balance
                    
                    **Curva ROC:**
                    - Muestra el rendimiento en diferentes umbrales de decisión
                    - AUC (Área bajo la curva): 0.5 = aleatorio, 1.0 = perfecto
                    
                    **Curva Precision-Recall:**
                    - Especialmente útil para clases desbalanceadas
                    - Muestra el trade-off entre precisión y recall
                    
                    **VP = Verdaderos Positivos, FP = Falsos Positivos, FN = Falsos Negativos**
                """)

        # Visualización avanzada - Predicciones correctas e incorrectas
        st.markdown("### Visualización de Predicciones")

        # Construir mapping seguro label -> name
        label_to_name = {labels[i]: target_names[i]
                         for i in range(len(labels))}

        # Crear dataframe con resultados
        results_df = pd.DataFrame({
            'Real': [label_to_name.get(v, str(v)) for v in y_test],
            'Predicción': [label_to_name.get(v, str(v)) for v in y_pred],
            'Correcto': (np.array(y_test) == np.array(y_pred))
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
        prec_by_class = {}
        for name in target_names:
            prec_by_class[name] = report.get(name, {}).get('precision', 0.0)
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
        code_prec = PRECISION_CODE
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
        code_pred = PRED_VS_REAL_CODE
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
        code_err = ERROR_DISTRIBUTION_CODE
        show_code_with_download(
            code_err, "Código para generar este gráfico", "distribucion_errores.py")

        with st.expander("📊 Explicación de métricas de regresión"):
            st.markdown("""
            **R² Score (Coeficiente de Determinación):**
            - Mide qué tan bien el modelo explica la variabilidad de los datos
            - Indica qué proporción de la varianza en la variable dependiente es predecible. 1 es predicción perfecta, 0 significa que el modelo no es mejor que predecir la media.
            - Rango: 0 a 1 (valores negativos indican un modelo muy malo)
            - **Interpretación:**
                - R² = 1.0: El modelo explica perfectamente toda la variabilidad
                - R² = 0.8: El modelo explica el 80% de la variabilidad (muy bueno)
                - R² = 0.5: El modelo explica el 50% de la variabilidad (moderado)
                - R² = 0.0: El modelo no explica nada de la variabilidad
            
            **MAE (Error Absoluto Medio):**
            - Promedio de las diferencias absolutas entre valores reales y predichos
            - Se expresa en las mismas unidades que la variable objetivo
            - **Interpretación:** Valores más bajos = mejor modelo
            
            **RMSE (Raíz del Error Cuadrático Medio):**
            - Similar al MAE pero penaliza más los errores grandes
            - Se expresa en las mismas unidades que la variable objetivo
            - **Interpretación:** Valores más bajos = mejor modelo
            """)

    # Mostrar interpretación contextual
    st.markdown("### 🔍 Interpretación de Resultados")

    if model_type != "Clasificación":
        # Métricas para regresión
        mse = mean_squared_error(y_test, y_pred)
        rmse_value = np.sqrt(mse)
        mae_value = mean_absolute_error(y_test, y_pred)
        r2_value = r2_score(y_test, y_pred)

        interpretation = "**Resumen del Modelo:**\n"
        interpretation += f"- Tu modelo de regresión lineal explica **{r2_value*100:.1f}%** de la variabilidad en los datos\n"
        interpretation += f"- En promedio, las predicciones se desvían **{mae_value:.2f} unidades** del valor real (MAE)\n"
        interpretation += f"- La raíz del error cuadrático medio es **{rmse_value:.2f} unidades** (RMSE)\n"

        if rmse_value > mae_value * 1.5:
            interpretation += "\n- ⚠️ El RMSE es significativamente mayor que el MAE, lo que indica la presencia de algunos errores grandes"

        st.info(interpretation)

    else:
        acc_value = report["accuracy"]
        prec_value = report['weighted avg']["precision"]
        rec_value = report['weighted avg']["recall"]
        f1_value = report['weighted avg']['f1-score']

        interpretation = "**Resumen del Modelo:**\n"
        interpretation += f"- Tu modelo clasifica correctamente **{acc_value*100:.1f}%** de los casos\n"
        interpretation += f"- De las predicciones positivas, **{prec_value*100:.1f}%** son correctas (Precisión)\n"
        interpretation += f"- Detecta **{rec_value*100:.1f}%** de todos los casos positivos reales (Recall)\n"

        if f1_value > 0:
            interpretation += f"- El F1-Score (balance entre precisión y recall) es **{f1_value:.3f}**\n\n"

        # Análisis de balance entre precisión y recall
        if abs(prec_value - rec_value) > 0.1:
            if prec_value > rec_value:
                interpretation += "- ⚖️ **Balance:** El modelo es más preciso pero menos sensible (más conservador)\n"
                interpretation += "- 💡 **Sugerencia:** Si es importante detectar todos los casos positivos, considera ajustar el umbral de decisión\n\n"
            else:
                interpretation += "- ⚖️ **Balance:** El modelo es más sensible pero menos preciso (más liberal)\n"
                interpretation += "- 💡 **Sugerencia:** Si es importante evitar falsos positivos, considera ajustar el umbral de decisión\n\n"
        else:
            interpretation += "- ⚖️ **Balance:** Buen equilibrio entre precisión y recall\n\n"

        # Análisis específico del accuracy
        if acc_value < 0.6:
            interpretation += "🔍 **Recomendaciones para mejorar:**\n"
            interpretation += "- Revisar la calidad y cantidad de datos de entrenamiento\n"
            interpretation += "- Considerar ingeniería de características adicionales\n"
            interpretation += "- Probar diferentes algoritmos de clasificación\n"
            interpretation += "- Verificar si hay desbalance de clases\n"

        st.info(interpretation)


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
        node_index = node_indicator.indices[node_indicator.indptr[0]
            :node_indicator.indptr[1]]

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
        code_path = DECISION_PATH_CODE
        show_code_with_download(
            code_path, "Código para generar el camino de decisión", "camino_decision.py")
    except Exception as e:
        st.error(f"Error al mostrar el camino de decisión: {str(e)}")
        st.info(
            "Intenta reformatear los datos de entrada o verificar que el modelo sea compatible.")


def neural_network_diagnostics(history, config):
    st.markdown("### 🔍 Diagnóstico del Entrenamiento")

    # Calcular métricas diagnósticas
    diagnostics = analyze_training_diagnostics(history, config)

    # Mostrar diagnósticos en pestañas
    st.markdown("### 🎯 Estado General")
    show_general_health(diagnostics, history, config)

    st.markdown("### 📊 Tendencias")
    show_trend_analysis(diagnostics, history)

    st.markdown("### ⚠️ Alertas")
    show_training_alerts(diagnostics, history, config)

    # SECCIÓN 5: RECOMENDACIONES INTELIGENTES
    st.markdown("### 💡 Recomendaciones")
    recommendations = generate_training_recommendations(
        diagnostics, history, config)

    if recommendations['excellent']:
        st.success("🌟 " + recommendations['excellent'])
    elif recommendations['good']:
        st.info("✅ " + recommendations['good'])
    elif recommendations['warning']:
        st.warning("⚠️ " + recommendations['warning'])
    elif recommendations['critical']:
        st.error("🚨 " + recommendations['critical'])

    # SECCIÓN 6: ACCIONES SUGERIDAS
    if recommendations.get('actions'):
        with st.expander("🔧 Acciones Sugeridas", expanded=False):
            for action in recommendations['actions']:
                st.markdown(f"• {action}")


def analyze_training_diagnostics(history, config):
    """Analiza el historial INCLUYENDO métricas de rendimiento real."""
    diagnostics = {}

    loss_values = history.history['loss']
    epochs = len(loss_values)

    # 1. Análisis de convergencia de pérdida (como antes)
    if epochs >= 10:
        early_loss = np.mean(loss_values[:epochs//4])
        late_loss = np.mean(loss_values[-epochs//4:])
        convergence_rate = (early_loss - late_loss) / early_loss
    else:
        convergence_rate = (loss_values[0] - loss_values[-1]) / loss_values[0]

    diagnostics['convergence_rate'] = convergence_rate
    diagnostics['loss_converged'] = convergence_rate > 0.01

    # 2. Análisis de sobreajuste (como antes)
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        train_loss = loss_values

        window = min(5, epochs//2)
        recent_gap = np.mean(val_loss[-window:]) - \
            np.mean(train_loss[-window:])
        relative_gap = recent_gap / np.mean(train_loss[-window:])

        diagnostics['overfitting_gap'] = relative_gap
        diagnostics['is_overfitting'] = relative_gap > 0.15
    else:
        diagnostics['overfitting_gap'] = 0
        diagnostics['is_overfitting'] = False

    # 3. Análisis de estabilidad (como antes)
    if epochs >= 5:
        recent_losses = loss_values[-5:]
        stability = np.std(recent_losses) / np.mean(recent_losses)
        diagnostics['stability'] = stability
        diagnostics['loss_stable'] = stability < 0.05
    else:
        diagnostics['stability'] = float('inf')
        diagnostics['loss_stable'] = False

    # 🚀 4. NUEVO: ANÁLISIS DE RENDIMIENTO REAL
    if config['task_type'] == 'Clasificación':
        if 'accuracy' in history.history:
            # Accuracy de entrenamiento
            train_acc = history.history['accuracy'][-1]
            diagnostics['final_train_accuracy'] = train_acc

            # Accuracy de validación (si existe)
            if 'val_accuracy' in history.history:
                val_acc = history.history['val_accuracy'][-1]
                diagnostics['final_val_accuracy'] = val_acc

                # 🎯 CRITERIOS REALISTAS DE CALIDAD
                diagnostics['good_train_accuracy'] = train_acc > 0.7
                diagnostics['good_val_accuracy'] = val_acc > 0.7
                diagnostics['excellent_val_accuracy'] = val_acc > 0.85

                # Detectar sobreajuste por accuracy
                acc_gap = train_acc - val_acc
                diagnostics['accuracy_overfitting'] = acc_gap > 0.15

            else:
                # Solo tenemos accuracy de entrenamiento
                diagnostics['good_train_accuracy'] = train_acc > 0.7
                diagnostics['good_val_accuracy'] = False  # No disponible
                diagnostics['excellent_val_accuracy'] = False
                diagnostics['accuracy_overfitting'] = False
        else:
            # No hay métricas de accuracy
            diagnostics['final_train_accuracy'] = None
            diagnostics['good_train_accuracy'] = False
            diagnostics['good_val_accuracy'] = False
            diagnostics['excellent_val_accuracy'] = False
            diagnostics['accuracy_overfitting'] = False

    else:  # Regresión
        if 'mae' in history.history:
            final_mae = history.history['mae'][-1]
            diagnostics['final_mae'] = final_mae

            if 'val_mae' in history.history:
                val_mae = history.history['val_mae'][-1]
                diagnostics['final_val_mae'] = val_mae

                # Para regresión, necesitamos contexto del rango de datos
                # Por ahora, usamos heurísticas generales
                diagnostics['good_mae'] = val_mae < np.mean(
                    history.history['mae'][:5])
            else:
                diagnostics['good_mae'] = final_mae < np.mean(
                    history.history['mae'][:5])
        else:
            diagnostics['good_mae'] = False

    # 🎯 5. EVALUACIÓN INTEGRAL DE CALIDAD
    # Ahora consideramos TANTO la curva de pérdida COMO el rendimiento real

    if config['task_type'] == 'Clasificación':
        # Para clasificación, el accuracy es lo más importante
        if diagnostics.get('good_val_accuracy', False):
            diagnostics['overall_quality'] = 'excellent' if diagnostics.get(
                'excellent_val_accuracy', False) else 'good'
        elif diagnostics.get('good_train_accuracy', False):
            diagnostics['overall_quality'] = 'moderate'  # Solo bueno en train
        else:
            diagnostics['overall_quality'] = 'poor'  # Accuracy bajo
    else:
        # Para regresión
        if diagnostics.get('good_mae', False):
            diagnostics['overall_quality'] = 'good'
        else:
            diagnostics['overall_quality'] = 'poor'

    # 6. Combinar criterios de pérdida Y rendimiento
    diagnostics['converged'] = (
        diagnostics['loss_converged'] and
        diagnostics['overall_quality'] in ['excellent', 'good']
    )

    diagnostics['is_stable'] = (
        diagnostics['loss_stable'] and
        diagnostics['overall_quality'] != 'poor'
    )

    return diagnostics


def show_general_health(diagnostics, history, config):
    """Muestra el estado general INCLUYENDO rendimiento real."""

    health_score = 0
    total_checks = 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Checklist de Salud:**")

        # 1. Convergencia (pérdida + rendimiento)
        if diagnostics['converged']:
            st.success("✅ Modelo convergió con buen rendimiento")
            health_score += 1
        else:
            if diagnostics['loss_converged']:
                st.warning("⚠️ Pérdida convergió pero rendimiento bajo")
            else:
                st.error("❌ Modelo no convergió suficientemente")
        total_checks += 1

        # 2. Sobreajuste (pérdida + accuracy)
        overfitting_detected = diagnostics['is_overfitting'] or diagnostics.get(
            'accuracy_overfitting', False)
        if not overfitting_detected:
            st.success("✅ Sin signos de sobreajuste")
            health_score += 1
        else:
            if diagnostics['is_overfitting']:
                st.error(
                    f"❌ Sobreajuste en pérdida (gap: {diagnostics['overfitting_gap']*100:.1f}%)")
            if diagnostics.get('accuracy_overfitting', False):
                train_acc = diagnostics.get('final_train_accuracy', 0)
                val_acc = diagnostics.get('final_val_accuracy', 0)
                st.error(
                    f"❌ Sobreajuste en accuracy (train: {train_acc:.3f}, val: {val_acc:.3f})")
        total_checks += 1

        # 3. Estabilidad
        if diagnostics['is_stable']:
            st.success("✅ Entrenamiento estable")
            health_score += 1
        else:
            st.warning(
                f"⚠️ Entrenamiento inestable (CV: {diagnostics['stability']*100:.1f}%)")
        total_checks += 1

        # 🚀 4. NUEVO: RENDIMIENTO REAL
        if config['task_type'] == 'Clasificación':
            if diagnostics.get('excellent_val_accuracy', False):
                st.success("🌟 Excelente accuracy de validación")
                health_score += 1
            elif diagnostics.get('good_val_accuracy', False):
                st.success("✅ Buen accuracy de validación")
                health_score += 1
            elif diagnostics.get('good_train_accuracy', False):
                st.warning("⚠️ Solo buen accuracy en entrenamiento")
            else:
                st.error("❌ Accuracy bajo (modelo no está aprendiendo bien)")
        else:
            if diagnostics.get('good_mae', False):
                st.success("✅ Error de regresión aceptable")
                health_score += 1
            else:
                st.error("❌ Error de regresión alto")
        total_checks += 1

    with col2:
        st.markdown("**🎯 Puntuación de Salud:**")

        health_percentage = (health_score / total_checks) * 100

        # 🎯 NUEVA LÓGICA: Considerar rendimiento real
        overall_quality = diagnostics.get('overall_quality', 'poor')

        if overall_quality == 'excellent' and health_percentage >= 75:
            st.success(f"🌟 Excelente: {health_percentage:.0f}%")
            health_status = "🌟 Modelo listo para producción"
        elif overall_quality == 'good' and health_percentage >= 60:
            st.info(f"👍 Bueno: {health_percentage:.0f}%")
            health_status = "👍 Modelo con buen rendimiento"
        elif overall_quality == 'moderate':
            st.warning(f"⚠️ Moderado: {health_percentage:.0f}%")
            health_status = "⚠️ Modelo necesita validación adicional"
        else:
            st.error(f"🚨 Crítico: {health_percentage:.0f}%")
            health_status = "🚨 Modelo no está funcionando correctamente"

        st.info(health_status)

        # Mostrar métricas específicas
        if config['task_type'] == 'Clasificación':
            if 'final_val_accuracy' in diagnostics and diagnostics['final_val_accuracy'] is not None:
                val_acc = diagnostics['final_val_accuracy']
                delta_color = "normal" if val_acc > 0.7 else "inverse"
                st.metric("🎯 Accuracy Validación",
                          f"{val_acc:.3f}",
                          delta_color=delta_color,
                          help="Métrica más importante para clasificación")
            elif 'final_train_accuracy' in diagnostics and diagnostics['final_train_accuracy'] is not None:
                train_acc = diagnostics['final_train_accuracy']
                st.metric("🎯 Accuracy Entrenamiento",
                          f"{train_acc:.3f}",
                          help="Solo disponible accuracy de entrenamiento")
        else:
            if 'final_val_mae' in diagnostics:
                st.metric("📏 MAE Validación",
                          f"{diagnostics['final_val_mae']:.4f}",
                          help="Error promedio en validación")


def generate_training_recommendations(diagnostics, history, config):
    """Genera recomendaciones basadas en pérdida Y rendimiento real."""

    recommendations = {
        'excellent': None,
        'good': None,
        'warning': None,
        'critical': None,
        'actions': []
    }

    # 🎯 NUEVA LÓGICA: Priorizar rendimiento real sobre curvas
    overall_quality = diagnostics.get('overall_quality', 'poor')

    # Contar problemas reales
    real_issues = []

    if not diagnostics['converged']:
        real_issues.append('convergencia')
    if diagnostics['is_overfitting'] or diagnostics.get('accuracy_overfitting', False):
        real_issues.append('sobreajuste')
    if not diagnostics['is_stable']:
        real_issues.append('estabilidad')
    if overall_quality == 'poor':
        real_issues.append('rendimiento_bajo')

    # Generar recomendaciones basadas en problemas reales
    if overall_quality == 'excellent' and len(real_issues) == 0:
        recommendations['excellent'] = "¡Entrenamiento excelente! Modelo con alto rendimiento y buenas curvas."
        recommendations['actions'] = [
            "✅ El modelo está listo para producción",
            "📊 Considera hacer validación cruzada para confirmar robustez",
            "🚀 Puedes proceder a hacer predicciones con confianza"
        ]
    elif overall_quality in ['good'] and len(real_issues) <= 1:
        recommendations['good'] = "Entrenamiento bueno con rendimiento satisfactorio."

        # Acciones específicas basadas en el problema
        if 'sobreajuste' in real_issues:
            recommendations['actions'].extend([
                "🔧 Añadir más regularización (dropout, L1/L2)",
                "📈 Aumentar datos de entrenamiento si es posible"
            ])
        elif 'estabilidad' in real_issues:
            recommendations['actions'].extend([
                "📉 Reducir learning rate para mayor estabilidad",
                "📦 Aumentar batch size"
            ])
        else:
            recommendations['actions'].append("👍 Continuar con este enfoque")

    elif overall_quality == 'moderate' or len(real_issues) == 2:
        recommendations['warning'] = "Rendimiento moderado. El modelo funciona pero tiene limitaciones importantes."
        recommendations['actions'].extend([
            "📊 Revisar datos de validación - pueden no ser representativos",
            "🔄 Considerar reentrenar con diferentes hiperparámetros",
            "🎯 Evaluar si la arquitectura es apropiada para el problema"
        ])
    else:
        # Rendimiento crítico
        if overall_quality == 'poor':
            recommendations['critical'] = f"⚠️ CRÍTICO: El modelo tiene muy bajo rendimiento (accuracy ≤ 70%). Las curvas pueden verse bien pero el modelo no está aprendiendo correctamente."
        else:
            recommendations['critical'] = "El entrenamiento tiene múltiples problemas serios."

        recommendations['actions'].extend([
            "🚨 PRIORIDAD: Revisar datos de entrada y preprocesamiento",
            "🏗️ Simplificar arquitectura del modelo",
            "📚 Verificar que el problema sea realmente solucionable con estos datos",
            "🔄 Considerar cambiar completamente de enfoque"
        ])

    # Añadir acciones específicas para problemas de rendimiento
    if overall_quality == 'poor':
        recommendations['actions'].insert(
            0, "🎯 El modelo no está aprendiendo patrones útiles - revisar datos y arquitectura")

    return recommendations


def show_trend_analysis(diagnostics, history):
    """Muestra análisis de tendencias."""

    loss_values = history.history['loss']
    epochs = len(loss_values)

    st.markdown("### 📈 Análisis de Tendencias por Fase")
    with st.expander("📚 ¿Qué significan las tendencias de pérdida?", expanded=False):
        st.markdown("""
        ### 🎯 **Interpretación de Tendencias por Fase:**
        
        **🟢 Fase Inicial (Primeras épocas):**
        - **Descendente rápido:** ✅ Excelente - El modelo está aprendiendo patrones básicos
        - **Descendente lento:** ⚠️ Learning rate muy bajo o datos complejos
        - **Ascendente:** 🚨 Learning rate muy alto o problema en los datos
        
        **🟡 Fase Media (Épocas intermedias):**
        - **Descendente sostenido:** ✅ Aprendizaje progresivo saludable
        - **Plateau temprano:** ⚠️ Posible saturación o learning rate muy bajo
        - **Fluctuaciones:** 📊 Normal, pero pueden indicar batch size pequeño
        
        **🔴 Fase Final (Últimas épocas):**
        - **Estabilizado:** 🎯 Ideal - Convergencia alcanzada
        - **Descendente:** 📈 Aún aprendiendo - Considera más épocas
        - **Ascendente:** 🚨 Sobreajuste - Detener entrenamiento antes
        
        ### 📊 **Patrones de Calidad:**
        - **Curva logarítmica suave:** Patrón ideal de aprendizaje
        - **Escalones descendentes:** Learning rate scheduling efectivo
        - **Zigzag descendente:** Normal con batch gradient descent
        - **Valle en U:** Posible learning rate muy alto inicialmente
        """)
    if epochs >= 10:
        col1, col2, col3 = st.columns(3)
        # Dividir en segmentos para análisis
        early_segment = loss_values[:epochs//3]
        middle_segment = loss_values[epochs//3:2*epochs//3]
        late_segment = loss_values[2*epochs//3:]

        early_trend = np.polyfit(
            range(len(early_segment)), early_segment, 1)[0]
        middle_trend = np.polyfit(
            range(len(middle_segment)), middle_segment, 1)[0]
        late_trend = np.polyfit(
            range(len(late_segment)), late_segment, 1)[0]

        with col1:
            # Interpretación detallada para fase inicial
            direction = "Descendente" if early_trend < 0 else "Ascendente"

            if early_trend < -0.1:
                trend_quality = "🚀 Excelente"
                trend_help = "Aprendizaje inicial muy efectivo"
            elif early_trend < -0.01:
                trend_quality = "✅ Bueno"
                trend_help = "Aprendizaje inicial satisfactorio"
            elif early_trend < 0:
                trend_quality = "⚠️ Lento"
                trend_help = "Aprendizaje inicial lento - considera aumentar learning rate"
            else:
                trend_quality = "🚨 Problema"
                trend_help = "Pérdida aumentando - revisar learning rate y datos"

            st.metric("🟢 Inicio (33%)",
                      f"{early_trend:.6f}",
                      f"{direction} - {trend_quality}",
                      help=trend_help)
        with col2:
            # Interpretación detallada para fase media
            direction = "Descendente" if middle_trend < 0 else "Ascendente"

            if middle_trend < -0.01:
                trend_quality = "📈 Progresando"
                trend_help = "Aprendizaje continuo saludable"
            elif middle_trend < 0:
                trend_quality = "📊 Lento"
                trend_help = "Aprendizaje desacelerando - normal en fases medias"
            elif abs(middle_trend) < 0.001:
                trend_quality = "🎯 Plateau"
                trend_help = "Posible convergencia temprana"
            else:
                trend_quality = "⚠️ Subiendo"
                trend_help = "Pérdida aumentando - posible sobreajuste"

            st.metric("🟡 Medio (33%)",
                      f"{middle_trend:.6f}",
                      f"{direction} - {trend_quality}",
                      help=trend_help)
        with col3:
            # Interpretación detallada para fase final
            direction = "Descendente" if late_trend < 0 else "Ascendente"

            if abs(late_trend) < 0.001:
                trend_quality = "🎯 Convergido"
                trend_help = "Excelente - modelo estabilizado"
            elif late_trend < -0.01:
                trend_quality = "📈 Mejorando"
                trend_help = "Aún aprendiendo - considera más épocas"
            elif late_trend < 0:
                trend_quality = "📊 Lento"
                trend_help = "Mejora marginal - cerca de convergencia"
            else:
                trend_quality = "🚨 Sobreajuste"
                trend_help = "Pérdida aumentando - detener entrenamiento"

            st.metric("🔴 Final (33%)",
                      f"{late_trend:.6f}",
                      f"{direction} - {trend_quality}",
                      help=trend_help)

         # Análisis comparativo entre fases
        st.markdown("### 🔄 Análisis Comparativo")

        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            st.markdown("**📊 Velocidad de Aprendizaje:**")

            # Comparar velocidades de aprendizaje
            speeds = [abs(early_trend), abs(middle_trend), abs(late_trend)]
            phase_names = ["Inicial", "Media", "Final"]
            fastest_phase = phase_names[speeds.index(max(speeds))]

            st.info(f"🏃‍♂️ **Fase más activa:** {fastest_phase}")

            # Detectar aceleración o desaceleración
            if abs(early_trend) > abs(middle_trend) > abs(late_trend):
                st.success("✅ Desaceleración natural - Patrón ideal")
            elif abs(late_trend) > abs(early_trend):
                st.warning("⚠️ Aceleración tardía - Revisar parámetros")
            else:
                st.info("📊 Patrón mixto de aprendizaje")

        with col_comp2:
            st.markdown("**🎯 Consistencia:**")

            # Análisis de consistencia
            trend_consistency = np.std([early_trend, middle_trend, late_trend])

            if trend_consistency < 0.01:
                st.success("🎯 Alta consistencia - Aprendizaje estable")
            elif trend_consistency < 0.1:
                st.info("📊 Consistencia moderada - Normal")
            else:
                st.warning("⚠️ Baja consistencia - Aprendizaje irregular")

            # Detectar patrones específicos
            if early_trend < 0 and middle_trend < 0 and late_trend >= 0:
                st.error("🚨 Patrón de sobreajuste detectado")
            elif all(t < 0 for t in [early_trend, middle_trend, late_trend]):
                st.success("✅ Mejora sostenida en todas las fases")
    else:
        st.info("📊 Historial muy corto para análisis detallado")
        st.markdown("""
        **🔍 Para un análisis completo necesitas:**
        - ✅ Al menos 10 épocas de entrenamiento
        - 📊 Datos de validación (recomendado)
        - 🎯 Métricas adicionales según el tipo de problema
        """)

    # Sección de patrones detectados con más detalle
    st.markdown("### 🔄 Patrones de Comportamiento Detectados")

    # Detectar patrones más específicos
    patterns = []
    recommendations = []

    # Análisis de plateau
    if diagnostics.get('plateau', False):
        patterns.append(
            "🎯 **Plateau alcanzado:** El modelo ha llegado a su límite de aprendizaje")
        recommendations.append(
            "💡 Considera early stopping o cambiar arquitectura")

    # Análisis de convergencia
    convergence_rate = diagnostics.get('convergence_rate', 0)
    if convergence_rate > 0.5:
        patterns.append(
            "🚀 **Convergencia rápida:** Excelente capacidad de aprendizaje")
        recommendations.append("✅ Parámetros bien ajustados")
    elif convergence_rate > 0.1:
        patterns.append(
            "📈 **Convergencia moderada:** Aprendizaje progresivo saludable")
        recommendations.append("👍 Rendimiento satisfactorio")
    elif convergence_rate > 0.01:
        patterns.append("🐌 **Convergencia lenta:** Aprendizaje gradual")
        recommendations.append(
            "⚠️ Considera aumentar learning rate o revisar datos")
    else:
        patterns.append(
            "❌ **Sin convergencia:** Modelo no está aprendiendo efectivamente")
        recommendations.append("🚨 Revisar completamente configuración y datos")

    # Análisis de sobreajuste
    if diagnostics.get('is_overfitting', False):
        overfitting_gap = diagnostics.get('overfitting_gap', 0) * 100
        patterns.append(
            f"📊 **Sobreajuste progresivo:** Gap train/val del {overfitting_gap:.1f}%")
        recommendations.append("🛑 Implementar regularización o early stopping")

    # Análisis de estabilidad
    if not diagnostics.get('is_stable', True):
        stability = diagnostics.get('stability', 0) * 100
        patterns.append(
            f"📈 **Entrenamiento inestable:** Variabilidad del {stability:.1f}%")
        recommendations.append("🔧 Reducir learning rate o aumentar batch size")

    # Mostrar patrones y recomendaciones
    if patterns:
        col_pat1, col_pat2 = st.columns(2)

        with col_pat1:
            st.markdown("**🔍 Patrones Identificados:**")
            for pattern in patterns:
                st.markdown(f"• {pattern}")

        with col_pat2:
            st.markdown("**💡 Recomendaciones:**")
            for recommendation in recommendations:
                st.markdown(f"• {recommendation}")
    else:
        st.success(
            "🎉 No se detectaron patrones problemáticos - ¡Entrenamiento saludable!")

    # Sección de contexto educativo adicional
    with st.expander("🎓 Contexto Educativo: ¿Cómo interpretar estos números?", expanded=False):
        st.markdown("""
        ### 📐 **Entendiendo los Valores de Tendencia:**
        
        **Valores Negativos (Descendente):**
        - `-0.1` o menor: 🚀 Aprendizaje muy rápido
        - `-0.01` a `-0.1`: ✅ Aprendizaje saludable  
        - `-0.001` a `-0.01`: 📊 Aprendizaje gradual
        - `-0.0001` a `-0.001`: 🎯 Convergencia fina
        
        **Valores Cerca de Cero:**
        - `-0.0001` a `+0.0001`: 🎯 Convergencia ideal
        
        **Valores Positivos (Ascendente):**
        - `+0.0001` a `+0.001`: ⚠️ Ligero deterioro
        - `+0.001` a `+0.01`: 🚨 Problema moderado
        - `+0.01` o mayor: 💥 Problema grave
        
        ### 🔬 **Factores que Afectan las Tendencias:**
        
        **Learning Rate:**
        - Muy alto → Oscilaciones o divergencia
        - Muy bajo → Convergencia lenta
        - Optimal → Descenso suave y rápido
        
        **Batch Size:**
        - Pequeño → Más ruido, convergencia irregular
        - Grande → Menos ruido, convergencia suave
        - Optimal → Balance entre velocidad y estabilidad
        
        **Arquitectura del Modelo:**
        - Muy simple → Plateau temprano (underfitting)
        - Muy compleja → Sobreajuste tardío
        - Apropiada → Convergencia saludable sin sobreajuste
        """)


def show_training_alerts(diagnostics, history, config):
    """Muestra alertas y problemas detectados."""

    alerts = []

    # Alertas críticas
    if not diagnostics['converged']:
        alerts.append({
            'level': 'error',
            'message': 'Modelo no convergió suficientemente',
            'action': 'Aumentar número de épocas o ajustar learning rate'
        })

    if diagnostics['is_overfitting']:
        alerts.append({
            'level': 'error',
            'message': f'Sobreajuste detectado (gap: {diagnostics["overfitting_gap"]*100:.1f}%)',
            'action': 'Reducir complejidad del modelo, añadir regularización o más datos'
        })

    # Alertas de advertencia
    if not diagnostics['is_stable']:
        alerts.append({
            'level': 'warning',
            'message': f'Entrenamiento inestable (variabilidad: {diagnostics["stability"]*100:.1f}%)',
            'action': 'Reducir learning rate o aumentar batch size'
        })

    if diagnostics.get('plateau', False):
        alerts.append({
            'level': 'info',
            'message': 'Plateau detectado en las últimas épocas',
            'action': 'El modelo puede haber alcanzado su límite de aprendizaje'
        })

    # Mostrar alertas
    if not alerts:
        st.success("🎉 ¡No se detectaron problemas significativos!")
    else:
        for alert in alerts:
            if alert['level'] == 'error':
                st.error(f"🚨 **{alert['message']}**\n💡 {alert['action']}")
            elif alert['level'] == 'warning':
                st.warning(f"⚠️ **{alert['message']}**\n💡 {alert['action']}")
            else:
                st.info(f"ℹ️ **{alert['message']}**\n💡 {alert['action']}")
