from mltutor.desktop import ui as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from itertools import cycle


def plot_roc_curve(y_test, y_pred_proba, average="macro", class_names=None):
    """
    Plots the ROC curve and Precision-Recall curve for the given true labels and predicted probabilities.
    Supports both binary and multi-class classification with macro/micro averaging.

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    average : str, default="macro"
        Averaging method for multi-class ('macro', 'micro')
    class_names : list, optional
        Names of the classes
    """
    n_classes = len(np.unique(y_test))

    if n_classes == 2:
        # Clasificación binaria
        _plot_binary_roc_pr(y_test, y_pred_proba)
    else:
        # Clasificación multiclase
        _plot_multiclass_roc_pr(y_test, y_pred_proba,
                                average, class_names, n_classes)

    # Añadir explicación detallada al final
    _add_detailed_explanation()


def _plot_binary_roc_pr(y_test, y_pred_proba):
    """
    Plots ROC and Precision-Recall curves for binary classification.
    """
    # Crear subplots para ROC y Precision-Recall
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Curva ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Clasificador Aleatorio')

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
    ax1.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    ax1.set_title('Curva ROC', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])

    ax2.plot(recall, precision, color='darkgreen', lw=2,
             label=f'Curva P-R (AP = {avg_precision:.3f})')
    ax2.axhline(y=np.sum(y_test)/len(y_test), color='navy', lw=2, linestyle='--',
                label=f'Baseline ({np.sum(y_test)/len(y_test):.3f})')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax2.set_ylabel('Precision (Precisión)', fontsize=12)
    ax2.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    st.markdown("### 📈 Curvas de Rendimiento")

    # Mostrar con 80% del ancho
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.pyplot(fig, use_container_width=True)

    _display_binary_interpretation(roc_auc, avg_precision, y_test)


def _plot_multiclass_roc_pr(y_test, y_pred_proba, average, class_names, n_classes):
    """
    Plots ROC and Precision-Recall curves for multi-class classification with macro/micro averaging.
    """
    # Opciones de visualización para multiclase
    st.markdown("### ⚙️ Opciones de Visualización")

    col1, col2 = st.columns(2)
    with col1:
        show_individual = st.checkbox(
            "Mostrar curvas por clase individual", value=True)
    with col2:
        averaging_methods = st.multiselect(
            "Métodos de promediado:",
            ["macro", "micro"],
            default=["macro", "micro"] if average in [
                "macro", "micro"] else ["macro"]
        )

    if not averaging_methods:
        averaging_methods = ["macro"]  # Default fallback

    # Preparar datos para multiclase
    if class_names is None:
        class_names = [f"Clase {i}" for i in range(n_classes)]

    # Binarizar las etiquetas para One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    if n_classes == 2:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    # Crear subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Colores para las clases
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green',
                    'purple', 'brown', 'pink', 'gray', 'olive'])

    # Diccionarios para almacenar valores de AUC y AP
    roc_auc_scores = {}
    ap_scores = {}

    # Curvas ROC por clase individual
    if show_individual:
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[f'clase_{i}'] = roc_auc

            ax1.plot(fpr, tpr, color=color, lw=2, alpha=0.7,
                     label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

            # Precision-Recall por clase
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
            ap_scores[f'clase_{i}'] = ap

            ax2.plot(recall, precision, color=color, lw=2, alpha=0.7,
                     label=f'{class_names[i]} (AP = {ap:.3f})')

    # Curvas promedio
    for avg_method in averaging_methods:
        if avg_method == "micro":
            # Calcular curvas micro-promedio
            fpr_micro, tpr_micro, _ = roc_curve(
                y_test_bin.ravel(), y_pred_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            roc_auc_scores['micro'] = roc_auc_micro

            precision_micro, recall_micro, _ = precision_recall_curve(
                y_test_bin.ravel(), y_pred_proba.ravel())
            ap_micro = average_precision_score(
                y_test_bin.ravel(), y_pred_proba.ravel())
            ap_scores['micro'] = ap_micro

            ax1.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
                     label=f'Micro-promedio (AUC = {roc_auc_micro:.3f})')
            ax2.plot(recall_micro, precision_micro, color='deeppink', linestyle=':', linewidth=4,
                     label=f'Micro-promedio (AP = {ap_micro:.3f})')

        elif avg_method == "macro":
            # Calcular curvas macro-promedio
            all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_pred_proba[:, i])[0]
                                               for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)

            mean_tpr /= n_classes
            roc_auc_macro = auc(all_fpr, mean_tpr)
            roc_auc_scores['macro'] = roc_auc_macro

            ap_macro = np.mean([average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
                               for i in range(n_classes)])
            ap_scores['macro'] = ap_macro

            ax1.plot(all_fpr, mean_tpr, color='navy', linestyle='--', linewidth=4,
                     label=f'Macro-promedio (AUC = {roc_auc_macro:.3f})')

            # Para precision-recall macro, usamos el promedio simple de AP
            ax2.axhline(y=ap_macro, color='navy', linestyle='--', linewidth=4,
                        label=f'Macro-promedio (AP = {ap_macro:.3f})')

    # Línea de referencia aleatoria
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5,
             label='Clasificador Aleatorio')

    # Configurar axes ROC
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
    ax1.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    ax1.set_title('Curvas ROC Multi-clase', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Configurar axes Precision-Recall
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax2.set_ylabel('Precision (Precisión)', fontsize=12)
    ax2.set_title('Curvas Precision-Recall Multi-clase',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    st.markdown("### 📈 Curvas de Rendimiento Multi-clase")

    # Mostrar figura
    col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
    with col2:
        st.pyplot(fig, use_container_width=True)

    _display_multiclass_interpretation(
        roc_auc_scores, ap_scores, averaging_methods, class_names)


def _display_binary_interpretation(roc_auc, avg_precision, y_test):
    """
    Display interpretation for binary classification metrics.
    """
    # Interpretación de las curvas
    st.markdown("### 📋 Interpretación de las Curvas")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Curva ROC:**")
        if roc_auc >= 0.9:
            st.success(f"🎯 Excelente discriminación (AUC = {roc_auc:.3f})")
        elif roc_auc >= 0.8:
            st.info(f"👍 Buena discriminación (AUC = {roc_auc:.3f})")
        elif roc_auc >= 0.7:
            st.warning(f"⚠️ Discriminación moderada (AUC = {roc_auc:.3f})")
        else:
            st.error(f"❌ Discriminación pobre (AUC = {roc_auc:.3f})")

    with col2:
        st.markdown("**Curva Precision-Recall:**")
        baseline_precision = np.sum(y_test)/len(y_test)
        if avg_precision >= baseline_precision + 0.3:
            st.success(f"🎯 Excelente (AP = {avg_precision:.3f})")
        elif avg_precision >= baseline_precision + 0.1:
            st.info(
                f"👍 Buena mejora sobre baseline (AP = {avg_precision:.3f})")
        elif avg_precision >= baseline_precision:
            st.warning(f"⚠️ Mejora marginal (AP = {avg_precision:.3f})")
        else:
            st.error(f"❌ Por debajo del baseline (AP = {avg_precision:.3f})")


def _display_multiclass_interpretation(roc_auc_scores, ap_scores, averaging_methods, class_names):
    """
    Display interpretation for multi-class classification metrics.
    """
    st.markdown("### 📋 Interpretación de Métricas Multi-clase")

    # Mostrar métricas por método de promediado
    for avg_method in averaging_methods:
        if avg_method in roc_auc_scores:
            st.markdown(f"#### {avg_method.title()}-promedio")

            col1, col2 = st.columns(2)

            with col1:
                roc_auc = roc_auc_scores[avg_method]
                st.markdown("**ROC AUC:**")
                if roc_auc >= 0.9:
                    st.success(f"🎯 Excelente (AUC = {roc_auc:.3f})")
                elif roc_auc >= 0.8:
                    st.info(f"👍 Buena (AUC = {roc_auc:.3f})")
                elif roc_auc >= 0.7:
                    st.warning(f"⚠️ Moderada (AUC = {roc_auc:.3f})")
                else:
                    st.error(f"❌ Pobre (AUC = {roc_auc:.3f})")

            with col2:
                if avg_method in ap_scores:
                    ap = ap_scores[avg_method]
                    st.markdown("**Average Precision:**")
                    if ap >= 0.8:
                        st.success(f"🎯 Excelente (AP = {ap:.3f})")
                    elif ap >= 0.6:
                        st.info(f"👍 Buena (AP = {ap:.3f})")
                    elif ap >= 0.4:
                        st.warning(f"⚠️ Moderada (AP = {ap:.3f})")
                    else:
                        st.error(f"❌ Pobre (AP = {ap:.3f})")

    # Tabla de métricas por clase
    if any(key.startswith('clase_') for key in roc_auc_scores.keys()):
        st.markdown("#### 📊 Métricas por Clase")

        metrics_data = []
        for i, class_name in enumerate(class_names):
            roc_key = f'clase_{i}'
            if roc_key in roc_auc_scores:
                metrics_data.append({
                    'Clase': class_name,
                    'ROC AUC': round(roc_auc_scores[roc_key], 3),
                    'Average Precision': round(ap_scores[roc_key], 3)
                })

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)

    # Explicación de macro vs micro
    with st.expander("ℹ️ ¿Macro vs Micro promedio?", expanded=False):
        st.markdown("""
        **Macro-promedio:**
        - Calcula la métrica para cada clase independientemente
        - Luego promedia los resultados (todas las clases tienen el mismo peso)
        - Útil cuando todas las clases son igualmente importantes
        - Más sensible al rendimiento en clases minoritarias

        **Micro-promedio:**
        - Agrega las contribuciones de todas las clases para calcular la métrica
        - Da más peso a las clases con más muestras
        - Útil cuando las clases más frecuentes son más importantes
        - Menos sensible a clases desbalanceadas

        **¿Cuál usar?**
        - **Macro** cuando todas las clases son igualmente importantes
        - **Micro** cuando algunas clases son más importantes que otras
        - **Ambos** para una evaluación completa
        """)


def _add_detailed_explanation():
    """
    Add detailed explanation about ROC and Precision-Recall curves.
    """
    # Explicación detallada sobre las curvas de rendimiento
    with st.expander("ℹ️ ¿Cómo interpretar las Curvas de Rendimiento?", expanded=False):
        st.markdown("""
        **Curva ROC (Receiver Operating Characteristic)**

        **¿Qué muestra?**
        - **Eje X:** Tasa de Falsos Positivos (FPR) = FP / (FP + TN)
        - **Eje Y:** Tasa de Verdaderos Positivos (TPR) = TP / (TP + FN) = Sensibilidad/Recall
        - **Línea diagonal:** Rendimiento de un clasificador aleatorio
        - **AUC (Área Bajo la Curva):** Métrica resumen del rendimiento

        **Interpretación:**
        - **AUC = 1.0:** Clasificador perfecto
        - **AUC = 0.9-1.0:** Excelente discriminación
        - **AUC = 0.8-0.9:** Buena discriminación
        - **AUC = 0.7-0.8:** Discriminación aceptable
        - **AUC = 0.5:** Equivalente a adivinar al azar
        - **AUC < 0.5:** Peor que adivinar (pero se puede invertir)

        **¿Cuándo usar ROC?**
        - Cuando las clases están relativamente balanceadas
        - Para comparar modelos rápidamente
        - Cuando te importa el rendimiento general

        ---

        **Curva Precision-Recall (P-R)**

        **¿Qué muestra?**
        - **Eje X:** Recall (Sensibilidad) = TP / (TP + FN)
        - **Eje Y:** Precision (Precisión) = TP / (TP + FP)
        - **Línea horizontal:** Baseline (proporción de casos positivos)
        - **AP (Average Precision):** Métrica resumen del rendimiento

        **Interpretación:**
        - **AP alto:** Buen balance entre precisión y recall
        - **Curva cerca del ángulo superior derecho:** Excelente rendimiento
        - **Por encima del baseline:** Mejor que una predicción aleatoria

        **¿Cuándo usar P-R?**
        - ✅ **Clases desbalanceadas** (muchos más negativos que positivos)
        - ✅ Cuando los **falsos positivos son costosos**
        - ✅ Para datasets con **pocos casos positivos**
        - ✅ En problemas como **detección de fraude, diagnóstico médico**

        **Comparación ROC vs P-R:**
        - **ROC** es más optimista con clases desbalanceadas
        - **P-R** es más conservadora y realista
        - **P-R** se enfoca más en el rendimiento de la clase minoritaria
        - Usar **ambas** para una evaluación completa

        ---

        **Multi-clase: Macro vs Micro**

        **Estrategia One-vs-Rest:**
        - Cada clase se trata como positiva vs todas las demás como negativas
        - Se calculan métricas para cada clase por separado
        - Luego se combinan usando macro o micro promedio

        **Macro-promedio:**
        - Promedio simple de las métricas de cada clase
        - Todas las clases tienen el mismo peso
        - Más sensible a clases minoritarias
        - Útil cuando todas las clases son igualmente importantes

        **Micro-promedio:**
        - Agregación global de TP, FP, FN de todas las clases
        - Las clases con más muestras tienen más peso
        - Menos sensible a desbalance de clases
        - Útil cuando algunas clases son más importantes
        """)


def plot_threshold_analysis(y_true, y_pred_proba, class_names):
    # Verificar que tenemos datos válidos
    if y_pred_proba is not None and len(y_pred_proba) > 0:
        unique_classes = np.unique(y_true)

        # Para clasificación binaria
        if len(unique_classes) == 2:
            # Explicación detallada sobre distribución de probabilidades
            with st.expander("ℹ️ ¿Cómo interpretar la Distribución de Probabilidades?", expanded=False):
                st.markdown("""
                    **¿Qué muestra este gráfico?**
                    
                    Este histograma muestra cómo el modelo asigna probabilidades a cada muestra del conjunto de prueba, 
                    separado por la clase real a la que pertenece cada muestra.
                    
                    **Elementos del gráfico:**
                    - **Histograma azul:** Distribución de probabilidades para muestras que realmente pertenecen a la clase positiva
                    - **Histograma rojo:** Distribución de probabilidades para muestras que realmente pertenecen a la clase negativa  
                    - **Línea roja vertical:** Umbral de decisión (0.5 por defecto)
                    - **Eje X:** Probabilidad asignada por el modelo (0 = clase negativa, 1 = clase positiva)
                    - **Eje Y:** Cantidad de muestras
                    
                    **Interpretación ideal:**
                    - ✅ **Buena separación:** Los histogramas no se superponen mucho
                    - ✅ **Clase negativa:** Concentrada cerca de 0 (izquierda)
                    - ✅ **Clase positiva:** Concentrada cerca de 1 (derecha)
                    - ✅ **Pocas muestras cerca del umbral (0.5):** Indica confianza en las predicciones
                    
                    **Problemas a identificar:**
                    - ⚠️ **Mucha superposición:** Indica dificultad para separar las clases
                    - ⚠️ **Concentración en el centro (0.3-0.7):** El modelo está inseguro
                    - ⚠️ **Distribución uniforme:** El modelo no está aprendiendo patrones útiles
                    
                    **Aplicaciones prácticas:**
                    - Identificar si el modelo está confiado en sus predicciones
                    - Evaluar si cambiar el umbral de decisión podría mejorar el rendimiento
                    - Detectar casos donde el modelo necesita más datos o características
                    """)
            fig, ax = plt.subplots(figsize=(12, 6))

            # Obtener probabilidades de la clase positiva
            prob_class_1 = y_pred_proba[:, 1]

            # Separar por clase real - usar los valores únicos reales
            mask_class_0 = (y_true == unique_classes[0])
            mask_class_1 = (y_true == unique_classes[1])

            prob_class_0_real = prob_class_1[mask_class_0]
            prob_class_1_real = prob_class_1[mask_class_1]

            # Crear histogramas solo si hay datos
            if len(prob_class_0_real) > 0:
                ax.hist(prob_class_0_real, bins=20, alpha=0.7,
                        label=f'Clase {class_names[0] if class_names and len(class_names) > 0 else unique_classes[0]} (Real)',
                        color='lightcoral', edgecolor='black')

            if len(prob_class_1_real) > 0:
                ax.hist(prob_class_1_real, bins=20, alpha=0.7,
                        label=f'Clase {class_names[1] if class_names and len(class_names) > 1 else unique_classes[1]} (Real)',
                        color='lightblue', edgecolor='black')

            # Línea del umbral de decisión
            ax.axvline(x=0.5, color='red', linestyle='--',
                       linewidth=2, label='Umbral de decisión (0.5)')

            # Configurar el gráfico
            ax.set_xlabel(
                'Probabilidad de Clase Positiva', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.set_title('Distribución de Probabilidades Predichas por Clase Real',
                         fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Asegurar límites apropiados
            ax.set_xlim(0, 1)

            plt.tight_layout()

            # Mostrar el gráfico
            col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
            with col2:
                st.pyplot(fig, use_container_width=True)

            # Limpiar la figura
            plt.close(fig)

            # Análisis de separación
            if len(prob_class_0_real) > 0 and len(prob_class_1_real) > 0:
                # Contar solapamiento en la zona de incertidumbre (0.3-0.7)
                overlap_0 = np.sum(
                    (prob_class_0_real > 0.3) & (prob_class_0_real < 0.7))
                overlap_1 = np.sum(
                    (prob_class_1_real > 0.3) & (prob_class_1_real < 0.7))
                total_overlap = overlap_0 + overlap_1

                overlap_percentage = total_overlap / len(y_true)

                # Métricas adicionales de separación
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Muestras en zona incierta",
                              f"{total_overlap}/{len(y_true)}")
                with col2:
                    st.metric("Porcentaje de incertidumbre",
                              f"{overlap_percentage:.1%}")
                with col3:
                    conf_threshold = 0.8  # 80% de confianza
                    high_conf = np.sum((prob_class_1 < 0.2) | (
                        prob_class_1 > conf_threshold))
                    st.metric("Predicciones confiables",
                              f"{high_conf}/{len(y_true)}")

                # Interpretación
                if overlap_percentage < 0.2:
                    st.success(
                        "✅ Excelente separación entre clases - El modelo está muy confiado en sus predicciones")
                elif overlap_percentage < 0.4:
                    st.info("👍 Buena separación entre clases")
                else:
                    st.warning(
                        "⚠️ Las clases se superponen significativamente - Considera ajustar el umbral de decisión")

        elif len(unique_classes) > 2:
            # Para clasificación multiclase
            st.info(
                "**Nota:** Clasificación multiclase detectada. Mostrando distribución de probabilidades para cada clase.")
            # Se crea un histograma diferente para cada clase en figuras separadas.
            with st.expander("ℹ️ ¿Cómo interpretar la Distribución de Probabilidades Multiclase?", expanded=False):
                st.markdown("""
                La distribución de probabilidades para cada clase muestra la confianza del modelo al predecir la clase a la que pertenece una muestra. 
                En un escenario de clasificación multiclase, el modelo asigna una probabilidad a cada clase posible, y la clase predicha suele ser la de mayor probabilidad.

                **Interpretación:**
                - Cada barra en cada histograma representa la frecuencia de las probabilidades predichas para una clase específica.
                - Un modelo bien calibrado tendrá una alta concentración de muestras en los extremos (cerca de 0 o 1) para la clase correcta.
                - Distribuciones más planas o con picos intermedios sugieren que el modelo presenta incertidumbre o confusión entre clases.

                **Consejos:**
                - Presta atención a las clases con solapamiento en sus distribuciones, ya que esto puede indicar áreas de incertidumbre.
                - Considera ajustar el umbral de decisión para mejorar la precisión en clases específicas.
                """)

            n_classes = len(unique_classes)
            n_cols = min(3, n_classes)
            n_rows = (n_classes + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

            # Manejar caso de una sola clase
            if n_classes == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()

            for i, class_val in enumerate(unique_classes):
                if i < len(axes):
                    ax_sub = axes[i]

                    # Probabilidades para esta clase
                    class_probs = y_pred_proba[:, i]

                    ax_sub.hist(class_probs, bins=20, alpha=0.7,
                                color=plt.cm.Set3(i), edgecolor='black')

                    class_label = class_names[i] if class_names and i < len(
                        class_names) else f"Clase {class_val}"
                    ax_sub.set_title(
                        f'Probabilidades para {class_label}')
                    ax_sub.set_xlabel('Probabilidad')
                    ax_sub.set_ylabel('Frecuencia')
                    ax_sub.grid(True, alpha=0.3)
                    ax_sub.set_xlim(0, 1)

            # Ocultar subplots vacíos
            for i in range(n_classes, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
            with col2:
                st.pyplot(fig, use_container_width=True)

            plt.close(fig)
        else:
            st.error(
                "Error: Datos de clasificación insuficientes para crear la visualización")

    else:
        st.error(
            "Error: No hay probabilidades predichas disponibles. Asegúrate de que el modelo esté entrenado correctamente.")

    # Análisis de umbrales de decisión para clasificación binaria
    if len(np.unique(y_true)) == 2:
        st.markdown("### 🎯 Análisis de Umbrales de Decisión")

        # Explicación detallada sobre umbrales de decisión
        with st.expander("ℹ️ ¿Cómo interpretar el Análisis de Umbrales?", expanded=False):
            st.markdown("""
            **¿Qué es el umbral de decisión?**
            
            El umbral de decisión es el valor que determina cuándo el modelo clasifica una muestra como 
            positiva o negativa. Por defecto, este umbral es **0.5**:
            - **Probabilidad ≥ 0.5** → Clase Positiva
            - **Probabilidad < 0.5** → Clase Negativa
            
            **¿Por qué cambiar el umbral?**
            
            El umbral por defecto (0.5) no siempre es óptimo. Dependiendo del problema, 
            puede ser beneficioso ajustarlo:
            
            **📈 Umbral más alto (0.6, 0.7, 0.8):**
            - ✅ **Mayor Precisión:** Menos falsos positivos
            - ✅ **Predicciones más conservadoras:** Solo clasifica como positivo cuando está muy seguro
            - ⚠️ **Menor Recall:** Puede perder casos positivos reales
            - **Útil cuando:** Los falsos positivos son muy costosos (ej: diagnóstico médico, inversiones)
            
            **📉 Umbral más bajo (0.3, 0.4):**
            - ✅ **Mayor Recall:** Detecta más casos positivos reales
            - ✅ **Predicciones más sensibles:** No se pierde tantos casos positivos
            - ⚠️ **Menor Precisión:** Más falsos positivos
            - **Útil cuando:** Los falsos negativos son muy costosos (ej: detección de fraude, seguridad)
            
            **Métricas mostradas:**
            - **Accuracy:** Porcentaje total de predicciones correctas
            - **Precision:** De las predicciones positivas, cuántas son correctas
            - **Recall:** De los casos positivos reales, cuántos detectamos
            - **F1-Score:** Balance entre precisión y recall
            
            **¿Cómo elegir el umbral óptimo?**
            1. **Maximizar F1-Score:** Balance general entre precisión y recall
            2. **Maximizar Precision:** Si los falsos positivos son costosos
            3. **Maximizar Recall:** Si los falsos negativos son costosos
            4. **Considerar el contexto:** Costos reales de errores en tu dominio
            
            **Ejemplo práctico:**
            - **Email spam:** Prefiere falsos positivos (email importante en spam) que falsos negativos
            - **Diagnóstico médico:** Prefiere falsos positivos (más pruebas) que falsos negativos (enfermedad no detectada)
            - **Recomendaciones:** Balance entre no molestar (precisión) y no perder oportunidades (recall)
            """)

        # Calcular métricas para diferentes umbrales
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []

        for threshold in thresholds:
            y_pred_thresh = (
                y_pred_proba[:, 1] >= threshold).astype(int)

            if len(np.unique(y_pred_thresh)) > 1:  # Evitar división por cero
                precision = precision_score(
                    y_true, y_pred_thresh, zero_division=0)
                recall = recall_score(
                    y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh,
                              zero_division=0)
                accuracy = accuracy_score(y_true, y_pred_thresh)

                threshold_metrics.append({
                    'Umbral': threshold,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })

        if threshold_metrics:
            df_thresholds = pd.DataFrame(threshold_metrics)

            # Encontrar el mejor umbral por F1-Score
            best_f1_idx = df_thresholds['F1-Score'].idxmax()
            best_threshold = df_thresholds.loc[best_f1_idx, 'Umbral']

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Umbral Actual", "0.50")
                st.metric("Umbral Óptimo (F1)",
                          f"{best_threshold:.2f}")

                if abs(best_threshold - 0.5) > 0.1:
                    st.info(
                        f"💡 Considera ajustar el umbral a {best_threshold:.2f} para mejorar el F1-Score")

            with col2:
                # Mostrar tabla de umbrales (seleccionados)
                display_thresholds = df_thresholds[df_thresholds['Umbral'].isin(
                    [0.3, 0.5, 0.7])].copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    display_thresholds[col] = display_thresholds[col].apply(
                        lambda x: f"{x:.3f}")

                st.markdown("**Comparación de Umbrales:**")
                st.dataframe(
                    display_thresholds, hide_index=True, use_container_width=True)
