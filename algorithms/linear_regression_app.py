import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from dataset_manager import load_data, preprocess_data
from algorithms.dataset_tab import run_dataset_tab
from utils import create_info_box, get_image_download_link, show_code_with_download
from model_training import train_linear_model
from model_evaluation import show_detailed_evaluation


def run_linear_regression_app():
    """Ejecuta la aplicación específica de regresión (lineal y logística)."""
    st.header("📊 Regresión")
    st.markdown(
        "Aprende sobre regresión lineal y logística de forma interactiva")

    # Información sobre regresión
    with st.expander("ℹ️ ¿Qué es la Regresión?", expanded=False):
        st.markdown("""
        La regresión es un conjunto de algoritmos de aprendizaje supervisado utilizados para predecir valores numéricos continuos (regresión lineal) o clasificar elementos en categorías (regresión logística).

        **Características principales:**
        - **Regresión Lineal**: Establece una relación lineal entre las variables independientes (características) y la variable dependiente (objetivo)
        - **Regresión Logística**: Utiliza la función logística para modelar la probabilidad de pertenencia a una clase
        - Ambos tipos minimizan funciones de costo específicas para encontrar los mejores parámetros
        - Son interpretables: los coeficientes indican la importancia y dirección del efecto de cada característica
        - No requieren escalado de datos, aunque puede mejorar la convergencia

        **Tipos de regresión:**
        - **Regresión Lineal Simple**: Una sola característica predictora
        - **Regresión Lineal Múltiple**: Múltiples características predictoras
        - **Regresión Logística**: Para problemas de clasificación binaria o multiclase

        **Limitaciones:**
        - Asume una relación lineal entre variables (regresión lineal)
        - Sensible a valores atípicos (outliers)
        - Puede sufrir de multicolinealidad cuando las características están correlacionadas
        """)

    # Variables para almacenar datos
    dataset_loaded = False
    X, y, feature_names, class_names, dataset_info, task_type = None, None, None, None, None, None

    # Inicializar el estado de la pestaña activa si no existe
    if 'active_tab_lr' not in st.session_state:
        st.session_state.active_tab_lr = 0

    # Crear pestañas para organizar la información
    tab_options = [
        "📊 Datos",
        "🏋️ Entrenamiento",
        "📈 Evaluación",
        "📉 Visualización",
        "🔍 Coeficientes",
        "🔮 Predicciones",
        "💾 Exportar Modelo"
    ]

    # Crear contenedor para los botones de las pestañas
    tab_cols = st.columns(len(tab_options))

    # Estilo CSS para los botones de pestañas (Regresión)
    st.markdown("""
    <style>
    div.tab-button-lr > button {
        border-radius: 4px 4px 0 0;
        padding: 10px;
        width: 100%;
        white-space: nowrap;
        background-color: #F0F2F6;
        border-bottom: 2px solid #E0E0E0;
        color: #333333;
    }
    div.tab-button-lr-active > button {
        background-color: #E3F2FD !important;
        border-bottom: 2px solid #1E88E5 !important;
        font-weight: bold !important;
        color: #1E88E5 !important;
    }
    div.tab-button-lr > button:hover {
        background-color: #E8EAF6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Crear botones para las pestañas
    for i, (tab_name, col) in enumerate(zip(tab_options, tab_cols)):
        button_key = f"tab_lr_{i}"
        button_style = "tab-button-lr-active" if st.session_state.active_tab_lr == i else "tab-button-lr"
        is_active = st.session_state.active_tab_lr == i

        with col:
            st.markdown(f"<div class='{button_style}'>",
                        unsafe_allow_html=True)
            # Usar type="primary" para el botón activo
            if st.button(tab_name, key=button_key, use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_tab_lr = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Separador visual
    st.markdown("---")

    ###########################################
    # Pestaña de Datos                        #
    ###########################################
    run_dataset_tab(st.session_state.active_tab_lr)

    ###########################################
    # Pestaña de Entrenamiento                #
    ###########################################
    if st.session_state.active_tab_lr == 1:
        st.header("Configuración del Modelo")

        # Inicializar session state variables
        if 'dataset_option_lr' not in st.session_state:
            st.session_state.dataset_option_lr = st.session_state.selected_dataset
        if 'model_type_lr' not in st.session_state:
            st.session_state.model_type_lr = "Linear"
        if 'is_trained_lr' not in st.session_state:
            st.session_state.is_trained_lr = False

        # Cargar datos para la vista previa si cambia el dataset o si no se ha cargado
        if st.session_state.selected_dataset != st.session_state.dataset_option_lr or not dataset_loaded:
            try:
                X, y, feature_names, class_names, dataset_info, task_type = load_data(
                    st.session_state.selected_dataset)

                st.session_state.dataset_option_lr = st.session_state.selected_dataset
                dataset_loaded = True

                # Mostrar información del dataset
                st.markdown("### Información del Dataset")
                st.markdown(create_info_box(dataset_info),
                            unsafe_allow_html=True)

                # Determinar el tipo de modelo según el task_type detectado
                st.markdown("### Tipo de Modelo Lineal")

                # Usar botones para seleccionar el tipo de modelo
                tipo_col1, tipo_col2 = st.columns(2)

                with tipo_col1:
                    is_linear = True
                    if "model_type_lr" in st.session_state:
                        is_linear = st.session_state.model_type_lr == "Linear"

                    if st.button("📈 Regresión Lineal",
                                 key="btn_linear",
                                 type="primary" if is_linear else "secondary",
                                 use_container_width=True,
                                 help="Para predecir valores numéricos continuos"):
                        model_type = "Linear"
                        st.session_state.model_type_lr = model_type
                        st.rerun()

                with tipo_col2:
                    is_logistic = False
                    if "model_type_lr" in st.session_state:
                        is_logistic = st.session_state.model_type_lr == "Logistic"

                    if st.button("🏷️ Regresión Logística",
                                 key="btn_logistic",
                                 type="primary" if is_logistic else "secondary",
                                 use_container_width=True,
                                 help="Para predecir categorías o clases"):
                        model_type = "Logistic"
                        st.session_state.model_type_lr = model_type
                        st.rerun()

                # Obtener el valor actual del tipo de modelo
                model_type = st.session_state.get('model_type_lr', "Linear")

                # Mostrar sugerencia basada en el tipo de tarea detectado
                if task_type == "Clasificación" and model_type == "Linear":
                    st.warning(
                        "Este dataset parece ser más adecuado para Regresión Logística. La selección actual podría no ofrecer resultados óptimos.")
                elif task_type == "Regresión" and model_type == "Logistic":
                    st.warning(
                        "Este dataset parece ser más adecuado para Regresión Lineal. La selección actual podría no ofrecer resultados óptimos.")

            except Exception as e:
                st.error(f"Error al cargar el dataset: {str(e)}")
                dataset_loaded = False

        # Parámetros del modelo
        st.markdown("### Parámetros del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            # Para regresión logística, mostrar parámetro max_iter
            if st.session_state.get('model_type_lr', 'Linear') == "Logistic":
                max_iter = st.slider(
                    "Máximo de Iteraciones:",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                    help="Número máximo de iteraciones para el optimizador de regresión logística."
                )
            else:
                st.info(
                    "La regresión lineal no requiere configuración adicional de iteraciones.")

        with col2:
            test_size = st.slider(
                "Tamaño del Conjunto de Prueba:",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Proporción de datos que se usará para evaluar el modelo."
            )

        if st.button("🚀 Entrenar Modelo", key="train_lr_button", type="primary"):
            if dataset_loaded and X is not None and y is not None:
                with st.spinner("Entrenando modelo..."):
                    try:
                        Xtrain, Xtest, ytrain, ytest = preprocess_data(
                            X, y, test_size)
                        # Entrenar el modelo
                        model_type = st.session_state.get(
                            'model_type_lr', 'Linear')
                        if model_type == "Logistic":
                            model = train_linear_model(
                                Xtrain, ytrain,
                                model_type=model_type,
                                max_iter=max_iter
                            )
                        else:
                            model = train_linear_model(
                                Xtrain, ytrain,
                                model_type=model_type
                            )

                        # Guardar en session state
                        st.session_state.model_lr = model
                        st.session_state.X_train_lr = Xtrain
                        st.session_state.X_test_lr = Xtest
                        st.session_state.y_train_lr = ytrain
                        st.session_state.y_test_lr = ytest
                        st.session_state.feature_names_lr = feature_names
                        st.session_state.class_names_lr = class_names
                        st.session_state.task_type_lr = task_type
                        st.session_state.model_trained_lr = True

                        st.success("¡Modelo entrenado exitosamente!")

                    except Exception as e:
                        st.error(f"Error al entrenar el modelo: {str(e)}")
            else:
                st.error("Por favor, carga un dataset válido primero.")

    ###########################################
    # Pestaña de Evaluación                   #
    ###########################################
    elif st.session_state.active_tab_lr == 2:
        st.header("Evaluación del Modelo")

        if not st.session_state.get('model_trained_lr', False):
            st.warning(
                "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
        else:
            y_test = st.session_state.get('y_test_lr')
            model_type = st.session_state.get('model_type_lr', 'Linear')
            model_type = "Regresión" if model_type == "Linear" else "Clasificación"
            class_names = st.session_state.class_names_lr

            # Obtener predicciones del modelo
            y_pred = st.session_state.model_lr.predict(
                st.session_state.X_test_lr)

            # Mostrar evaluación detallada del modelo
            show_detailed_evaluation(
                y_test,
                y_pred,
                class_names,
                model_type
            )

    ###########################################
    # Pestaña de Visualización                #
    ###########################################
    elif st.session_state.active_tab_lr == 3:
        st.header("Visualizaciones")

        if st.session_state.get('model_trained_lr', False):
            model_type = st.session_state.get('model_type_lr', 'Linear')
            X_test = st.session_state.get('X_test_lr')
            y_test = st.session_state.get('y_test_lr')
            X_train = st.session_state.get('X_train_lr')
            y_train = st.session_state.get('y_train_lr')
            model = st.session_state.get('model_lr')

            # Información sobre las visualizaciones
            with st.expander("ℹ️ ¿Cómo interpretar estas visualizaciones?", expanded=False):
                if model_type == "Linear":
                    st.markdown("""
                    **Gráfico de Predicciones vs Valores Reales:**
                    - Cada punto representa una predicción del modelo
                    - La línea roja diagonal representa predicciones perfectas
                    - **Interpretación:**
                      - Puntos cerca de la línea roja = buenas predicciones
                      - Puntos dispersos = predicciones menos precisas
                      - Patrones sistemáticos fuera de la línea pueden indicar problemas del modelo
                    
                    **Gráfico de Residuos:**
                    - Muestra la diferencia entre valores reales y predicciones
                    - **Interpretación:**
                      - Residuos cerca de cero = buenas predicciones
                      - Patrones en los residuos pueden indicar que el modelo lineal no es adecuado
                      - Distribución aleatoria alrededor de cero es ideal
                    """)
                else:
                    st.markdown("""
                    **Matriz de Confusión:**
                    - Muestra predicciones correctas e incorrectas por clase
                    - **Interpretación:**
                      - Diagonal principal = predicciones correctas
                      - Fuera de la diagonal = errores del modelo
                      - Colores más intensos = mayor cantidad de casos
                    
                    **Curva ROC (si es binaria):**
                    - Muestra el rendimiento del clasificador en diferentes umbrales
                    - **Interpretación:**
                      - Línea más cerca de la esquina superior izquierda = mejor modelo
                      - Área bajo la curva (AUC) cercana a 1 = excelente modelo
                    """)

            if model_type == "Linear" and X_test is not None and y_test is not None and model is not None:
                y_pred = model.predict(X_test)

                # Crear visualizaciones con mejor tamaño
                st.markdown("### 📊 Gráfico de Predicciones vs Valores Reales")

                fig, ax = plt.subplots(figsize=(12, 8))

                # Scatter plot con mejor estilo
                ax.scatter(y_test, y_pred, alpha=0.6, s=50,
                           edgecolors='black', linewidth=0.5)

                # Línea de predicción perfecta
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val],
                        'r--', lw=2, label='Predicción Perfecta')

                # Personalización del gráfico
                ax.set_xlabel('Valores Reales', fontsize=12)
                ax.set_ylabel('Predicciones', fontsize=12)
                ax.set_title('Predicciones vs Valores Reales',
                             fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Añadir estadísticas al gráfico
                r2_value = st.session_state.get('metrics_lr', {}).get('r2', 0)
                ax.text(0.05, 0.95, f'R² = {r2_value:.4f}',
                        transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Mostrar con 80% del ancho
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Gráfico de residuos
                st.markdown("### 📈 Análisis de Residuos")

                # Información explicativa sobre los residuos
                with st.expander("ℹ️ ¿Cómo interpretar el análisis de residuos?", expanded=False):
                    st.markdown("""
                    **¿Qué son los residuos?**
                    Los residuos son las diferencias entre los valores reales y las predicciones del modelo:
                    `Residuo = Valor Real - Predicción`
                    
                    **Gráfico de Residuos vs Predicciones:**
                    - **Ideal:** Los puntos deben estar distribuidos aleatoriamente alrededor de la línea y=0
                    - **Problema:** Si ves patrones (curvas, abanicos), puede indicar:
                      - El modelo no captura relaciones no lineales
                      - Heterocedasticidad (varianza no constante)
                      - Variables importantes omitidas
                    
                    **Histograma de Residuos:**
                    - **Ideal:** Distribución normal (campana) centrada en 0
                    - **Problema:** Si la distribución está sesgada o tiene múltiples picos:
                      - Puede indicar que el modelo no es apropiado
                      - Sugiere la presencia de outliers o datos problemáticos
                    
                    **Línea roja punteada:** Marca el residuo = 0 (predicción perfecta)
                    **Media de residuos:** Debería estar cerca de 0 para un modelo bien calibrado
                    """)

                residuals = y_test - y_pred

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Residuos vs Predicciones
                ax1.scatter(y_pred, residuals, alpha=0.6, s=50,
                            edgecolors='black', linewidth=0.5)
                ax1.axhline(y=0, color='r', linestyle='--',
                            lw=2, label='Residuo = 0')
                ax1.set_xlabel('Predicciones', fontsize=12)
                ax1.set_ylabel('Residuos (Real - Predicción)', fontsize=12)
                ax1.set_title('Residuos vs Predicciones',
                              fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Añadir estadísticas al gráfico
                residual_std = residuals.std()
                ax1.text(0.05, 0.95, f'Desv. Estándar: {residual_std:.3f}',
                         transform=ax1.transAxes, fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

                # Histograma de residuos
                ax2.hist(residuals, bins=20, alpha=0.7,
                         edgecolor='black', color='skyblue')
                ax2.axvline(residuals.mean(), color='red', linestyle='--',
                            lw=2, label=f'Media: {residuals.mean():.3f}')
                ax2.axvline(0, color='green', linestyle='-',
                            lw=2, alpha=0.7, label='Ideal (0)')
                ax2.set_xlabel('Residuos', fontsize=12)
                ax2.set_ylabel('Frecuencia', fontsize=12)
                ax2.set_title('Distribución de Residuos',
                              fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                # Mostrar con 80% del ancho
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Interpretación automática de los residuos
                st.markdown("### 🔍 Interpretación de los Residuos")

                mean_residual = abs(residuals.mean())
                std_residual = residuals.std()

                interpretation = []

                if mean_residual < 0.1 * std_residual:
                    interpretation.append(
                        "✅ **Media de residuos cercana a 0:** El modelo está bien calibrado")
                else:
                    interpretation.append(
                        "⚠️ **Media de residuos alejada de 0:** El modelo puede tener sesgo sistemático")

                # Calcular R² de los residuos para detectar patrones
                from scipy import stats
                if len(residuals) > 10:
                    slope, _, r_value, _, _ = stats.linregress(
                        y_pred, residuals)
                    if abs(r_value) < 0.1:
                        interpretation.append(
                            "✅ **Sin correlación entre residuos y predicciones:** Buen ajuste lineal")
                    else:
                        interpretation.append(
                            "⚠️ **Correlación detectada en residuos:** Puede haber relaciones no lineales")

                # Test de normalidad simplificado (basado en asimetría)
                skewness = abs(stats.skew(residuals))
                if skewness < 1:
                    interpretation.append(
                        "✅ **Distribución de residuos aproximadamente normal**")
                else:
                    interpretation.append(
                        "⚠️ **Distribución de residuos sesgada:** Revisar outliers o transformaciones")

                for item in interpretation:
                    st.markdown(f"- {item}")

                if mean_residual >= 0.1 * std_residual or abs(r_value) >= 0.1 or skewness >= 1:
                    st.info(
                        "💡 **Sugerencias de mejora:** Considera probar transformaciones de variables, añadir características polinómicas, o usar modelos no lineales.")

            elif model_type == "Logistic" and X_test is not None and y_test is not None and model is not None:
                from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Matriz de Confusión
                st.markdown("### 📊 Matriz de Confusión")

                cm = confusion_matrix(y_test, y_pred)
                class_names = st.session_state.get('class_names_lr', [])

                fig, ax = plt.subplots(figsize=(10, 8))

                # Crear mapa de calor
                try:
                    import seaborn as sns
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                ax=ax, cbar_kws={'shrink': 0.8},
                                xticklabels=class_names if class_names else True,
                                yticklabels=class_names if class_names else True)
                except ImportError:
                    # Fallback to matplotlib if seaborn is not available
                    im = ax.imshow(cm, cmap='Blues', aspect='auto')
                    # Add text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            text = ax.text(j, i, f'{cm[i, j]}',
                                           ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
                    ax.set_xticks(range(cm.shape[1]))
                    ax.set_yticks(range(cm.shape[0]))
                    if class_names:
                        ax.set_xticklabels(class_names)
                        ax.set_yticklabels(class_names)
                ax.set_title('Matriz de Confusión',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Predicciones', fontsize=12)
                ax.set_ylabel('Valores Reales', fontsize=12)

                # Mostrar con 80% del ancho
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Análisis detallado de la matriz de confusión
                if len(np.unique(y_test)) == 2:
                    tn, fp, fn, tp = cm.ravel()

                    # Métricas derivadas de la matriz de confusión
                    st.markdown("### 🔍 Análisis Detallado de la Matriz")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Verdaderos Positivos", f"{tp}")
                        st.caption(
                            "Casos positivos correctamente identificados")
                    with col2:
                        st.metric("Falsos Positivos", f"{fp}")
                        st.caption(
                            "Casos negativos clasificados como positivos")
                    with col3:
                        st.metric("Falsos Negativos", f"{fn}")
                        st.caption(
                            "Casos positivos clasificados como negativos")
                    with col4:
                        st.metric("Verdaderos Negativos", f"{tn}")
                        st.caption(
                            "Casos negativos correctamente identificados")

                    # Interpretación de errores
                    if fp > fn:
                        st.warning(
                            "⚠️ El modelo tiende a clasificar más casos como positivos (más falsos positivos que falsos negativos)")
                    elif fn > fp:
                        st.warning(
                            "⚠️ El modelo tiende a ser más conservador (más falsos negativos que falsos positivos)")
                    else:
                        st.success(
                            "✅ El modelo tiene un balance equilibrado entre falsos positivos y negativos")

                # Curvas ROC y Precision-Recall
                if len(np.unique(y_test)) == 2:
                    from sklearn.metrics import roc_curve, auc, average_precision_score

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
                    precision, recall, _ = precision_recall_curve(
                        y_test, y_pred_proba[:, 1])
                    avg_precision = average_precision_score(
                        y_test, y_pred_proba[:, 1])

                    ax2.plot(recall, precision, color='darkgreen', lw=2,
                             label=f'Curva P-R (AP = {avg_precision:.3f})')
                    ax2.axhline(y=np.sum(y_test)/len(y_test), color='navy', lw=2, linestyle='--',
                                label=f'Baseline ({np.sum(y_test)/len(y_test):.3f})')

                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 1.05])
                    ax2.set_xlabel('Recall (Sensibilidad)', fontsize=12)
                    ax2.set_ylabel('Precision (Precisión)', fontsize=12)
                    ax2.set_title('Curva Precision-Recall',
                                  fontsize=14, fontweight='bold')
                    ax2.legend(loc="lower left")
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()

                    st.markdown("### 📈 Curvas de Rendimiento")

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
                        """)

                    # Mostrar con 80% del ancho
                    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                    with col2:
                        st.pyplot(fig, use_container_width=True)

                    # Interpretación de las curvas
                    st.markdown("### 📋 Interpretación de las Curvas")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Curva ROC:**")
                        if roc_auc >= 0.9:
                            st.success(
                                f"🎯 Excelente discriminación (AUC = {roc_auc:.3f})")
                        elif roc_auc >= 0.8:
                            st.info(
                                f"👍 Buena discriminación (AUC = {roc_auc:.3f})")
                        elif roc_auc >= 0.7:
                            st.warning(
                                f"⚠️ Discriminación moderada (AUC = {roc_auc:.3f})")
                        else:
                            st.error(
                                f"❌ Discriminación pobre (AUC = {roc_auc:.3f})")

                    with col2:
                        st.markdown("**Curva Precision-Recall:**")
                        baseline_precision = np.sum(y_test)/len(y_test)
                        if avg_precision >= baseline_precision + 0.3:
                            st.success(
                                f"🎯 Excelente (AP = {avg_precision:.3f})")
                        elif avg_precision >= baseline_precision + 0.1:
                            st.info(
                                f"👍 Buena mejora sobre baseline (AP = {avg_precision:.3f})")
                        elif avg_precision >= baseline_precision:
                            st.warning(
                                f"⚠️ Mejora marginal (AP = {avg_precision:.3f})")
                        else:
                            st.error(
                                f"❌ Por debajo del baseline (AP = {avg_precision:.3f})")

                # Análisis de probabilidades de predicción
                st.markdown("### 📊 Distribución de Probabilidades")

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

                # Verificar que tenemos datos válidos
                if y_pred_proba is not None and len(y_pred_proba) > 0:
                    unique_classes = np.unique(y_test)

                    # Para clasificación binaria
                    if len(unique_classes) == 2:
                        fig, ax = plt.subplots(figsize=(12, 6))

                        # Obtener probabilidades de la clase positiva
                        prob_class_1 = y_pred_proba[:, 1]

                        # Separar por clase real - usar los valores únicos reales
                        mask_class_0 = (y_test == unique_classes[0])
                        mask_class_1 = (y_test == unique_classes[1])

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

                            overlap_percentage = total_overlap / len(y_test)

                            # Métricas adicionales de separación
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Muestras en zona incierta",
                                          f"{total_overlap}/{len(y_test)}")
                            with col2:
                                st.metric("Porcentaje de incertidumbre",
                                          f"{overlap_percentage:.1%}")
                            with col3:
                                conf_threshold = 0.8  # 80% de confianza
                                high_conf = np.sum((prob_class_1 < 0.2) | (
                                    prob_class_1 > conf_threshold))
                                st.metric("Predicciones confiables",
                                          f"{high_conf}/{len(y_test)}")

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
                if len(np.unique(y_test)) == 2:
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
                            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                            precision = precision_score(
                                y_test, y_pred_thresh, zero_division=0)
                            recall = recall_score(
                                y_test, y_pred_thresh, zero_division=0)
                            f1 = f1_score(y_test, y_pred_thresh,
                                          zero_division=0)
                            accuracy = accuracy_score(y_test, y_pred_thresh)

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

        else:
            st.info("Entrena un modelo primero para ver las visualizaciones.")

    ###########################################
    # Pestaña de Coeficientes                 #
    ###########################################
    elif st.session_state.active_tab_lr == 4:
        st.header("Coeficientes del Modelo")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])
            model_type = st.session_state.get('model_type_lr', 'Linear')

            # Información sobre los coeficientes
            with st.expander("ℹ️ ¿Cómo interpretar los coeficientes?", expanded=False):
                if model_type == "Linear":
                    st.markdown("""
                    # 📈 **Coeficientes en Regresión Lineal**
                    
                    ## 🎯 **¿Qué representan los coeficientes?**
                    
                    Los coeficientes son los **parámetros aprendidos** por el modelo que determinan cómo cada característica 
                    influye en la predicción final. La fórmula de regresión lineal es:
                    
                    ```
                    Predicción = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ
                    ```
                    
                    Donde cada **βᵢ** es un coeficiente que indica el cambio en la variable objetivo 
                    por cada unidad de cambio en la característica correspondiente.
                    
                    ## 🔍 **Interpretación Detallada:**
                    
                    ### **Valor del Coeficiente (Magnitud):**
                    - **Valor absoluto grande (ej: |5.2|):** La característica tiene **alta influencia**
                    - **Valor absoluto pequeño (ej: |0.1|):** La característica tiene **baja influencia**
                    - **Valor cero (0.0):** La característica **no influye** en la predicción
                    
                    ### **Signo del Coeficiente (Dirección):**
                    - **Positivo (+):** 📈 **Relación directa** - A mayor valor de X, mayor valor de Y
                    - **Negativo (-):** 📉 **Relación inversa** - A mayor valor de X, menor valor de Y
                    
                    ### **Unidades:**
                    Los coeficientes mantienen las **unidades originales**. Si predices precios en euros 
                    y una característica está en metros², un coeficiente de 150 significa 
                    **+150 euros por cada metro² adicional**.
                    
                    ## 💡 **Ejemplos Prácticos:**
                    
                    **🏠 Predicción de Precios de Casas:**
                    - `Tamaño = +150`: Cada m² adicional aumenta el precio en 150€
                    - `Antigüedad = -500`: Cada año adicional reduce el precio en 500€
                    - `Habitaciones = +2000`: Cada habitación adicional aumenta el precio en 2000€
                    
                    **📊 Predicción de Ventas:**
                    - `Presupuesto_Marketing = +1.5`: Cada euro en marketing genera 1.5€ en ventas
                    - `Competencia = -0.8`: Cada competidor adicional reduce ventas en 0.8€
                    
                    ## ⚠️ **Limitaciones y Consideraciones:**
                    
                    1. **Correlación ≠ Causalidad:** Un coeficiente alto no implica que la característica cause el resultado
                    2. **Escalas diferentes:** Características con diferentes escalas pueden tener coeficientes incomparables
                    3. **Multicolinealidad:** Características correlacionadas pueden tener coeficientes inestables
                    4. **Outliers:** Valores extremos pueden afectar significativamente los coeficientes
                    
                    ## 🎛️ **Cómo usar esta información:**
                    
                    - **Identificar factores clave:** Coeficientes con mayor valor absoluto
                    - **Validar intuición:** ¿Los signos coinciden con el conocimiento del dominio?
                    - **Tomar decisiones:** ¿En qué características enfocar esfuerzos?
                    - **Detectar problemas:** ¿Hay coeficientes que no tienen sentido?
                    """)
                else:
                    st.markdown("""
                    # 🎯 **Coeficientes en Regresión Logística**
                    
                    ## 🎯 **¿Qué representan los coeficientes?**
                    
                    En regresión logística, los coeficientes representan el **cambio en log-odds** 
                    (logaritmo de las probabilidades) por cada unidad de cambio en la característica. 
                    La fórmula es:
                    
                    ```
                    log(odds) = β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ
                    odds = e^(β₀ + β₁×X₁ + β₂×X₂ + ... + βₙ×Xₙ)
                    probabilidad = odds / (1 + odds)
                    ```
                    
                    ## 🔍 **Interpretación de Log-Odds:**
                    
                    ### **Valor del Coeficiente:**
                    - **Positivo (+):** 📈 **Aumenta** la probabilidad de la clase positiva
                    - **Negativo (-):** 📉 **Disminuye** la probabilidad de la clase positiva
                    - **Cero (0.0):** **No afecta** la probabilidad
                    
                    ### **Magnitud del Coeficiente:**
                    - **|β| > 2:** **Efecto muy fuerte** (odds ratio > 7.4)
                    - **1 < |β| < 2:** **Efecto fuerte** (odds ratio entre 2.7 y 7.4)
                    - **0.5 < |β| < 1:** **Efecto moderado** (odds ratio entre 1.6 y 2.7)
                    - **|β| < 0.5:** **Efecto débil** (odds ratio < 1.6)
                    
                    ## 📊 **Conversión a Odds Ratios (Más Intuitivo):**
                    
                    Los **Odds Ratios** se calculan como `e^β` y son más fáciles de interpretar:
                    
                    ### **Interpretación de Odds Ratios:**
                    - **OR = 1:** La característica **no tiene efecto**
                    - **OR > 1:** La característica **aumenta** las odds de la clase positiva
                    - **OR < 1:** La característica **disminuye** las odds de la clase positiva
                    
                    ### **Ejemplos de Odds Ratios:**
                    - **OR = 2.0:** Duplica las odds (100% de aumento)
                    - **OR = 1.5:** Aumenta las odds en 50%
                    - **OR = 0.5:** Reduce las odds a la mitad (50% de reducción)
                    - **OR = 0.2:** Reduce las odds en 80%
                    
                    ## 💡 **Ejemplos Prácticos:**
                    
                    **🏥 Diagnóstico Médico (Predicción de Enfermedad):**
                    - `Edad = +0.05 (OR=1.05)`: Cada año adicional aumenta las odds en 5%
                    - `Ejercicio = -1.2 (OR=0.30)`: Hacer ejercicio reduce las odds en 70%
                    - `Fumador = +1.8 (OR=6.05)`: Ser fumador multiplica las odds por 6
                    
                    **📧 Clasificación de Spam:**
                    - `Palabras_Sospechosas = +2.3 (OR=10.0)`: Multiplica las odds de spam por 10
                    - `Remitente_Conocido = -1.5 (OR=0.22)`: Reduce las odds de spam en 78%
                    
                    **💳 Detección de Fraude:**
                    - `Hora_Inusual = +1.1 (OR=3.0)`: Triplica las odds de fraude
                    - `Ubicacion_Habitual = -2.0 (OR=0.14)`: Reduce las odds de fraude en 86%
                    
                    ## ⚠️ **Limitaciones y Consideraciones:**
                    
                    1. **Interpretación no lineal:** El efecto en probabilidades no es constante
                    2. **Asunciones de linealidad:** En el espacio log-odds, no en probabilidades
                    3. **Interacciones ignoradas:** Los efectos pueden depender de otras variables
                    4. **Escalas importantes:** Normalizar puede facilitar la interpretación
                    
                    ## 🎛️ **Cómo usar esta información:**
                    
                    ### **Para el Negocio:**
                    - **Identificar factores de riesgo:** Coeficientes positivos altos
                    - **Encontrar factores protectores:** Coeficientes negativos altos
                    - **Priorizar intervenciones:** Enfocarse en variables con mayor impacto
                    
                    ### **Para el Modelo:**
                    - **Validar coherencia:** ¿Los signos tienen sentido del dominio?
                    - **Detectar overfitting:** ¿Coeficientes extremadamente grandes?
                    - **Selección de características:** ¿Coeficientes cerca de cero?
                    
                    ## 🧮 **Fórmula de Conversión:**
                    ```python
                    # De coeficiente a odds ratio
                    odds_ratio = np.exp(coeficiente)
                    
                    # Cambio porcentual en odds
                    cambio_porcentual = (odds_ratio - 1) * 100
                    ```
                    """)

            if model is not None and hasattr(model, 'coef_'):
                try:
                    # Preparar datos de coeficientes de forma robusta
                    coef_raw = model.coef_
                    class_names = st.session_state.get('class_names_lr', [])

                    # Para regresión logística binaria, coef_ tiene forma (1, n_features)
                    # Para regresión logística multiclase, coef_ tiene forma (n_classes, n_features)
                    # Para regresión lineal, coef_ tiene forma (n_features,)

                    if len(coef_raw.shape) == 2:
                        # Es una matriz 2D (regresión logística)
                        if coef_raw.shape[0] == 1:
                            # Clasificación binaria: tomar la primera (y única) fila
                            coefficients = coef_raw[0]
                            is_multiclass = False
                        else:
                            # Clasificación multiclase: mostrar nota informativa
                            # Usar primera clase por defecto
                            coefficients = coef_raw[0]
                            is_multiclass = True

                            st.info(
                                f"**Nota:** Este es un modelo de clasificación multiclase con {coef_raw.shape[0]} clases. Mostrando los coeficientes para la primera clase ({class_names[0] if class_names else 'Clase 0'}).")

                            # Opción para seleccionar qué clase mostrar
                            if class_names and len(class_names) == coef_raw.shape[0]:
                                selected_class = st.selectbox(
                                    "Selecciona la clase para mostrar coeficientes:",
                                    options=range(len(class_names)),
                                    format_func=lambda x: class_names[x],
                                    index=0
                                )
                                coefficients = coef_raw[selected_class]
                    else:
                        # Es un vector 1D (regresión lineal)
                        coefficients = coef_raw
                        is_multiclass = False

                    # Asegurar que coefficients es un array 1D
                    coefficients = np.array(coefficients).flatten()

                    # Verificar que las longitudes coincidan
                    if len(coefficients) != len(feature_names):
                        st.error(
                            f"Error: Se encontraron {len(coefficients)} coeficientes pero {len(feature_names)} características.")
                        st.error(f"Forma de coef_: {coef_raw.shape}")
                        st.error(f"Características: {feature_names}")
                        return

                    coef_df = pd.DataFrame({
                        'Característica': feature_names,
                        'Coeficiente': coefficients,
                        'Valor_Absoluto': np.abs(coefficients)
                    })

                except Exception as e:
                    st.error(f"Error al procesar los coeficientes: {str(e)}")
                    st.error(f"Forma de model.coef_: {model.coef_.shape}")
                    st.error(
                        f"Número de características: {len(feature_names)}")
                    return
                coef_df = coef_df.sort_values(
                    'Valor_Absoluto', ascending=False)

                # Añadir interpretación
                coef_df['Efecto'] = coef_df['Coeficiente'].apply(
                    lambda x: '📈 Positivo' if x > 0 else '📉 Negativo'
                )
                coef_df['Importancia'] = coef_df['Valor_Absoluto'].apply(
                    lambda x: '🔥 Alta' if x > coef_df['Valor_Absoluto'].quantile(0.75)
                    else ('🔶 Media' if x > coef_df['Valor_Absoluto'].quantile(0.25) else '🔹 Baja')
                )

                # Mostrar tabla de coeficientes
                st.markdown("### 📊 Tabla de Coeficientes")

                # Crear tabla mejorada con información adicional para regresión logística
                if model_type == "Logistic":
                    # Agregar odds ratios y cambios porcentuales
                    coef_df['Odds_Ratio'] = np.exp(coef_df['Coeficiente'])
                    coef_df['Cambio_Porcentual'] = (
                        coef_df['Odds_Ratio'] - 1) * 100

                    # Interpretación de la magnitud del efecto
                    def interpretar_efecto_logistico(odds_ratio):
                        if odds_ratio > 7.4:  # |coef| > 2
                            return '🔥 Muy Fuerte'
                        elif odds_ratio > 2.7:  # |coef| > 1
                            return '🔥 Fuerte'
                        elif odds_ratio > 1.6 or odds_ratio < 0.625:  # |coef| > 0.5
                            return '🔶 Moderado'
                        else:
                            return '🔹 Débil'

                    coef_df['Fuerza_Efecto'] = coef_df['Odds_Ratio'].apply(
                        interpretar_efecto_logistico)

                    # Formatear la tabla para regresión logística
                    display_df = coef_df[[
                        'Característica', 'Coeficiente', 'Odds_Ratio', 'Cambio_Porcentual', 'Efecto', 'Fuerza_Efecto']].copy()
                    display_df['Coeficiente'] = display_df['Coeficiente'].apply(
                        lambda x: f"{x:.4f}")
                    display_df['Odds_Ratio'] = display_df['Odds_Ratio'].apply(
                        lambda x: f"{x:.3f}")
                    display_df['Cambio_Porcentual'] = display_df['Cambio_Porcentual'].apply(
                        lambda x: f"{x:+.1f}%")

                    # Renombrar columnas para mayor claridad
                    display_df = display_df.rename(columns={
                        'Odds_Ratio': 'Odds Ratio',
                        'Cambio_Porcentual': 'Cambio en Odds',
                        'Fuerza_Efecto': 'Fuerza del Efecto'
                    })

                    st.dataframe(
                        display_df, use_container_width=True, hide_index=True)

                    # Explicación de la tabla para regresión logística
                    with st.expander("📖 ¿Cómo leer esta tabla?", expanded=False):
                        st.markdown("""
                        **Columnas de la tabla:**
                        
                        - **Coeficiente:** Valor original del modelo (log-odds)
                        - **Odds Ratio:** `e^coeficiente` - Más fácil de interpretar
                        - **Cambio en Odds:** Cambio porcentual en las odds
                        - **Efecto:** Dirección del efecto (positivo/negativo)
                        - **Fuerza del Efecto:** Magnitud del impacto
                        
                        **Interpretación de Odds Ratio:**
                        - **OR = 1.0:** Sin efecto
                        - **OR > 1.0:** Aumenta las odds (efecto positivo)
                        - **OR < 1.0:** Disminuye las odds (efecto negativo)
                        
                        **Ejemplos:**
                        - **OR = 2.0:** Duplica las odds (+100%)
                        - **OR = 1.5:** Aumenta las odds en 50%
                        - **OR = 0.5:** Reduce las odds a la mitad (-50%)
                        - **OR = 0.2:** Reduce las odds en 80%
                        """)

                else:
                    # Tabla estándar para regresión lineal
                    display_df = coef_df[[
                        'Característica', 'Coeficiente', 'Efecto', 'Importancia']].copy()
                    display_df['Coeficiente'] = display_df['Coeficiente'].apply(
                        lambda x: f"{x:.4f}")

                    st.dataframe(
                        display_df, use_container_width=True, hide_index=True)

                    # Explicación de la tabla para regresión lineal
                    with st.expander("📖 ¿Cómo leer esta tabla?", expanded=False):
                        st.markdown("""
                        **Columnas de la tabla:**
                        
                        - **Coeficiente:** Cambio en la variable objetivo por unidad de cambio en la característica
                        - **Efecto:** Dirección de la relación (positiva/negativa)
                        - **Importancia:** Magnitud relativa del efecto
                        
                        **Interpretación directa:**
                        - Un coeficiente de **+50** significa que cada unidad adicional de esa característica 
                          aumenta la predicción en 50 unidades
                        - Un coeficiente de **-20** significa que cada unidad adicional de esa característica 
                          disminuye la predicción en 20 unidades
                        """)

                # Mostrar intercepto si existe
                if hasattr(model, 'intercept_'):
                    st.markdown("### 🎯 Intercepto del Modelo")
                    intercept = model.intercept_[0] if hasattr(
                        model.intercept_, '__len__') else model.intercept_

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Intercepto", f"{intercept:.4f}")

                    if model_type == "Logistic":
                        with col2:
                            intercept_odds_ratio = np.exp(intercept)
                            st.metric("Odds Ratio Base",
                                      f"{intercept_odds_ratio:.3f}")

                        st.info(
                            f"**Interpretación:** Cuando todas las características son 0, el log-odds base es {intercept:.4f} "
                            f"(Odds Ratio = {intercept_odds_ratio:.3f})")
                    else:
                        st.info(
                            f"**Interpretación:** Cuando todas las características son 0, el modelo predice un valor de {intercept:.4f}")

                # Gráfico de coeficientes
                st.markdown("### 📈 Visualización de Coeficientes")

                fig, ax = plt.subplots(
                    figsize=(12, max(6, len(feature_names) * 0.4)))

                # Crear gráfico de barras horizontal
                colors = ['#ff6b6b' if x <
                          0 else '#4ecdc4' for x in coef_df['Coeficiente']]
                bars = ax.barh(range(
                    len(coef_df)), coef_df['Coeficiente'], color=colors, alpha=0.7, edgecolor='black')

                # Personalización
                ax.set_yticks(range(len(coef_df)))
                ax.set_yticklabels(coef_df['Característica'], fontsize=10)
                ax.set_xlabel('Valor del Coeficiente', fontsize=12)
                ax.set_title(
                    'Coeficientes del Modelo (ordenados por importancia)', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-',
                           alpha=0.8, linewidth=1)
                ax.grid(True, alpha=0.3, axis='x')

                # Añadir valores en las barras
                for i, (bar, coef) in enumerate(zip(bars, coef_df['Coeficiente'])):
                    width = bar.get_width()
                    ax.text(width + (0.01 * (max(coef_df['Coeficiente']) - min(coef_df['Coeficiente']))) if width >= 0
                            else width - (0.01 * (max(coef_df['Coeficiente']) - min(coef_df['Coeficiente']))),
                            bar.get_y() + bar.get_height()/2,
                            f'{coef:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)

                # Leyenda
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#4ecdc4', alpha=0.7,
                          label='Efecto Positivo'),
                    Patch(facecolor='#ff6b6b', alpha=0.7,
                          label='Efecto Negativo')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

                plt.tight_layout()

                # Mostrar con 80% del ancho
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                with col2:
                    st.pyplot(fig, use_container_width=True)

                # Análisis de importancia
                st.markdown("### 🔍 Análisis Detallado de Importancia")

                # Identificar las características más importantes
                top_features = coef_df.head(5)  # Mostrar top 5 en lugar de 3

                # Análisis diferenciado por tipo de modelo
                if model_type == "Logistic":
                    st.markdown(
                        "#### 🎯 **Características que MÁS aumentan la probabilidad:**")
                    positive_features = coef_df[coef_df['Coeficiente'] > 0].head(
                        3)

                    if len(positive_features) > 0:
                        for i, row in positive_features.iterrows():
                            odds_ratio = np.exp(row['Coeficiente'])
                            cambio_pct = (odds_ratio - 1) * 100

                            # Determinar la intensidad del efecto
                            if odds_ratio > 3:
                                intensidad = "🔥 **FUERTE**"
                            elif odds_ratio > 1.5:
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{row['Coeficiente']:.4f}`
                            - Odds Ratio: `{odds_ratio:.3f}`
                            - **Impacto:** Cada unidad adicional **multiplica las odds por {odds_ratio:.2f}** (aumenta {cambio_pct:+.1f}%)
                            """)
                    else:
                        st.info(
                            "No hay características que aumenten significativamente la probabilidad.")

                    st.markdown(
                        "#### 📉 **Características que MÁS disminuyen la probabilidad:**")
                    negative_features = coef_df[coef_df['Coeficiente'] < 0].head(
                        3)

                    if len(negative_features) > 0:
                        for i, row in negative_features.iterrows():
                            odds_ratio = np.exp(row['Coeficiente'])
                            reduccion_pct = (1 - odds_ratio) * 100

                            # Determinar la intensidad del efecto
                            if odds_ratio < 0.33:  # Reduce a menos de 1/3
                                intensidad = "🔥 **FUERTE**"
                            elif odds_ratio < 0.67:  # Reduce a menos de 2/3
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{row['Coeficiente']:.4f}`
                            - Odds Ratio: `{odds_ratio:.3f}`
                            - **Impacto:** Cada unidad adicional **reduce las odds en {reduccion_pct:.1f}%**
                            """)
                    else:
                        st.info(
                            "No hay características que disminuyan significativamente la probabilidad.")

                    # Resumen ejecutivo para regresión logística
                    st.markdown("#### 📋 **Resumen Ejecutivo:**")

                    # Característica más influyente
                    most_influential = coef_df.iloc[0]
                    odds_ratio_most = np.exp(most_influential['Coeficiente'])

                    if most_influential['Coeficiente'] > 0:
                        impacto_desc = f"multiplica las odds por {odds_ratio_most:.2f}"
                    else:
                        reduccion = (1 - odds_ratio_most) * 100
                        impacto_desc = f"reduce las odds en {reduccion:.1f}%"

                    st.success(f"""
                    🎯 **Factor más determinante:** `{most_influential['Característica']}`
                    
                    Esta característica es la que mayor impacto tiene en las predicciones del modelo.
                    Cada unidad adicional {impacto_desc}.
                    """)

                    # Verificar balance de efectos
                    n_positive = len(coef_df[coef_df['Coeficiente'] > 0])
                    n_negative = len(coef_df[coef_df['Coeficiente'] < 0])

                    if n_positive == 0:
                        st.warning(
                            "⚠️ **Atención:** Todas las características reducen la probabilidad. Verifica que el modelo esté bien entrenado.")
                    elif n_negative == 0:
                        st.warning(
                            "⚠️ **Atención:** Todas las características aumentan la probabilidad. Verifica que el modelo esté bien entrenado.")
                    else:
                        st.info(
                            f"✅ **Balance:** {n_positive} características aumentan y {n_negative} disminuyen la probabilidad.")

                else:  # Regresión lineal
                    st.markdown(
                        "#### 📈 **Características que MÁS aumentan el valor predicho:**")
                    positive_features = coef_df[coef_df['Coeficiente'] > 0].head(
                        3)

                    if len(positive_features) > 0:
                        for i, row in positive_features.iterrows():
                            coef_val = row['Coeficiente']

                            # Determinar la intensidad del efecto
                            abs_coef = abs(coef_val)
                            if abs_coef > coef_df['Valor_Absoluto'].quantile(0.8):
                                intensidad = "🔥 **FUERTE**"
                            elif abs_coef > coef_df['Valor_Absoluto'].quantile(0.6):
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{coef_val:.4f}`
                            - **Impacto:** Cada unidad adicional **aumenta la predicción en {coef_val:.3f} unidades**
                            """)
                    else:
                        st.info(
                            "No hay características que aumenten significativamente el valor predicho.")

                    st.markdown(
                        "#### 📉 **Características que MÁS disminuyen el valor predicho:**")
                    negative_features = coef_df[coef_df['Coeficiente'] < 0].head(
                        3)

                    if len(negative_features) > 0:
                        for i, row in negative_features.iterrows():
                            coef_val = row['Coeficiente']

                            # Determinar la intensidad del efecto
                            abs_coef = abs(coef_val)
                            if abs_coef > coef_df['Valor_Absoluto'].quantile(0.8):
                                intensidad = "🔥 **FUERTE**"
                            elif abs_coef > coef_df['Valor_Absoluto'].quantile(0.6):
                                intensidad = "🔶 **MODERADO**"
                            else:
                                intensidad = "🔹 **DÉBIL**"

                            st.markdown(f"""
                            **{i+1}. {row['Característica']}** {intensidad}
                            - Coeficiente: `{coef_val:.4f}`
                            - **Impacto:** Cada unidad adicional **disminuye la predicción en {abs(coef_val):.3f} unidades**
                            """)
                    else:
                        st.info(
                            "No hay características que disminuyan significativamente el valor predicho.")

                    # Resumen ejecutivo para regresión lineal
                    st.markdown("#### 📋 **Resumen Ejecutivo:**")

                    most_influential = coef_df.iloc[0]
                    coef_val = most_influential['Coeficiente']

                    if coef_val > 0:
                        impacto_desc = f"aumenta la predicción en {coef_val:.3f} unidades"
                    else:
                        impacto_desc = f"disminuye la predicción en {abs(coef_val):.3f} unidades"

                    st.success(f"""
                    🎯 **Factor más determinante:** `{most_influential['Característica']}`
                    
                    Esta característica tiene el mayor impacto absoluto en las predicciones.
                    Cada unidad adicional {impacto_desc}.
                    """)

                # Recomendaciones generales
                st.markdown("#### 💡 **Recomendaciones para la Acción:**")

                # Identificar características controlables vs no controlables
                top_3 = coef_df.head(3)

                recommendations = []
                for i, row in top_3.iterrows():
                    feature_name = row['Característica']
                    effect_direction = "aumentar" if row['Coeficiente'] > 0 else "reducir"
                    target = "probabilidad positiva" if model_type == "Logistic" else "valor predicho"

                    recommendations.append(
                        f"**{feature_name}:** {effect_direction.capitalize()} esta característica para {'incrementar' if row['Coeficiente'] > 0 else 'decrementar'} la {target}")

                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")

                # Warning para coeficientes extremos
                extreme_coefs = coef_df[coef_df['Valor_Absoluto']
                                        > coef_df['Valor_Absoluto'].quantile(0.95)]
                if len(extreme_coefs) > 0:
                    st.warning(f"""
                    ⚠️ **Atención - Coeficientes Extremos Detectados:**
                    
                    Las siguientes características tienen coeficientes muy altos: {', '.join(extreme_coefs['Característica'].tolist())}
                    
                    Esto podría indicar:
                    - Overfitting del modelo
                    - Escalas muy diferentes entre características
                    - Multicolinealidad entre variables
                    - Outliers en los datos
                    
                    **Recomendación:** Considera normalizar las características o revisar la calidad de los datos.
                    """)

            else:
                st.error("El modelo no tiene coeficientes disponibles.")
        else:
            st.info("Entrena un modelo primero para ver los coeficientes.")

    ###########################################
    # Pestaña de Predicciones                 #
    ###########################################
    elif st.session_state.active_tab_lr == 5:
        st.header("Hacer Predicciones")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])
            dataset_name = st.session_state.get('selected_dataset', '')
            X_train = st.session_state.get('X_train_lr')
            class_names = st.session_state.get('class_names_lr', [])
            model_type = st.session_state.get('model_type_lr', 'Linear')
            task_type = st.session_state.get('task_type_lr', 'Regresión')

            # Obtener metadata del dataset para adaptar los inputs
            from dataset_metadata import get_dataset_metadata
            metadata = get_dataset_metadata(dataset_name)
            feature_descriptions = metadata.get('feature_descriptions', {})
            value_mappings = metadata.get('value_mappings', {})
            original_to_display = metadata.get('original_to_display', {})
            categorical_features = metadata.get('categorical_features', [])

            st.markdown("### Ingresa los valores para hacer una predicción")

            # Analizar características si tenemos datos de entrenamiento
            feature_info = {}
            if X_train is not None:
                # Convertir a DataFrame si es necesario
                if hasattr(X_train, 'columns'):
                    column_names = X_train.columns
                else:
                    column_names = range(len(feature_names))

                for i, feature_display_name in enumerate(feature_names):
                    # Obtener el nombre original de la columna
                    if i < len(column_names):
                        original_col_name = column_names[i]
                    else:
                        original_col_name = feature_display_name

                    # Encontrar el nombre original usando reverse mapping
                    reverse_mapping = {v: k for k,
                                       v in original_to_display.items()}
                    if feature_display_name in reverse_mapping:
                        original_col_name = reverse_mapping[feature_display_name]

                    # Usar el índice para acceder a las columnas
                    if hasattr(X_train, 'iloc'):
                        feature_col = X_train.iloc[:, i]
                    else:
                        feature_col = X_train[:, i]

                    # Determinar tipo de característica
                    unique_values = len(set(feature_col)) if hasattr(
                        feature_col, '__iter__') else 10
                    unique_vals = sorted(list(set(feature_col))) if hasattr(
                        feature_col, '__iter__') else [0, 1]

                    # Verificar si es categórica según metadata
                    is_categorical_by_metadata = original_col_name in categorical_features

                    if unique_values <= 2:
                        feature_type = 'binary'
                    elif unique_values <= 10 and (all(isinstance(x, (int, float)) and x == int(x) for x in unique_vals) or is_categorical_by_metadata):
                        feature_type = 'categorical'
                    else:
                        feature_type = 'continuous'

                    # Preparar información de la característica
                    if feature_type in ['binary', 'categorical'] and original_col_name in value_mappings:
                        display_values = []
                        value_to_original = {}
                        for orig_val in unique_vals:
                            if orig_val in value_mappings[original_col_name]:
                                display_val = value_mappings[original_col_name][orig_val]
                                display_values.append(display_val)
                                value_to_original[display_val] = orig_val
                            else:
                                display_values.append(str(orig_val))
                                value_to_original[str(orig_val)] = orig_val

                        feature_info[i] = {
                            'type': feature_type,
                            'values': unique_vals,
                            'display_values': display_values,
                            'value_to_original': value_to_original,
                            'original_column': original_col_name,
                            'display_name': feature_display_name
                        }
                    else:
                        feature_info[i] = {
                            'type': feature_type,
                            'values': unique_vals,
                            'min': float(min(unique_vals)) if feature_type == 'continuous' else min(unique_vals),
                            'max': float(max(unique_vals)) if feature_type == 'continuous' else max(unique_vals),
                            'mean': float(sum(feature_col) / len(feature_col)) if feature_type == 'continuous' and hasattr(feature_col, '__iter__') else None,
                            'original_column': original_col_name,
                            'display_name': feature_display_name
                        }
            else:
                # Valores por defecto si no hay datos de entrenamiento
                for i, feature in enumerate(feature_names):
                    feature_info[i] = {
                        'type': 'continuous',
                        'min': 0.0,
                        'max': 10.0,
                        'mean': 5.0,
                        'original_column': feature,
                        'display_name': feature
                    }

            # Crear controles adaptativos para cada característica
            input_values = []
            cols = st.columns(min(3, len(feature_names)))

            for i, feature in enumerate(feature_names):
                with cols[i % len(cols)]:
                    info = feature_info.get(
                        i, {'type': 'continuous', 'min': 0.0, 'max': 10.0, 'mean': 5.0})

                    # Crear etiqueta con descripción si está disponible
                    original_col = info.get('original_column', feature)
                    description = feature_descriptions.get(original_col, '')
                    if description:
                        label = f"**{feature}**\n\n*{description}*"
                    else:
                        label = feature

                    if info['type'] == 'binary':
                        # Control para características binarias
                        if 'display_values' in info and 'value_to_original' in info:
                            selected_display = st.selectbox(
                                label,
                                options=info['display_values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                            value = info['value_to_original'][selected_display]
                        elif len(info['values']) == 2 and 0 in info['values'] and 1 in info['values']:
                            value = st.checkbox(label, key=f"pred_input_{i}")
                            value = 1 if value else 0
                        else:
                            value = st.selectbox(
                                label,
                                options=info['values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                        input_values.append(value)

                    elif info['type'] == 'categorical':
                        # Control para características categóricas
                        if 'display_values' in info and 'value_to_original' in info:
                            selected_display = st.selectbox(
                                label,
                                options=info['display_values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                            value = info['value_to_original'][selected_display]
                        else:
                            value = st.selectbox(
                                label,
                                options=info['values'],
                                index=0,
                                key=f"pred_input_{i}"
                            )
                        input_values.append(value)

                    else:  # continuous
                        # Control para características continuas
                        if 'min' in info and 'max' in info:
                            step = (info['max'] - info['min']) / \
                                100 if info['max'] != info['min'] else 0.1
                            default_val = info.get(
                                'mean', (info['min'] + info['max']) / 2)
                            value = st.slider(
                                label,
                                min_value=info['min'],
                                max_value=info['max'],
                                value=default_val,
                                step=step,
                                key=f"pred_input_{i}"
                            )
                        else:
                            value = st.number_input(
                                label,
                                value=0.0,
                                key=f"pred_input_{i}"
                            )
                        input_values.append(value)

            if st.button("🔮 Predecir", key="predict_lr_button"):
                try:
                    if model is not None:
                        prediction = model.predict([input_values])[0]

                        # For logistic regression, convert numeric prediction to class label
                        if task_type == 'Clasificación' and class_names is not None:
                            prediction_label = class_names[int(prediction)]
                            st.success(f"Predicción: {prediction_label}")
                        else:
                            # For regression, show numeric prediction
                            st.success(f"Predicción: {prediction:.4f}")
                    else:
                        st.error("Modelo no disponible")
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")
        else:
            st.info("Entrena un modelo primero para hacer predicciones.")

    ###########################################
    # Pestaña de Exportar                     #
    ###########################################
    elif st.session_state.active_tab_lr == 6:
        st.header("Exportar Modelo")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')

            col1, col2 = st.columns(2)

            with col2:
                if st.button("📥 Descargar Modelo (Pickle)", key="download_pickle_lr"):
                    pickle_data = export_model_pickle(model)
                    st.download_button(
                        label="Descargar modelo.pkl",
                        data=pickle_data,
                        file_name="linear_regression_model.pkl",
                        mime="application/octet-stream"
                    )

            with col1:
                if st.button("📄 Generar Código", key="generate_code_lr"):
                    # Generate complete code for linear models
                    model_type = st.session_state.get(
                        'model_type_lr', 'Linear')
                    feature_names = st.session_state.get(
                        'feature_names_lr', [])
                    class_names = st.session_state.get('class_names_lr', [])

                    if model_type == "Logistic":
                        code = generate_logistic_regression_code(
                            feature_names, class_names)
                    else:
                        code = generate_linear_regression_code(feature_names)

                    st.code(code, language="python")

                    # Download button for the code
                    st.download_button(
                        label="📥 Descargar código",
                        data=code,
                        file_name=f"{'logistic' if model_type == 'Logistic' else 'linear'}_regression_code.py",
                        mime="text/plain"
                    )
        else:
            st.info("Entrena un modelo primero para exportarlo.")
