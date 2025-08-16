import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression

from dataset.dataset_manager import load_data, preprocess_data
from dataset.dataset_tab import run_dataset_tab
from utils import create_info_box, get_image_download_link, show_code_with_download
from algorithms.model_training import train_linear_model
from algorithms.model_evaluation import show_detailed_evaluation
from viz.roc import plot_roc_curve, plot_threshold_analysis
from viz.residual import plot_predictions, plot_residuals
from viz.decision_boundary import plot_decision_boundary, plot_decision_surface
from ui import create_button_panel, create_prediction_interface


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
                st.session_state.class_names = class_names
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
                        if st.session_state.get('model_type_lr', 'Linear') == "Logistic":
                            st.session_state.max_iter = max_iter
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
            feature_names = st.session_state.get('feature_names_lr', [])

            if model_type == "Linear" and X_test is not None and y_test is not None and model is not None:

                viz_options = [
                    ("📈 Análisis de Residuos", "Residuos", "viz_residuals"),
                    ("🌐 Superficie de Predicción", "Superficie", "viz_surface")
                ]

                viz_type = create_button_panel(viz_options)

                if viz_type == "Residuos":
                    y_pred = model.predict(X_test)
                    # Crear visualizaciones con mejor tamaño
                    st.markdown(
                        "### 📊 Gráfico de Predicciones vs Valores Reales")
                    plot_predictions(y_test, y_pred)

                    # Gráfico de residuos
                    st.markdown("### 📈 Análisis de Residuos")
                    plot_residuals(y_test, y_pred)
                elif viz_type == "Superficie":
                    model_2d = LinearRegression()
                    # Superficie de predicción
                    plot_decision_surface(
                        model_2d, feature_names, X_train, y_train)

            elif model_type == "Logistic" and X_test is not None and y_test is not None and model is not None:

                viz_options = [
                    ("📉 Curva ROC", "ROC", "viz_roc"),
                    ("🌈 Frontera de Decisión", "Frontera", "viz_boundary"),
                    ("📊 Distribución de Probabilidades", "Probs", "viz_prob")
                ]

                viz_type = create_button_panel(viz_options)

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                class_names = st.session_state.class_names

                if viz_type == "ROC":
                    # Curva ROC
                    plot_roc_curve(y_test, y_pred_proba,
                                   class_names=class_names)

                elif viz_type == "Frontera":
                    # Frontera de decisión
                    # Verificar que es un modelo de clasificación y que está entrenado
                    if not st.session_state.get('is_trained', False):
                        st.warning(
                            "Primero debes entrenar un modelo en la pestaña '🏋️ Entrenamiento'.")
                    else:

                        # Crear un nuevo modelo entrenado solo con las 2 características seleccionadas
                        # para que sea compatible con DecisionBoundaryDisplay
                        try:
                            model_2d = LogisticRegression(
                                max_iter=st.session_state.max_iter, random_state=42)

                            plot_decision_boundary(
                                model_2d,
                                st.session_state.X_train_lr,
                                st.session_state.y_train_lr,
                                st.session_state.feature_names_lr,
                                st.session_state.class_names_lr
                            )
                        except Exception as e:
                            st.error(
                                f"Error al mostrar la visualización de frontera de decisión: {str(e)}")
                            st.info("""
                            La frontera de decisión requiere:
                            - Un modelo de clasificación entrenado
                            - Exactamente 2 características para visualizar
                            - Datos de entrenamiento válidos
                            """)
                            st.exception(
                                e)  # Mostrar detalles del error para debugging

                elif viz_type == "Probs":
                    # Análisis de probabilidades de predicción
                    st.markdown("### 📊 Distribución de Probabilidades")
                    plot_threshold_analysis(
                        y_test, y_pred_proba, class_names=class_names)

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
        st.header("🔮 Predicciones con Nuevos Datos")

        if st.session_state.get('model_trained_lr', False):
            model = st.session_state.get('model_lr')
            feature_names = st.session_state.get('feature_names_lr', [])
            dataset_name = st.session_state.get('selected_dataset', '')
            X_train = st.session_state.get('X_train_lr')
            class_names = st.session_state.get('class_names_lr', [])
            model_type = st.session_state.get('model_type_lr', 'Linear')
            task_type = st.session_state.get('task_type_lr', 'Regresión')

            create_prediction_interface(
                model,
                feature_names,
                class_names,
                task_type,
                # Pasar datos de entrenamiento para rangos dinámicos
                X_train,
                # Pasar nombre del dataset para metadata
                dataset_name
            )
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
