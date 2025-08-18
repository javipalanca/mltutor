# ====== FUNCIONES EXTRA =========


def show_network_anatomy_tab(viz_mode):
    """Análisis profundo de la estructura y pesos de la red."""
    st.subheader("🧠 Anatomía de la Red Neuronal")

    model = st.session_state.nn_model
    config = st.session_state.nn_config

    if viz_mode == "learning":
        st.info("""
        🎓 **¿Qué estás viendo?**
        Los pesos son las "conexiones" entre neuronas. Valores altos = conexiones fuertes.
        Los sesgos son "umbrales" que cada neurona necesita superar para activarse.
        """)

    # VISUALIZACIÓN INTERACTIVA DE PESOS
    col1, col2 = st.columns([3, 1])

    with col1:
        # Selector de capa
        layer_names = [f"Capa {i}: {layer.name}" for i, layer in enumerate(
            model.layers) if len(layer.get_weights()) > 0]
        selected_layer_idx = st.selectbox("Selecciona una capa:", range(
            len(layer_names)), format_func=lambda x: layer_names[x])

        # Obtener capa seleccionada
        layers_with_weights = [
            layer for layer in model.layers if len(layer.get_weights()) > 0]
        selected_layer = layers_with_weights[selected_layer_idx]
        weights, biases = selected_layer.get_weights()

        # Visualización interactiva de pesos
        show_interactive_weights_visualization(
            weights, biases, selected_layer.name, viz_mode)

    with col2:
        st.markdown("#### 📊 Estadísticas de la Capa")

        # Estadísticas de pesos
        weight_stats = {
            "Media": np.mean(weights),
            "Std": np.std(weights),
            "Min": np.min(weights),
            "Max": np.max(weights),
            "Zeros": np.sum(np.abs(weights) < 1e-6),
            "Total": weights.size
        }

        for stat, value in weight_stats.items():
            if stat == "Zeros":
                st.metric(stat, f"{value}/{weight_stats['Total']}")
            else:
                st.metric(stat, f"{value:.4f}")

        # Análisis de salud de la capa
        if viz_mode in ["analysis", "comparison"]:
            analyze_layer_health(weights, biases, selected_layer.name)


def show_enhanced_decision_surface_tab(viz_mode):
    """Superficie de decisión con herramientas interactivas."""
    st.subheader("🎯 Comportamiento de Decisión")

    model = st.session_state.nn_model
    X_test, y_test = st.session_state.nn_test_data
    feature_names = st.session_state.nn_feature_names
    config = st.session_state.nn_config

    if config['task_type'] != 'Clasificación':
        st.info("💡 Esta visualización está optimizada para problemas de clasificación.")
        show_regression_behavior_analysis(
            model, X_test, y_test, feature_names, viz_mode)
        return

    if len(feature_names) < 2:
        st.warning(
            "⚠️ Se necesitan al menos 2 características para visualizar la superficie de decisión.")
        return

    # SELECTOR INTERACTIVO DE CARACTERÍSTICAS
    st.markdown("#### 🎛️ Configuración de Visualización")

    col1, col2, col3 = st.columns(3)

    with col1:
        feature_x = st.selectbox(
            "Característica X:", feature_names, key="decision_x")
    with col2:
        feature_y = st.selectbox("Característica Y:", feature_names,
                                 index=1 if len(feature_names) > 1 else 0, key="decision_y")
    with col3:
        resolution = st.slider("Resolución:", 20, 200, 50,
                               help="Mayor resolución = más detalle pero más lento")

    if feature_x == feature_y:
        st.warning("⚠️ Selecciona características diferentes para X e Y")
        return

    # VISUALIZACIÓN INTERACTIVA
    col_viz1, col_viz2 = st.columns([2, 1])

    with col_viz1:
        # Crear superficie de decisión interactiva
        show_interactive_decision_surface(
            model, X_test, y_test, feature_names,
            feature_x, feature_y, resolution, viz_mode
        )

    with col_viz2:
        st.markdown("#### 🎮 Experimentación")

        # Punto de prueba interactivo
        st.markdown("**Prueba un punto personalizado:**")

        # Crear inputs para todas las características
        test_point = {}
        x_idx = feature_names.index(feature_x)
        y_idx = feature_names.index(feature_y)

        # Valores por defecto basados en los datos
        x_min, x_max = X_test[:, x_idx].min(), X_test[:, x_idx].max()
        y_min, y_max = X_test[:, y_idx].min(), X_test[:, y_idx].max()

        test_point[feature_x] = st.slider(
            f"{feature_x}:",
            float(x_min), float(x_max),
            float((x_min + x_max) / 2),
            key="test_x"
        )

        test_point[feature_y] = st.slider(
            f"{feature_y}:",
            float(y_min), float(y_max),
            float((y_min + y_max) / 2),
            key="test_y"
        )

        # Para otras características, usar valores promedio
        for i, feat in enumerate(feature_names):
            if feat not in test_point:
                test_point[feat] = float(X_test[:, i].mean())

        # Hacer predicción
        if st.button("🔮 Predecir Punto"):
            make_live_prediction(model, test_point, feature_names, config)


def show_live_activations_tab(viz_mode):
    """Visualización en tiempo real de activaciones de neuronas."""
    st.subheader("⚡ Activaciones en Tiempo Real")

    model = st.session_state.nn_model
    X_test, y_test = st.session_state.nn_test_data
    feature_names = st.session_state.nn_feature_names

    if viz_mode == "learning":
        st.info("""
        🎓 **¿Qué son las activaciones?**
        Son los valores que produce cada neurona cuando procesa una entrada.
        Valores altos = neurona muy activa, valores bajos = neurona poco activa.
        """)

    # SELECTOR DE MUESTRA
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("#### 🎯 Selección de Muestra")

        sample_method = st.radio(
            "Método:",
            ["Muestra específica", "Muestra aleatoria", "Entrada personalizada"]
        )

        if sample_method == "Muestra específica":
            sample_idx = st.slider("Índice de muestra:", 0, len(X_test)-1, 0)
            input_data = X_test[sample_idx:sample_idx+1]

        elif sample_method == "Muestra aleatoria":
            if st.button("🎲 Nueva muestra aleatoria"):
                st.session_state.random_sample_idx = np.random.randint(
                    0, len(X_test))

            if 'random_sample_idx' not in st.session_state:
                st.session_state.random_sample_idx = 0

            sample_idx = st.session_state.random_sample_idx
            input_data = X_test[sample_idx:sample_idx+1]

        else:  # Entrada personalizada
            input_data = create_custom_input_widget(X_test, feature_names)

    with col2:
        # VISUALIZACIÓN DE ACTIVACIONES
        if 'input_data' in locals():
            show_network_activations_live(model, input_data, viz_mode)


def show_interpretability_tab(viz_mode):
    """Herramientas de interpretabilidad y explicabilidad."""
    st.subheader("🔍 Interpretabilidad del Modelo")

    model = st.session_state.nn_model
    X_test, y_test = st.session_state.nn_test_data
    feature_names = st.session_state.nn_feature_names
    config = st.session_state.nn_config

    # ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
    st.markdown("### 📊 Importancia de Características")

    if viz_mode == "learning":
        st.info("""
        🎓 **¿Qué mide la importancia?**
        Te dice qué características son más relevantes para las predicciones del modelo.
        Usa técnicas como permutación para ver cómo cambia la precisión al "mezclar" cada característica.
        """)

    importance_method = st.selectbox(
        "Método de análisis:",
        ["Permutation Importance", "SHAP Values",
            "Gradient-based", "Occlusion Analysis"]
    )

    if st.button(f"🔍 Calcular {importance_method}"):
        with st.spinner(f"Calculando {importance_method}..."):
            if importance_method == "Permutation Importance":
                show_permutation_importance(
                    model, X_test, y_test, feature_names, config)
            elif importance_method == "SHAP Values":
                show_shap_analysis(model, X_test, feature_names, config)
            elif importance_method == "Gradient-based":
                show_gradient_importance(model, X_test, feature_names, config)
            else:  # Occlusion Analysis
                show_occlusion_analysis(model, X_test, feature_names, config)


def show_interactive_simulator_tab(viz_mode):
    """Simulador interactivo del funcionamiento de la red."""
    st.subheader("🎮 Simulador Interactivo de Red Neuronal")

    st.info("""
    🎮 **Simulador de Red Neuronal**
    Experimenta con diferentes entradas y observa cómo se comporta cada capa de la red en tiempo real.
    """)

    model = st.session_state.nn_model
    X_test, y_test = st.session_state.nn_test_data
    feature_names = st.session_state.nn_feature_names
    config = st.session_state.nn_config

    # CONFIGURACIÓN DEL SIMULADOR
    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.markdown("#### ⚙️ Configuración")

        # Modo de simulación
        sim_mode = st.radio(
            "Modo de simulación:",
            ["Paso a paso", "Tiempo real", "Comparación"]
        )

        # Velocidad de animación
        if sim_mode == "Tiempo real":
            animation_speed = st.slider("Velocidad:", 0.1, 2.0, 1.0, 0.1)

        # Entrada de datos
        st.markdown("**Entrada personalizada:**")
        custom_input = create_interactive_input_panel(X_test, feature_names)

        if st.button("▶️ Ejecutar Simulación"):
            st.session_state.run_simulation = True

    with sim_col2:
        # VISUALIZACIÓN DE LA SIMULACIÓN
        if st.session_state.get('run_simulation', False):
            if sim_mode == "Paso a paso":
                show_step_by_step_simulation(model, custom_input, config)
            elif sim_mode == "Tiempo real":
                show_realtime_simulation(
                    model, custom_input, config, animation_speed)
            else:  # Comparación
                show_comparison_simulation(model, custom_input, X_test, config)


# ===== FUNCIONES AUXILIARES PARA VISUALIZACIONES =====

def show_interactive_training_plot(history, config, viz_mode):
    """Gráfico interactivo del historial de entrenamiento."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Crear subplot con métricas múltiples
    metrics = list(history.history.keys())

    # Separar métricas de entrenamiento y validación
    train_metrics = [m for m in metrics if not m.startswith('val_')]
    val_metrics = [m for m in metrics if m.startswith('val_')]

    # Crear figura con subplots
    fig = make_subplots(
        rows=len(train_metrics), cols=1,
        subplot_titles=[f"{metric.title()}" for metric in train_metrics],
        vertical_spacing=0.08
    )

    epochs = list(range(1, len(history.history['loss']) + 1))

    for i, metric in enumerate(train_metrics):
        row = i + 1

        # Métrica de entrenamiento
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history.history[metric],
                mode='lines+markers',
                name=f'Entrenamiento {metric}',
                line=dict(color='blue', width=2),
                hovertemplate=f'<b>Época %{{x}}</b><br>{metric}: %{{y:.6f}}<extra></extra>'
            ),
            row=row, col=1
        )

        # Métrica de validación si existe
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history.history[val_metric],
                    mode='lines+markers',
                    name=f'Validación {metric}',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate=f'<b>Época %{{x}}</b><br>Val {metric}: %{{y:.6f}}<extra></extra>'
                ),
                row=row, col=1
            )

    # Configurar layout
    fig.update_layout(
        height=300 * len(train_metrics),
        title="📈 Evolución del Entrenamiento (Interactivo)",
        showlegend=True,
        hovermode='x unified'
    )

    # Añadir herramientas interactivas
    fig.update_layout(
        xaxis=dict(title="Época"),
        yaxis=dict(title="Valor"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Análisis automático basado en el gráfico
    if viz_mode in ["analysis", "interactive"]:
        analyze_training_behavior_interactive(history, epochs)


def show_interactive_weights_visualization(weights, biases, layer_name, viz_mode):
    """Visualización interactiva de pesos y sesgos."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Crear visualización de heatmap interactivo
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Pesos (Weights)", "Sesgos (Biases)"],
        column_widths=[0.8, 0.2]
    )

    # Heatmap de pesos
    fig.add_trace(
        go.Heatmap(
            z=weights.T if weights.ndim > 1 else weights.reshape(1, -1),
            colorscale='RdBu',
            zmid=0,
            name="Pesos",
            hovertemplate='<b>Neurona entrada: %{x}</b><br>Neurona salida: %{y}<br>Peso: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Visualización de sesgos
    fig.add_trace(
        go.Heatmap(
            z=biases.reshape(-1, 1),
            colorscale='Viridis',
            name="Sesgos",
            hovertemplate='<b>Neurona: %{y}</b><br>Sesgo: %{z:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"🧠 Análisis de Pesos - {layer_name}",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Histogramas interactivos
    if viz_mode in ["analysis", "interactive"]:
        show_weight_distribution_analysis(weights, biases, layer_name)


def create_custom_input_widget(X_test, feature_names):
    """Crea un widget para entrada personalizada."""
    st.markdown("**Crea tu entrada personalizada:**")

    input_values = []
    cols = st.columns(min(3, len(feature_names)))

    for i, feature in enumerate(feature_names):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Calcular rango basado en datos
            min_val = float(X_test[:, i].min())
            max_val = float(X_test[:, i].max())
            default_val = float(X_test[:, i].mean())

            value = st.number_input(
                f"{feature}:",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"custom_input_{i}"
            )
            input_values.append(value)

    return np.array(input_values).reshape(1, -1)


def show_network_activations_live(model, input_data, viz_mode):
    """Muestra activaciones de la red en tiempo real."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Crear modelo para extraer activaciones intermedias
    layer_outputs = []
    layer_names = []

    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'activation') and len(layer.get_weights()) > 0:
            layer_outputs.append(layer.output)
            layer_names.append(f"Capa {i}: {layer.name}")

    if not layer_outputs:
        st.warning("No hay capas con activaciones disponibles")
        return

    # Crear modelo de activaciones
    try:
        activation_model = tf.keras.Model(
            inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(input_data, verbose=0)

        # Asegurar que activations sea una lista
        if not isinstance(activations, list):
            activations = [activations]

        # Crear visualización
        fig = make_subplots(
            rows=len(activations), cols=1,
            subplot_titles=layer_names,
            vertical_spacing=0.05
        )

        for i, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            row = i + 1

            # Aplanar activaciones para visualización
            act_flat = activation.flatten()

            # Crear heatmap de activaciones
            fig.add_trace(
                go.Heatmap(
                    z=activation.reshape(
                        1, -1) if activation.ndim == 1 else activation,
                    colorscale='Viridis',
                    name=layer_name,
                    hovertemplate=f'<b>{layer_name}</b><br>Neurona: %{{x}}<br>Activación: %{{z:.4f}}<extra></extra>'
                ),
                row=row, col=1
            )

        fig.update_layout(
            title="⚡ Activaciones en Tiempo Real",
            height=150 * len(activations),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas de activaciones
        if viz_mode in ["analysis", "interactive"]:
            show_activation_statistics(activations, layer_names)

    except Exception as e:
        st.error(f"Error creando modelo de activaciones: {str(e)}")


def show_step_by_step_simulation(model, input_data, config):
    """Simulación paso a paso del procesamiento."""
    st.markdown("### 🎬 Simulación Paso a Paso")

    # Controles de simulación
    step_col1, step_col2 = st.columns([1, 4])

    with step_col1:
        if 'simulation_step' not in st.session_state:
            st.session_state.simulation_step = 0

        total_layers = len(
            [layer for layer in model.layers if len(layer.get_weights()) > 0])

        if st.button("▶️ Siguiente Paso"):
            st.session_state.simulation_step = min(
                st.session_state.simulation_step + 1, total_layers)

        if st.button("⏮️ Reiniciar"):
            st.session_state.simulation_step = 0

        st.write(f"Paso: {st.session_state.simulation_step}/{total_layers}")

    with step_col2:
        # Mostrar el procesamiento hasta el paso actual
        show_processing_up_to_step(
            model, input_data, st.session_state.simulation_step)


# ===== FUNCIONES AUXILIARES ADICIONALES =====

def analyze_training_period(history, epoch_range, selected_metrics):
    """Analiza un período específico del entrenamiento."""
    start_epoch, end_epoch = epoch_range

    st.markdown(
        f"#### 🔍 Análisis del período: Épocas {start_epoch+1} - {end_epoch+1}")

    for metric in selected_metrics:
        if metric in history.history:
            period_values = history.history[metric][start_epoch:end_epoch+1]

            improvement = period_values[-1] - period_values[0]
            avg_improvement_per_epoch = improvement / \
                len(period_values) if len(period_values) > 1 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{metric} - Cambio Total", f"{improvement:.6f}")
            with col2:
                st.metric(f"{metric} - Promedio por Época",
                          f"{avg_improvement_per_epoch:.6f}")
            with col3:
                volatility = np.std(period_values)
                st.metric(f"{metric} - Volatilidad", f"{volatility:.6f}")


def detect_training_patterns(history, diagnostics):
    """Detecta patrones automáticamente en el entrenamiento."""
    patterns = {
        'detected': [],
        'alerts': [],
        'recommendations': []
    }

    loss_values = history.history['loss']
    epochs = len(loss_values)

    # Detectar convergencia rápida
    if epochs >= 10:
        early_loss = np.mean(loss_values[:epochs//4])
        mid_loss = np.mean(loss_values[epochs//4:epochs//2])

        if (early_loss - mid_loss) / early_loss > 0.5:
            patterns['detected'].append(
                "🚀 Convergencia rápida en fase inicial")

    # Detectar plateau
    if epochs >= 20:
        recent_slope = np.polyfit(
            range(epochs//2, epochs), loss_values[epochs//2:], 1)[0]
        if abs(recent_slope) < 0.001:
            patterns['detected'].append("🎯 Plateau alcanzado")
            patterns['recommendations'].append(
                "💡 Considera usar early stopping")

    # Detectar oscilaciones
    if epochs >= 10:
        recent_losses = loss_values[-10:]
        if np.std(recent_losses) / np.mean(recent_losses) > 0.1:
            patterns['alerts'].append(
                "⚠️ Oscilaciones en pérdida - Learning rate puede ser alto")

    return patterns


def make_live_prediction(model, test_point, feature_names, config):
    """Hace una predicción en tiempo real y muestra el proceso."""

    # Preparar datos
    input_array = np.array([test_point[feat]
                           for feat in feature_names]).reshape(1, -1)

    # Normalizar si hay scaler
    if 'nn_scaler' in st.session_state:
        scaler = st.session_state.nn_scaler
        input_array = scaler.transform(input_array)

    # Hacer predicción
    prediction = model.predict(input_array, verbose=0)

    # Mostrar resultado
    if config['task_type'] == 'Clasificación':
        # Interpretar predicción de clasificación
        if config['output_size'] == 1:
            prob = prediction[0][0]
            predicted_class = 1 if prob > 0.5 else 0
            st.success(
                f"🎯 **Predicción**: Clase {predicted_class} (Probabilidad: {prob:.3f})")
        else:
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            # Mostrar probabilidades de todas las clases
            st.success(
                f"🎯 **Predicción**: Clase {predicted_class} (Confianza: {confidence:.3f})")

            # Mostrar distribución de probabilidades
            if 'nn_class_names' in st.session_state:
                class_names = st.session_state.nn_class_names
                prob_df = pd.DataFrame({
                    'Clase': class_names[:len(prediction[0])],
                    'Probabilidad': prediction[0]
                }).sort_values('Probabilidad', ascending=False)

                st.dataframe(prob_df, use_container_width=True)
    else:
        # Regresión
        value = prediction[0][0]
        st.success(f"🎯 **Predicción**: {value:.4f}")

    # Mostrar valores de entrada utilizados
    with st.expander("📊 Valores de entrada utilizados"):
        input_df = pd.DataFrame({
            'Característica': feature_names,
            'Valor': [test_point[feat] for feat in feature_names]
        })
        st.dataframe(input_df, use_container_width=True)


def analyze_layer_health(weights, biases, layer_name):
    """Analiza la salud de una capa específica."""
    st.markdown("**🏥 Análisis de Salud:**")

    # Detectar neuronas muertas
    dead_neurons = np.sum(np.abs(weights).max(axis=0) < 1e-6)
    if dead_neurons > 0:
        st.error(f"💀 {dead_neurons} neuronas muertas detectadas")
    else:
        st.success("✅ No hay neuronas muertas")

    # Detectar saturación
    saturated = np.sum(np.abs(weights) > 10)
    if saturated > weights.size * 0.1:
        st.warning(f"⚠️ {saturated} pesos saturados (>10)")
    else:
        st.success("✅ Pesos en rango saludable")

    # Detectar gradiente explosivo
    if np.max(np.abs(weights)) > 100:
        st.error("💥 Posible gradiente explosivo")

    # Distribución de pesos
    weight_std = np.std(weights)
    if weight_std < 0.01:
        st.warning("⚠️ Pesos muy uniformes - puede indicar poco aprendizaje")
    elif weight_std > 10:
        st.warning("⚠️ Pesos muy dispersos - puede indicar inestabilidad")
    else:
        st.success(
            f"✅ Distribución de pesos saludable (std: {weight_std:.4f})")


# ...existing code...

def create_interactive_input_panel(X_test, feature_names):
    """Crea un panel interactivo para entrada personalizada de datos."""
    st.markdown("**Configura los valores de entrada:**")

    input_values = {}

    # Crear controles para cada característica
    n_features = len(feature_names)
    n_cols = min(3, n_features)  # Máximo 3 columnas
    cols = st.columns(n_cols)

    for i, feature in enumerate(feature_names):
        col_idx = i % n_cols
        with cols[col_idx]:
            # Calcular estadísticas de la característica
            min_val = float(X_test[:, i].min())
            max_val = float(X_test[:, i].max())
            mean_val = float(X_test[:, i].mean())
            std_val = float(X_test[:, i].std())

            # Crear slider con rango apropiado
            value = st.slider(
                f"{feature}:",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100,
                key=f"interactive_input_{i}",
                help=f"Rango: [{min_val:.3f}, {max_val:.3f}], Media: {mean_val:.3f}"
            )
            input_values[feature] = value

    # Convertir a array numpy
    input_array = np.array([input_values[feat]
                           for feat in feature_names]).reshape(1, -1)

    # Mostrar resumen de valores
    with st.expander("📊 Resumen de valores seleccionados"):
        summary_df = pd.DataFrame({
            'Característica': feature_names,
            'Valor Seleccionado': [input_values[feat] for feat in feature_names],
            'Valor Medio Dataset': [X_test[:, i].mean() for i in range(len(feature_names))],
            'Desviación': [abs(input_values[feat] - X_test[:, i].mean()) for i, feat in enumerate(feature_names)]
        })
        st.dataframe(summary_df, use_container_width=True)

    return input_array


def show_realtime_simulation(model, custom_input, config, animation_speed):
    """Muestra una simulación en tiempo real del procesamiento de la red."""
    st.markdown("### ⚡ Simulación en Tiempo Real")

    try:
        # Crear modelo para activaciones intermedias
        layer_outputs = []
        layer_names = []

        for i, layer in enumerate(model.layers):
            if len(layer.get_weights()) > 0:  # Solo capas con pesos
                layer_outputs.append(layer.output)
                layer_names.append(f"Capa {i+1}")

        if not layer_outputs:
            st.warning("No hay capas disponibles para mostrar")
            return

        activation_model = tf.keras.Model(
            inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(custom_input, verbose=0)

        if not isinstance(activations, list):
            activations = [activations]

        # Mostrar activaciones con animación simulada
        st.markdown("#### 🧠 Flujo de Información por Capas")

        for i, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.markdown(f"**{layer_name}**")
                    st.metric("Neuronas Activas",
                              f"{np.sum(activation > 0.1)}/{activation.size}")
                    st.metric("Activación Media", f"{np.mean(activation):.3f}")
                    st.metric("Activación Máxima", f"{np.max(activation):.3f}")

                with col2:
                    # Crear visualización de barras para las activaciones
                    if activation.size <= 50:  # Solo si no hay demasiadas neuronas
                        import plotly.graph_objects as go

                        fig = go.Figure(data=go.Bar(
                            x=list(range(activation.size)),
                            y=activation.flatten(),
                            marker_color=activation.flatten(),
                            colorscale='Viridis'
                        ))

                        fig.update_layout(
                            title=f"Activaciones - {layer_name}",
                            xaxis_title="Neurona",
                            yaxis_title="Activación",
                            height=200,
                            showlegend=False
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Para capas con muchas neuronas, mostrar histograma
                        import plotly.graph_objects as go

                        fig = go.Figure(data=go.Histogram(
                            x=activation.flatten(),
                            nbinsx=20,
                            marker_color='lightblue'
                        ))

                        fig.update_layout(
                            title=f"Distribución de Activaciones - {layer_name}",
                            xaxis_title="Valor de Activación",
                            yaxis_title="Frecuencia",
                            height=200
                        )

                        st.plotly_chart(fig, use_container_width=True)

                # Simular "tiempo" entre capas
                if i < len(activations) - 1:
                    st.markdown("⬇️")

        # Mostrar predicción final
        final_prediction = model.predict(custom_input, verbose=0)

        st.markdown("#### 🎯 Resultado Final")
        if config['task_type'] == 'Clasificación':
            if config['output_size'] == 1:
                prob = final_prediction[0][0]
                predicted_class = 1 if prob > 0.5 else 0
                st.success(
                    f"🏆 **Clase Predicha**: {predicted_class} (Probabilidad: {prob:.3f})")
            else:
                predicted_class = np.argmax(final_prediction[0])
                confidence = np.max(final_prediction[0])
                st.success(
                    f"🏆 **Clase Predicha**: {predicted_class} (Confianza: {confidence:.3f})")

                # Mostrar todas las probabilidades
                prob_cols = st.columns(min(len(final_prediction[0]), 5))
                # Máximo 5 clases
                for i, prob in enumerate(final_prediction[0][:5]):
                    with prob_cols[i]:
                        st.metric(f"Clase {i}", f"{prob:.3f}")
        else:
            value = final_prediction[0][0]
            st.success(f"🏆 **Valor Predicho**: {value:.4f}")

    except Exception as e:
        st.error(f"Error en simulación en tiempo real: {str(e)}")


def show_comparison_simulation(model, custom_input, X_test, config):
    """Muestra una comparación entre la entrada personalizada y muestras del dataset."""
    st.markdown("### 🔍 Simulación Comparativa")

    try:
        # Seleccionar algunas muestras aleatorias para comparar
        n_samples = min(3, len(X_test))
        random_indices = np.random.choice(
            len(X_test), n_samples, replace=False)
        comparison_samples = X_test[random_indices]

        # Hacer predicciones para todas las muestras
        all_inputs = np.vstack([custom_input, comparison_samples])
        predictions = model.predict(all_inputs, verbose=0)

        st.markdown("#### 📊 Comparación de Predicciones")

        # Crear tabla comparativa
        comparison_data = []

        for i, (input_data, pred) in enumerate(zip(all_inputs, predictions)):
            if i == 0:
                sample_type = "🎯 Tu Entrada"
            else:
                sample_type = f"📝 Muestra {i}"

            if config['task_type'] == 'Clasificación':
                if config['output_size'] == 1:
                    prob = pred[0]
                    predicted_class = 1 if prob > 0.5 else 0
                    result = f"Clase {predicted_class} ({prob:.3f})"
                else:
                    predicted_class = np.argmax(pred)
                    confidence = np.max(pred)
                    result = f"Clase {predicted_class} ({confidence:.3f})"
            else:
                result = f"{pred[0]:.4f}"

            comparison_data.append({
                'Tipo': sample_type,
                'Predicción': result,
                'Confianza': np.max(pred) if config['task_type'] == 'Clasificación' else pred[0]
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Visualización comparativa
        st.markdown("#### 📈 Visualización Comparativa")

        if config['task_type'] == 'Clasificación':
            # Gráfico de barras para probabilidades/confianzas
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Bar(
                x=[row['Tipo'] for row in comparison_data],
                y=[row['Confianza'] for row in comparison_data],
                marker_color=['red' if 'Tu Entrada' in row['Tipo']
                              else 'lightblue' for row in comparison_data]
            ))

            fig.update_layout(
                title="Confianza de Predicciones",
                xaxis_title="Muestra",
                yaxis_title="Confianza",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # Gráfico de líneas para valores de regresión
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Scatter(
                x=[row['Tipo'] for row in comparison_data],
                y=[row['Confianza'] for row in comparison_data],
                mode='markers+lines',
                marker=dict(size=10, color=[
                            'red' if 'Tu Entrada' in row['Tipo'] else 'blue' for row in comparison_data])
            ))

            fig.update_layout(
                title="Valores Predichos",
                xaxis_title="Muestra",
                yaxis_title="Valor Predicho",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error en simulación comparativa: {str(e)}")


def show_processing_up_to_step(model, input_data, current_step):
    """Muestra el procesamiento de la red hasta un paso específico."""
    st.markdown(f"### 🎬 Procesamiento hasta el Paso {current_step}")

    try:
        if current_step == 0:
            st.info("🏁 **Paso 0**: Estado inicial - No hay procesamiento aún")
            return

        # Obtener capas con pesos
        layers_with_weights = [
            layer for layer in model.layers if len(layer.get_weights()) > 0]

        if current_step > len(layers_with_weights):
            current_step = len(layers_with_weights)

        # Crear modelo parcial hasta el paso actual
        layer_outputs = []
        for i in range(current_step):
            if i < len(layers_with_weights):
                layer_outputs.append(layers_with_weights[i].output)

        if not layer_outputs:
            st.warning("No hay capas para mostrar en este paso")
            return

        partial_model = tf.keras.Model(
            inputs=model.input, outputs=layer_outputs)
        partial_outputs = partial_model.predict(input_data, verbose=0)

        if not isinstance(partial_outputs, list):
            partial_outputs = [partial_outputs]

        # Mostrar el procesamiento paso a paso
        st.markdown("#### 🔄 Flujo de Información")

        for i, output in enumerate(partial_outputs):
            step_num = i + 1
            layer_name = f"Capa {step_num}"

            with st.container():
                st.markdown(f"**{layer_name}** (Paso {step_num})")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Neuronas", output.shape[-1])
                    st.metric("Activas", np.sum(output > 0.1))
                    st.metric("Media", f"{np.mean(output):.3f}")

                with col2:
                    # Visualización simple de las activaciones
                    if output.size <= 20:
                        # Para pocas neuronas, mostrar barras
                        activations = output.flatten()
                        fig, ax = plt.subplots(figsize=(8, 3))
                        bars = ax.bar(range(len(activations)), activations)
                        ax.set_xlabel('Neurona')
                        ax.set_ylabel('Activación')
                        ax.set_title(f'Activaciones - {layer_name}')

                        # Colorear barras según activación
                        for bar, act in zip(bars, activations):
                            if act > 0.5:
                                bar.set_color('green')
                            elif act > 0.1:
                                bar.set_color('orange')
                            else:
                                bar.set_color('red')

                        st.pyplot(fig)
                        plt.close()
                    else:
                        # Para muchas neuronas, mostrar estadísticas
                        st.write(
                            f"Activaciones: Min={np.min(output):.3f}, Max={np.max(output):.3f}")
                        st.write(
                            f"Percentiles: P25={np.percentile(output, 25):.3f}, P75={np.percentile(output, 75):.3f}")

                if i < len(partial_outputs) - 1:
                    st.markdown("⬇️")

        # Mostrar próximo paso
        if current_step < len(layers_with_weights):
            st.info(
                f"▶️ **Próximo paso**: Procesamiento en Capa {current_step + 1}")
        else:
            st.success(
                "🏁 **Procesamiento completo**: Todas las capas procesadas")

    except Exception as e:
        st.error(f"Error mostrando procesamiento paso a paso: {str(e)}")


def show_regression_behavior_analysis(model, X_test, y_test, feature_names, viz_mode):
    """Análisis específico para problemas de regresión."""
    st.markdown("#### 📊 Análisis de Comportamiento - Regresión")

    if viz_mode == "learning":
        st.info("""
        🎓 **Análisis de Regresión:**
        En regresión, analizamos cómo el modelo mapea las entradas a valores continuos.
        Observamos patrones en residuos y predicciones vs valores reales.
        """)

    try:
        # Hacer predicciones
        y_pred = model.predict(X_test, verbose=0).flatten()

        # Análisis de residuos
        residuals = y_test - y_pred

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de predicciones vs reales
            import plotly.graph_objects as go

            fig = go.Figure()

            # Línea perfecta (y = x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(dash='dash', color='red')
            ))

            # Puntos reales vs predichos
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predicciones',
                marker=dict(color='blue', opacity=0.6)
            ))

            fig.update_layout(
                title="Predicciones vs Valores Reales",
                xaxis_title="Valores Reales",
                yaxis_title="Predicciones",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Histograma de residuos
            fig = go.Figure(data=go.Histogram(
                x=residuals,
                nbinsx=20,
                marker_color='lightgreen'
            ))

            fig.update_layout(
                title="Distribución de Residuos",
                xaxis_title="Residuo (Real - Predicho)",
                yaxis_title="Frecuencia",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Métricas de rendimiento
        st.markdown("#### 📏 Métricas de Rendimiento")

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("MSE", f"{mse:.4f}", help="Error Cuadrático Medio")
        with metric_col2:
            st.metric("MAE", f"{mae:.4f}", help="Error Absoluto Medio")
        with metric_col3:
            st.metric("R²", f"{r2:.4f}", help="Coeficiente de Determinación")

    except Exception as e:
        st.error(f"Error en análisis de regresión: {str(e)}")


def show_interactive_decision_surface(model, X_test, y_test, feature_names,
                                      feature_x, feature_y, resolution, viz_mode):
    """Crea una superficie de decisión interactiva."""
    try:
        x_idx = feature_names.index(feature_x)
        y_idx = feature_names.index(feature_y)

        # Crear grid para la superficie de decisión
        x_min, x_max = X_test[:, x_idx].min(
        ) - 0.1, X_test[:, x_idx].max() + 0.1
        y_min, y_max = X_test[:, y_idx].min(
        ) - 0.1, X_test[:, y_idx].max() + 0.1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Crear puntos para predicción
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Para otras características, usar valores promedio
        full_grid = np.zeros((grid_points.shape[0], len(feature_names)))
        full_grid[:, x_idx] = grid_points[:, 0]
        full_grid[:, y_idx] = grid_points[:, 1]

        for i, feature in enumerate(feature_names):
            if i not in [x_idx, y_idx]:
                full_grid[:, i] = X_test[:, i].mean()

        # Normalizar si hay scaler
        if 'nn_scaler' in st.session_state:
            scaler = st.session_state.nn_scaler
            full_grid = scaler.transform(full_grid)

        # Hacer predicciones
        with st.spinner("Calculando superficie de decisión..."):
            Z = model.predict(full_grid, verbose=0)

            if Z.shape[1] == 1:
                Z = Z.ravel()
            else:
                Z = np.argmax(Z, axis=1)

            Z = Z.reshape(xx.shape)

        # Crear visualización interactiva
        import plotly.graph_objects as go

        fig = go.Figure()

        # Superficie de decisión
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=Z,
            colorscale='Viridis',
            opacity=0.6,
            name='Superficie de Decisión',
            showscale=True
        ))

        # Puntos de datos reales
        if 'nn_scaler' in st.session_state:
            # Desnormalizar para mostrar
            X_display = st.session_state.nn_scaler.inverse_transform(X_test)
        else:
            X_display = X_test

        fig.add_trace(go.Scatter(
            x=X_display[:, x_idx],
            y=X_display[:, y_idx],
            mode='markers',
            marker=dict(
                color=y_test,
                size=8,
                colorscale='Set1',
                line=dict(width=1, color='black')
            ),
            name='Datos Reales',
            text=[f'Clase: {int(y)}' for y in y_test],
            hovertemplate=f'<b>{feature_x}</b>: %{{x:.3f}}<br><b>{feature_y}</b>: %{{y:.3f}}<br>%{{text}}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Superficie de Decisión: {feature_x} vs {feature_y}",
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        if viz_mode == "learning":
            st.info("""
            💡 **Interpretación de la Superficie:**
            - Los colores representan las predicciones del modelo
            - Los puntos son los datos reales
            - Las fronteras suaves indican que el modelo ha aprendido patrones complejos
            - Zonas muy "pixeladas" pueden indicar sobreajuste
            """)

    except Exception as e:
        st.error(f"Error creando superficie de decisión: {str(e)}")


# Función para generar código del modelo entrenado
def generate_neural_network_code(config, label_encoder):
    """Genera código Python para el modelo entrenado."""

    code = f"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NeuralNetworkPredictor:
    \"\"\"
    Predictor de Red Neuronal Entrenada
    Arquitectura: {config['architecture']}
    Tipo: {config['task_type']}
    \"\"\"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if {config['task_type'] == 'Clasificación'} else None
        self.feature_names = {st.session_state.get('nn_feature_names', [])}
        
        # Configuración del modelo
        self.config = {config}
        
        # Construir modelo
        self._build_model()
    
    def _build_model(self):
        \"\"\"Construye la arquitectura del modelo.\"\"\"
        model = tf.keras.Sequential()
        
        # Capa de entrada
        model.add(tf.keras.layers.Input(shape=({config['input_size']},)))
        
        # Capas ocultas
        architecture = {config['architecture']}
        for i, neurons in enumerate(architecture[1:-1]):
            model.add(tf.keras.layers.Dense(
                neurons, 
                activation='{config['activation']}',
                name=f'hidden_{{i+1}}'
            ))
            model.add(tf.keras.layers.Dropout({config['dropout_rate']}))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(
            {config['output_size']}, 
            activation='{config['output_activation']}',
            name='output'
        ))
        
        # Compilar
        model.compile(
            optimizer='{config['optimizer']}',
            loss={'binary_crossentropy' if config['task_type'] == 'Clasificación' and config['output_size'] == 1 else 'categorical_crossentropy' if config['task_type'] == 'Clasificación' else 'mse'},
            metrics=['accuracy'] if {config['task_type'] == 'Clasificación'} else ['mae']
        )
        
        self.model = model
        
        # NOTA: Aquí debes cargar los pesos entrenados
        # self.model.load_weights('path_to_weights.h5')
    
    def predict(self, X):
        \"\"\"Realiza predicciones en nuevos datos.\"\"\"
        # Normalizar datos
        X_scaled = self.scaler.transform(X)
        
        # Predicción
        predictions = self.model.predict(X_scaled)
        
        if self.config['task_type'] == 'Clasificación':
            if self.config['output_size'] == 1:
                # Clasificación binaria
                return (predictions > 0.5).astype(int).flatten()
            else:
                # Clasificación multiclase
                return np.argmax(predictions, axis=1)
        else:
            # Regresión
            return predictions.flatten()
    
    def predict_proba(self, X):
        \"\"\"Retorna probabilidades para clasificación.\"\"\"
        if self.config['task_type'] != 'Clasificación':
            raise ValueError("predict_proba solo disponible para clasificación")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# Ejemplo de uso:
# predictor = NeuralNetworkPredictor()
# predictions = predictor.predict(new_data)
"""

    return code


# Código para cargar modelo completo
LOAD_NN = """
import pickle
import numpy as np

# Cargar modelo completo
with open('neural_network_complete.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extraer componentes
model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
config = model_data['config']
feature_names = model_data['feature_names']

# Función para hacer predicciones
def predict_new_data(new_data):
    \"\"\"
    Hace predicciones en nuevos datos.
    new_data: array-like, shape (n_samples, n_features)
    \"\"\"
    # Normalizar datos
    X_scaled = scaler.transform(new_data)
    
    # Predicción
    predictions = model.predict(X_scaled)
    
    # Interpretar según el tipo de problema
    if config['task_type'] == 'Clasificación':
        if config['output_size'] == 1:
            # Clasificación binaria
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Clasificación multiclase
            return np.argmax(predictions, axis=1)
    else:
        # Regresión
        return predictions.flatten()

# Ejemplo de uso:
# new_sample = [[value1, value2, ...]]  # Tus nuevos datos
# prediction = predict_new_data(new_sample)
# print(f"Predicción: {prediction}")
"""
