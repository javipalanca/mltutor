import streamlit as st
import streamlit.components.v1 as components


def evaluate_nn(model, X_test, y_test, task_type):
    # Mostrar métricas básicas con explicaciones
    st.markdown("#### 📊 Resultados del Entrenamiento")

    if task_type == "Clasificación":
        test_loss, test_acc = model.evaluate(
            X_test, y_test, verbose=0)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("🎯 Precisión en Test", f"{test_acc:.3f}",
                      help="Porcentaje de predicciones correctas en datos nunca vistos")
        with col_m2:
            st.metric("📉 Pérdida en Test", f"{test_loss:.3f}",
                      help="Qué tan 'equivocada' está la red en promedio")
    else:
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        test_loss = test_loss[0] if isinstance(test_loss, list) else test_loss
        st.metric("📉 Error en Test", f"{test_loss:.3f}")

    # Gráfico de entrenamiento en tiempo real
    st.markdown("### 📈 Progreso del Entrenamiento")
    history = st.session_state.nn_history
    plot_training_history(history, task_type)


def create_neural_network_visualization(architecture, activation, output_activation, task_type):
    """
    Crea una visualización dinámica de la arquitectura de red neuronal usando HTML5 Canvas.
    """
    try:
        # Colores para diferentes elementos
        colors = {
            'input': '#4ECDC4',
            'hidden': '#45B7D1',
            'output': '#FF6B6B',
            'connection': '#BDC3C7',
            'text': '#2C3E50'
        }

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .nn-container {{
                    max-width: 100%;
                    margin: 0 auto;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                }}
                .canvas-container {{
                    position: relative;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    margin: 10px 0;
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    width: 100%;
                    overflow: hidden;
                }}
                #nnCanvas {{
                    display: block;
                    width: 100%;
                    height: auto;
                    max-width: 100%;
                }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    font-size: 14px;
                }}
                .layer-info {{
                    display: flex;
                    gap: 15px;
                    margin-top: 10px;
                    flex-wrap: wrap;
                    justify-content: center;
                    font-size: 12px;
                }}
                .layer-item {{
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    padding: 5px 10px;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .layer-color {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                }}
            </style>
        </head>
        <body>
            <div class="nn-container">
                <div class="info-box">
                    <strong>🧠 Arquitectura de Red Neuronal</strong><br>
                    Visualización dinámica de la estructura de la red para {task_type.lower()}
                </div>
                
                <div class="canvas-container">
                    <canvas id="nnCanvas"></canvas>
                </div>
                
                <div class="layer-info">
                    <div class="layer-item">
                        <div class="layer-color" style="background-color: {colors['input']}"></div>
                        <span>Capa de Entrada</span>
                    </div>
                    <div class="layer-item">
                        <div class="layer-color" style="background-color: {colors['hidden']}"></div>
                        <span>Capas Ocultas ({activation.upper()})</span>
                    </div>
                    <div class="layer-item">
                        <div class="layer-color" style="background-color: {colors['output']}"></div>
                        <span>Capa de Salida ({output_activation.upper()})</span>
                    </div>
                </div>
            </div>

            <script>
                const canvas = document.getElementById('nnCanvas');
                const ctx = canvas.getContext('2d');
                
                // Arquitectura de la red
                const architecture = {architecture};
                const maxNeurons = Math.max(...architecture);
                
                // Función para redimensionar el canvas
                function resizeCanvas() {{
                    const container = document.querySelector('.canvas-container');
                    const containerWidth = container.clientWidth - 4;
                    const aspectRatio = 2/1;
                    const canvasHeight = Math.max(300, containerWidth / aspectRatio);
                    
                    canvas.width = containerWidth;
                    canvas.height = canvasHeight;
                    canvas.style.width = containerWidth + 'px';
                    canvas.style.height = canvasHeight + 'px';
                    
                    drawNetwork();
                }}
                
                // Función para dibujar la red neuronal
                function drawNetwork() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    const margin = 40;
                    const layerWidth = (canvas.width - 2 * margin) / (architecture.length - 1);
                    const maxRadius = Math.min(20, canvas.width / (architecture.length * 8));
                    
                    // Dibujar conexiones primero
                    ctx.strokeStyle = '{colors['connection']}';
                    ctx.lineWidth = 1;
                    ctx.globalAlpha = 0.3;
                    
                    for (let i = 0; i < architecture.length - 1; i++) {{
                        const currentLayerSize = architecture[i];
                        const nextLayerSize = architecture[i + 1];
                        
                        const currentX = margin + i * layerWidth;
                        const nextX = margin + (i + 1) * layerWidth;
                        
                        for (let j = 0; j < currentLayerSize; j++) {{
                            const currentY = getNodeY(j, currentLayerSize);
                            
                            for (let k = 0; k < nextLayerSize; k++) {{
                                const nextY = getNodeY(k, nextLayerSize);
                                
                                ctx.beginPath();
                                ctx.moveTo(currentX, currentY);
                                ctx.lineTo(nextX, nextY);
                                ctx.stroke();
                            }}
                        }}
                    }}
                    
                    ctx.globalAlpha = 1.0;
                    
                    // Dibujar nodos
                    architecture.forEach((layerSize, layerIndex) => {{
                        const x = margin + layerIndex * layerWidth;
                        const radius = Math.min(maxRadius, Math.max(8, (canvas.height - 2 * margin) / (maxNeurons * 3)));
                        
                        // Color según tipo de capa
                        let color;
                        if (layerIndex === 0) {{
                            color = '{colors['input']}';
                        }} else if (layerIndex === architecture.length - 1) {{
                            color = '{colors['output']}';
                        }} else {{
                            color = '{colors['hidden']}';
                        }}
                        
                        // Dibujar nodos de la capa
                        for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {{
                            const y = getNodeY(nodeIndex, layerSize);
                            
                            // Nodo
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.arc(x, y, radius, 0, 2 * Math.PI);
                            ctx.fill();
                            
                            // Borde
                            ctx.strokeStyle = '#2C3E50';
                            ctx.lineWidth = 2;
                            ctx.stroke();
                        }}
                        
                        // Etiqueta de capa
                        ctx.fillStyle = '{colors['text']}';
                        ctx.font = `bold ${{Math.max(10, canvas.width / 50)}}px Arial`;
                        ctx.textAlign = 'center';
                        
                        let layerLabel;
                        if (layerIndex === 0) {{
                            layerLabel = `Entrada\\n(${{layerSize}})`;
                        }} else if (layerIndex === architecture.length - 1) {{
                            layerLabel = `Salida\\n(${{layerSize}})`;
                        }} else {{
                            layerLabel = `Oculta ${{layerIndex}}\\n(${{layerSize}})`;
                        }}
                        
                        // Dibujar texto en múltiples líneas
                        const lines = layerLabel.split('\\n');
                        lines.forEach((line, lineIndex) => {{
                            ctx.fillText(line, x, canvas.height - 25 + lineIndex * 15);
                        }});
                    }});
                }}
                
                // Función auxiliar para calcular posición Y de un nodo
                function getNodeY(nodeIndex, layerSize) {{
                    const margin = 40;
                    const availableHeight = canvas.height - 2 * margin - 60; // Espacio para etiquetas
                    
                    if (layerSize === 1) {{
                        return margin + availableHeight / 2;
                    }}
                    
                    const spacing = availableHeight / (layerSize + 1);
                    return margin + spacing * (nodeIndex + 1);
                }}
                
                // Inicialización
                resizeCanvas();
                
                // Redimensionar cuando cambie el tamaño de ventana
                window.addEventListener('resize', function() {{
                    setTimeout(resizeCanvas, 100);
                }});
                
                // Observer para detectar cambios en el contenedor
                if (window.ResizeObserver) {{
                    const resizeObserver = new ResizeObserver(entries => {{
                        for (let entry of entries) {{
                            if (entry.target.querySelector('#nnCanvas')) {{
                                resizeCanvas();
                            }}
                        }}
                    }});
                    resizeObserver.observe(document.querySelector('.canvas-container'));
                }}
            </script>
        </body>
        </html>
        """

        components.html(html_code, height=400, scrolling=False)

    except Exception as e:
        st.error(f"Error en la visualización de red neuronal: {str(e)}")


def calculate_network_parameters(architecture):
    """Calcula el número total de parámetros en la red."""
    total_params = 0
    for i in range(len(architecture) - 1):
        # Pesos: current_layer * next_layer + Sesgos: next_layer
        weights = architecture[i] * architecture[i + 1]
        biases = architecture[i + 1]
        total_params += weights + biases
    return total_params


def plot_training_history(history, task_type):
    """Grafica el historial de entrenamiento."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import streamlit as st

        # Crear subplots
        if task_type == 'Clasificación':
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pérdida durante el Entrenamiento',
                                'Precisión durante el Entrenamiento')
            )

            # Pérdida
            fig.add_trace(
                go.Scatter(
                    y=history.history['loss'], name='Entrenamiento', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'], name='Validación', line=dict(color='red')),
                row=1, col=1
            )

            # Precisión
            fig.add_trace(
                go.Scatter(y=history.history['accuracy'], name='Entrenamiento', line=dict(
                    color='blue'), showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_accuracy'], name='Validación', line=dict(
                    color='red'), showlegend=False),
                row=1, col=2
            )

            fig.update_yaxes(title_text="Pérdida", row=1, col=1)
            fig.update_yaxes(title_text="Precisión", row=1, col=2)

        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    'Pérdida (MSE) durante el Entrenamiento', 'Error Absoluto Medio')
            )

            # MSE
            fig.add_trace(
                go.Scatter(
                    y=history.history['loss'], name='Entrenamiento', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'], name='Validación', line=dict(color='red')),
                row=1, col=1
            )

            # MAE
            fig.add_trace(
                go.Scatter(y=history.history['mae'], name='Entrenamiento', line=dict(
                    color='blue'), showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_mae'], name='Validación', line=dict(
                    color='red'), showlegend=False),
                row=1, col=2
            )

            fig.update_yaxes(title_text="MSE", row=1, col=1)
            fig.update_yaxes(title_text="MAE", row=1, col=2)

        fig.update_xaxes(title_text="Época")
        fig.update_layout(height=400, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error graficando historial: {str(e)}")


def show_neural_network_evaluation():
    """Muestra la evaluación detallada del modelo de red neuronal."""

    # Tips educativos sobre evaluación
    st.info("""
    🎓 **Evaluación de Redes Neuronales:**
    - **Accuracy**: Porcentaje de predicciones correctas (para clasificación)
    - **Matriz de Confusión**: Muestra qué clases se confunden entre sí
    - **MSE/MAE**: Errores promedio para regresión
    - **Datos de test**: Nunca vistos durante entrenamiento, miden la capacidad real
    """)

    try:
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
        from plotly.subplots import make_subplots

        model = st.session_state.nn_model
        X_test, y_test = st.session_state.nn_test_data
        scaler = st.session_state.nn_scaler
        label_encoder = st.session_state.nn_label_encoder
        config = st.session_state.nn_config

        # Hacer predicciones
        y_pred = model.predict(X_test, verbose=0)

        # Métricas según el tipo de tarea
        if config['task_type'] == 'Clasificación':
            # Obtener el tamaño de salida de forma segura
            output_size = safe_get_output_size(config)

            # Para clasificación - detectar formato de y_test
            # One-hot encoded (multiclase)
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_test_classes = np.argmax(y_test, axis=1)
            else:  # Binaria o multiclase sin one-hot
                if output_size == 1:  # Binaria con 1 neurona
                    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                    y_test_classes = y_test.flatten()
                else:  # Multiclase sin one-hot (sparse)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_test_classes = y_test.flatten()

            # Accuracy
            accuracy = np.mean(y_pred_classes == y_test_classes)

            # Mostrar métricas principales con explicaciones
            st.markdown("### 🎯 Métricas Principales")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%",
                          help="Porcentaje de predicciones correctas en datos nunca vistos")

                # Interpretación del accuracy
                if accuracy >= 0.9:
                    st.success("🌟 **Excelente**: Tu red predice muy bien")
                elif accuracy >= 0.8:
                    st.success("✅ **Muy Bueno**: Predicciones muy confiables")
                elif accuracy >= 0.7:
                    st.warning("⚠️ **Bueno**: Predicciones aceptables")
                elif accuracy >= 0.6:
                    st.warning("🟡 **Regular**: Hay margen de mejora")
                else:
                    st.error("🔴 **Bajo**: Considera ajustar el modelo")

            with col2:
                # Calcular confianza promedio
                # One-hot multiclase
                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                    confidence = np.mean(np.max(y_pred, axis=1))
                elif output_size == 1:  # Binaria
                    confidence = np.mean(np.maximum(
                        y_pred.flatten(), 1 - y_pred.flatten()))
                else:  # Multiclase sparse
                    confidence = np.mean(np.max(y_pred, axis=1))
                st.metric("🎲 Confianza Promedio",
                          f"{confidence:.4f}", f"{confidence*100:.2f}%")

            with col3:
                # Número de predicciones correctas
                correct_preds = np.sum(y_pred_classes == y_test_classes)
                st.metric("✅ Predicciones Correctas",
                          f"{correct_preds}/{len(y_test_classes)}")

            # Matriz de confusión
            st.subheader("🔍 Matriz de Confusión")

            try:
                cm = confusion_matrix(y_test_classes, y_pred_classes)

                # Obtener nombres de clases
                if label_encoder and hasattr(label_encoder, 'classes_'):
                    class_names = list(label_encoder.classes_)
                else:
                    # Determinar clases basado en los datos únicos
                    all_classes = sorted(
                        set(list(y_test_classes) + list(y_pred_classes)))
                    class_names = [f"Clase {i}" for i in all_classes]

                # Ajustar class_names al tamaño de la matriz si es necesario
                if len(class_names) != cm.shape[0]:
                    class_names = [f"Clase {i}" for i in range(cm.shape[0])]

                # Crear heatmap de la matriz de confusión
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    annotation_text=cm,
                    colorscale='Blues',
                    showscale=True
                )

                fig_cm.update_layout(
                    title='Matriz de Confusión',
                    xaxis_title='Predicciones',
                    yaxis_title='Valores Reales',
                    height=500
                )

                st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as cm_error:
                st.error(
                    f"❌ Error creando matriz de confusión: {str(cm_error)}")
                st.info(
                    "La matriz de confusión no pudo generarse. El modelo funciona correctamente pero hay un problema con la visualización.")

            # Reporte de clasificación detallado
            st.subheader("📋 Reporte de Clasificación")

            if label_encoder:
                target_names = label_encoder.classes_
            else:
                target_names = [f"Clase {i}" for i in range(
                    len(np.unique(y_test_classes)))]

            # Generar reporte
            report = classification_report(
                y_test_classes, y_pred_classes,
                target_names=target_names,
                output_dict=True
            )

            # Mostrar métricas por clase
            metrics_data = []
            for class_name in target_names:
                if class_name in report:
                    metrics_data.append({
                        'Clase': class_name,
                        'Precisión': f"{report[class_name]['precision']:.4f}",
                        'Recall': f"{report[class_name]['recall']:.4f}",
                        'F1-Score': f"{report[class_name]['f1-score']:.4f}",
                        'Soporte': report[class_name]['support']
                    })

            st.dataframe(metrics_data, use_container_width=True)

            # Métricas macro y weighted
            st.subheader("📊 Métricas Agregadas")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **Macro Average:**
                - Precisión: {report['macro avg']['precision']:.4f}
                - Recall: {report['macro avg']['recall']:.4f}
                - F1-Score: {report['macro avg']['f1-score']:.4f}
                """)

            with col2:
                st.info(f"""
                **Weighted Average:**
                - Precisión: {report['weighted avg']['precision']:.4f}
                - Recall: {report['weighted avg']['recall']:.4f}
                - F1-Score: {report['weighted avg']['f1-score']:.4f}
                """)

        else:
            # Para regresión
            y_pred_flat = y_pred.flatten()
            y_test_flat = y_test.flatten()

            # Métricas de regresión
            mse = mean_squared_error(y_test_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_flat, y_pred_flat)
            r2 = r2_score(y_test_flat, y_pred_flat)

            # Mostrar métricas principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 R² Score", f"{r2:.4f}")

            with col2:
                st.metric("📏 MAE", f"{mae:.4f}")

            with col3:
                st.metric("📐 RMSE", f"{rmse:.4f}")

            with col4:
                st.metric("🎯 MSE", f"{mse:.4f}")

            # Gráficos de evaluación para regresión
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Predicciones vs Valores Reales',
                                'Distribución de Residuos')
            )

            # Scatter plot de predicciones vs reales
            fig.add_trace(
                go.Scatter(
                    x=y_test_flat,
                    y=y_pred_flat,
                    mode='markers',
                    name='Predicciones',
                    marker=dict(size=8, opacity=0.6)
                ),
                row=1, col=1
            )

            # Línea de referencia y = x
            min_val = min(y_test_flat.min(), y_pred_flat.min())
            max_val = max(y_test_flat.max(), y_pred_flat.max())

            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Línea Ideal',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )

            # Histograma de residuos
            residuals = y_test_flat - y_pred_flat
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Residuos',
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text="Valores Reales", row=1, col=1)
            fig.update_yaxes(title_text="Predicciones", row=1, col=1)
            fig.update_xaxes(title_text="Residuos", row=1, col=2)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=2)

            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        # Información del modelo
        st.subheader("🔧 Información del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Arquitectura:**
            - Capas: {len(config['architecture'])}
            - Neuronas por capa: {config['architecture']}
            - Función de activación: {config['activation']}
            - Activación de salida: {config['output_activation']}
            """)

        with col2:
            total_params = calculate_network_parameters(config['architecture'])
            st.info(f"""
            **Parámetros:**
            - Total de parámetros: {total_params:,}
            - Optimizador: {config['optimizer']}
            - Dropout: {config['dropout_rate']}
            - Batch size: {config['batch_size']}
            """)

        # Botón para generar código Python de evaluación
        st.markdown("### 💻 Código Python")
        if st.button("📝 Generar Código de Evaluación", use_container_width=True):
            # Generar código Python para evaluación
            code = generate_neural_network_evaluation_code(
                config, st.session_state.nn_feature_names, st.session_state.nn_class_names
            )

            st.markdown("#### 🐍 Código Python para Evaluación")
            st.code(code, language='python')

            # Botón para descargar el código
            st.download_button(
                label="💾 Descargar Código de Evaluación",
                data=code,
                file_name=f"evaluacion_red_neuronal_{config['task_type'].lower()}.py",
                mime="text/plain"
            )

        # Navegación
        st.markdown("---")
        st.markdown("### 🧭 Navegación")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("🔙 Volver a Entrenamiento", use_container_width=True):
                st.session_state.active_tab_nn = 2
                st.rerun()
        with col_nav2:
            if st.button("🎯 Ver Visualizaciones", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 4
                st.rerun()

    except Exception as e:
        st.error(f"Error en la evaluación: {str(e)}")
        st.info(
            "Asegúrate de que TensorFlow esté instalado y el modelo esté entrenado correctamente.")


def show_neural_network_visualizations():
    """Muestra visualizaciones avanzadas del modelo."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    # Tips educativos sobre visualizaciones
    st.info("""
    🎓 **Visualizaciones de Redes Neuronales:**
    - **Historial de entrenamiento**: Muestra cómo evoluciona el aprendizaje
    - **Pesos y sesgos**: Revelan qué ha aprendido cada neurona
    - **Superficie de decisión**: Cómo la red separa las clases (2D)
    - **Análisis de capas**: Activaciones y patrones internos
    
    🔧 **Reparación Automática**: Esta función incluye inicialización automática del modelo para prevenir errores comunes.
    """)

    try:
        import tensorflow as tf
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler

        model = st.session_state.nn_model
        history = st.session_state.nn_history
        config = st.session_state.nn_config

        # SOLUCIÓN DEFINITIVA PARA EL ERROR DE TENSORFLOW
        st.info("🔄 Inicializando modelo para visualizaciones...")

        # Obtener datos de test
        X_test, _ = st.session_state.nn_test_data

        # ESTRATEGIA DEFINITIVA: FORZAR CONSTRUCCIÓN COMPLETA
        try:
            # Preparar datos de muestra
            sample_data = X_test[:1].astype(np.float32)

            # PASO 1: Forzar la construcción del modelo de forma agresiva
            st.info("🏗️ Forzando construcción del modelo...")

            # DIAGNÓSTICO INICIAL
            st.info(
                f"Estado inicial: built={getattr(model, 'built', False)}, input={'definido' if hasattr(model, 'input') and model.input is not None else 'None'}")

            # ESTRATEGIA ESPECÍFICA PARA EL PROBLEMA DETECTADO
            # Cuando model.built=True pero model.input=None
            construction_success = False

            for attempt in range(3):
                try:
                    st.info(f"Intento {attempt + 1}/3 de construcción...")

                    # ESTRATEGIA 1: Forzar predicción simple
                    _ = model.predict(sample_data, verbose=0)

                    # ESTRATEGIA 2: Si input sigue siendo None, forzar build con input_shape
                    if not hasattr(model, 'input') or model.input is None:
                        st.info("🔧 Aplicando fix específico para input=None...")
                        # Reconstruir completamente el modelo
                        model.build(input_shape=(None, X_test.shape[1]))

                        # Forzar que el modelo "vea" datos reales
                        _ = model(sample_data)  # Llamada directa

                    # ESTRATEGIA 3: Si aún no funciona, usar _set_inputs (método interno)
                    if not hasattr(model, 'input') or model.input is None:
                        st.info("🔧 Aplicando fix avanzado...")
                        # Método interno de TensorFlow para forzar input
                        try:
                            model._set_inputs(sample_data)
                        except:
                            # Si falla, intentar con fit en modo dummy usando la función de pérdida CORRECTA
                            try:
                                # Determinar la función de pérdida correcta según la arquitectura
                                output_size = model.layers[-1].units if hasattr(
                                    model.layers[-1], 'units') else 1

                                if output_size == 1:
                                    # Clasificación binaria o regresión
                                    if hasattr(model.layers[-1], 'activation') and 'sigmoid' in str(model.layers[-1].activation):
                                        loss_func = 'binary_crossentropy'
                                    else:
                                        loss_func = 'mse'  # Regresión
                                else:
                                    # Clasificación multiclase
                                    loss_func = 'sparse_categorical_crossentropy'

                                st.info(
                                    f"🔧 Usando {loss_func} para reparación...")
                                model.compile(optimizer='adam', loss=loss_func)

                                # Crear dummy target con el tamaño correcto
                                if loss_func == 'sparse_categorical_crossentropy':
                                    dummy_target = np.zeros(
                                        (1,), dtype=np.int32)  # Clase 0
                                elif loss_func == 'binary_crossentropy':
                                    dummy_target = np.zeros(
                                        (1, 1), dtype=np.float32)  # Probabilidad 0
                                else:  # MSE
                                    dummy_target = np.zeros(
                                        (1, output_size), dtype=np.float32)

                                model.fit(sample_data, dummy_target,
                                          epochs=1, verbose=0)
                            except Exception as fit_error:
                                st.warning(
                                    f"Fix avanzado falló: {str(fit_error)}")
                                pass

                    # VERIFICACIÓN CRÍTICA
                    if hasattr(model, 'input') and model.input is not None:
                        st.success(
                            f"✅ Construcción exitosa en intento {attempt + 1}")
                        construction_success = True
                        break
                    else:
                        st.warning(
                            f"⚠️ Intento {attempt + 1} falló: input sigue siendo None")

                except Exception as build_error:
                    st.warning(
                        f"⚠️ Intento {attempt + 1} falló: {str(build_error)}")
                    if attempt == 2:  # Último intento
                        # Último recurso: recrear el modelo completamente
                        st.info("🔥 ÚLTIMO RECURSO: Intentando recrear modelo...")
                        try:
                            # Obtener pesos del modelo actual
                            weights = model.get_weights()

                            # Recrear arquitectura desde config
                            config = st.session_state.nn_config

                            import tensorflow as tf
                            new_model = tf.keras.Sequential()
                            new_model.add(tf.keras.layers.Input(
                                shape=(config['input_size'],)))

                            # Recrear capas densas (excluyendo Input y Dropout)
                            layer_idx = 0
                            for i, layer in enumerate(model.layers):
                                if hasattr(layer, 'units'):  # Es una capa Dense
                                    if layer_idx == 0:  # Primera capa densa
                                        new_model.add(tf.keras.layers.Dense(
                                            layer.units,
                                            activation=layer.activation.__name__
                                        ))
                                    else:  # Capas siguientes
                                        new_model.add(tf.keras.layers.Dense(
                                            layer.units,
                                            activation=layer.activation.__name__
                                        ))
                                    layer_idx += 1
                                elif hasattr(layer, 'rate'):  # Es Dropout
                                    new_model.add(
                                        tf.keras.layers.Dropout(layer.rate))

                            # Compilar nuevo modelo con la configuración CORRECTA
                            if config['task_type'] == 'Clasificación':
                                if config.get('output_size', 1) == 1:
                                    loss_func = 'binary_crossentropy'
                                else:
                                    loss_func = 'sparse_categorical_crossentropy'
                            else:
                                loss_func = 'mse'

                            new_model.compile(optimizer='adam', loss=loss_func)

                            # Forzar construcción
                            _ = new_model.predict(sample_data, verbose=0)

                            # Copiar pesos si es posible
                            try:
                                new_model.set_weights(weights)
                                st.info("✅ Pesos copiados exitosamente")
                            except:
                                st.warning(
                                    "⚠️ No se pudieron copiar los pesos")

                            # Reemplazar modelo en session_state
                            st.session_state.nn_model = new_model
                            model = new_model

                            if model.input is not None:
                                st.success("🎉 Modelo recreado exitosamente!")
                                construction_success = True
                                break

                        except Exception as recreate_error:
                            st.error(
                                f"❌ Recreación falló: {str(recreate_error)}")

                            # ÚLTIMO ÚLTIMO RECURSO: Modelo completamente nuevo SIN copiar pesos
                            st.info(
                                "🚨 ÚLTIMO RECURSO EXTREMO: Modelo completamente nuevo...")
                            try:
                                config = st.session_state.nn_config

                                # Crear modelo minimalista garantizado que SIEMPRE funciona
                                minimal_model = tf.keras.Sequential([
                                    tf.keras.layers.Input(
                                        shape=(config['input_size'],)),
                                    tf.keras.layers.Dense(
                                        32, activation='relu'),
                                    tf.keras.layers.Dense(
                                        config['output_size'], activation=config['output_activation'])
                                ])

                                # Compilar con configuración correcta
                                if config['task_type'] == 'Clasificación':
                                    loss_func = 'sparse_categorical_crossentropy' if config[
                                        'output_size'] > 1 else 'binary_crossentropy'
                                else:
                                    loss_func = 'mse'

                                minimal_model.compile(
                                    optimizer='adam', loss=loss_func, metrics=['accuracy'])

                                # Forzar construcción INMEDIATA
                                dummy_input = np.zeros(
                                    (1, config['input_size']), dtype=np.float32)
                                _ = minimal_model.predict(
                                    dummy_input, verbose=0)

                                # VERIFICAR que funcione
                                if minimal_model.input is not None:
                                    st.warning(
                                        "⚠️ Modelo minimal creado (PERDISTE los pesos entrenados)")
                                    st.info(
                                        "💡 Este modelo te permitirá ver las visualizaciones, pero necesitarás reentrenar")
                                    st.session_state.nn_model = minimal_model
                                    model = minimal_model
                                    construction_success = True
                                    break
                                else:
                                    st.error(
                                        "🚨 IMPOSIBLE: Incluso el modelo minimal falló")

                            except Exception as minimal_error:
                                st.error(
                                    f"❌ Modelo minimal falló: {str(minimal_error)}")
                                st.error(
                                    "🚨 ERROR CRÍTICO: TensorFlow no funciona correctamente en este entorno")
                                pass

            if not construction_success:
                raise Exception(
                    "Fallo total en construcción después de todos los intentos")

            # PASO 2: Verificación EXHAUSTIVA que el modelo funcione
            # No solo verificar model.input, sino que REALMENTE funcione
            test_prediction = model.predict(sample_data, verbose=0)

            # Verificar que las capas están construidas
            layers_built = all(getattr(layer, 'built', True)
                               for layer in model.layers)

            if model.input is not None and layers_built and test_prediction is not None:
                st.success("✅ Modelo completamente construido y funcional")

                # PASO 3: Crear modelo de activaciones CON VERIFICACIÓN ROBUSTA
                if len(model.layers) > 2:  # Al menos Input + Hidden + Output
                    try:
                        # Identificar capas válidas para activaciones (excluir Input y Output)
                        intermediate_layers = []
                        for i, layer in enumerate(model.layers):
                            # Excluir la primera capa (Input) y la última (Output)
                            if i > 0 and i < len(model.layers) - 1:
                                if hasattr(layer, 'output') and layer.output is not None:
                                    intermediate_layers.append(layer.output)

                        if intermediate_layers:
                            # Crear modelo de activaciones
                            activation_model = tf.keras.Model(
                                inputs=model.input,
                                outputs=intermediate_layers
                            )

                            # VERIFICAR que el modelo de activaciones funcione
                            test_activations = activation_model.predict(
                                sample_data, verbose=0)

                            # Guardar solo si funciona
                            st.session_state.activation_model = activation_model
                            st.success(
                                f"✅ Modelo de activaciones creado ({len(intermediate_layers)} capas)")
                        else:
                            st.warning(
                                "⚠️ No hay capas intermedias válidas para análisis")
                            st.session_state.activation_model = None
                    except Exception as activation_error:
                        st.warning("⚠️ Error creando modelo de activaciones")
                        st.caption(f"Detalle: {str(activation_error)}")
                        st.session_state.activation_model = None
                else:
                    st.info("ℹ️ Red muy simple, análisis de capas limitado")
                    st.session_state.activation_model = None

            else:
                # Si llegamos aquí, hay un problema fundamental
                error_details = []
                if model.input is None:
                    error_details.append("model.input es None")
                if not layers_built:
                    error_details.append("capas no construidas")
                if test_prediction is None:
                    error_details.append("predicción falló")

                raise Exception(
                    f"Modelo no funcional: {', '.join(error_details)}")

        except Exception as error:
            st.error("❌ FALLO CRÍTICO: El modelo no se puede inicializar")
            st.markdown("### 🚨 Diagnóstico del Error")
            st.code(f"Error: {str(error)}")

            # Diagnóstico técnico detallado
            st.markdown("### 🔬 Estado Técnico del Modelo")
            try:
                st.write(f"- **Tipo de modelo**: {type(model).__name__}")
                st.write(
                    f"- **Modelo construido**: {getattr(model, 'built', 'Desconocido')}")
                st.write(
                    f"- **Input definido**: {model.input is not None if hasattr(model, 'input') else 'No disponible'}")
                st.write(
                    f"- **Número de capas**: {len(model.layers) if hasattr(model, 'layers') else 'Desconocido'}")

                if hasattr(model, 'layers'):
                    st.write("- **Estado de capas**:")
                    for i, layer in enumerate(model.layers):
                        built_status = getattr(layer, 'built', 'Desconocido')
                        st.write(
                            f"  - Capa {i+1} ({layer.__class__.__name__}): {built_status}")

            except Exception as diag_error:
                st.write(f"Error en diagnóstico: {diag_error}")

            st.markdown("### 💡 Solución Obligatoria")
            st.error(
                "**El modelo está corrupto o mal construido. DEBES reentrenarlo desde cero.**")
            st.markdown("""
            **Pasos para solucionarlo:**
            1. Ve a la pestaña **'Entrenamiento'**
            2. Reentrena el modelo completamente
            3. NO uses modelos guardados previamente
            4. Regresa a esta pestaña después del entrenamiento
            """)

            if st.button("🔙 Ir a Reentrenar Modelo", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 2
                st.rerun()

            return

        # CREAR PESTAÑAS DE VISUALIZACIÓN UNA VEZ QUE EL MODELO ESTÁ REPARADO
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Historial de Entrenamiento",
            "🧠 Pesos y Sesgos",
            "🎯 Superficie de Decisión",
            "📉 Análisis de Capas"
        ])

        with viz_tab1:
            st.subheader("📊 Historial de Entrenamiento Detallado")
            st.markdown("📈 **¿Cómo aprendió tu red neuronal?**")

            # Explicación sobre el historial
            with st.expander("💡 ¿Cómo interpretar estas gráficas?"):
                st.markdown("""
                **Gráfica de Pérdida (Loss):**
                - **Bajando**: La red está aprendiendo ✅
                - **Estable**: Ha convergido 🎯
                - **Subiendo**: Posible sobreajuste ⚠️
                - **Gap grande entre train/val**: Sobreajuste 🚨
                
                **Gráfica de Accuracy (clasificación) o MAE (regresión):**
                - **Subiendo**: Mejorando en las predicciones ✅
                - **Plateau**: Ha alcanzado su límite 📊
                - **Train > Val**: Normal, pero gap grande = sobreajuste ⚠️
                """)

            plot_training_history(history, config['task_type'])

            # Información adicional del entrenamiento
            st.markdown("#### 📊 Estadísticas del Entrenamiento")
            col1, col2, col3 = st.columns(3)

            with col1:
                final_loss = history.history['loss'][-1]
                initial_loss = history.history['loss'][0]
                improvement = ((initial_loss - final_loss) /
                               initial_loss) * 100
                st.metric("🔴 Pérdida Final (Entrenamiento)", f"{final_loss:.6f}",
                          f"-{improvement:.1f}% desde inicio")

            with col2:
                if 'val_loss' in history.history:
                    final_val_loss = history.history['val_loss'][-1]
                    overfitting_gap = final_val_loss - final_loss
                    st.metric("🟡 Pérdida Final (Validación)", f"{final_val_loss:.6f}",
                              f"Gap: {overfitting_gap:.6f}")

                    # Interpretación del gap
                    if overfitting_gap < 0.01:
                        st.success("✅ **Sin sobreajuste**: Gap muy pequeño")
                    elif overfitting_gap < 0.05:
                        st.warning("⚠️ **Sobreajuste leve**: Gap aceptable")
                    else:
                        st.error("🚨 **Sobreajuste**: Gap significativo")

            with col3:
                epochs_trained = len(history.history['loss'])
                st.metric("⏱️ Épocas Entrenadas", epochs_trained)

                # ¿Paró por early stopping?
                if 'nn_config' in st.session_state:
                    max_epochs = st.session_state.get('training_epochs', 100)
                    if epochs_trained < max_epochs:
                        st.caption(
                            "🛑 **Early Stopping**: Paró automáticamente")
                    else:
                        st.caption("🔄 **Completó todas las épocas**")

        with viz_tab2:
            st.subheader("🧠 Análisis de Pesos y Sesgos")
            st.markdown("🔍 **¿Qué ha aprendido cada neurona?**")

            # Explicación sobre pesos
            with st.expander("💡 ¿Qué significan los pesos?"):
                st.markdown("""
                **Pesos (Weights):**
                - **Valores altos**: Conexiones importantes entre neuronas
                - **Valores cercanos a 0**: Conexiones débiles o irrelevantes
                - **Valores negativos**: Relaciones inversas
                - **Distribución**: Indica si la red está bien inicializada
                
                **Sesgos (Biases):**
                - **Valores altos**: Neurona se activa fácilmente
                - **Valores bajos**: Neurona es más selectiva
                - **Distribución**: Debe ser razonable, no extrema
                """)

            # Obtener pesos de todas las capas
            layer_weights = []
            layer_biases = []

            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    weights = layer.get_weights()
                    if len(weights) >= 2:  # Pesos y sesgos
                        layer_weights.append(weights[0])
                        layer_biases.append(weights[1])

            if layer_weights:
                # Crear gráficos para cada capa
                for i, (weights, biases) in enumerate(zip(layer_weights, layer_biases)):
                    st.markdown(f"#### 📊 Capa {i+1}")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Histograma de pesos
                        fig_weights = go.Figure()
                        fig_weights.add_trace(go.Histogram(
                            x=weights.flatten(),
                            nbinsx=50,
                            name=f'Pesos Capa {i+1}',
                            opacity=0.7
                        ))
                        fig_weights.update_layout(
                            title=f'Distribución de Pesos - Capa {i+1}',
                            xaxis_title='Valor de Peso',
                            yaxis_title='Frecuencia',
                            height=300
                        )
                        st.plotly_chart(fig_weights, use_container_width=True)

                        # Estadísticas de pesos
                        st.caption(f"📊 **Estadísticas**: Media={np.mean(weights):.4f}, "
                                   f"Std={np.std(weights):.4f}, "
                                   f"Min={np.min(weights):.4f}, "
                                   f"Max={np.max(weights):.4f}")

                    with col2:
                        # Histograma de sesgos
                        fig_biases = go.Figure()
                        fig_biases.add_trace(go.Histogram(
                            x=biases.flatten(),
                            nbinsx=20,
                            name=f'Sesgos Capa {i+1}',
                            opacity=0.7,
                            marker_color='orange'
                        ))
                        fig_biases.update_layout(
                            title=f'Distribución de Sesgos - Capa {i+1}',
                            xaxis_title='Valor de Sesgo',
                            yaxis_title='Frecuencia',
                            height=300
                        )
                        st.plotly_chart(fig_biases, use_container_width=True)

                        # Estadísticas de sesgos
                        st.caption(f"📊 **Estadísticas**: Media={np.mean(biases):.4f}, "
                                   f"Std={np.std(biases):.4f}, "
                                   f"Min={np.min(biases):.4f}, "
                                   f"Max={np.max(biases):.4f}")

                # Análisis general
                st.markdown("#### 🔍 Análisis General de la Red")
                all_weights = np.concatenate(
                    [w.flatten() for w in layer_weights])
                all_biases = np.concatenate(
                    [b.flatten() for b in layer_biases])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Pesos Promedio",
                              f"{np.mean(all_weights):.6f}")
                with col2:
                    st.metric("📊 Desv. Std. Pesos",
                              f"{np.std(all_weights):.6f}")
                with col3:
                    dead_neurons = np.sum(np.abs(all_weights) < 1e-6)
                    st.metric("💀 Pesos ~0", f"{dead_neurons}")

                # Salud de la red
                if np.std(all_weights) < 0.01:
                    st.error(
                        "🚨 **Problema**: Pesos muy pequeños, la red puede no haber aprendido")
                elif np.std(all_weights) > 2:
                    st.warning(
                        "⚠️ **Atención**: Pesos muy grandes, posible inestabilidad")
                else:
                    st.success("✅ **Saludable**: Distribución de pesos normal")
            else:
                st.warning("No se pudieron extraer los pesos del modelo")

        with viz_tab3:
            st.subheader("🎯 Superficie de Decisión")
            st.markdown(
                "🗺️ **¿Cómo divide tu red el espacio de características?**")

            # Verificar si es clasificación para mostrar superficie de decisión
            if config.get('task_type', 'Clasificación') == 'Clasificación':
                # Si hay más de 2 características, permitir seleccionar 2
                if config['input_size'] > 2:
                    st.info(
                        "💡 Tu dataset tiene más de 2 características. Selecciona 2 para visualizar la superficie de decisión.")

                    # Obtener nombres de características
                    if 'nn_feature_names' in st.session_state:
                        feature_names = st.session_state.nn_feature_names
                    else:
                        feature_names = [
                            f'Característica {i+1}' for i in range(config['input_size'])]

                    st.markdown("### Selección de Características")
                    col1, col2 = st.columns(2)

                    with col1:
                        feature1 = st.selectbox(
                            "Primera característica:",
                            feature_names,
                            index=0,
                            key="viz_feature1_nn"
                        )

                    with col2:
                        feature2 = st.selectbox(
                            "Segunda característica:",
                            feature_names,
                            index=min(1, len(feature_names) - 1),
                            key="viz_feature2_nn"
                        )

                    if feature1 != feature2:
                        # Obtener datos de test para la visualización
                        X_test, y_test = st.session_state.nn_test_data

                        # Obtener índices de las características seleccionadas
                        feature_idx = [feature_names.index(
                            feature1), feature_names.index(feature2)]

                        # Extraer las características seleccionadas
                        X_2d = X_test[:, feature_idx]

                        # Generar superficie de decisión
                        try:
                            st.info("🎨 Generando superficie de decisión...")

                            # Crear malla de puntos para la superficie
                            h = 0.02  # tamaño del paso en la malla
                            x_min, x_max = X_2d[:, 0].min(
                            ) - 0.5, X_2d[:, 0].max() + 0.5
                            y_min, y_max = X_2d[:, 1].min(
                            ) - 0.5, X_2d[:, 1].max() + 0.5
                            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                                 np.arange(y_min, y_max, h))

                            # Para hacer predicciones en la malla, necesitamos crear puntos completos
                            # con valores promedio para las características no seleccionadas
                            X_full_test = X_test.copy()
                            mesh_points = []

                            for i in range(xx.ravel().shape[0]):
                                # Usar valores promedio
                                point = np.mean(X_full_test, axis=0)
                                # Primera característica seleccionada
                                point[feature_idx[0]] = xx.ravel()[i]
                                # Segunda característica seleccionada
                                point[feature_idx[1]] = yy.ravel()[i]
                                mesh_points.append(point)

                            mesh_points = np.array(mesh_points)

                            # Hacer predicciones en la malla
                            Z = model.predict(mesh_points, verbose=0)

                            # Si es clasificación multiclase, tomar la clase con mayor probabilidad
                            if len(Z.shape) > 1 and Z.shape[1] > 1:
                                Z = np.argmax(Z, axis=1)
                            else:
                                # Para clasificación binaria
                                Z = (Z > 0.5).astype(int).ravel()

                            Z = Z.reshape(xx.shape)

                            # Crear la visualización
                            fig, ax = plt.subplots(figsize=(10, 8))

                            # Dibujar la superficie de decisión
                            contourf = ax.contourf(
                                xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')

                            # Añadir los puntos de datos reales
                            if 'nn_class_names' in st.session_state and st.session_state.nn_class_names:
                                class_names = st.session_state.nn_class_names
                                # Mapear y_test a índices de clase si es necesario
                                if hasattr(y_test, 'shape') and len(y_test.shape) > 1:
                                    y_plot = np.argmax(y_test, axis=1)
                                else:
                                    y_plot = y_test

                                # Crear scatter plot por clase
                                unique_classes = np.unique(y_plot)
                                colors = plt.cm.Set1(
                                    np.linspace(0, 1, len(unique_classes)))

                                for i, class_idx in enumerate(unique_classes):
                                    mask = y_plot == class_idx
                                    class_name = class_names[class_idx] if class_idx < len(
                                        class_names) else f'Clase {class_idx}'
                                    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                                               c=[colors[i]], label=class_name,
                                               edgecolors='black', s=50, alpha=0.9)
                            else:
                                ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test,
                                           cmap='RdYlBu', edgecolors='black', s=50, alpha=0.9)

                            # Configurar etiquetas y título
                            ax.set_xlabel(feature1, fontsize=12)
                            ax.set_ylabel(feature2, fontsize=12)
                            ax.set_title(
                                f'Superficie de Decisión de Red Neuronal\n{feature1} vs {feature2}', fontsize=14)
                            ax.grid(True, alpha=0.3)

                            # Añadir leyenda si hay nombres de clase
                            if 'nn_class_names' in st.session_state and st.session_state.nn_class_names:
                                ax.legend(bbox_to_anchor=(
                                    1.05, 1), loc='upper left')

                            # Añadir colorbar para la superficie
                            plt.colorbar(contourf, ax=ax,
                                         label='Predicción de Clase')

                            plt.tight_layout()
                            st.pyplot(fig)

                            # Información adicional
                            st.success(
                                "✅ Superficie de decisión generada exitosamente")
                            st.info(f"""
                            🔍 **Información de la visualización:**
                            - **Características mostradas:** {feature1} vs {feature2}
                            - **Otras características:** Se mantienen en sus valores promedio
                            - **Colores de fondo:** Regiones de decisión de la red neuronal
                            - **Puntos:** Datos reales de prueba
                            - **Fronteras:** Límites donde la red cambia de decisión
                            """)

                            # Interpretación de la superficie
                            with st.expander("💡 ¿Cómo interpretar la superficie de decisión?"):
                                st.markdown("""
                                **Colores de fondo:**
                                - Cada color representa una clase diferente que predice la red
                                - Las transiciones suaves indican fronteras de decisión graduales
                                - Las transiciones bruscas indican fronteras más definidas
                                
                                **Puntos de datos:**
                                - Muestran dónde están ubicados los datos reales en este espacio 2D
                                - Puntos del mismo color deberían estar en regiones del mismo color de fondo
                                - Puntos en la región "incorrecta" indican errores de clasificación
                                
                                **Complejidad de las fronteras:**
                                - Fronteras muy complejas pueden indicar sobreajuste
                                - Fronteras muy simples pueden indicar subajuste
                                - Lo ideal son fronteras que capturen el patrón sin ser excesivamente complejas
                                """)

                        except Exception as e:
                            st.error(
                                f"❌ Error al generar la superficie de decisión: {str(e)}")
                            st.info(
                                "💡 Intenta con diferentes características o verifica que el modelo esté correctamente entrenado.")

                    else:
                        st.warning(
                            "⚠️ Por favor selecciona dos características diferentes.")

                else:
                    # Dataset con 2 o menos características - mostrar directamente
                    st.info("🎨 Generando superficie de decisión...")
                    st.markdown("""
                    **Superficie de Decisión 2D:**
                    - Cada color representa una clase predicha
                    - Los puntos son tus datos de entrenamiento
                    - Las fronteras muestran cómo la red separa las clases
                    - Fronteras suaves = red bien generalizada
                    - Fronteras muy complejas = posible sobreajuste
                    """)

                    # Aquí se podría implementar la visualización directa para datasets 2D
                    st.info(
                        "💡 Implementación completa para datasets 2D próximamente.")

            else:
                # Para tareas de regresión
                st.info("🏔️ **Superficie de Predicción para Regresión**")
                st.markdown("""
                Para tareas de regresión, se puede visualizar una superficie de predicción que muestra 
                cómo varían las predicciones numéricas en el espacio de características.
                """)

                if config['input_size'] > 2:
                    st.markdown(
                        "💡 Selecciona 2 características para visualizar la superficie de predicción.")
                    # Aquí se podría implementar similar lógica para regresión
                    st.info(
                        "🚧 Implementación de superficie de predicción para regresión próximamente.")
                else:
                    st.info(
                        "🚧 Implementación de superficie de predicción próximamente.")

        with viz_tab4:
            st.subheader("📉 Análisis de Capas")
            st.markdown("🔬 **Activaciones y patrones internos de la red**")

            # Explicación sobre activaciones
            with st.expander("💡 ¿Qué son las activaciones?"):
                st.markdown("""
                **Activaciones:**
                - **Valores que producen las neuronas** cuando procesan datos
                - **Primeras capas**: Detectan características básicas
                - **Capas intermedias**: Combinan características en patrones
                - **Última capa**: Decisión final o predicción
                
                **Qué buscar:**
                - **Muchos ceros**: Neuronas "muertas" (problema)
                - **Valores extremos**: Saturación (problema)
                - **Distribución balanceada**: Red saludable ✅
                """)

            # USAR EL MODELO DE ACTIVACIONES PRE-CREADO EN LA INICIALIZACIÓN
            try:
                # Obtener datos de test
                X_test, y_test = st.session_state.nn_test_data
                sample_size = min(100, len(X_test))
                X_sample = X_test[:sample_size]

                # Verificar que el modelo tiene suficientes capas
                if len(model.layers) <= 1:
                    st.warning(
                        "⚠️ El modelo tiene muy pocas capas para análisis detallado")
                    return

                # VERIFICAR SI HAY MODELO DE ACTIVACIONES PRE-CREADO
                activation_model = None

                # Método 1: Modelo creado durante el entrenamiento
                if hasattr(model, '_activation_model_ready'):
                    activation_model = model._activation_model_ready
                    st.success(
                        "✅ Usando modelo de activaciones preparado durante el entrenamiento")

                # Método 2: Modelo guardado en session_state
                elif 'activation_model' in st.session_state and st.session_state.activation_model is not None:
                    activation_model = st.session_state.activation_model
                    st.success(
                        "✅ Usando modelo de activaciones de session_state")

                # Método 3: Crear on-demand si no existe
                else:
                    st.info("🔧 Creando modelo de activaciones...")
                    try:
                        import tensorflow as tf
                        intermediate_layers = []
                        for i, layer in enumerate(model.layers):
                            # Excluir la primera capa (Input) y la última (Output)
                            if i > 0 and i < len(model.layers) - 1:
                                if hasattr(layer, 'output') and layer.output is not None:
                                    intermediate_layers.append(layer.output)

                        if intermediate_layers:
                            activation_model = tf.keras.Model(
                                inputs=model.input,
                                outputs=intermediate_layers
                            )
                            # Verificar que funcione
                            _ = activation_model.predict(
                                X_sample[:1], verbose=0)
                            st.session_state.activation_model = activation_model
                            st.success(
                                "✅ Modelo de activaciones creado exitosamente")
                        else:
                            st.warning(
                                "⚠️ No hay capas intermedias válidas para análisis")
                            return
                    except Exception as create_error:
                        st.error(
                            f"❌ Error creando modelo de activaciones: {str(create_error)}")
                        st.info(
                            "💡 El modelo necesita ser reentrenado para análisis de capas")
                        return

                # Si llegamos aquí, tenemos un modelo de activaciones válido
                if activation_model is None:
                    st.error("❌ No se pudo obtener modelo de activaciones")
                    return

                # Obtener activaciones usando el modelo pre-creado
                activations = activation_model.predict(X_sample, verbose=0)

                if not isinstance(activations, list):
                    activations = [activations]

                st.success(
                    f"✅ Análisis de {len(activations)} capas completado exitosamente")

                # Mostrar estadísticas por capa
                for i, activation in enumerate(activations):
                    st.markdown(f"#### 📊 Capa {i+1} - Activaciones")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔥 Media", f"{np.mean(activation):.4f}")
                    with col2:
                        st.metric("📊 Desv. Std.", f"{np.std(activation):.4f}")
                    with col3:
                        dead_ratio = np.mean(activation == 0) * 100
                        st.metric("💀 % Neuronas Muertas", f"{dead_ratio:.1f}%")
                    with col4:
                        saturated_ratio = np.mean(activation >= 0.99) * 100
                        st.metric("🔴 % Saturadas", f"{saturated_ratio:.1f}%")

                    # Interpretación de la salud
                    if dead_ratio > 50:
                        st.error(
                            f"🚨 **Problema en Capa {i+1}**: Muchas neuronas muertas")
                    elif dead_ratio > 20:
                        st.warning(
                            f"⚠️ **Atención en Capa {i+1}**: Algunas neuronas muertas")
                    else:
                        st.success(
                            f"✅ **Capa {i+1} Saludable**: Buena activación")

            except Exception as e:
                error_msg = str(e)
                if "never been called" in error_msg or "no defined input" in error_msg:
                    st.error(
                        "🚨 **Error de Inicialización del Modelo Sequential**")
                    st.markdown("""
                    **¿Qué significa este error?**
                    - El modelo Sequential de TensorFlow no ha sido completamente inicializado
                    - Las capas no conocen el tamaño de sus entradas
                    - Se necesita hacer al menos una predicción para construir el modelo
                    
                    **Solución Automática:**
                    La función incluye reparación automática que debería resolver esto.
                    Si persiste, usa el botón de 'Reparar Modelo' en la sección de errores abajo.
                    """)
                    st.info(
                        "💡 **Tip:** Este error es común con modelos Sequential recién cargados y tiene solución automática.")
                else:
                    st.error(f"❌ Error inesperado en el análisis: {str(e)}")
                    st.info(
                        "🔧 Esto puede indicar un problema con la arquitectura del modelo.")

                st.markdown(f"**Error técnico:** {error_msg}")

        # Botón para generar código de visualización
        st.markdown("### 💻 Código Python")
        if st.button("📝 Generar Código de Visualización", use_container_width=True):
            code = generate_neural_network_visualization_code(config)
            st.markdown("#### 🐍 Código Python para Visualizaciones")
            st.code(code, language='python')

            st.download_button(
                label="💾 Descargar Código de Visualización",
                data=code,
                file_name="visualizaciones_red_neuronal.py",
                mime="text/plain"
            )

        # Navegación
        st.markdown("---")
        st.markdown("### 🧭 Navegación")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("🔙 Volver a Evaluación", use_container_width=True):
                st.session_state.active_tab_nn = 3
                st.rerun()
        with col_nav2:
            if st.button("🔮 Hacer Predicciones", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 5
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error en las visualizaciones: {str(e)}")

        # Diagnóstico detallado del error
        error_type = type(e).__name__
        error_msg = str(e)

        st.markdown("### 🔍 Diagnóstico del Error")

        if "never been called" in error_msg or "no defined input" in error_msg:
            st.error("🚨 **Problema de Inicialización del Modelo**")
            st.markdown("""
            **Causa del problema:**
            - El modelo Sequential no ha sido completamente inicializado
            - Las capas no tienen sus formas de entrada definidas
            - Se necesita hacer al menos una predicción para construir el modelo
            """)

            # Botón de reparación automática
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔧 Reparar Modelo Automáticamente", type="primary", key="auto_repair"):
                    try:
                        st.info("🔄 Iniciando reparación exhaustiva del modelo...")

                        # Obtener datos de test
                        if 'nn_test_data' in st.session_state:
                            X_test, y_test = st.session_state.nn_test_data

                            # Forzar construcción del modelo con múltiples estrategias MEJORADAS
                            with st.spinner("Aplicando estrategias de reparación..."):
                                progress_bar = st.progress(0)

                                # Estrategia 1: Predicción simple
                                progress_bar.progress(20)
                                _ = model.predict(X_test[:1], verbose=0)

                                # Estrategia 2: Llamada directa al modelo
                                progress_bar.progress(40)
                                _ = model(X_test[:1])

                                # Estrategia 3: Predicción con batch más grande
                                progress_bar.progress(60)
                                batch_size = min(10, len(X_test))
                                _ = model.predict(
                                    X_test[:batch_size], verbose=0)

                                # Estrategia 4: Compilar explícitamente si es necesario
                                progress_bar.progress(80)
                                if not model.built:
                                    model.build(input_shape=(
                                        None, X_test.shape[1]))

                                # Estrategia 5: Verificar input tensor
                                if model.input is None:
                                    model._set_inputs(X_test[:1])

                                # Estrategia 6: Forzar todas las capas
                                for i, layer in enumerate(model.layers):
                                    if hasattr(layer, 'built') and not layer.built:
                                        if i == 0:
                                            layer.build(input_shape=(
                                                None, X_test.shape[1]))
                                        else:
                                            prev_output = model.layers[i -
                                                                       1].output_shape
                                            layer.build(prev_output)

                                progress_bar.progress(100)

                            # Verificación EXHAUSTIVA que el modelo esté funcionando
                            st.info("✅ Verificando reparación...")

                            # Test múltiples operaciones
                            test_pred = model.predict(X_test[:5], verbose=0)
                            _ = model.get_weights()

                            # Test crítico: modelo de activaciones
                            if len(model.layers) > 1:
                                layer_outputs = [
                                    layer.output for layer in model.layers[:-1]]
                                test_activation_model = tf.keras.Model(
                                    inputs=model.input, outputs=layer_outputs)
                                _ = test_activation_model.predict(
                                    X_test[:1], verbose=0)

                            st.success("🎉 ¡Modelo reparado exitosamente!")
                            st.info(
                                "✅ El modelo está completamente inicializado y listo para todas las visualizaciones.")

                            # Botón para recargar visualizaciones
                            if st.button("🔄 Recargar Visualizaciones", type="primary"):
                                st.rerun()

                        else:
                            st.error(
                                "❌ No se encontraron datos de test para reparar el modelo")

                    except Exception as repair_error:
                        st.error(
                            f"❌ Error durante la reparación: {repair_error}")

                        # Diagnóstico específico del error
                        if "never been called" in str(repair_error):
                            st.warning(
                                "🔧 **Error persistente de inicialización**")
                            st.markdown("""
                            **Estrategias adicionales:**
                            1. Reinicia la aplicación completamente
                            2. Reentrena el modelo desde cero  
                            3. Verifica que TensorFlow esté actualizado
                            4. Prueba con un dataset más pequeño
                            """)
                        else:
                            st.info("💡 Intenta reentrenar el modelo desde cero.")

            with col2:
                st.markdown("**💡 Solución manual:**")
                st.markdown("""
                1. Ve a la pestaña **'Entrenamiento'**
                2. Reentrena el modelo desde cero
                3. Regresa a esta pestaña
                4. Las visualizaciones deberían funcionar
                """)

                if st.button("🔙 Ir a Entrenamiento", key="go_training"):
                    st.session_state.active_tab_nn = 2
                    st.rerun()

        else:
            # Otros tipos de errores
            st.warning("⚠️ **Error Inesperado**")
            st.code(f"Tipo: {error_type}\nMensaje: {error_msg}")

            st.markdown("""
            **Posibles soluciones:**
            - Verifica que TensorFlow esté instalado correctamente
            - Asegúrate de que el modelo esté entrenado
            - Intenta reentrenar el modelo
            - Reinicia la aplicación si persiste el problema
            """)

        # Información técnica adicional
        with st.expander("🔬 Información Técnica Detallada"):
            try:
                st.write("**Estado del Modelo:**")
                st.write(f"- Tipo: {type(model).__name__}")
                st.write(
                    f"- Construido: {getattr(model, 'built', 'No disponible')}")
                st.write(
                    f"- Número de capas: {len(model.layers) if hasattr(model, 'layers') else 'No disponible'}")

                if hasattr(model, 'input'):
                    st.write(
                        f"- Input definido: {'✅' if model.input is not None else '❌'}")

                if hasattr(model, 'layers'):
                    st.write("**Estado de las Capas:**")
                    for i, layer in enumerate(model.layers):
                        layer_built = getattr(layer, 'built', False)
                        st.write(
                            f"  - Capa {i+1} ({layer.__class__.__name__}): {'✅' if layer_built else '❌'}")

            except Exception as debug_error:
                st.write(f"Error obteniendo información: {debug_error}")

        st.info("💡 **Tip**: Este error es común con modelos Sequential. La reparación automática debería resolverlo.")


def show_neural_network_predictions():
    """Interfaz para hacer predicciones con el modelo entrenado."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    try:
        import numpy as np
        import pandas as pd
        import scipy.stats

        model = st.session_state.nn_model
        scaler = st.session_state.nn_scaler
        label_encoder = st.session_state.nn_label_encoder
        config = st.session_state.nn_config

        if 'nn_df' not in st.session_state or 'nn_target_col' not in st.session_state:
            st.error("No hay datos disponibles para hacer predicciones.")
            return

        df = st.session_state.nn_df
        target_col = st.session_state.nn_target_col
        feature_cols = [col for col in df.columns if col != target_col]

        st.header("🎯 Hacer Predicciones")

        # Tabs para diferentes tipos de predicción
        pred_tab1, pred_tab2, pred_tab3 = st.tabs([
            "🔍 Predicción Individual",
            "📊 Predicción por Lotes",
            "🎲 Exploración Interactiva"
        ])

        with pred_tab1:
            st.subheader("🔍 Predicción Individual")
            st.markdown("Introduce los valores para cada característica:")

            # Crear inputs para cada característica
            input_values = {}

            # Organizar en columnas
            num_cols = min(3, len(feature_cols))
            cols = st.columns(num_cols)

            for i, feature in enumerate(feature_cols):
                col_idx = i % num_cols

                with cols[col_idx]:
                    # Obtener estadísticas de la característica
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())

                    input_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100,
                        key=f"nn_pred_{feature}"
                    )

                    st.caption(
                        f"Min: {min_val:.2f}, Max: {max_val:.2f}, Media: {mean_val:.2f}")

            # Botón para hacer predicción
            if st.button("🚀 Hacer Predicción", type="primary"):
                # Preparar datos para predicción
                input_array = np.array(
                    [[input_values[feature] for feature in feature_cols]])
                input_scaled = scaler.transform(input_array)

                # Hacer predicción
                prediction = model.predict(input_scaled, verbose=0)

                # Mostrar resultados
                st.success("✅ Predicción completada")

                if config['task_type'] == 'Clasificación':
                    output_size = safe_get_output_size(config)
                    if output_size > 2:  # Multiclase
                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class_idx]

                        if label_encoder:
                            predicted_class = label_encoder.inverse_transform(
                                [predicted_class_idx])[0]
                        else:
                            predicted_class = f"Clase {predicted_class_idx}"

                        # Mostrar resultado principal
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("🎯 Clase Predicha", predicted_class)

                        with col2:
                            st.metric(
                                "🎲 Confianza", f"{confidence:.4f}", f"{confidence*100:.2f}%")

                        # Mostrar probabilidades para todas las clases
                        st.subheader("📊 Probabilidades por Clase")

                        prob_data = []
                        for i, prob in enumerate(prediction[0]):
                            if label_encoder:
                                class_name = label_encoder.inverse_transform([i])[
                                    0]
                            else:
                                class_name = f"Clase {i}"

                            prob_data.append({
                                'Clase': class_name,
                                'Probabilidad': f"{prob:.4f}",
                                'Porcentaje': f"{prob*100:.2f}%"
                            })

                        st.dataframe(prob_data, use_container_width=True)

                        # Gráfico de barras de probabilidades
                        import plotly.graph_objects as go

                        class_names = [item['Clase'] for item in prob_data]
                        probabilities = [float(item['Probabilidad'])
                                         for item in prob_data]

                        fig = go.Figure(data=[
                            go.Bar(x=class_names, y=probabilities,
                                   marker_color=['red' if i == predicted_class_idx else 'lightblue'
                                                 for i in range(len(class_names))])
                        ])

                        fig.update_layout(
                            title="Distribución de Probabilidades",
                            xaxis_title="Clases",
                            yaxis_title="Probabilidad",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:  # Binaria
                        probability = prediction[0][0]
                        predicted_class_idx = 1 if probability > 0.5 else 0

                        if label_encoder:
                            predicted_class = label_encoder.inverse_transform(
                                [predicted_class_idx])[0]
                        else:
                            predicted_class = f"Clase {predicted_class_idx}"

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("🎯 Clase Predicha", predicted_class)

                        with col2:
                            st.metric("🎲 Probabilidad", f"{probability:.4f}")

                        with col3:
                            confidence = max(probability, 1 - probability)
                            st.metric(
                                "✨ Confianza", f"{confidence:.4f}", f"{confidence*100:.2f}%")

                else:  # Regresión
                    predicted_value = prediction[0][0]

                    st.metric("🎯 Valor Predicho", f"{predicted_value:.6f}")

                    # Información adicional para regresión
                    target_stats = df[target_col].describe()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.info(f"📊 **Estadísticas del Target:**\n"
                                f"- Media: {target_stats['mean']:.4f}\n"
                                f"- Mediana: {target_stats['50%']:.4f}")

                    with col2:
                        st.info(f"📏 **Rango de Datos:**\n"
                                f"- Mínimo: {target_stats['min']:.4f}\n"
                                f"- Máximo: {target_stats['max']:.4f}")

                    with col3:
                        deviation_from_mean = abs(
                            predicted_value - target_stats['mean'])
                        st.info(f"🎯 **Análisis:**\n"
                                f"- Desviación de la media: {deviation_from_mean:.4f}\n"
                                f"- Percentil aproximado: {scipy.stats.percentileofscore(df[target_col], predicted_value):.1f}%")

        with pred_tab2:
            st.subheader("📊 Predicción por Lotes")

            st.markdown(
                "Sube un archivo CSV con nuevos datos para hacer predicciones en lote:")

            uploaded_file = st.file_uploader(
                "Selecciona archivo CSV",
                type=['csv'],
                key="nn_batch_predictions"
            )

            if uploaded_file is not None:
                try:
                    # Cargar datos
                    new_df = pd.read_csv(uploaded_file)

                    st.success(
                        f"✅ Archivo cargado: {new_df.shape[0]} filas, {new_df.shape[1]} columnas")

                    # Verificar que las columnas coincidan
                    missing_features = set(feature_cols) - set(new_df.columns)
                    extra_features = set(new_df.columns) - set(feature_cols)

                    if missing_features:
                        st.error(
                            f"❌ Faltan características: {', '.join(missing_features)}")
                    elif extra_features:
                        st.warning(
                            f"⚠️ Características adicionales (serán ignoradas): {', '.join(extra_features)}")
                        # Seleccionar solo las características necesarias
                        new_df = new_df[feature_cols]

                    if not missing_features:
                        # Mostrar vista previa
                        st.dataframe(new_df.head(), use_container_width=True)

                        if st.button("🚀 Generar Predicciones", type="primary"):
                            # Procesar datos
                            new_data_scaled = scaler.transform(new_df)

                            # Hacer predicciones
                            batch_predictions = model.predict(
                                new_data_scaled, verbose=0)

                            # Procesar resultados según el tipo de tarea
                            if config['task_type'] == 'Clasificación':
                                output_size = safe_get_output_size(config)
                                if output_size > 2:
                                    predicted_classes_idx = np.argmax(
                                        batch_predictions, axis=1)
                                    confidences = np.max(
                                        batch_predictions, axis=1)

                                    if label_encoder:
                                        predicted_classes = label_encoder.inverse_transform(
                                            predicted_classes_idx)
                                    else:
                                        predicted_classes = [
                                            f"Clase {idx}" for idx in predicted_classes_idx]

                                    results_df = new_df.copy()
                                    results_df['Predicción'] = predicted_classes
                                    results_df['Confianza'] = confidences

                                else:  # Binaria
                                    probabilities = batch_predictions.flatten()
                                    predicted_classes_idx = (
                                        probabilities > 0.5).astype(int)
                                    confidences = np.maximum(
                                        probabilities, 1 - probabilities)

                                    if label_encoder:
                                        predicted_classes = label_encoder.inverse_transform(
                                            predicted_classes_idx)
                                    else:
                                        predicted_classes = [
                                            f"Clase {idx}" for idx in predicted_classes_idx]

                                    results_df = new_df.copy()
                                    results_df['Predicción'] = predicted_classes
                                    results_df['Probabilidad'] = probabilities
                                    results_df['Confianza'] = confidences

                            else:  # Regresión
                                predicted_values = batch_predictions.flatten()

                                results_df = new_df.copy()
                                results_df['Predicción'] = predicted_values

                            # Mostrar resultados
                            st.success(
                                f"✅ Predicciones generadas para {len(results_df)} muestras")
                            st.dataframe(results_df, use_container_width=True)

                            # Botón para descargar resultados
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Resultados",
                                data=csv_results,
                                file_name="predicciones_neural_network.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Error procesando archivo: {str(e)}")

            else:
                # Mostrar formato esperado
                st.info("📋 **Formato esperado del archivo CSV:**")

                sample_data = df[feature_cols].head(3)
                st.dataframe(sample_data, use_container_width=True)

                st.markdown(
                    "El archivo debe contener las siguientes columnas:")
                st.code(", ".join(feature_cols))

        with pred_tab3:
            st.subheader("🎲 Exploración Interactiva")

            # Información educativa sobre la exploración interactiva
            with st.expander("ℹ️ ¿Qué es la Exploración Interactiva?", expanded=False):
                st.markdown("""
                **La exploración interactiva** te permite entender cómo el modelo neural toma decisiones:
                
                🔍 **¿Para qué sirve?**
                - Ver cómo cada característica influye en las predicciones
                - Identificar patrones y comportamientos del modelo
                - Detectar posibles sesgos o comportamientos inesperados
                - Comprender la sensibilidad del modelo a cambios en los datos
                
                📊 **¿Cómo interpretar los resultados?**
                - **Líneas ascendentes**: La característica tiene correlación positiva
                - **Líneas descendentes**: La característica tiene correlación negativa  
                - **Líneas planas**: La característica tiene poco impacto
                - **Cambios abruptos**: Puntos de decisión críticos del modelo
                
                💡 **Consejos de uso:**
                - Prueba diferentes muestras base para ver patrones generales
                - Observa qué características causan mayores cambios
                - Busca comportamientos inesperados o poco realistas
                """)

            st.markdown(
                "🎯 **Explora cómo cambian las predicciones al modificar diferentes características:**")

            # Seleccionar una muestra base
            st.markdown("**1. 📍 Selecciona una muestra base:**")

            st.info("💡 **Tip:** La muestra base es tu punto de referencia. Todas las exploraciones mostrarán cómo cambian las predicciones desde este punto inicial.")

            sample_idx = st.selectbox(
                "Índice de muestra:",
                range(len(df)),
                format_func=lambda x: f"Muestra {x}",
                key="nn_interactive_sample"
            )

            base_sample = df.iloc[sample_idx][feature_cols].to_dict()

            # Mostrar valores base
            st.markdown("**2. 📋 Valores base de la muestra:**")
            st.caption(
                "Estos son los valores de todas las características para la muestra seleccionada:")
            base_df = pd.DataFrame([base_sample])
            st.dataframe(base_df, use_container_width=True)

            # Hacer predicción base
            base_array = np.array([[base_sample[feature]
                                  for feature in feature_cols]])
            base_scaled = scaler.transform(base_array)
            base_prediction = model.predict(base_scaled, verbose=0)

            if config['task_type'] == 'Clasificación':
                output_size = safe_get_output_size(config)
                if output_size > 2:
                    base_class_idx = np.argmax(base_prediction[0])
                    base_confidence = base_prediction[0][base_class_idx]

                    if label_encoder:
                        base_class = label_encoder.inverse_transform([base_class_idx])[
                            0]
                    else:
                        base_class = f"Clase {base_class_idx}"

                    st.info(
                        f"🎯 **Predicción Base:** {base_class} (Confianza: {base_confidence:.3f})")
                else:
                    base_prob = base_prediction[0][0]
                    base_class_idx = 1 if base_prob > 0.5 else 0

                    if label_encoder:
                        base_class = label_encoder.inverse_transform([base_class_idx])[
                            0]
                    else:
                        base_class = f"Clase {base_class_idx}"

                    st.info(
                        f"🎯 **Predicción Base:** {base_class} (Probabilidad: {base_prob:.3f})")
            else:
                base_value = base_prediction[0][0]
                st.info(f"🎯 **Predicción Base:** {base_value:.6f}")

            # Seleccionar característica para explorar
            st.markdown("**3. 🔍 Explora el efecto de una característica:**")

            st.info("🎯 **Objetivo:** Verás cómo cambia la predicción cuando modificas solo UNA característica, manteniendo todas las demás constantes. Esto te ayuda a entender la importancia relativa de cada variable.")

            feature_to_explore = st.selectbox(
                "Característica a explorar:",
                feature_cols,
                key="nn_explore_feature",
                help="Selecciona la característica cuyo efecto quieres analizar en las predicciones"
            )

            # Crear rango de valores para la característica seleccionada
            min_val = float(df[feature_to_explore].min())
            max_val = float(df[feature_to_explore].max())

            # Generar valores para exploración
            exploration_values = np.linspace(min_val, max_val, 50)
            exploration_predictions = []

            for val in exploration_values:
                # Crear muestra modificada
                modified_sample = base_sample.copy()
                modified_sample[feature_to_explore] = val

                # Hacer predicción
                modified_array = np.array(
                    [[modified_sample[feature] for feature in feature_cols]])
                modified_scaled = scaler.transform(modified_array)
                pred = model.predict(modified_scaled, verbose=0)

                if config['task_type'] == 'Clasificación':
                    output_size = safe_get_output_size(config)
                    if output_size > 2:
                        pred_class_idx = np.argmax(pred[0])
                        confidence = pred[0][pred_class_idx]
                        exploration_predictions.append(
                            (pred_class_idx, confidence))
                    else:
                        prob = pred[0][0]
                        exploration_predictions.append(prob)
                else:
                    exploration_predictions.append(pred[0][0])

            # Crear visualización
            import plotly.graph_objects as go

            fig = go.Figure()

            if config['task_type'] == 'Clasificación':
                output_size = safe_get_output_size(config)
                if output_size > 2:
                    # Multiclase: mostrar clase predicha y confianza
                    classes = [pred[0] for pred in exploration_predictions]
                    confidences = [pred[1] for pred in exploration_predictions]

                    fig.add_trace(go.Scatter(
                        x=exploration_values,
                        y=classes,
                        mode='lines+markers',
                        name='Clase Predicha',
                        yaxis='y1'
                    ))

                    fig.add_trace(go.Scatter(
                        x=exploration_values,
                        y=confidences,
                        mode='lines+markers',
                        name='Confianza',
                        yaxis='y2',
                        line=dict(color='red')
                    ))

                    fig.update_layout(
                        title=f'Efecto de {feature_to_explore} en la Predicción',
                        xaxis_title=feature_to_explore,
                        yaxis=dict(title='Clase Predicha', side='left'),
                        yaxis2=dict(title='Confianza',
                                    side='right', overlaying='y'),
                        height=500
                    )
                else:
                    # Binaria: mostrar probabilidad
                    fig.add_trace(go.Scatter(
                        x=exploration_values,
                        y=exploration_predictions,
                        mode='lines+markers',
                        name='Probabilidad'
                    ))

                    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                                  annotation_text="Umbral de decisión")

                    fig.update_layout(
                        title=f'Efecto de {feature_to_explore} en la Probabilidad',
                        xaxis_title=feature_to_explore,
                        yaxis_title='Probabilidad',
                        height=500
                    )
            else:
                # Regresión
                fig.add_trace(go.Scatter(
                    x=exploration_values,
                    y=exploration_predictions,
                    mode='lines+markers',
                    name='Predicción'
                ))

                fig.update_layout(
                    title=f'Efecto de {feature_to_explore} en la Predicción',
                    xaxis_title=feature_to_explore,
                    yaxis_title='Valor Predicho',
                    height=500
                )

            # Marcar el valor base
            base_val = base_sample[feature_to_explore]
            fig.add_vline(x=base_val, line_dash="dash", line_color="green",
                          annotation_text="Valor Base")

            st.plotly_chart(fig, use_container_width=True)

            # Análisis interpretativo
            st.markdown("**📈 Análisis de Resultados:**")

            # Calcular estadísticas del efecto
            if config['task_type'] == 'Clasificación':
                output_size = safe_get_output_size(config)
                if output_size <= 2:
                    pred_range = max(exploration_predictions) - \
                        min(exploration_predictions)
                    volatility = np.std(exploration_predictions)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rango de Probabilidades", f"{pred_range:.3f}",
                                  help="Diferencia entre la probabilidad máxima y mínima observada")
                    with col2:
                        st.metric("Volatilidad", f"{volatility:.3f}",
                                  help="Qué tan variables son las predicciones (desviación estándar)")

                    if pred_range > 0.3:
                        st.success(
                            f"🎯 **Característica muy influyente:** '{feature_to_explore}' tiene un gran impacto en las predicciones")
                    elif pred_range > 0.1:
                        st.warning(
                            f"📊 **Característica moderadamente influyente:** '{feature_to_explore}' tiene un impacto moderado")
                    else:
                        st.info(
                            f"📉 **Característica poco influyente:** '{feature_to_explore}' tiene poco impacto en las predicciones")
            else:
                pred_range = max(exploration_predictions) - \
                    min(exploration_predictions)
                pred_mean = np.mean(exploration_predictions)
                relative_impact = (pred_range / abs(pred_mean)
                                   ) * 100 if pred_mean != 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rango de Predicciones", f"{pred_range:.6f}")
                with col2:
                    st.metric("Impacto Relativo", f"{relative_impact:.1f}%")

                if relative_impact > 20:
                    st.success(
                        f"🎯 **Característica muy influyente:** '{feature_to_explore}' causa cambios significativos")
                elif relative_impact > 5:
                    st.warning(
                        f"📊 **Característica moderadamente influyente:** '{feature_to_explore}' tiene impacto moderado")
                else:
                    st.info(
                        f"📉 **Característica poco influyente:** '{feature_to_explore}' tiene poco impacto")

            # Consejos interpretativos
            with st.expander("💡 Consejos para Interpretar los Resultados", expanded=False):
                st.markdown(f"""
                **🔍 Analizando '{feature_to_explore}':**
                
                ✅ **Buenas señales:**
                - Cambios graduales y suaves en las predicciones
                - Comportamiento consistente con el conocimiento del dominio
                - Relaciones monotónicas (siempre creciente o decreciente)
                
                ⚠️ **Señales de alerta:**
                - Cambios muy abruptos sin explicación lógica
                - Comportamientos contradictorios al conocimiento experto
                - Excesiva sensibilidad a pequeños cambios
                
                **🎯 Próximos pasos:**
                1. Prueba con diferentes muestras base para confirmar patrones
                2. Explora otras características para comparar importancias
                3. Si encuentras comportamientos extraños, considera reentrenar el modelo
                4. Documenta los insights para mejorar futuras versiones del modelo
                """)

    except Exception as e:
        st.error(f"Error en las predicciones: {str(e)}")
        st.info("Asegúrate de que el modelo esté entrenado correctamente.")


def safe_get_output_size(config):
    """
    Extrae el tamaño de salida de forma segura para evitar errores de comparación de arrays.
    """
    try:
        output_size = config['output_size']
        # Si es un array o lista, tomar el primer elemento
        if hasattr(output_size, '__len__') and not isinstance(output_size, (str, bytes)):
            return int(output_size[0]) if len(output_size) > 0 else 1
        # Si es un escalar
        return int(output_size)
    except:
        return 1
