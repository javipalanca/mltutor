import streamlit as st
import streamlit.components.v1 as components

import tensorflow as tf
import numpy as np


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


def initialize_model_safely():
    """Inicializa el modelo de forma robusta y completa."""
    try:
        model = st.session_state.nn_model
        config = st.session_state.nn_config
        X_test, _ = st.session_state.nn_test_data
        sample_data = X_test[:1].astype(np.float32)

        # Método 1: Hacer una predicción simple para inicializar
        _ = model.predict(sample_data, verbose=0)

        # Método 2: Verificar y forzar construcción del modelo
        if not hasattr(model, 'input') or model.input is None:
            model.build(input_shape=(None, X_test.shape[1]))

        # Método 3: Llamar al modelo directamente (TensorFlow 2.x)
        if hasattr(model, 'call') and not hasattr(model, 'input'):
            _ = model(sample_data)

        # Método 4: Compilar si no está compilado
        if not hasattr(model, 'optimizer') or model.optimizer is None:
            config = st.session_state.nn_config
            model.compile(
                optimizer=config.get('optimizer', 'adam'),
            )

        # Verificación final
        if hasattr(model, 'input') and model.input is not None:
            return True, "✅ Modelo inicializado correctamente"
        else:
            return False, "❌ Error de inicialización: No se pudo acceder a model.input"

    except Exception as e:
        return False, f"❌ Error de inicialización: {str(e)}"


def show_training_history_tab():
    """Muestra el historial de entrenamiento."""
    st.subheader("📊 Historial de Entrenamiento")

   # SECCIÓN 1: INTERPRETACIÓN EDUCATIVA EXPANDIDA
    with st.expander("💡 Interpretación del Entrenamiento", expanded=False):
        st.markdown("""
        📚 **Guía para Interpretar los Gráficos:**
        
        **🔴 Pérdida (Loss):**
        - **Bajando:** ✅ El modelo está aprendiendo correctamente
        - **Estable:** 🎯 El modelo ha convergido (terminó de aprender)
        - **Subiendo:** 🚨 Posible sobreajuste o learning rate muy alto
        
        **🟡 Validación vs Entrenamiento:**
        - **Líneas cercanas:** ✅ Generalización saludable
        - **Gap creciente:** 🚨 Sobreajuste (memorización vs aprendizaje)
        - **Validación mejor que entrenamiento:** 🤔 Posible error en datos
        
        **📈 Accuracy/Métricas:**
        - **Crecimiento sostenido:** ✅ Aprendizaje progresivo
        - **Plateau:** 🎯 Límite del modelo alcanzado
        - **Fluctuaciones grandes:** ⚠️ Batch size muy pequeño o datos ruidosos
        
        **🎯 Señales de Calidad:**
        - ✅ Loss decreciente y suave
        - ✅ Gap train/val menor al 10%
        - ✅ Métricas estables al final
        """)

    history = st.session_state.nn_history
    config = st.session_state.nn_config

    plot_training_history(history, config['task_type'])

    # Estadísticas del entrenamiento
    col1, col2, col3, col4 = st.columns(4)
    final_loss = history.history['loss'][-1]
    initial_loss = history.history['loss'][0]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100

    # Calcular estadísticas robustas
    loss_values = history.history['loss']
    epochs_total = len(loss_values)

    # Detectar convergencia (últimas 5 épocas)
    convergence_window = min(5, epochs_total // 2)
    recent_losses = loss_values[-convergence_window:]
    loss_stability = np.std(recent_losses) / np.mean(recent_losses) * 100

    with col1:
        st.metric("🔴 Pérdida Final",
                  f"{final_loss:.6f}",
                  f"-{improvement:.1f}%",
                  help="Pérdida en la última época vs primera época")
    with col2:
        if 'val_loss' in history.history:
            final_val_loss = history.history['val_loss'][-1]
            gap = final_val_loss - final_loss
            gap_percentage = (gap / final_loss) * 100

            # Color del delta basado en el gap
            delta_color = "normal" if abs(gap_percentage) < 10 else "inverse"

            st.metric("🟡 Pérdida Validación",
                      f"{final_val_loss:.6f}",
                      f"Gap: {gap:.6f} ({gap_percentage:+.1f}%)",
                      delta_color=delta_color,
                      help="Diferencia entre validación y entrenamiento indica sobreajuste")

    with col3:
        st.metric("⏱️ Épocas", epochs_total,
                  help="Número total de épocas de entrenamiento")
    with col4:
        # Indicador de estabilidad
        if loss_stability < 1:
            stability_emoji = "🎯"
            stability_text = "Estable"
        elif loss_stability < 5:
            stability_emoji = "📊"
            stability_text = "Moderado"
        else:
            stability_emoji = "📈"
            stability_text = "Variable"

        st.metric(f"{stability_emoji} Estabilidad",
                  stability_text,
                  f"{loss_stability:.1f}% CV",
                  help="Variabilidad en las últimas épocas (menor = más estable)")


def show_weights_analysis_tab():
    """Muestra el análisis de pesos y sesgos."""
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    st.subheader("🧠 Análisis de Pesos y Sesgos")

    with st.expander("💡 Cómo interpretar esta pestaña", expanded=False):
        st.markdown("""
        Esta sección te ayuda a entender **qué está aprendiendo** la red:

        • **Histograma de Pesos:** Muestra la distribución de las conexiones entre neuronas.
          - Distribución centrada en 0 y con forma aproximadamente normal suele indicar un entrenamiento saludable.
          - Pesos extremadamente grandes pueden provocar inestabilidad o sobreajuste.
          - Pesos casi todos muy cercanos a 0 pueden indicar que la red no aprendió (underfitting) o exceso de regularización.

        • **Histograma de Sesgos:** Indica desplazamientos aprendidos por cada neurona.
          - Sesgos suelen estar más concentrados; valores extremos pueden saturar activaciones.

        • **Líneas verticales:**
          - Línea gris: valor 0 (referencia neutra)
          - Línea verde discontinua: media actual

        • **% Casi Cero:** Conexiones cuyo valor absoluto < 1e-3. Un porcentaje moderado (<60%) es normal; demasiado alto puede indicar red infraentrenada.
        """)

    with st.expander("⚙️ Opciones de visualización", expanded=False):
        colc1, colc2, colc3 = st.columns(3)
        with colc1:
            bins_w = st.slider("Bins pesos", 10, 120, 50, key="bins_weights")
        with colc2:
            bins_b = st.slider("Bins sesgos", 5, 60, 20, key="bins_biases")
        with colc3:
            log_y = st.checkbox("Escala log en frecuencia", value=False,
                                help="Útil si hay valores muy concentrados")

    model = st.session_state.nn_model

    # Extraer pesos de forma robusta
    try:
        layer_weights, layer_biases = [], []
        layer_summaries = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                if len(weights) >= 2:
                    W, b = weights[0], weights[1]
                    layer_weights.append(W)
                    layer_biases.append(b)
                    near_zero_ratio = np.mean(np.abs(W) < 1e-3) * 100
                    layer_summaries.append({
                        'Capa': i+1,
                        'Forma Pesos': str(W.shape),
                        'Media Pesos': float(np.mean(W)),
                        'Std Pesos': float(np.std(W)),
                        '% |w|<1e-3': round(near_zero_ratio, 1),
                        'Media Sesgos': float(np.mean(b)),
                        'Std Sesgos': float(np.std(b)),
                        'Num Parámetros': W.size + b.size
                    })
                    st.caption(
                        f"✅ Capa {i+1}: {W.shape} pesos, {b.shape} sesgos")

        if layer_weights:
            # Tabla resumen
            st.markdown("### 📋 Resumen por Capa")
            summary_df = pd.DataFrame(layer_summaries)
            st.dataframe(summary_df, use_container_width=True)

            # Análisis global
            all_weights = np.concatenate([w.flatten() for w in layer_weights])
            global_std = np.std(all_weights)
            global_mean = np.mean(all_weights)
            near_zero_global = np.mean(np.abs(all_weights) < 1e-3) * 100

            colg1, colg2, colg3, colg4 = st.columns(4)
            with colg1:
                st.metric("Media global", f"{global_mean:.4f}")
            with colg2:
                st.metric("Std global", f"{global_std:.4f}")
            with colg3:
                st.metric("% |w|<1e-3", f"{near_zero_global:.1f}%")
            with colg4:
                st.metric("Capas analizadas", len(layer_weights))

            if global_std < 0.01:
                st.warning(
                    "🟡 Muchos pesos muy pequeños: posible underfitting o regularización alta")
            elif global_std > 2:
                st.warning(
                    "⚠️ Pesos con varianza muy alta: posible inestabilidad del entrenamiento")
            else:
                st.success("✅ Varianza de pesos dentro de un rango saludable")

            st.markdown("### 🔍 Distribución por Capa")
            for i, (weights, biases) in enumerate(zip(layer_weights, layer_biases)):
                st.markdown(f"#### 📊 Capa {i+1}")
                col1, col2 = st.columns(2)

                w_flat = weights.flatten()
                b_flat = biases.flatten()
                w_mean, w_std = np.mean(w_flat), np.std(w_flat)
                b_mean, b_std = np.mean(b_flat), np.std(b_flat)
                w_near_zero = np.mean(np.abs(w_flat) < 1e-3) * 100

                with col1:
                    fig_w = go.Figure()
                    fig_w.add_trace(go.Histogram(
                        x=w_flat, nbinsx=bins_w, name='Pesos', marker_color='#1f77b4'
                    ))
                    # Líneas de referencia
                    fig_w.add_vline(x=0, line_color='gray', line_width=1)
                    fig_w.add_vline(x=w_mean, line_color='green', line_dash='dash',
                                    annotation_text='Media', annotation_position='top right')
                    fig_w.update_layout(
                        title=f'Distribución Pesos Capa {i+1}',
                        height=320,
                        xaxis_title="Valor de peso",
                        yaxis_title="Frecuencia",
                        yaxis_type='log' if log_y else 'linear',
                        bargap=0.02,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_w, use_container_width=True)
                    st.caption(
                        f"Media={w_mean:.4f} | Std={w_std:.4f} | %≈0={w_near_zero:.1f}% | Rango=[{w_flat.min():.3f}, {w_flat.max():.3f}]")

                with col2:
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Histogram(
                        x=b_flat, nbinsx=bins_b, name='Sesgos', marker_color='#ff7f0e'
                    ))
                    fig_b.add_vline(x=0, line_color='gray', line_width=1)
                    fig_b.add_vline(x=b_mean, line_color='green', line_dash='dash',
                                    annotation_text='Media', annotation_position='top right')
                    fig_b.update_layout(
                        title=f'Distribución Sesgos Capa {i+1}',
                        height=320,
                        xaxis_title="Valor de sesgo",
                        yaxis_title="Frecuencia",
                        yaxis_type='log' if log_y else 'linear',
                        bargap=0.02,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_b, use_container_width=True)
                    st.caption(
                        f"Media={b_mean:.4f} | Std={b_std:.4f} | Rango=[{b_flat.min():.3f}, {b_flat.max():.3f}]")

            st.markdown("### 🩺 Evaluación General")
            insights = []
            if near_zero_global > 75:
                insights.append(
                    "Muchos pesos casi nulos → considera reducir regularización o entrenar más épocas.")
            if global_std < 0.01:
                insights.append(
                    "Varianza extremadamente baja → el modelo quizá no aprendió patrones útiles.")
            if global_std > 2:
                insights.append(
                    "Varianza muy alta → riesgo de explosión de gradientes/saturación.")
            if not insights:
                insights.append(
                    "Distribución equilibrada de pesos y sesgos. No se detectan anomalías destacables.")

            for msg in insights:
                st.write(f"• {msg}")
        else:
            st.warning("⚠️ No se encontraron capas con pesos entrenables")

    except Exception as weights_error:
        st.error(f"❌ Error analizando pesos: {weights_error}")


def show_decision_surface_tab():
    """Muestra la superficie de decisión."""
    import numpy as np
    import matplotlib.pyplot as plt

    st.subheader("🎯 Superficie de Decisión")

    model = st.session_state.nn_model
    config = st.session_state.nn_config

    if config.get('task_type') == 'Clasificación':
        if config['input_size'] > 2:
            feature_names = st.session_state.get('nn_feature_names',
                                                 [f'Característica {i+1}' for i in range(config['input_size'])])

            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Primera característica:", feature_names,
                                        index=0, key="viz_f1")
            with col2:
                feature2 = st.selectbox("Segunda característica:", feature_names,
                                        index=min(1, len(feature_names)-1), key="viz_f2")

            if feature1 != feature2:
                try:
                    X_test, _ = st.session_state.nn_test_data

                    # Extraer características seleccionadas
                    feature_idx = [feature_names.index(
                        feature1), feature_names.index(feature2)]
                    X_2d = X_test[:, feature_idx]

                    # Crear malla
                    h = 0.02
                    x_min, x_max = X_2d[:, 0].min(
                    ) - 0.5, X_2d[:, 0].max() + 0.5
                    y_min, y_max = X_2d[:, 1].min(
                    ) - 0.5, X_2d[:, 1].max() + 0.5
                    xx, yy = np.meshgrid(
                        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                    # Crear puntos para predicción con valores promedio
                    mesh_points = []
                    mean_values = np.mean(X_test, axis=0)

                    for i in range(xx.ravel().shape[0]):
                        point = mean_values.copy()
                        point[feature_idx[0]] = xx.ravel()[i]
                        point[feature_idx[1]] = yy.ravel()[i]
                        mesh_points.append(point)

                    mesh_points = np.array(mesh_points)
                    Z = model.predict(mesh_points, verbose=0)

                    # Procesar predicciones
                    if len(Z.shape) > 1 and Z.shape[1] > 1:
                        Z = np.argmax(Z, axis=1)
                    else:
                        Z = (Z > 0.5).astype(int).ravel()
                    Z = Z.reshape(xx.shape)

                    # Crear visualización
                    fig, ax = plt.subplots(figsize=(10, 8))
                    contourf = ax.contourf(
                        xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')

                    # Agregar puntos de datos
                    _, y_test = st.session_state.nn_test_data
                    y_plot = np.argmax(y_test, axis=1) if len(
                        y_test.shape) > 1 else y_test
                    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_plot, cmap='RdYlBu',
                               edgecolors='black', s=50, alpha=0.9)

                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    ax.set_title(
                        f'Superficie de Decisión: {feature1} vs {feature2}')
                    plt.colorbar(contourf, ax=ax)
                    st.pyplot(fig)

                    st.success("✅ Superficie generada exitosamente")

                except Exception as surf_error:
                    st.error(f"❌ Error: {surf_error}")
            else:
                st.warning("⚠️ Selecciona características diferentes")
        else:
            st.info("💡 Implementación para datasets 2D próximamente")
    else:
        st.info("🏔️ Superficie de predicción para regresión próximamente")


def show_layer_activations_tab():
    """Muestra el análisis de activaciones de capas."""
    import tensorflow as tf
    import plotly.graph_objects as go
    import numpy as np

    st.subheader("📉 Análisis de Activaciones y Neuronas Muertas")

    with st.expander("💡 ¿Qué verás aquí?", expanded=False):
        st.markdown("""
        Esta pestaña muestra **cómo se activan las neuronas internas** y detecta:

        * 💀 **Neuronas muertas:** Nunca se activan (todas sus salidas ≈ 0 en ReLU) → No aportan nada.
        * 😴 **Neuronas casi constantes:** Producen (casi) siempre el mismo valor → Muy poca contribución.
        * 🔴 **Saturación:** Muchas activaciones muy altas (por ejemplo >0.99 en sigmoides) → Gradientes débiles.

        **¿Por qué ocurren las neuronas muertas?**
        - Learning rate alto que empuja los pesos a regiones sin recuperación
        - Inicialización poco favorable
        - Demasiada regularización (L1/L2 / dropout mal configurado)
        - Arquitectura sobredimensionada

        **Cómo mitigarlo:** Reduce LR, elimina capas innecesarias, cambia inicialización o usa activaciones como LeakyReLU / ELU.
        """)

    model = st.session_state.nn_model

    # Controles
    with st.expander("⚙️ Parámetros de Análisis", expanded=False):
        colc1, colc2, colc3, colc4 = st.columns(4)
        with colc1:
            sample_size = st.slider(
                "Muestras", 10, 200, 50, help="Número de ejemplos de test usados para calcular activaciones")
        with colc2:
            dead_threshold = st.number_input("Umbral muerta (<=)", 0.0, 1.0, 0.0, 0.01,
                                             help="Si TODAS las activaciones de la neurona están por debajo o iguales a este valor, se marca como muerta")
        with colc3:
            constant_std = st.number_input("Std constante <", 1e-6, 1.0, 1e-3, format="%.6f",
                                           help="Si la desviación estándar de una neurona es menor a este valor, se considera casi constante")
        with colc4:
            show_network_map = st.checkbox("Mostrar mapa de red", value=True)

        colr1, colr2, colr3 = st.columns(3)
        with colr1:
            jitter_strength = st.slider(
                "Dispersión nodos", 0, 50, 18, help="Separación horizontal aleatoria (sin aumentar altura). 0 = columnas rectas.")
        with colr2:
            connection_sample_pct = st.slider(
                "% conexiones dibujadas", 10, 100, 70,
                help="Para capas muy densas reduce líneas dibujadas manteniendo estructura (submuestreo determinista)")
        with colr3:
            adaptive_radius = st.checkbox(
                "Radio adaptativo", value=True, help="Reduce el tamaño de nodos en capas con muchas neuronas para evitar solapamiento")

        colgA, colgB, colgC = st.columns(3)
        with colgA:
            group_large_layers = st.checkbox(
                "Agrupar capas grandes", value=True, help="Divide internamente capas con muchas neuronas en varias subcolumnas para reducir apelotonamiento sin aumentar altura")
        with colgB:
            min_spacing = st.slider(
                "Espaciado mínimo vertical", 12, 50, 22, help="Altura mínima entre neuronas dentro de una subcolumna")
        with colgC:
            max_columns = st.slider(
                "Máx subcolumnas capa", 1, 6, 3, help="Límite superior de subcolumnas usadas al agrupar capas grandes")

    try:
        X_test, _ = st.session_state.nn_test_data
        sample_size = min(sample_size, len(X_test))
        sample_data = X_test[:sample_size].astype(np.float32)

        # Identificar capas densas ocultas (excluye entrada implícita y última capa de salida)
        hidden_layers = []
        # excluye capa de salida
        for idx, layer in enumerate(model.layers[:-1]):
            # evitar capa de entrada (InputLayer / primera densa)
            if hasattr(layer, 'units') and idx > 0:
                hidden_layers.append((idx, layer))

        if not hidden_layers:
            st.warning("⚠️ No hay capas ocultas densas para analizar")
            return

        layer_stats = []
        dead_neurons_per_layer = []
        constant_neurons_per_layer = []
        saturation_per_layer = []
        activations_cache = []

        for layer_index, layer in hidden_layers:
            try:
                partial_model = tf.keras.Model(
                    inputs=model.input, outputs=layer.output)
                acts = partial_model.predict(sample_data, verbose=0)
                activations_cache.append(acts)

                # Calcular métricas por neurona
                neuron_means = acts.mean(axis=0)
                neuron_stds = acts.std(axis=0)
                # Dead: todas activaciones <= dead_threshold
                dead_mask = (acts <= dead_threshold).all(axis=0)
                constant_mask = (~dead_mask) & (neuron_stds < constant_std)
                # Saturación (heurística simple: activaciones muy altas ~1.0 si rango 0-1)
                sat_mask = (acts >= 0.99).mean(axis=0) > 0.5 if acts.max(
                ) <= 1.2 else np.zeros_like(dead_mask, dtype=bool)

                dead_idx = np.where(dead_mask)[0].tolist()
                constant_idx = np.where(constant_mask)[0].tolist()
                sat_ratio = (acts >= 0.99).mean() * \
                    100 if acts.max() <= 1.2 else 0.0

                layer_stats.append({
                    'Capa': layer_index + 1,
                    'Neuronas': acts.shape[1],
                    'Muertas': len(dead_idx),
                    '% Muertas': len(dead_idx) / acts.shape[1] * 100,
                    'Constantes': len(constant_idx),
                    '% Const': len(constant_idx) / acts.shape[1] * 100,
                    '% Saturación≈': sat_ratio,
                    'Media Act.': float(neuron_means.mean()),
                    'Std Act.': float(neuron_stds.mean())
                })
                dead_neurons_per_layer.append(dead_idx)
                constant_neurons_per_layer.append(constant_idx)
                saturation_per_layer.append(sat_ratio)
            except Exception as layer_err:
                st.error(f"Error capa {layer_index+1}: {layer_err}")
                continue

        # Resumen
        import pandas as pd
        st.markdown("### 📋 Resumen por Capa Oculta")
        df_stats = pd.DataFrame(layer_stats)
        st.dataframe(df_stats, use_container_width=True)

        # Interpretación global
        total_dead = sum(s['Muertas'] for s in layer_stats)
        total_neurons = sum(s['Neuronas'] for s in layer_stats)
        dead_pct = total_dead / total_neurons * 100 if total_neurons else 0
        if dead_pct > 50:
            st.error(
                f"🚨 Alta proporción de neuronas muertas ({dead_pct:.1f}%) → Revisa LR, inicialización o arquitectura")
        elif dead_pct > 20:
            st.warning(
                f"⚠️ Proporción moderada de neuronas muertas ({dead_pct:.1f}%)")
        else:
            st.success(f"✅ Neuronas muertas bajo control ({dead_pct:.1f}%)")

        # Detalle por capa con histogramas
        st.markdown("### 🔍 Detalle de Activaciones")
        for (layer_index, layer), acts in zip(hidden_layers, activations_cache):
            with st.expander(f"Capa {layer_index+1} ({layer.name})", expanded=False):
                colm1, colm2, colm3, colm4 = st.columns(4)
                with colm1:
                    st.metric("Neuronas", acts.shape[1])
                with colm2:
                    st.metric(
                        "Muertas", f"{len(dead_neurons_per_layer[hidden_layers.index((layer_index, layer))])}")
                with colm3:
                    st.metric(
                        "Constantes", f"{len(constant_neurons_per_layer[hidden_layers.index((layer_index, layer))])}")
                with colm4:
                    st.metric(
                        "Saturación %", f"{saturation_per_layer[hidden_layers.index((layer_index, layer))]:.1f}")

                if st.checkbox(f"Mostrar histograma activaciones capa {layer_index+1}", key=f"hist_act_{layer_index}"):
                    fig_act = go.Figure()
                    fig_act.add_trace(go.Histogram(
                        x=acts.flatten(), nbinsx=50, name='Activaciones'))
                    fig_act.update_layout(
                        height=280, title=f"Distribución Activaciones Capa {layer_index+1}")
                    st.plotly_chart(fig_act, use_container_width=True)

                if st.checkbox(f"Mostrar medias por neurona capa {layer_index+1}", key=f"means_{layer_index}"):
                    neuron_means = acts.mean(axis=0)
                    fig_mean = go.Figure(go.Bar(y=neuron_means, name='Media'))
                    fig_mean.update_layout(
                        height=260, title=f"Medias de Activación (Capa {layer_index+1})", xaxis_title='Neurona', yaxis_title='Media')
                    st.plotly_chart(fig_mean, use_container_width=True)

        # Mapa de red con calaveras
        if show_network_map:
            st.markdown("### 🗺️ Mapa de Neuronas Muertas (💀) y Constantes (😴)")
            # Preparar datos para HTML
            architecture = []
            # input size estimado
            try:
                input_dim = model.layers[0].input_shape[-1]
            except Exception:
                input_dim = st.session_state.nn_config.get('input_size', 0)
            architecture.append(int(input_dim))
            for idx, layer in hidden_layers:
                architecture.append(int(layer.units))
            # capa salida
            out_units = getattr(model.layers[-1], 'units', 1)
            architecture.append(int(out_units))

            dead_js = dead_neurons_per_layer
            const_js = constant_neurons_per_layer

            # Construir información de pesos/sesgos por neurona para tooltip
            import json
            neuron_info = []  # índice 0 capa entrada (sin pesos entrantes)
            neuron_info.append([])
            # Dense layers en orden (todas las capas con kernel incluyendo salida)
            dense_layers = [ly for ly in model.layers if hasattr(ly, 'kernel')]
            for d_idx, d_layer in enumerate(dense_layers):
                W, b = d_layer.get_weights()[:2]
                layer_list = []
                for j in range(W.shape[1]):
                    w_col = W[:, j]
                    info_item = {
                        'bias': float(b[j]),
                        'mean_w': float(np.mean(w_col)),
                        'std_w': float(np.std(w_col)),
                        'max_abs_w': float(np.max(np.abs(w_col)))
                    }
                    # Marcar estado si es capa oculta (excluye salida)
                    arch_layer_index = d_idx + 1  # arquitectura incluye input al inicio
                    if arch_layer_index < len(architecture) - 1:  # capa oculta
                        hidden_list_index = arch_layer_index - 1  # dead_js index
                        if hidden_list_index < len(dead_js):
                            if j in dead_js[hidden_list_index]:
                                info_item['estado'] = 'muerta'
                            elif j in const_js[hidden_list_index]:
                                info_item['estado'] = 'constante'
                            else:
                                info_item['estado'] = 'activa'
                        else:
                            info_item['estado'] = 'activa'
                    else:
                        info_item['estado'] = 'salida'
                    layer_list.append(info_item)
                neuron_info.append(layer_list)

            weight_info_js = json.dumps(neuron_info)

            # Calcular ancho ideal del iframe (más capas => más ancho), con límites razonables
            try:
                comp_width = min(
                    1600, max(700, 260 + 180 * (len(architecture) - 1)))
            except Exception:
                comp_width = 1000

            html_map = f"""
                        <style>
                            .nn-dead-wrapper {{
                                width:100%;
                                max-width:100%;
                                margin:0 auto;
                                border:1px solid #ddd;
                                padding:12px 12px 4px 12px;
                                border-radius:8px;
                                background:#fafafa;
                                box-sizing:border-box;
                            }}
                            #deadCanvas {{
                                width:100% !important;
                                height:460px !important;
                                display:block;
                            }}
                        </style>
                        <div class='nn-dead-wrapper'>
                            <canvas id='deadCanvas'></canvas>
                            <div style='font-size:12px;margin-top:6px;'>Leyenda: <span style='color:#1976d2'>● Activa</span>  <span style='opacity:0.35'>● Inactiva</span>  💀 Muerta  😴 Constante</div>
                        </div>
                        <script>
                            const arch = {architecture};
                            const dead = {dead_js};
                            const constantN = {const_js};
                            const jitterStrength = {jitter_strength};
                            const connectionSample = {connection_sample_pct} / 100.0; // proporción de conexiones a dibujar
                            const adaptiveRadius = {str(adaptive_radius).lower()};
                            const groupLarge = {str(group_large_layers).lower()};
                            const minSpacing = {min_spacing};
                            const maxCols = {max_columns};
                            const neuronInfo = {weight_info_js};
                            const canvas = document.getElementById('deadCanvas');
                            const ctx = canvas.getContext('2d');
                            let __nnData = null; // coords y radios
                            // PRNG determinista simple basado en capa y neurona
                            function hash(l, i, j=0) {{
                                let h = l * 374761393 + i * 668265263 + j * 2147483647;
                                h = (h ^ (h >> 13)) * 1274126177;
                                h = (h ^ (h >> 16));
                                return (h >>> 0) / 4294967295; // [0,1)
                            }}
                            function resize() {{
                                const parentW = canvas.parentElement.getBoundingClientRect().width;
                                canvas.width = parentW;
                                canvas.height = 460;
                                draw();
                            }}
                            function nodeY(idx, total, h, margin) {{
                                if(total===1) return h/2;
                                const spacing = (h - 2*margin)/(total-1);
                                return margin + idx*spacing;
                            }}
                            function draw() {{
                                ctx.clearRect(0,0,canvas.width,canvas.height);
                                // Márgenes dinámicos para aprovechar ancho sin pegarse a bordes
                                const marginY = 50;
                                const marginX = Math.min(80, Math.max(40, canvas.width*0.05));
                                const layerGap = (canvas.width - 2*marginX) / (arch.length-1);
                                // radios por capa
                                const layerR = arch.map(n => adaptiveRadius ? Math.min(14, Math.max(5, (canvas.height/(n+6)))) : 10);
                                // Precalcular coordenadas de neuronas por capa con posible agrupación
                                const coords = [];
                                for(let l=0; l<arch.length; l++) {{
                                    const n = arch[l];
                                    const baseX = marginX + l*layerGap;
                                    const layerCoords = [];
                                    if(groupLarge && n > 0) {{
                                        const availableH = canvas.height - 2*marginY;
                                        const maxPerCol = Math.max(1, Math.floor(availableH / minSpacing));
                                        let cols = Math.ceil(n / maxPerCol);
                                        cols = Math.min(cols, maxCols);
                                        const perCol = maxPerCol; // base capacity per col
                                        const colOffset = Math.min(32, layerGap * 0.25); // separación horizontal entre subcolumnas
                                        for(let i=0;i<n;i++) {{
                                            const col = Math.floor(i / perCol);
                                            const row = i % perCol;
                                            // Ajustar filas reales en última columna
                                            const itemsInCol = (col === cols-1) ? (n - col*perCol) : perCol;
                                            const y = itemsInCol === 1 ? (canvas.height/2) : (marginY + row * ((canvas.height - 2*marginY)/(itemsInCol-1)));
                                            const jitter = jitterStrength ? (hash(l,i)-0.5) * jitterStrength : 0;
                                            const x = baseX + (col - (cols-1)/2) * colOffset + jitter;
                                            layerCoords.push({{x,y,r:layerR[l]}});
                                        }}
                                    }} else {{
                                        for(let i=0;i<n;i++) {{
                                            const jitter = jitterStrength ? (hash(l,i)-0.5) * jitterStrength : 0;
                                            const x = baseX + jitter;
                                            const y = nodeY(i, n, canvas.height, marginY);
                                            layerCoords.push({{x,y,r:layerR[l]}});
                                        }}
                                    }}
                                    coords.push(layerCoords);
                                }}
                                __nnData = {{coords, layerR}};
                                ctx.globalAlpha = 0.15;
                                // Dibujar conexiones usando coords
                                for(let l=0; l<arch.length-1; l++) {{
                                    const leftCount = arch[l];
                                    const rightCount = arch[l+1];
                                    for(let i=0; i<leftCount; i++) {{
                                        const c1 = coords[l][i];
                                        for(let j=0; j<rightCount; j++) {{
                                            if(connectionSample < 0.999) {{
                                                const pr = hash(l,i,j);
                                                if(pr > connectionSample) continue;
                                            }}
                                            const c2 = coords[l+1][j];
                                            ctx.beginPath();
                                            ctx.moveTo(c1.x, c1.y);
                                            ctx.lineTo(c2.x, c2.y);
                                            ctx.strokeStyle = '#777';
                                            ctx.lineWidth = 1; ctx.stroke();
                                        }}
                                    }}
                                }}
                                ctx.globalAlpha = 1;
                                for(let l=0; l<arch.length; l++) {{
                                    const isHidden = l>0 && l<arch.length-1;
                                    for(let i=0;i<arch[l];i++) {{
                                        const c = coords[l][i];
                                        const x = c.x; const y = c.y;
                                        // Radio adaptativo: depende de densidad vertical y ancho (si está activado)
                                        const r = layerR[l];
                                        let alpha = 1;
                                        let emoji = '';
                                        if(isHidden) {{
                                            const layerHiddenIndex = l-1;
                                            const deadList = dead[layerHiddenIndex] || [];
                                            const constList = constantN[layerHiddenIndex] || [];
                                            if(deadList.includes(i)) {{ alpha = 0.25; emoji = '💀'; }}
                                            else if(constList.includes(i)) {{ alpha = 0.45; emoji = '😴'; }}
                                        }}
                                        ctx.globalAlpha = alpha;
                                        ctx.beginPath();
                                        ctx.fillStyle = l===0? '#4ECDC4' : (l===arch.length-1? '#FF6B6B' : '#1976d2');
                                        ctx.arc(x,y,r,0,Math.PI*2);
                                        ctx.fill();
                                        ctx.strokeStyle = '#1b1f23';
                                        ctx.lineWidth = 1.3; ctx.stroke();
                                        ctx.globalAlpha = 1;
                                        if(emoji) {{ ctx.font='14px sans-serif'; ctx.fillText(emoji, x-8, y+5); }}
                                    }}
                                    ctx.font = '13px sans-serif';
                                    ctx.fillStyle = '#333';
                                    let label = l===0? 'Entrada' : (l===arch.length-1? 'Salida' : 'Oculta '+(l));
                                    ctx.fillText(label + ' ('+ arch[l] +')', marginX + l*layerGap - 35, 18);
                                }}
                            }}
                            window.addEventListener('resize', () => {{ clearTimeout(window.__deadT); window.__deadT = setTimeout(resize, 120); }});
                            resize();
                            // Tooltip
                            const wrapper = canvas.parentElement;
                            const tip = document.createElement('div');
                            tip.className = 'nn-tooltip';
                            tip.style.cssText = 'position:absolute;pointer-events:none;background:#fff;border:1px solid #ccc;padding:6px 8px;font-size:12px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.15);display:none;max-width:220px;line-height:1.2;';
                            wrapper.style.position = 'relative';
                            wrapper.appendChild(tip);
                            function formatVal(v) {{ return (Math.abs(v) < 1e-4 ? v.toExponential(2) : v.toFixed(4)); }}
                            function handleMove(ev) {{
                                if(!__nnData) return;
                                const rect = canvas.getBoundingClientRect();
                                const mx = ev.clientX - rect.left;
                                const my = ev.clientY - rect.top;
                                let found = null;
                                for(let l=0; l<__nnData.coords.length; l++) {{
                                    for(let i=0;i<__nnData.coords[l].length;i++) {{
                                        const c = __nnData.coords[l][i];
                                        const dx = mx - c.x; const dy = my - c.y;
                                        if(dx*dx + dy*dy <= (c.r+4)*(c.r+4)) {{ found = {{l,i,c}}; break; }}
                                    }}
                                    if(found) break;
                                }}
                                if(found) {{
                                    const l = found.l; const i = found.i;
                                    if(l === 0) {{ tip.innerHTML = `<strong>Capa Entrada</strong><br>Índice: ${{i}}`; }}
                                    else if(neuronInfo[l] && neuronInfo[l][i]) {{
                                        const info = neuronInfo[l][i];
                                        const estado = info.estado || 'n/a';
                                        tip.innerHTML = `<strong>Capa ${{l===arch.length-1? 'Salida':'Oculta '+l}}</strong> · Neurona ${{i}}<br>`+
                                            `Bias: ${{formatVal(info.bias)}}<br>`+
                                            `w̄: ${{formatVal(info.mean_w)}} | σ: ${{formatVal(info.std_w)}}<br>`+
                                            `|w|max: ${{formatVal(info.max_abs_w)}}<br>`+
                                            `Estado: ${{estado}}`;
                                    }} else {{
                                        tip.innerHTML = `<strong>Capa ${{l}}</strong> · Neurona ${{i}}`;
                                    }}
                                    tip.style.left = (mx + 14) + 'px';
                                    tip.style.top = (my + 14) + 'px';
                                    tip.style.display = 'block';
                                }} else {{
                                    tip.style.display = 'none';
                                }}
                            }}
                            canvas.addEventListener('mousemove', handleMove);
                            canvas.addEventListener('mouseleave', () => {{ tip.style.display='none'; }});
                        </script>
                        """
            components.html(html_map, height=510,
                            scrolling=False, width=comp_width)

        with st.expander("🧪 ¿Qué hacer si hay muchas neuronas muertas?", expanded=False):
            st.markdown("""
            **Acciones recomendadas:**
            1. Disminuye el learning rate
            2. Usa activaciones alternativas (LeakyReLU, ELU)
            3. Reduce profundidad o neuronas redundantes
            4. Revisa inicialización (HeNormal para ReLU)
            5. Entrena más épocas si la pérdida aún baja
            """)

    except Exception as e:
        st.error(f"❌ Error en análisis de activaciones: {e}")
        st.info(
            "Verifica que el modelo esté entrenado y que existan capas ocultas densas.")


def show_neural_network_visualizations():
    """Muestra visualizaciones avanzadas del modelo de forma simplificada."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    st.info("""
    🎓 **Visualizaciones de Redes Neuronales:**
    - **Historial**: Evolución del aprendizaje - **Pesos**: Lo que aprendió cada neurona
    - **Superficie**: Cómo separa las clases (2D) - **Capas**: Activaciones internas
    """)

    try:
        # Intentar inicialización
        # success, message = initialize_model_safely()

        # if success:
        #    st.success(message)
        # else:
        #    st.error(message)

        # CREAR PESTAÑAS DE VISUALIZACIÓN
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Historial", "🧠 Pesos", "🎯 Superficie", "📉 Capas"
        ])

        # TAB 1: HISTORIAL DE ENTRENAMIENTO
        with viz_tab1:
            show_training_history_tab()

        # TAB 2: PESOS Y SESGOS
        with viz_tab2:
            show_weights_analysis_tab()

        # TAB 3: SUPERFICIE DE DECISIÓN
        with viz_tab3:
            show_decision_surface_tab()

        # TAB 4: ANÁLISIS DE CAPAS
        with viz_tab4:
            show_layer_activations_tab()

        # NAVEGACIÓN
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔙 Volver a Evaluación", use_container_width=True):
                st.session_state.active_tab_nn = 3
                st.rerun()
        with col2:
            if st.button("🔮 Hacer Predicciones", type="primary", use_container_width=True):
                st.session_state.active_tab_nn = 5
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error en visualizaciones: {str(e)}")

        if "never been called" in str(e):
            st.markdown("""
            **🔧 Solución:** El modelo necesita inicialización completa.
            1. Reentrena el modelo desde cero
            2. O usa el botón de reparación automática
            """)

            if st.button("🔙 Ir a Entrenamiento", type="primary"):
                st.session_state.active_tab_nn = 2
                st.rerun()


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
                if output_size > 2:  # Multiclase
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
