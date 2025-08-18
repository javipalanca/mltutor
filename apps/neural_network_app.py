import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from dataset.dataset_manager import load_data
from dataset.dataset_tab import run_select_dataset, show_dataset_info, run_explore_dataset_tab
from utils import create_info_box, get_image_download_link, show_code_with_download
from algorithms.code_examples import generate_neural_network_architecture_code, generate_neural_network_evaluation_code
from algorithms.model_training import train_neural_network
from algorithms.model_evaluation import show_detailed_evaluation, neural_network_diagnostics
from algorithms.export import show_neural_network_export
from viz.nn import (
    evaluate_nn,
    show_neural_network_evaluation,
    create_neural_network_visualization,
    calculate_network_parameters,
    show_neural_network_visualizations,
    show_training_history_tab,
    show_weights_analysis_tab,
    show_decision_surface_tab,
    show_layer_activations_tab
)
from ui import create_button_panel, create_prediction_interface
from apps.navbar import navbar


def run_neural_networks_app():
    """Ejecuta la aplicación específica de redes neuronales."""
    st.header("🧠 Redes Neuronales")
    st.markdown(
        "Aprende sobre redes neuronales artificiales de forma visual e interactiva")

    # Información sobre redes neuronales
    with st.expander("ℹ️ ¿Qué son las Redes Neuronales?", expanded=False):
        st.markdown("""
        Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano.
        Están compuestas por nodos (neuronas) interconectados que procesan información de manera paralela.

        **Características principales:**
        - **Neuronas**: Unidades básicas que reciben entradas, las procesan y generan salidas
        - **Capas**: Organizan las neuronas en estructuras jerárquicas (entrada, ocultas, salida)
        - **Pesos y Sesgos**: Parámetros que se ajustan durante el entrenamiento
        - **Funciones de Activación**: Determinan la salida de cada neurona
        - **Backpropagation**: Algoritmo para entrenar la red ajustando pesos y sesgos

        **Ventajas:**
        - Pueden modelar relaciones no lineales complejas
        - Excelentes para reconocimiento de patrones
        - Adaptables a diferentes tipos de problemas
        - Capaces de aproximar cualquier función continua

        **Desventajas:**
        - Requieren grandes cantidades de datos
        - Pueden ser "cajas negras" (difíciles de interpretar)
        - Propensos al sobreajuste
        - Requieren mucho poder computacional
        """)

    # Sistema de pestañas
    tab_names = [
        "📊 Datos",
        "🏗️ Arquitectura",
        "⚙️ Entrenamiento",
        "📈 Evaluación",
        "🎯 Visualización",
        "🔮 Predicciones",
        "💾 Exportar"
    ]

    # Inicializar estado de pestañas si no existe
    if 'active_tab_nn' not in st.session_state:
        st.session_state.active_tab_nn = 0

    # Crear pestañas visuales personalizadas
    cols = st.columns(len(tab_names))
    for i, (col, tab_name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(
                tab_name,
                key=f"tab_nn_{i}",
                use_container_width=True,
                type="primary" if st.session_state.active_tab_nn == i else "secondary"
            ):
                st.session_state.active_tab_nn = i
                st.rerun()

    st.markdown("---")

    ###########################################
    #     Pestaña de Datos                    #
    ###########################################
    tips = """
        🎓 **Tips para Redes Neuronales:**
        - Las redes neuronales funcionan mejor con **datos normalizados** (valores entre 0 y 1 o -1 y 1)
        - Necesitan **suficientes datos** para entrenar bien (mínimo 100 ejemplos por clase)
        - Son excelentes para **patrones complejos** y **relaciones no lineales**
        - Pueden funcionar tanto para **clasificación** como para **regresión**
        """
    if st.session_state.active_tab_nn == 0:
        st.header("📊 Selección y Preparación de Datos")
        st.info(tips)
        run_select_dataset()
        run_explore_dataset_tab()

        # Botones de navegación
        navbar("active_tab_nn", None, "Continuar a Arquitectura")
    else:
        # Para otras pestañas, mostrar qué dataset está seleccionado actualmente
        if hasattr(st.session_state, 'selected_dataset'):
            st.info(
                f"📊 **Dataset actual:** {st.session_state.selected_dataset}")
            st.markdown("---")

    ###########################################
    #     Pestaña de Arquitectura             #
    ###########################################
    if st.session_state.active_tab_nn == 1:
        st.header("🏗️ Diseño de la Arquitectura de la Red")
        st.markdown("### 🎛️ Configuración de la Red Neuronal")

        try:
            # Cargar datos usando la función load_data común
            dataset_name = st.session_state.selected_dataset
            X, y, feature_names, class_names, dataset_info, task_type = load_data(
                dataset_name)

            # Crear DataFrame
            df = X  # pd.DataFrame(X, columns=feature_names)

            df = df.reset_index(drop=True)
            y = y.reset_index(drop=True)
            # Determinar el nombre de la columna objetivo
            if task_type == "Clasificación":
                # Para clasificación, usar el nombre de la variable objetivo del dataset_info
                target_col = 'target'  # Nombre por defecto
                if hasattr(dataset_info, 'target_names'):
                    target_col = 'target'
                df[target_col] = y
            else:
                # Para regresión
                target_col = 'target'
                df[target_col] = y

            # Almacenar información del dataset
            st.session_state.nn_df = df
            st.session_state.nn_target_col = target_col
            st.session_state.nn_feature_names = feature_names
            st.session_state.nn_class_names = class_names
            st.session_state.nn_task_type = task_type
            st.session_state.nn_dataset_info = dataset_info

        except Exception as e:
            st.error(f"Error al cargar el dataset: {e}")

        # Preparar datos básicos para mostrar dimensiones
        X = df.drop(columns=[target_col])
        y = df[target_col]
        input_size = X.shape[1]

        if task_type == "Clasificación":
            num_classes = len(y.unique())
            if num_classes == 2:
                output_size = 1  # Para clasificación binaria
                st.info(
                    f"📊 **Entrada**: {input_size} características → **Salida**: {output_size} neurona (clasificación binaria)")
            else:
                output_size = num_classes  # Para clasificación multiclase
                st.info(
                    f"📊 **Entrada**: {input_size} características → **Salida**: {output_size} clases")
        else:
            output_size = 1
            st.info(
                f"📊 **Entrada**: {input_size} características → **Salida**: {output_size} valor numérico")

        # Tips sobre dimensiones
        with st.expander("💡 ¿Cómo decidir el tamaño de la red?"):
            st.markdown(f"""
            **Reglas generales para tu dataset ({df.shape[0]} muestras, {input_size} características):**
            
            🔢 **Neuronas por capa oculta:**
            - Pequeño: {input_size//2} - {input_size} neuronas
            - Mediano: {input_size} - {input_size*2} neuronas  
            - Grande: {input_size*2} - {input_size*4} neuronas
            
            📚 **Número de capas:**
            - 1-2 capas: Problemas simples, linealmente separables
            - 2-3 capas: Problemas moderadamente complejos (recomendado para empezar)
            - 4+ capas: Problemas muy complejos (requiere muchos datos)
            
            ⚖️ **Balance capacidad vs. datos:**
            - Más parámetros que datos → riesgo de sobreajuste
            - Tu dataset: {df.shape[0]} muestras, mantén parámetros < {df.shape[0]//10}
            """)

        # Configuración de arquitectura
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### ⚙️ Configuración de Capas")

            # Número de capas ocultas con explicación
            num_hidden_layers = st.slider(
                "Número de capas ocultas",
                min_value=1, max_value=5, value=2,
                help="💡 Más capas = mayor capacidad de aprender patrones complejos, pero también mayor riesgo de sobreajuste"
            )

            # Sugerencia basada en el número de capas
            if num_hidden_layers == 1:
                st.caption(
                    "🟦 **1 capa**: Ideal para problemas linealmente separables")
            elif num_hidden_layers == 2:
                st.caption(
                    "🟨 **2 capas**: Recomendado para la mayoría de problemas")
            elif num_hidden_layers >= 3:
                st.caption(
                    "🟥 **3+ capas**: Solo para problemas muy complejos con muchos datos")

            # Configuración de cada capa oculta con explicaciones
            hidden_layers = []
            for i in range(num_hidden_layers):
                # Calcular sugerencia inteligente
                suggested_size = max(10, input_size // (i+1))
                if task_type == "Clasificación" and num_classes > 2:
                    suggested_size = max(suggested_size, num_classes * 2)

                neurons = st.slider(
                    f"Neuronas en capa oculta {i+1}",
                    min_value=1, max_value=256,
                    value=min(suggested_size, 64),  # Limitar valor por defecto
                    key=f"layer_{i}",
                    help=f"💡 Sugerencia para capa {i+1}: {suggested_size} neuronas"
                )
                hidden_layers.append(neurons)

            # Función de activación con explicaciones detalladas
            st.markdown("#### 🧮 Función de Activación (Capas Ocultas)")
            activation = st.selectbox(
                "Función de activación",
                ["relu", "tanh", "sigmoid"],
                help="💡 ReLU es la más popular y efectiva para la mayoría de problemas"
            )

            # Explicaciones sobre funciones de activación
            if activation == "relu":
                st.success(
                    "✅ **ReLU**: Rápida, evita el problema del gradiente que desaparece. Recomendada.")
            elif activation == "tanh":
                st.info(
                    "ℹ️ **Tanh**: Salida entre -1 y 1. Buena para datos normalizados.")
            elif activation == "sigmoid":
                st.warning(
                    "⚠️ **Sigmoid**: Puede causar gradientes que desaparecen. Úsala solo si es necesario.")

            # Función de activación de salida - AHORA SELECCIONABLE
            st.markdown("#### 🎯 Función de Activación de Salida")

            # Opciones disponibles según el tipo de tarea
            if task_type == "Clasificación":
                output_options = ["sigmoid", "softmax", "linear", "tanh"]
                if output_size == 1:  # Clasificación binaria
                    recommended = "sigmoid"
                    default_index = 0
                else:  # Clasificación multiclase
                    recommended = "softmax"
                    default_index = 1
            else:
                output_options = ["linear", "sigmoid", "tanh", "softmax"]
                recommended = "linear"
                default_index = 0

            output_activation = st.selectbox(
                "Función de activación de salida",
                output_options,
                index=default_index,
                help=f"💡 Función recomendada para {task_type.lower()}: **{recommended}**"
            )

            # Validaciones y avisos
            show_warning = False
            warning_message = ""

            if task_type == "Clasificación":
                if output_size == 1:  # Clasificación binaria
                    if output_activation == "sigmoid":
                        st.success(
                            "✅ Sigmoid es ideal para clasificación binaria")
                    elif output_activation == "softmax":
                        show_warning = True
                        warning_message = "⚠️ Softmax no es recomendada para clasificación binaria (1 neurona). Considera usar Sigmoid."
                    elif output_activation == "linear":
                        show_warning = True
                        warning_message = "⚠️ Linear puede causar problemas en clasificación. Considera usar Sigmoid."
                    elif output_activation == "tanh":
                        show_warning = True
                        warning_message = "⚠️ Tanh puede funcionar pero Sigmoid es más estándar para clasificación binaria."

                else:  # Clasificación multiclase
                    if output_activation == "softmax":
                        st.success(
                            "✅ Softmax es ideal para clasificación multiclase")
                    elif output_activation == "sigmoid":
                        st.warning(
                            "⚠️ Sigmoid en multiclase requiere 'binary_crossentropy' por clase. Softmax es más estándar.")
                    elif output_activation == "linear":
                        show_warning = True
                        warning_message = "⚠️ Linear no es apropiada para clasificación. Usa Softmax."
                    elif output_activation == "tanh":
                        show_warning = True
                        warning_message = "⚠️ Tanh no es estándar para clasificación multiclase. Softmax es recomendada."

            else:  # Regresión
                if output_activation == "linear":
                    st.success("✅ Linear es ideal para regresión")
                elif output_activation == "sigmoid":
                    st.warning(
                        "⚠️ Sigmoid limita la salida a [0,1]. Solo útil si tus valores objetivo están en este rango.")
                elif output_activation == "tanh":
                    st.warning(
                        "⚠️ Tanh limita la salida a [-1,1]. Solo útil si tus valores objetivo están en este rango.")
                elif output_activation == "softmax":
                    show_warning = True
                    warning_message = "⚠️ Softmax no es apropiada para regresión. Las salidas suman 1. Usa Linear."

            # Mostrar advertencia crítica si es necesario
            if show_warning:
                st.error(warning_message)

        with col2:
            st.markdown("#### 🎨 Visualización de la Arquitectura")

            # Crear arquitectura completa
            architecture = [input_size] + hidden_layers + [output_size]

            # Guardar configuración en session state
            st.session_state.nn_architecture = {
                'layers': architecture,
                'activation': activation,
                'output_activation': output_activation,
                'input_size': input_size,
                'output_size': output_size,
                'task_type': task_type
            }

            # Visualizar la red neuronal dinámicamente
            create_neural_network_visualization(
                architecture, activation, output_activation, task_type)

        # Configuración adicional con explicaciones detalladas
        st.markdown("### ⚙️ Configuración Adicional")

        st.markdown("📚 **Parámetros importantes para el entrenamiento:**")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown("#### 🛡️ Regularización")
            dropout_rate = st.slider(
                "Tasa de Dropout",
                min_value=0.0, max_value=0.8, value=0.2, step=0.1,
                help="💡 Dropout previene sobreajuste eliminando aleatoriamente neuronas durante entrenamiento"
            )

            # Explicación del dropout
            if dropout_rate == 0.0:
                st.caption("🔴 **Sin Dropout**: Mayor riesgo de sobreajuste")
            elif dropout_rate <= 0.2:
                st.caption("🟢 **Dropout Ligero**: Bueno para datasets grandes")
            elif dropout_rate <= 0.5:
                st.caption(
                    "🟡 **Dropout Moderado**: Recomendado para la mayoría de casos")
            else:
                st.caption(
                    "🟠 **Dropout Alto**: Solo para datasets muy pequeños")

        with col4:
            st.markdown("#### 📦 Procesamiento")
            batch_size = st.selectbox(
                "Tamaño de Batch",
                [16, 32, 64, 128, 256],
                index=2,  # 64 por defecto
                help="💡 Número de muestras procesadas antes de actualizar los pesos"
            )

            # Sugerencias según el tamaño del dataset
            dataset_size = df.shape[0]
            if batch_size >= dataset_size // 4:
                st.caption(
                    "🔴 **Batch Grande**: Puede ser lento pero más estable")
            elif batch_size >= 32:
                st.caption(
                    "🟢 **Batch Óptimo**: Buen balance velocidad/estabilidad")
            else:
                st.caption("🟡 **Batch Pequeño**: Más rápido pero más ruidoso")

        with col5:
            st.markdown("#### 🚀 Optimización")
            optimizer = st.selectbox(
                "Optimizador",
                ["adam", "sgd", "rmsprop"],
                help="💡 Algoritmo para actualizar los pesos de la red"
            )

            # Explicaciones sobre optimizadores
            if optimizer == "adam":
                st.caption(
                    "🟢 **Adam**: Adaptativo, recomendado para la mayoría de casos")
            elif optimizer == "sgd":
                st.caption(
                    "🟡 **SGD**: Clásico, requiere ajuste fino del learning rate")
            elif optimizer == "rmsprop":
                st.caption(
                    "🟦 **RMSprop**: Bueno para RNNs y problemas específicos")

        # Tips sobre la configuración
        with st.expander("💡 Tips para optimizar tu configuración"):
            st.markdown(f"""
            **Para tu dataset específico ({dataset_size} muestras):**
            
            🎯 **Batch Size recomendado:**
            - Dataset pequeño (<1000): 16-32
            - Dataset mediano (1000-10000): 32-64
            - Dataset grande (>10000): 64-128
            - Tu dataset: {dataset_size} muestras → Recomendado: {32 if dataset_size < 1000 else 64 if dataset_size < 10000 else 128}
            
            🛡️ **Dropout recomendado:**
            - Pocos datos: 0.3-0.5 (más regularización)
            - Muchos datos: 0.1-0.2 (menos regularización)
            - Dataset balanceado: 0.2-0.3
            
            🚀 **Optimizador:**
            - **Adam**: Mejor opción general, se adapta automáticamente
            - **SGD**: Úsalo solo si tienes experiencia ajustando learning rates
            - **RMSprop**: Alternativa a Adam, a veces funciona mejor en problemas específicos
            """)

        # Guardar configuración completa
        st.session_state.nn_config = {
            'architecture': architecture,
            'activation': activation,
            'output_activation': output_activation,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'task_type': task_type,
            'input_size': input_size,
            'output_size': output_size
        }

        # Resumen de la configuración con análisis
        st.markdown("### 📋 Resumen de la Arquitectura")

        total_params = calculate_network_parameters(architecture)

        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("🔢 Total de Parámetros", f"{total_params:,}",
                      help="Número total de pesos y sesgos que la red aprenderá")
        with col7:
            st.metric("📚 Capas Totales", len(architecture),
                      help="Entrada + Ocultas + Salida")
        with col8:
            complexity_ratio = total_params / dataset_size if dataset_size > 0 else 0
            complexity_level = "Baja" if complexity_ratio < 0.1 else "Media" if complexity_ratio < 1 else "Alta"
            st.metric("⚖️ Complejidad", complexity_level,
                      help=f"Ratio parámetros/datos: {complexity_ratio:.2f}")
        with col8:
            st.metric("🧠 Tipo de Red", "Perceptrón Multicapa")

        # Mostrar detalles de cada capa
        st.markdown("#### 📊 Detalles por Capa")
        layer_details = []
        for i, (current, next_size) in enumerate(zip(architecture[:-1], architecture[1:])):
            if i == 0:
                layer_type = "Entrada"
                params = 0
            elif i == len(architecture) - 2:
                layer_type = "Salida"
                params = current * next_size + next_size
            else:
                layer_type = f"Oculta {i}"
                params = current * next_size + next_size

            layer_details.append({
                "Capa": layer_type,
                "Neuronas": current if i < len(architecture) - 1 else next_size,
                "Parámetros": params,
                "Activación": "Entrada" if i == 0 else (output_activation if i == len(architecture) - 2 else activation)
            })

        st.dataframe(pd.DataFrame(layer_details), use_container_width=True)

        # Análisis de complejidad y recomendaciones
        st.markdown("#### 🔍 Análisis de Complejidad")

        # Análisis del ratio parámetros/datos
        if complexity_ratio < 0.1:
            st.success(
                f"✅ **Complejidad Óptima**: Tu red tiene {total_params:,} parámetros para {dataset_size} muestras (ratio: {complexity_ratio:.3f}). Bajo riesgo de sobreajuste.")
        elif complexity_ratio < 1:
            st.warning(
                f"⚠️ **Complejidad Media**: Tu red tiene {total_params:,} parámetros para {dataset_size} muestras (ratio: {complexity_ratio:.3f}). Monitorea el sobreajuste.")
        else:
            st.error(
                f"🚨 **Complejidad Alta**: Tu red tiene {total_params:,} parámetros para {dataset_size} muestras (ratio: {complexity_ratio:.3f}). Alto riesgo de sobreajuste. Considera reducir el tamaño de la red.")

        # Botón para generar código Python
        st.markdown("### 💻 Código Python")
        if st.button("📝 Generar Código de la Arquitectura", use_container_width=True):
            # Generar código Python para la arquitectura
            code = generate_neural_network_architecture_code(
                architecture, activation, output_activation, dropout_rate,
                optimizer, batch_size, task_type, st.session_state.nn_feature_names
            )

            st.markdown("#### 🐍 Código Python Generado")
            st.code(code, language='python')

            # Botón para descargar el código
            st.download_button(
                label="💾 Descargar Código Python",
                data=code,
                file_name=f"red_neuronal_arquitectura_{task_type.lower()}.py",
                mime="text/plain"
            )

        # Botones de navegación
        navbar("active_tab_nn", "Volver a Datos", "Continuar a Entrenamiento",
               next_note="**¿Listo para entrenar?** ¡Tu arquitectura está configurada!")

    ###########################################
    #     Pestaña de Entrenamiento            #
    ###########################################
    elif st.session_state.active_tab_nn == 2:
        st.header("⚙️ Entrenamiento de la Red Neuronal")

        if 'nn_config' not in st.session_state:
            st.warning("⚠️ Primero debes configurar la arquitectura de la red.")
            if st.button("🔙 Ir a Arquitectura"):
                st.session_state.active_tab_nn = 1
                st.rerun()
            return

        # Tips educativos sobre entrenamiento
        st.info("""
        🎓 **Conceptos Clave del Entrenamiento:**
        - **Learning Rate**: Controla qué tan rápido aprende la red (muy alto = inestable, muy bajo = lento)
        - **Épocas**: Cuántas veces la red ve todos los datos (más épocas ≠ siempre mejor)
        - **Validación**: Datos separados para monitorear si la red está generalizando bien
        - **Early Stopping**: Para evitar sobreajuste, para cuando la validación no mejora
        """)

        st.markdown("### 🎛️ Parámetros de Entrenamiento")

        # Información del dataset para sugerencias
        df = st.session_state.nn_df
        dataset_size = df.shape[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📈 **Learning Rate**")
            learning_rate = st.selectbox(
                "Tasa de Aprendizaje",
                [0.001, 0.01, 0.1, 0.3],
                index=0,
                help="💡 0.001 es seguro para empezar. Valores más altos pueden acelerar el entrenamiento pero causar inestabilidad"
            )

            # Explicación del learning rate seleccionado
            if learning_rate == 0.001:
                st.caption("🟢 **Conservador**: Aprendizaje lento pero estable")
            elif learning_rate == 0.01:
                st.caption(
                    "🟡 **Moderado**: Buen balance velocidad/estabilidad")
            elif learning_rate == 0.1:
                st.caption("🟠 **Agresivo**: Rápido pero puede ser inestable")
            else:
                st.caption("🔴 **Muy Alto**: Solo para casos especiales")

        with col2:
            st.markdown("#### 🔄 **Épocas**")
            # Sugerir épocas basado en tamaño del dataset
            suggested_epochs = min(200, max(50, dataset_size // 10))
            epochs = st.slider(
                "Épocas",
                min_value=10, max_value=500, value=min(100, suggested_epochs), step=10,
                help=f"💡 Sugerencia para tu dataset: ~{suggested_epochs} épocas"
            )

            # Explicación sobre las épocas
            if epochs < 50:
                st.caption(
                    "🟡 **Pocas épocas**: Puede que no aprenda completamente")
            elif epochs <= 150:
                st.caption("🟢 **Épocas adecuadas**: Buen balance")
            else:
                st.caption("🟠 **Muchas épocas**: Monitorea el sobreajuste")

        with col3:
            st.markdown("#### 🎯 **Validación**")
            validation_split = st.slider(
                "% Datos de Validación",
                min_value=10, max_value=40, value=20,
                help="💡 20% es estándar. Más datos = mejor validación, menos datos para entrenar"
            )

            # Calcular tamaños efectivos
            # 80% del total para entrenamiento
            train_size = int(
                dataset_size * (100 - validation_split) / 100 * 0.8)
            val_size = int(dataset_size * validation_split / 100)
            test_size = dataset_size - train_size - val_size

            st.caption(
                f"📊 **Distribución**: Train={train_size}, Val={val_size}, Test={test_size}")

        # Configuración avanzada con explicaciones detalladas
        with st.expander("⚙️ Configuración Avanzada - Técnicas para Mejorar el Entrenamiento", expanded=False):
            st.markdown("#### 🛡️ Técnicas de Regularización y Optimización")

            col4, col5 = st.columns(2)

            with col4:
                st.markdown("##### 🛑 Early Stopping")
                early_stopping = st.checkbox(
                    "Activar Parada Temprana",
                    value=True,
                    help="💡 Recomendado: Evita sobreajuste parando cuando la validación no mejora"
                )

                if early_stopping:
                    st.success(
                        "✅ **Early Stopping activado**: La red parará automáticamente cuando deje de mejorar")
                    patience = st.slider(
                        "Paciencia (épocas)",
                        min_value=5, max_value=50, value=10,
                        help="Épocas a esperar sin mejora antes de parar. Más paciencia = más oportunidades de mejorar"
                    )

                    if patience <= 5:
                        st.caption(
                            "🔴 **Impatiente**: Para rápido, puede interrumpir mejoras tardías")
                    elif patience <= 15:
                        st.caption("🟢 **Balanceado**: Buen equilibrio")
                    else:
                        st.caption(
                            "🟡 **Paciente**: Da muchas oportunidades, pero puede sobreajustar")
                else:
                    st.warning(
                        "⚠️ **Sin Early Stopping**: La red entrenará todas las épocas. Riesgo de sobreajuste.")

            with col5:
                st.markdown("##### 📉 Learning Rate Scheduler")
                reduce_lr = st.checkbox(
                    "Reducir Learning Rate Automáticamente",
                    value=True,
                    help="💡 Recomendado: Reduce la tasa de aprendizaje cuando no mejora"
                )

                if reduce_lr:
                    st.success(
                        "✅ **Scheduler activado**: La tasa de aprendizaje se reducirá automáticamente")
                    lr_factor = st.slider(
                        "Factor de Reducción",
                        min_value=0.1, max_value=0.9, value=0.5,
                        help="Factor por el que se multiplica la tasa. 0.5 = reduce a la mitad"
                    )

                    if lr_factor <= 0.3:
                        st.caption(
                            "🔴 **Reducción agresiva**: Cambios dramáticos")
                    elif lr_factor <= 0.7:
                        st.caption("🟢 **Reducción moderada**: Recomendado")
                    else:
                        st.caption("🟡 **Reducción suave**: Cambios graduales")
                else:
                    st.info(
                        "ℹ️ **Learning rate fijo**: Se mantendrá constante durante todo el entrenamiento")

            # Explicación sobre las técnicas
            st.markdown("---")
            st.markdown("#### 📚 ¿Por qué usar estas técnicas?")
            st.markdown("""
            - **Early Stopping**: Evita que la red memorice los datos (sobreajuste) parando cuando la performance en validación deja de mejorar
            - **Learning Rate Reduction**: Permite un ajuste fino hacia el final del entrenamiento cuando se está cerca del óptimo
            - **Combinadas**: Estas técnicas trabajan juntas para lograr el mejor modelo posible automáticamente
            """)

        # Botón de entrenamiento con explicación
        st.markdown("### 🚀 Iniciar Entrenamiento")
        st.markdown(
            "**¿Todo listo?** Tu red está configurada y lista para aprender de los datos.")

        if st.button("🧠 Entrenar Red Neuronal", type="primary", use_container_width=True):
            with st.spinner("🧠 Entrenando la red neuronal... Esto puede tomar unos minutos."):
                # Preparar datos
                df = st.session_state.nn_df
                target_col = st.session_state.nn_target_col
                task_type = st.session_state.nn_task_type

                try:
                    # Mostrar progreso del entrenamiento con pasos
                    progress_container = st.empty()

                    # Paso 1: Preparando datos
                    with progress_container.container():
                        st.info(
                            "🔄 **Paso 1/4**: Preparando y dividiendo los datos...")

                    # Llamar función de entrenamiento con callback de progreso
                    def update_progress(step, message):
                        with progress_container.container():
                            if step == 2:
                                st.info(f"🧠 **Paso {step}/4**: {message}")
                            elif step == 3:
                                st.info(f"⚙️ **Paso {step}/4**: {message}")
                            elif step == 4:
                                st.info(f"🚀 **Paso {step}/4**: {message}")
                            else:
                                st.info(f"🔄 **Paso {step}/4**: {message}")

                    # Entrenar el modelo con callback de progreso
                    model, history, X_test, y_test, scaler, label_encoder = train_neural_network(
                        df, target_col, st.session_state.nn_config,
                        learning_rate, epochs, validation_split/100,
                        early_stopping, patience if early_stopping else None,
                        reduce_lr, lr_factor if reduce_lr else None,
                        progress_callback=update_progress
                    )

                    # Guardar resultados
                    st.session_state.nn_model = model
                    st.session_state.nn_history = history
                    st.session_state.nn_test_data = (X_test, y_test)
                    st.session_state.nn_scaler = scaler
                    st.session_state.nn_label_encoder = label_encoder
                    st.session_state.model_trained_nn = True

                    # Limpiar el progreso y mostrar finalización
                    with progress_container.container():
                        if model is not None:
                            st.success(
                                "✅ **¡Entrenamiento completado!** Red neuronal lista para usar")

                    if model is not None:
                        st.success("🎉 ¡Red neuronal entrenada exitosamente!")

                except Exception as e:
                    # Limpiar el progreso en caso de error
                    with progress_container.container():
                        st.error("❌ **Error durante el entrenamiento**")

                    st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                    st.info(
                        "Intenta ajustar los parámetros o verificar el dataset.")

        # Botones de navegación
        if st.session_state.get('model_trained_nn', False):
            navbar("active_tab_nn", "Volver a Arquitectura", "Ver Evaluación")
        else:
            navbar("active_tab_nn", "Volver a Arquitectura", None)

    ###########################################
    #     Pestaña de Evaluación.              #
    ###########################################
    elif st.session_state.active_tab_nn == 3:
        st.header("📈 Evaluación del Modelo")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
            if st.button("🔙 Ir a Entrenamiento"):
                st.session_state.active_tab_nn = 2
                st.rerun()
        else:
            model = st.session_state.nn_model
            X_test = st.session_state.nn_test_data[0]
            y_test = st.session_state.nn_test_data[1]
            class_names = st.session_state.nn_class_names
            task_type = st.session_state.nn_config.get(
                'task_type', 'Clasificación')
            # === Predicciones y preparación de etiquetas para evaluación ===
            raw_pred = model.predict(X_test, verbose=0)

            if task_type == "Clasificación":
                # Determinar formato de y_test (one-hot vs etiquetas)
                y_test_arr = np.asarray(y_test)
                one_hot = (y_test_arr.ndim > 1 and y_test_arr.shape[1] > 1)

                # Procesar predicciones según forma de salida del modelo
                if raw_pred.ndim > 1 and raw_pred.shape[1] > 1:
                    # Salida multiclase (softmax / linear / tanh etc.)
                    y_pred_classes = np.argmax(raw_pred, axis=1)
                else:
                    # Salida binaria (1 neurona, normalmente sigmoid)
                    y_pred_classes = (raw_pred.ravel() > 0.5).astype(int)

                # Procesar etiquetas reales
                if one_hot:
                    y_test_classes = np.argmax(y_test_arr, axis=1)
                else:
                    # Ya son etiquetas enteras (binaria o sparse multiclase)
                    y_test_classes = y_test_arr.ravel()

                # Guardar para otras pestañas
                st.session_state.nn_y_pred = y_pred_classes

                # Primero evaluar con el formato original (para model.evaluate)
                evaluate_nn(model, X_test, y_test, task_type)
                # Mostrar evaluación detallada con clases procesadas
                show_detailed_evaluation(
                    y_test_classes, y_pred_classes, class_names, task_type)
            else:
                # Regresión
                y_pred_reg = raw_pred.ravel()
                st.session_state.nn_y_pred = y_pred_reg
                evaluate_nn(model, X_test, y_test, task_type)
                show_detailed_evaluation(
                    y_test, y_pred_reg, class_names, task_type)

            history = st.session_state.nn_history
            config = st.session_state.nn_config
            neural_network_diagnostics(history, config)

            navbar("active_tab_nn", "Volver a Entrenamiento", "Ver Visualización")

    ###########################################
    #     Pestaña de Visualización            #
    ###########################################
    elif st.session_state.active_tab_nn == 4:
        st.header("🎯 Visualizaciones")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
        else:
            st.info("""
            🎓 **Visualizaciones de Redes Neuronales:**
            - **Historial**: Evolución del aprendizaje 
            - **Pesos**: Lo que aprendió cada neurona
            - **Superficie**: Cómo separa las clases (2D) 
            - **Capas**: Activaciones internas
            """)

            # Usar botones para seleccionar el tipo de visualización
            viz_options = [
                ("📊 Historial", "Historial", "viz_history"),
                ("🧠 Análisis de Pesos", "Pesos", "viz_weights"),
                ("🎯 Superficie de Decisión", "Superficie", "viz_surface"),
                ("📉 Análisis de Activaciones", "Activaciones", "viz_activations"),
            ]

            viz_type = create_button_panel(viz_options)

            if viz_type == "Historial":
                show_training_history_tab()
            elif viz_type == "Pesos":
                show_weights_analysis_tab()
            elif viz_type == "Superficie":
                show_decision_surface_tab()
            elif viz_type == "Activaciones":
                show_layer_activations_tab()

        navbar("active_tab_nn", "Volver a Evaluación", "Ver Predicciones")

    ###########################################
    #     Pestaña de Predicciones             #
    ###########################################
    elif st.session_state.active_tab_nn == 5:
        st.header("🔮 Predicciones")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
        else:
            # Usar los datos ORIGINALES (antes de escalar) para preservar tipos/categorías.
            # nn_test_data[0] es un array escalado que pierde información categórica y provoca sliders.
            try:
                if 'nn_df' in st.session_state and 'nn_target_col' in st.session_state:
                    X_original_for_ui = st.session_state.nn_df.drop(
                        columns=[st.session_state.nn_target_col])
                else:
                    # Fallback: si algo falla, usar el array test (mantiene comportamiento previo)
                    X_original_for_ui = st.session_state.nn_test_data[0]
            except Exception:
                X_original_for_ui = st.session_state.nn_test_data[0]

            create_prediction_interface(
                st.session_state.nn_model,
                st.session_state.nn_feature_names,
                st.session_state.nn_class_names,
                st.session_state.get('nn_task_type', 'Clasificación'),
                # Pasar datos originales para que el tipo de feature se detecte correctamente
                X_original_for_ui,
                # Pasar nombre del dataset para metadata
                st.session_state.get('selected_dataset', 'Titanic')
            )

    ###########################################
    #     Pestaña de Exportar.                #
    ###########################################
    elif st.session_state.active_tab_nn == 6:
        st.header("💾 Exportar Modelo")
        if not st.session_state.get('model_trained_nn', False):
            st.warning("⚠️ Primero debes entrenar un modelo.")
        else:
            show_neural_network_export()


# ===== FUNCIONES PARA REDES NEURONALES =====
