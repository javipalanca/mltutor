import base64
import pickle
import io
import streamlit as st

from sklearn.tree import export_text

from utils import show_code_with_download, export_model_pickle
from algorithms.code_examples import (
    generate_tree_model_export_code,
    generate_regression_code,
    generate_neural_network_code,
    LOAD_NN,
)
from viz.nn import calculate_network_parameters
from ui import create_button_panel
import traceback
import tempfile
import zipfile
import os


def _zip_dir(src_dir: str) -> bytes:
    """Empaqueta recursivamente un directorio en un ZIP y devuelve los bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                fp = os.path.join(root, f)
                arc = os.path.relpath(fp, src_dir)
                zf.write(fp, arc)
    buf.seek(0)
    return buf.getvalue()


def convert_keras_to_tflite(model) -> bytes:
    """Convierte un modelo Keras a TFLite.

    Con Keras 3 (TF >= 2.16) TFLiteConverter.from_keras_model falla con
    "Functional object has no attribute _get_save_spec"; en ese caso se
    exporta primero a SavedModel y se convierte desde el directorio.
    """
    import tensorflow as tf

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()
    except AttributeError:
        with tempfile.TemporaryDirectory() as tmpdir:
            model.export(tmpdir)
            converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
            return converter.convert()


def export_saved_model_as_zip(model, safe_mode: bool):
    """
    Exporta un modelo Keras como SavedModel comprimido en ZIP.
    safe_mode=True intenta desactivar traces (save_traces=False) para evitar problemas
    con capas Lambda o funciones no serializables.
    """
    # Asegurar que el modelo esté construido
    try:
        if not getattr(model, 'built', False) and getattr(model, 'input_shape', None):
            import numpy as np
            dummy_shape = [
                1] + [d if d is not None else 1 for d in model.input_shape[1:]]
            _ = model.predict(np.zeros(dummy_shape), verbose=0)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, 'saved_model')
        import inspect
        # Determinar si el método save admite save_traces
        try:
            save_func = model.save
            sig = inspect.signature(save_func)
            has_save_traces = 'save_traces' in sig.parameters
        except Exception:
            has_save_traces = False
        try:
            if has_save_traces:
                model.save(model_dir, include_optimizer=True,
                           save_traces=not safe_mode)
            else:
                model.save(model_dir, include_optimizer=True)
        except (TypeError, ValueError):
            # Reintentar usando API moderna si model.save requiere extensión (Keras 3)
            try:
                import tensorflow as tf
                if hasattr(model, 'export'):
                    model.export(model_dir)
                else:
                    tf.saved_model.save(model, model_dir)
            except Exception as e2:
                raise RuntimeError(
                    f"Fallo exportando SavedModel (alternativas). Error: {e2!r}") from e2
        return _zip_dir(model_dir)


def export_tree_as_python_code(model, feature_names):
    """Exporta el árbol de decisión como una función Python con if-else."""
    def recurse(node, depth=0):
        indent = "    " * depth
        # Nodo hoja
        if model.tree_.children_left[node] == model.tree_.children_right[node]:
            value = model.tree_.value[node][0]
            if hasattr(model, 'classes_'):
                class_idx = int(value.argmax())
                class_name = model.classes_[class_idx] if hasattr(
                    model, 'classes_') else f"Clase_{class_idx}"
                probability = value[class_idx] / value.sum()
                return f'{indent}return "{class_name}"  # Probabilidad: {probability:.3f}'
            else:
                return f'{indent}return {value[0]:.6f}'
        # Nodo interno
        feature_idx = model.tree_.feature[node]
        threshold = model.tree_.threshold[node]
        feature_name = feature_names[
            feature_idx] if feature_names else f"feature_{feature_idx}"
        left_child = model.tree_.children_left[node]
        right_child = model.tree_.children_right[node]
        code = f'{indent}if {feature_name} <= {threshold:.6f}:\n'
        code += recurse(left_child, depth + 1) + '\n'
        code += f'{indent}else:\n'
        code += recurse(right_child, depth + 1)
        return code

    task_type = "clasificación" if hasattr(model, 'classes_') else "regresión"
    function_name = "predecir_clase" if hasattr(
        model, 'classes_') else "predecir_valor"
    params_doc = "\n".join(
        [f"\t{name}: float - {name}" for name in feature_names])
    code = f'''def {function_name}({", ".join(feature_names)}):
    """
    Función generada automáticamente que implementa el árbol de decisión entrenado.

    Parameters:
    {params_doc}

    Returns:
    {'    str - La clase predicha' if hasattr(model, 'classes_') else '    float - El valor predicho'}

    Árbol entrenado para {task_type} con {len(feature_names)} características.
    Profundidad máxima: {model.tree_.max_depth}
    Número de nodos: {model.tree_.node_count}
    """
{recurse(0)}

# Ejemplo de uso:
# resultado = {function_name}({", ".join([f"{name}=valor_{i+1}" for i, name in enumerate(feature_names)])})
# print(f"Predicción: {{resultado}}")
'''
    return code

    # Nodo interno
    feature_idx = model.tree_.feature[node]
    threshold = model.tree_.threshold[node]
    feature_name = feature_names[
        feature_idx] if feature_names else f"feature_{feature_idx}"

    left_child = model.tree_.children_left[node]
    right_child = model.tree_.children_right[node]

    code = f'{indent}if {feature_name} <= {threshold:.6f}:\n'
    code += recurse(left_child, depth + 1) + '\n'
    code += f'{indent}else:\n'
    code += recurse(right_child, depth + 1)

    return code

    # Generar el código completo
    task_type = "clasificación" if hasattr(model, 'classes_') else "regresión"
    function_name = "predecir_clase" if hasattr(
        model, 'classes_') else "predecir_valor"
    params_doc = "\n".join(
        [f"\t{name}: float - {name}" for name in feature_names])

    code = f'''def {function_name}({", ".join(feature_names)}):
    """
    Función generada automáticamente que implementa el árbol de decisión entrenado.

    Parameters:
    {params_doc}

    Returns:
    {'    str - La clase predicha' if hasattr(
        model, 'classes_') else '    float - El valor predicho'}

    Árbol entrenado para {task_type} con {len(feature_names)} características.
    Profundidad máxima: {model.tree_.max_depth}
    Número de nodos: {model.tree_.node_count}
    """
{recurse(0)}

# Ejemplo de uso:
# resultado = {function_name}({", ".join([f"{name}=valor_{i+1}" for i, name in enumerate(feature_names)])})
# print(f"Predicción: {{resultado}}")
'''

    return code


def display_tree_export_options(model, feature_names, class_names, task_type, max_depth, min_samples_split, criterion):
    """
    Muestra opciones para exportar el modelo usando solo librerías de terceros.
    """

    # Información del modelo antes de las opciones
    model_info = {
        'task_type': task_type,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'criterion': criterion,
        'n_features': len(feature_names),
        'n_nodes': model.tree_.node_count,
        'n_leaves': model.tree_.n_leaves
    }

    # Opciones de exportación
    st.markdown("#### Formatos de Exportación")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Exportar como pickle
        if st.button("📦 Exportar como Pickle", use_container_width=True):
            # Crear objeto para serializar
            model_data = {
                'model': model,
                'feature_names': feature_names,
                'class_names': class_names,
                'task_type': task_type,
                'model_info': model_info
            }

            # Serializar
            buffer = io.BytesIO()
            pickle.dump(model_data, buffer)
            buffer.seek(0)

            st.download_button(
                label="Descargar Modelo (.pkl)",
                data=buffer.getvalue(),
                file_name=f"decision_tree_model.pkl",
                mime="application/octet-stream"
            )

    with col2:
        # Exportar reglas como texto
        if st.button("📝 Exportar Reglas como Texto", use_container_width=True):
            tree_rules = export_text(
                model,
                feature_names=feature_names
            )

            st.download_button(
                label="Descargar Reglas (.txt)",
                data=tree_rules,
                file_name="decision_tree_rules.txt",
                mime="text/plain"
            )

    python_tree_code = export_tree_as_python_code(model, feature_names)

    with col3:
        # Nueva opción: Exportar como código Python con if-else
        if st.button("🐍 Exportar como Código Python", use_container_width=True):
            st.download_button(
                label="Descargar Código Python (.py)",
                data=python_tree_code,
                file_name="arbol_decision.py",
                mime="text/plain"
            )

    # Vista previa del código Python
    st.markdown("#### 👀 Vista Previa del Código Python")
    with st.expander("Ver código Python del árbol", expanded=False):
        st.code(python_tree_code, language='python')

        # Información adicional sobre el código generado
        st.info(f"""
        **Información del código generado:**
        - 🌳 **Nodos totales:** {model.tree_.node_count}
        - 🍃 **Nodos hoja:** {model.tree_.n_leaves}
        - 📏 **Profundidad máxima:** {model.tree_.max_depth}
        - 📊 **Características:** {len(feature_names)} ({', '.join(feature_names[:3])}{'...' if len(feature_names) > 3 else ''})
        - 🎯 **Tipo de tarea:** {task_type}

        Este código es completamente **autocontenido** y no requiere scikit-learn para ejecutarse.
        """)

    # Código Python para usar el modelo
    python_code = generate_tree_model_export_code(feature_names)

    show_code_with_download(
        python_code, "Código para Usar el Modelo", "usar_modelo.py")


def display_model_export_options(model):
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

            code = generate_regression_code()

            st.code(code, language="python")

            # Download button for the code
            st.download_button(
                label="📥 Descargar código",
                data=code,
                file_name=f"regression_code.py",
                mime="text/plain"
            )


def show_neural_network_export():
    """Permite exportar el modelo entrenado."""
    if 'nn_model' not in st.session_state or st.session_state.nn_model is None:
        st.warning(
            "⚠️ Primero debes entrenar un modelo en la pestaña 'Entrenamiento'")
        return

    try:
        import pickle
        import json
        from datetime import datetime

        model = st.session_state.nn_model
        scaler = st.session_state.nn_scaler
        label_encoder = st.session_state.nn_label_encoder
        config = st.session_state.nn_config

        # Información del modelo
        st.subheader("ℹ️ Información del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Arquitectura:**
            - Tipo: {config['task_type']}
            - Capas: {len(config['architecture'])}
            - Neuronas: {config['architecture']}
            - Activación: {config['activation']}
            - Optimizador: {config['optimizer']}
            """)

        with col2:
            total_params = calculate_network_parameters(config['architecture'])
            st.info(f"""
            **Parámetros:**
            - Total: {total_params:,}
            - Dropout: {config['dropout_rate']}
            - Batch size: {config['batch_size']}
            - Fecha entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)

        # Opciones de exportación
        st.subheader("📁 Opciones de Exportación")

        # Usar botones para seleccionar el tipo de visualización
        viz_options = [
            ("🤖 Modelo TensorFlow", "Tensorflow", "viz_tf"),
            ("📊 Modelo Completo", "Completo", "viz_complete"),
            ("📝 Código Python", "Python", "viz_python"),
            ("📋 Metadatos", "Metadatos", "viz_metadata")
        ]

        viz_type = create_button_panel(viz_options)

        if viz_type == "Tensorflow":
            st.markdown("**Exportar solo el modelo de TensorFlow:**")

            format_option = st.radio(
                "Formato:",
                ["SavedModel (.pb)", "HDF5 (.h5)",
                 "TensorFlow Lite (.tflite)"],
                key="nn_export_format"
            )
            safe_mode = st.checkbox(
                "Modo seguro (desactivar traces)",
                value=True,
                help="Útil si hay capas Lambda / funciones personalizadas que impiden la serialización"
            )

            if st.button("💾 Exportar Modelo TensorFlow", type="primary"):
                try:
                    if format_option == "SavedModel (.pb)":
                        zip_bytes = export_saved_model_as_zip(model, safe_mode)
                        st.download_button(
                            label="📥 Descargar SavedModel (ZIP)",
                            data=zip_bytes,
                            file_name="neural_network_savedmodel.zip",
                            mime="application/zip"
                        )
                        st.success("✅ SavedModel generado")
                    elif format_option == "HDF5 (.h5)":
                        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                            import inspect
                            try:
                                sig = inspect.signature(model.save)
                                has_save_traces = 'save_traces' in sig.parameters
                            except Exception:
                                has_save_traces = False
                            try:
                                if has_save_traces:
                                    model.save(
                                        tmp_file.name, include_optimizer=True, save_traces=not safe_mode)
                                else:
                                    model.save(tmp_file.name,
                                               include_optimizer=True)
                            except (TypeError, ValueError):
                                model.save(tmp_file.name,
                                           include_optimizer=True)
                            except Exception:
                                if not safe_mode and has_save_traces:
                                    try:
                                        model.save(
                                            tmp_file.name, include_optimizer=True, save_traces=False)
                                    except Exception:
                                        raise
                                else:
                                    raise
                            with open(tmp_file.name, 'rb') as f:
                                model_data = f.read()
                        st.download_button(
                            label="📥 Descargar Modelo HDF5",
                            data=model_data,
                            file_name="neural_network_model.h5",
                            mime="application/octet-stream"
                        )
                        st.success("✅ HDF5 generado")
                    elif format_option == "TensorFlow Lite (.tflite)":
                        tflite_model = convert_keras_to_tflite(model)
                        st.download_button(
                            label="📥 Descargar Modelo TFLite",
                            data=tflite_model,
                            file_name="neural_network_model.tflite",
                            mime="application/octet-stream"
                        )
                        st.success("✅ TFLite generado")
                        st.info(
                            "💡 TensorFlow Lite es ideal para dispositivos móviles y embebidos")
                except Exception as e:
                    st.error(f"Error exportando modelo: {e}")
                    with st.expander("Detalles del error"):
                        st.code("".join(traceback.format_exc()),
                                language="text")
                        layer_info = [f"{idx}: {layer.__class__.__name__} (name={layer.name})" for idx, layer in enumerate(
                            getattr(model, 'layers', []))]
                        st.code("\n".join(layer_info), language="text")

        elif viz_type == "Completo":
            st.markdown("**Exportar modelo completo con preprocesadores:**")
            st.info("Incluye el modelo, scaler, label encoder y configuración")

            if st.button("💾 Exportar Modelo Completo", type="primary"):
                try:
                    # Crear diccionario con todos los componentes
                    complete_model = {
                        'model': model,
                        'scaler': scaler,
                        'label_encoder': label_encoder,
                        'config': config,
                        'feature_names': st.session_state.get('nn_feature_names', []),
                        'export_date': datetime.now().isoformat(),
                        'version': '1.0'
                    }

                    # Serializar con pickle
                    model_data = pickle.dumps(complete_model)

                    st.download_button(
                        label="📥 Descargar Modelo Completo",
                        data=model_data,
                        file_name="neural_network_complete.pkl",
                        mime="application/octet-stream"
                    )

                    st.success("✅ Modelo completo exportado")
                    st.info(
                        "💡 Este archivo contiene todo lo necesario para hacer predicciones")

                except Exception as e:
                    st.error(f"Error exportando modelo completo: {str(e)}")

            # Mostrar código de ejemplo para cargar
            st.markdown("**Código para cargar el modelo:**")

            load_code = LOAD_NN

            st.code(load_code, language='python')

        elif viz_type == "Python":
            st.markdown("**Generar código Python independiente:**")

            if st.button("📝 Generar Código", type="primary"):
                try:
                    # Obtener pesos del modelo
                    weights_data = []
                    for layer in model.layers:
                        if hasattr(layer, 'get_weights') and layer.get_weights():
                            weights_data.append(layer.get_weights())

                    # Generar código
                    code = generate_neural_network_code(config, label_encoder)

                    st.code(code, language='python')

                    # Botón para descargar el código
                    st.download_button(
                        label="📥 Descargar Código",
                        data=code,
                        file_name="neural_network_predictor.py",
                        mime="text/plain"
                    )

                    st.warning(
                        "⚠️ El código generado es una plantilla. Debes implementar los pesos específicos del modelo entrenado.")

                except Exception as e:
                    st.error(f"Error generando código: {str(e)}")

        elif viz_type == "Metadatos":
            st.markdown("**Exportar metadatos del modelo:**")

            # Preparar metadatos
            if 'nn_history' in st.session_state:
                history = st.session_state.nn_history
                final_metrics = {
                    'final_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
                    'epochs_trained': len(history.history['loss'])
                }

                if config['task_type'] == 'Clasificación' and 'accuracy' in history.history:
                    final_metrics['final_accuracy'] = float(
                        history.history['accuracy'][-1])
                    if 'val_accuracy' in history.history:
                        final_metrics['final_val_accuracy'] = float(
                            history.history['val_accuracy'][-1])
            else:
                final_metrics = {}

            metadata = {
                'model_info': {
                    'type': 'Neural Network',
                    'task_type': config['task_type'],
                    'architecture': config['architecture'],
                    'total_parameters': calculate_network_parameters(config['architecture']),
                    'activation_function': config['activation'],
                    'output_activation': config['output_activation'],
                    'optimizer': config['optimizer'],
                    'dropout_rate': config['dropout_rate'],
                    'batch_size': config['batch_size']
                },
                'training_info': final_metrics,
                'data_info': {
                    'feature_names': st.session_state.get('nn_feature_names', []),
                    'target_column': st.session_state.get('nn_target_col', ''),
                    'num_features': config['input_size'],
                    'num_classes': config['output_size'] if config['task_type'] == 'Clasificación' else 1
                },
                'export_info': {
                    'export_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'framework': 'TensorFlow/Keras'
                }
            }

            # Mostrar metadatos
            st.json(metadata)

            # Botón para descargar metadatos
            metadata_json = json.dumps(metadata, indent=2)

            st.download_button(
                label="📥 Descargar Metadatos",
                data=metadata_json,
                file_name="neural_network_metadata.json",
                mime="application/json"
            )

        # Información adicional
        st.subheader("💡 Información Adicional")

        st.info("""
        **Recomendaciones para el uso del modelo:**

        1. **Modelo TensorFlow**: Ideal para integrar en aplicaciones que ya usan TensorFlow
        2. **Modelo Completo**: Incluye preprocesadores, perfecto para producción
        3. **Código Python**: Para entender la implementación o crear versiones optimizadas
        4. **Metadatos**: Para documentación y seguimiento del modelo

        **Consideraciones de versión:**
        - TensorFlow versión utilizada en entrenamiento
        - Compatibilidad con versiones futuras
        - Dependencias del entorno de producción
        """)

    except Exception as e:
        st.error(f"Error en la exportación: {str(e)}")
        st.info("Asegúrate de que el modelo esté entrenado correctamente.")
