import base64
import pickle
import io
import streamlit as st

from sklearn.tree import export_text

from utils import show_code_with_download, export_model_pickle
from algorithms.code_examples import (
    generate_tree_model_export_code,
    generate_regression_code,
)


def export_tree_as_python_code(model, feature_names):
    """
    Exporta el árbol de decisión como una función Python con if-else statements.
    """
    def recurse(node, depth=0):
        indent = "    " * depth

        # Si es un nodo hoja
        if model.tree_.children_left[node] == model.tree_.children_right[node]:
            value = model.tree_.value[node][0]
            if hasattr(model, 'classes_'):  # Clasificación
                class_idx = int(value.argmax())
                class_name = model.classes_[class_idx] if hasattr(
                    model, 'classes_') else f"Clase_{class_idx}"
                probability = value[class_idx] / value.sum()
                return f'{indent}return "{class_name}"  # Probabilidad: {probability:.3f}'
            else:  # Regresión
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
