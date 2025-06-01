"""
Este módulo contiene funciones de utilidad general para la aplicación MLTutor.
Incluye funciones para generar enlaces de descarga, exportar modelos y otras utilidades comunes.
"""

import base64
import io
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Función para generar enlace de descarga de imagen


def get_image_download_link(fig, filename, link_text):
    """
    Genera un enlace HTML para descargar una figura de matplotlib como imagen PNG.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figura a descargar
    filename : str
        Nombre del archivo sin extensión
    link_text : str
        Texto para mostrar en el enlace

    Returns:
    --------
    html : str
        Código HTML con el enlace de descarga
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}.png">{link_text}</a>'
    return href

# Funciones para exportar modelos


def export_model_pickle(model):
    """
    Exporta un modelo a formato pickle.

    Parameters:
    -----------
    model : object
        Modelo a exportar

    Returns:
    --------
    pickle_data : bytes
        Datos serializados del modelo
    """
    return pickle.dumps(model)


def export_model_onnx(model, num_features):
    """
    Exporta un modelo a formato ONNX.

    Parameters:
    -----------
    model : object
        Modelo a exportar
    num_features : int
        Número de características del modelo

    Returns:
    --------
    onnx_data : bytes
        Datos del modelo en formato ONNX, o None si hay error
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Configuración inicial
        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        # Convertir el modelo
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Serializar a bytes
        return onnx_model.SerializeToString()
    except Exception as e:
        print(f"Error al exportar a ONNX: {str(e)}")
        return None


def generate_model_code(model, tree_type, max_depth, min_samples_split, criterion, feature_names, class_names=None):
    """
    Genera código Python para recrear un modelo de árbol de decisión.

    Parameters:
    -----------
    model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    tree_type : str
        Tipo de árbol ('Clasificación' o 'Regresión')
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio de división del árbol
    feature_names : list
        Nombres de las características
    class_names : list, optional
        Nombres de las clases (solo para clasificación)

    Returns:
    --------
    code : str
        Código Python para recrear y utilizar el modelo
    """
    # Generar el código
    is_classifier = tree_type == "Clasificación"
    model_type = "DecisionTreeClassifier" if is_classifier else "DecisionTreeRegressor"

    code = f"""# Código para recrear el modelo de árbol de decisión
import numpy as np
from sklearn.tree import {model_type}

# Definir los nombres de las características
feature_names = {feature_names}

"""

    if is_classifier and class_names:
        code += f"# Definir los nombres de las clases\nclass_names = {class_names}\n\n"

    code += f"""# Crear el modelo
model = {model_type}(
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    criterion="{criterion}",
    random_state=42
)

# El modelo ya está entrenado con estos parámetros
# Para usarlo, debes entrenarlo con tus datos:
# model.fit(X_train, y_train)

# Ejemplo de uso para predicción:
# Para un nuevo ejemplo con las características:
# ejemplo = [[{', '.join(['0.0' for _ in feature_names])}]]
# predicción = model.predict(ejemplo)

"""

    if is_classifier:
        code += """# Para obtener probabilidades:
# probabilidades = model.predict_proba(ejemplo)
"""

    return code


def create_info_box(content):
    """
    Crea una caja de información con estilo personalizado.

    Parameters:
    -----------
    content : str
        Contenido de la caja de información

    Returns:
    --------
    html : str
        Código HTML con la caja de información
    """
    return f'<div class="info-box">{content}</div>'


def format_number(value, precision=4):
    """
    Formatea un número para su visualización.

    Parameters:
    -----------
    value : float
        Valor a formatear
    precision : int, default=4
        Número de decimales

    Returns:
    --------
    str
        Número formateado
    """
    if isinstance(value, (int, float)):
        if value == int(value):
            return str(int(value))
        else:
            return f"{value:.{precision}f}"
    return str(value)
