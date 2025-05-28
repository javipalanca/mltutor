import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import seaborn as sns
import io
import base64
from sklearn import tree
import textwrap
import pickle

# Comprobar si onnx está disponible
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="MLTutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para guardar estado global
if 'tree_model' not in st.session_state:
    st.session_state.tree_model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'tree_type' not in st.session_state:
    st.session_state.tree_type = "Clasificación"
if 'fig_width' not in st.session_state:
    st.session_state.fig_width = 14
if 'fig_height' not in st.session_state:
    st.session_state.fig_height = 10.5
if 'fig_size' not in st.session_state:
    st.session_state.fig_size = 3
if 'show_values' not in st.session_state:
    st.session_state.show_values = True
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'dataset_option' not in st.session_state:
    st.session_state.dataset_option = "Iris (clasificación de flores)"
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'criterion' not in st.session_state:
    st.session_state.criterion = "gini"
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 3
if 'min_samples_split' not in st.session_state:
    st.session_state.min_samples_split = 2
if 'test_size' not in st.session_state:
    st.session_state.test_size = 0.3

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .inference-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #FFE082;
    }
    .code-box {
        background-color: #ECEFF1;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        font-family: monospace;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        color: #666;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
        border-bottom: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown("<h1 class='main-header'>MLTutor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Plataforma de aprendizaje de Machine Learning</p>",
            unsafe_allow_html=True)

# Función para cargar datos


def load_data(dataset_name):
    if "Iris" in dataset_name:
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="species")
        class_names = data.target_names
        return X, y, data.feature_names, class_names, "Dataset Iris: 150 muestras, 4 características, 3 clases de flores"

    elif "Vino" in dataset_name:
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="wine_type")
        class_names = data.target_names
        return X, y, data.feature_names, class_names, "Dataset Vino: 178 muestras, 13 características, 3 tipos de vino"

    elif "Cáncer" in dataset_name:
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="malignant")
        class_names = data.target_names
        return X, y, data.feature_names, class_names, "Dataset Cáncer: 569 muestras, 30 características, diagnóstico binario"

# Función para generar una imagen descargable


def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">{text}</a>'
    return href

# Función para generar enlaces de descarga genéricos


def get_download_link(content, filename, text, mime="text/plain"):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{text}</a>'
    return href

# Función para generar código Python descargable


def get_code_download_link(code, filename, text):
    return get_download_link(code.encode(), f"{filename}.py", text, "text/x-python")

# Función para exportar el modelo en formato pickle


def export_model_pickle(model):
    output = io.BytesIO()
    pickle.dump(model, output)
    output.seek(0)
    return output.getvalue()

# Función para exportar el modelo en formato ONNX (si está disponible)


def export_model_onnx(model, feature_count):
    if not ONNX_AVAILABLE:
        return None

    # Definir el tipo de entrada
    initial_type = [('float_input', FloatTensorType([None, feature_count]))]

    # Convertir el modelo a ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Guardar en un buffer
    output = io.BytesIO()
    output.write(onnx_model.SerializeToString())
    output.seek(0)
    return output.getvalue()

# Función para generar código del modelo


def generate_model_code(tree_model, tree_type, max_depth, min_samples_split, criterion, feature_names, class_names=None):
    code = f"""# Código para crear y entrenar un árbol de decisión
import numpy as np
import pandas as pd
from sklearn.tree import {'DecisionTreeClassifier' if tree_type == 'Clasificación' else 'DecisionTreeRegressor'}
from sklearn.model_selection import train_test_split
{'from sklearn.metrics import accuracy_score, confusion_matrix, classification_report' if tree_type == 'Clasificación' else 'from sklearn.metrics import mean_squared_error, r2_score'}
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Definir el modelo
modelo = {'DecisionTreeClassifier' if tree_type == 'Clasificación' else 'DecisionTreeRegressor'}(
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    criterion='{criterion}',
    random_state=42
)

# 1. Cargar datos
X = pd.read_csv('tus_datos.csv')  # Características
y = pd.read_csv('tus_etiquetas.csv')  # Variable objetivo

# 2. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar el modelo
modelo.fit(X_train, y_train)

# 4. Evaluar el modelo
if '{tree_type}' == 'Clasificación':
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {{accuracy:.4f}}")
    print("\\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
else:
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Error cuadrático medio: {{mse:.4f}}")
    print(f"R² Score: {{r2:.4f}}")

# 5. Visualizar el árbol
plt.figure(figsize=(15, 10))
plot_tree(modelo, 
         feature_names={feature_names},
         {'class_names=' + str(class_names) + ',' if class_names is not None and tree_type == 'Clasificación' else ''}
         filled=True, 
         rounded=True)
plt.savefig('arbol_decision.png', dpi=300, bbox_inches='tight')
plt.show()

# Características importantes
importancia = dict(zip({feature_names}, modelo.feature_importances_))
for caracteristica, valor in sorted(importancia.items(), key=lambda x: x[1], reverse=True):
    print(f"{{caracteristica}}: {{valor:.4f}}")

# Función para realizar predicciones con nuevos datos
def predecir(nuevos_datos):
    \"\"\"
    Realiza predicciones utilizando el modelo entrenado
    
    Parámetros:
    -----------
    nuevos_datos : array-like o DataFrame
        Datos con los que realizar la predicción. Debe tener las mismas características que los datos de entrenamiento.
        
    Retorna:
    --------
    array
        Predicciones del modelo
    \"\"\"
    return modelo.predict(nuevos_datos)

# Ejemplo de predicción con nuevos datos:
nuevos_datos = [[5.1, 3.5, 1.4, 0.2]]  # Sustituye con tus propios valores
prediccion = predecir(nuevos_datos)
print(f"Predicción: {{prediccion}}")
"""
    return code

# Función para mostrar el resultado de la evaluación detallada


def show_detailed_evaluation(y_test, y_pred, class_names, tree_type):
    if tree_type == "Clasificación":
        # Calcular métricas
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Métricas globales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Exactitud (Accuracy)", f"{report['accuracy']:.4f}",
                      help="Proporción total de predicciones correctas")
        with col2:
            st.metric("Precisión media", f"{report['weighted avg']['precision']:.4f}",
                      help="Media ponderada de la precisión de cada clase")
        with col3:
            st.metric("Exhaustividad media", f"{report['weighted avg']['recall']:.4f}",
                      help="Media ponderada de la exhaustividad de cada clase")
        with col4:
            st.metric("F1-Score medio", f"{report['weighted avg']['f1-score']:.4f}",
                      help="Media armónica de precisión y exhaustividad")

        # Métricas por clase
        st.markdown("### Métricas por clase")

        # Excluir filas avg y accuracy del dataframe
        report_by_class = report_df.drop(
            ['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=class_names, yticklabels=class_names)
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')
        ax_cm.set_title('Matriz de Confusión')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig_cm)
            st.markdown(get_image_download_link(
                fig_cm, "matriz_confusion", "📥 Descargar matriz de confusión"), unsafe_allow_html=True)

        with col2:
            st.dataframe(report_by_class.style.format({
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1-score': '{:.4f}',
                'support': '{:.0f}'
            }))

            with st.expander("📊 Explicación de métricas"):
                st.markdown("""
                - **Precisión**: De todas las muestras que se predijeron como clase X, ¿qué porcentaje eran realmente X?
                - **Exhaustividad (Recall)**: De todas las muestras que realmente son clase X, ¿qué porcentaje se predijo correctamente?
                - **F1-Score**: Media armónica de precisión y exhaustividad. Útil cuando las clases están desbalanceadas.
                - **Support**: Número de muestras de cada clase en el conjunto de prueba.
                """)

        # Visualización avanzada - Predicciones correctas e incorrectas
        st.markdown("### Visualización de Predicciones")

        # Crear dataframe con resultados
        results_df = pd.DataFrame({
            'Real': [class_names[x] for x in y_test],
            'Predicción': [class_names[x] for x in y_pred],
            'Correcto': y_test == y_pred
        })

        # Mostrar algunas muestras
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("#### Muestras correctamente clasificadas")
            correct_samples = results_df[results_df['Correcto']].head(10)
            st.dataframe(correct_samples, height=300)

        with col_viz2:
            st.markdown("#### Muestras incorrectamente clasificadas")
            incorrect_samples = results_df[~results_df['Correcto']].head(10)
            if len(incorrect_samples) > 0:
                st.dataframe(incorrect_samples, height=300)
            else:
                st.info("¡Todas las muestras fueron clasificadas correctamente!")

        # Gráfico de precisión por clase
        fig_prec, ax_prec = plt.subplots(figsize=(10, 5))
        prec_by_class = {
            class_name: report[class_name]['precision'] for class_name in class_names}
        sns.barplot(x=list(prec_by_class.keys()), y=list(
            prec_by_class.values()), ax=ax_prec)
        ax_prec.set_ylim(0, 1)
        ax_prec.set_title('Precisión por clase')
        ax_prec.set_ylabel('Precisión')
        ax_prec.set_xlabel('Clase')

        st.pyplot(fig_prec)
    else:
        # Métricas para regresión
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred)**2) / \
            np.sum((y_test - np.mean(y_test))**2)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Error Cuadrático Medio (MSE)", f"{mse:.4f}",
                      help="Promedio de los errores al cuadrado. Penaliza más los errores grandes.")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}",
                      help="Raíz cuadrada del MSE. En las mismas unidades que la variable objetivo.")
        with col3:
            st.metric("Error Absoluto Medio (MAE)", f"{mae:.4f}",
                      help="Promedio de los errores absolutos. Menos sensible a valores atípicos.")
        with col4:
            st.metric("R² Score", f"{r2:.4f}",
                      help="Proporción de la varianza explicada por el modelo. 1 es predicción perfecta.")

        # Gráfico de predicciones vs valores reales
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(y_test, y_pred, alpha=0.5,
                             c=np.abs(y_test - y_pred), cmap='viridis')
        ax.plot([y_test.min(), y_test.max()], [
                y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Valores reales')
        ax.set_ylabel('Predicciones')
        ax.set_title('Predicciones vs Valores reales')
        plt.colorbar(scatter, ax=ax, label='Error absoluto')

        st.pyplot(fig)
        st.markdown(get_image_download_link(
            fig, "predicciones_vs_reales", "📥 Descargar gráfico"), unsafe_allow_html=True)

        # Distribución de errores
        fig_err, ax_err = plt.subplots(figsize=(8, 6))
        errors = y_test - y_pred
        sns.histplot(errors, kde=True, ax=ax_err)
        ax_err.axvline(x=0, color='r', linestyle='--')
        ax_err.set_title('Distribución de errores')
        ax_err.set_xlabel('Error (Real - Predicción)')

        st.pyplot(fig_err)

        with st.expander("📊 Explicación de métricas de regresión"):
            st.markdown("""
            - **MSE (Error Cuadrático Medio)**: Promedio de los errores al cuadrado. Penaliza más los errores grandes.
            - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del MSE. Está en las mismas unidades que la variable objetivo.
            - **MAE (Error Absoluto Medio)**: Promedio de los valores absolutos de los errores. Menos sensible a valores atípicos que MSE.
            - **R² Score**: Indica qué proporción de la varianza en la variable dependiente es predecible. 1 es predicción perfecta, 0 significa que el modelo no es mejor que predecir la media.
            """)


# Sidebar para configuración
with st.sidebar:
    st.header("MLTutor")

    # Selección del algoritmo
    st.subheader("Algoritmo")
    algorithm_type = st.selectbox(
        "Selecciona un algoritmo:",
        ("Árboles de Decisión", "Regresión Logística",
         "K-Nearest Neighbors", "Redes Neuronales"),
        index=0,  # Árboles de Decisión como opción por defecto
        help="Selecciona el algoritmo de machine learning que quieres explorar"
    )

    # Tipo de árbol - definido aquí para estar disponible globalmente
    tree_type = st.session_state.tree_type

    # Mostrar la configuración correspondiente al algoritmo seleccionado
    if algorithm_type == "Árboles de Decisión":
        st.subheader("Configuración - Árboles de Decisión")

        # Selección de dataset
        dataset_option = st.selectbox(
            "Dataset de ejemplo:",
            ("Iris (clasificación de flores)",
             "Vino (clasificación de vinos)", "Cáncer de mama (diagnóstico)")
        )

        if dataset_option != st.session_state.dataset_option:
            st.session_state.dataset_option = dataset_option
            st.session_state.is_trained = False  # Resetear estado de entrenamiento

        # Sección de configuración del modelo
        st.subheader("Parámetros del Árbol")

        # Tipo de árbol
        tree_type = st.radio(
            "Tipo de árbol:",
            ("Clasificación", "Regresión"),
            help="Clasificación para predecir categorías, Regresión para valores continuos"
        )

        if tree_type != st.session_state.tree_type:
            st.session_state.tree_type = tree_type
            st.session_state.is_trained = False  # Resetear estado de entrenamiento

        # Profundidad máxima
        max_depth = st.slider(
            "Profundidad máxima:",
            1, 10, 3,
            help="Controla la complejidad del árbol. Mayor profundidad puede llevar a sobreajuste."
        )
    else:
        # Mensaje de "Próximamente" para algoritmos no implementados
        st.markdown(f"""
        <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800; margin-bottom: 20px;">
            <h3 style="color: #FF9800; margin-top: 0;">🚧 Próximamente: {algorithm_type} 🚧</h3>
            <p>Esta sección está en desarrollo y estará disponible en futuras actualizaciones de MLTutor.</p>
            <p>Mientras tanto, puedes explorar los Árboles de Decisión que ya están implementados.</p>
        </div>
        """, unsafe_allow_html=True)

        # Imagen ilustrativa para el algoritmo seleccionado
        if algorithm_type == "Regresión Logística":
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_logistic_001.png",
                     caption="Ilustración de Regresión Logística")
        elif algorithm_type == "K-Nearest Neighbors":
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png",
                     caption="Ilustración de K-Nearest Neighbors")
        elif algorithm_type == "Redes Neuronales":
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_mlp_001.png",
                     caption="Ilustración de Redes Neuronales")

        # Redirigir al usuario a Árboles de Decisión
        st.subheader("Mientras tanto...")
        st.markdown(
            "Puedes seleccionar 'Árboles de Decisión' en el menú desplegable para explorar la funcionalidad implementada.")

        # Inicializar variables necesarias para evitar errores
        max_depth = 3  # valor por defecto
        min_samples_split = 2  # valor por defecto
        criterion = "gini"  # valor por defecto
        dataset_option = "Iris (clasificación de flores)"  # valor por defecto

    # Muestras mínimas para dividir
    min_samples_split = st.slider(
        "Muestras mínimas para dividir:",
        2, 20, 2,
        help="Número mínimo de muestras requeridas para dividir un nodo."
    )

    # Criterio
    if tree_type == "Clasificación":
        criterion_options = ["gini", "entropy"]
        criterion_desc = {
            "gini": "Mide la impureza (menos homogéneo = mayor impureza)",
            "entropy": "Mide la ganancia de información (entropía de Shannon)"
        }
    else:
        criterion_options = ["squared_error",
                             "friedman_mse", "absolute_error", "poisson"]
        criterion_desc = {
            "squared_error": "Minimiza el error cuadrático medio (MSE)",
            "friedman_mse": "Mejora la selección de características",
            "absolute_error": "Minimiza el error absoluto medio (MAE)",
            "poisson": "Para datos que siguen distribución de Poisson"
        }

    criterion = st.selectbox(
        "Criterio de división:",
        criterion_options,
        help="Método para evaluar la calidad de una división del nodo."
    )
    st.caption(criterion_desc[criterion])

    # Porcentaje de división de datos
    test_size = st.slider(
        "Porcentaje para prueba:",
        0.1, 0.5, 0.3, 0.05,
        help="Porcentaje de datos reservados para evaluar el modelo."
    )

    # Botón de entrenamiento
    train_button = st.button("Entrenar Árbol", type="primary")

    # Información educativa
    with st.expander("ℹ️ ¿Qué son los parámetros?"):
        st.markdown("""
        **Profundidad máxima**: Limita cuántos niveles puede tener el árbol. Un valor bajo crea un árbol simple que puede no capturar todos los patrones. Un valor alto puede crear un árbol complejo que se sobreajusta a los datos.
        
        **Muestras mínimas**: Número mínimo de muestras necesarias para dividir un nodo. Valores más altos previenen divisiones con pocos ejemplos, reduciendo el sobreajuste.
        
        **Criterio**: Método usado para evaluar la calidad de una división. En clasificación, 'gini' mide la impureza y 'entropy' la ganancia de información. En regresión, generalmente se usa el error cuadrático.
         **Porcentaje para prueba**: Fracción de datos que se reservan para evaluar el modelo. No se usan durante el entrenamiento.
        """)

# Solo mostrar el contenido principal para Árboles de Decisión
if algorithm_type == "Árboles de Decisión":
    # Cargar datos
    X, y, feature_names, class_names, data_info = load_data(dataset_option)

    # Mostrar información del dataset
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<div class='info-box'><b>{data_info}</b></div>", unsafe_allow_html=True)

        # Ver muestra de datos
        with st.expander("Ver muestra de datos"):
            st.dataframe(pd.concat([X, y], axis=1).head(10))

# Dividir datos y entrenar modelo si se presiona el botón
if train_button or st.session_state.is_trained:
    if train_button:  # Solo ejecutar división y entrenamiento si es un nuevo entrenamiento
        with st.spinner("Entrenando el árbol de decisión..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)

            # Guardar en session_state para reutilizar
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_names
            st.session_state.class_names = class_names

            if tree_type == "Clasificación":
                tree_model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    random_state=42
                )
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Guardar resultados de prueba
                st.session_state.test_results = {
                    "y_pred": y_pred,
                    "accuracy": accuracy
                }

            else:
                tree_model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    random_state=42
                )
                tree_model.fit(X_train, y_train)
                y_pred = tree_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)

                # Guardar resultados de prueba
                st.session_state.test_results = {
                    "y_pred": y_pred,
                    "mse": mse
                }

            # Guardar modelo entrenado
            st.session_state.tree_model = tree_model
            st.session_state.is_trained = True

    # Usar datos guardados en session_state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    feature_names = st.session_state.feature_names
    class_names = st.session_state.class_names
    tree_model = st.session_state.tree_model
    tree_type = st.session_state.tree_type

    # Mostrar métricas básicas
    if tree_type == "Clasificación":
        y_pred = st.session_state.test_results["y_pred"]
        accuracy = st.session_state.test_results["accuracy"]

        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Precisión", f"{accuracy:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Matriz de confusión compacta
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                        xticklabels=class_names, yticklabels=class_names)
            ax_cm.set_xlabel('Predicción')
            ax_cm.set_ylabel('Real')
            ax_cm.set_title('Matriz de Confusión')
            st.pyplot(fig_cm)
    else:
        y_pred = st.session_state.test_results["y_pred"]
        mse = st.session_state.test_results["mse"]

        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Error Cuadrático Medio", f"{mse:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Gráfico compacto de predicciones vs valores reales
            fig_pred, ax_pred = plt.subplots(figsize=(4, 3))
            ax_pred.scatter(y_test, y_pred, alpha=0.5)
            ax_pred.plot([y_test.min(), y_test.max()], [
                         y_test.min(), y_test.max()], 'r--')
            ax_pred.set_xlabel('Valores reales')
            ax_pred.set_ylabel('Predicciones')
            ax_pred.set_title('Predicciones vs Valores reales')
            st.pyplot(fig_pred)

    # Crear tabs para visualizaciones y utilidades
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Árbol de Decisión", "Importancia de Características", "Evaluación Detallada", "Inferencia", "Exportar Modelo", "Explicación"])

    with tab1:
        st.markdown(
            "<h2 class='sub-header'>Visualización del Árbol de Decisión</h2>", unsafe_allow_html=True)

        # Configuración de visualización
        col_viz1, col_viz2 = st.columns([1, 3])
        with col_viz1:
            fig_size = st.slider("Tamaño de la figura", 1,
                                 5, 3,
                                 help="Controla el tamaño global del árbol visualizado")

            # Calcular width y height basados en el tamaño seleccionado
            fig_width = 8 + (fig_size - 1) * 3  # 8, 11, 14, 17, 20
            fig_height = 6 + (fig_size - 1) * 2.25  # 6, 8.25, 10.5, 12.75, 15

            show_values = st.checkbox(
                "Mostrar valores", st.session_state.show_values)

            # Actualizar valores en session_state
            st.session_state.fig_width = fig_width
            st.session_state.fig_height = fig_height
            st.session_state.fig_size = fig_size
            st.session_state.show_values = show_values

        # Crear visualización del árbol
        with col_viz2:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plot_tree(tree_model,
                      feature_names=feature_names,
                      class_names=class_names if tree_type == "Clasificación" else None,
                      filled=True,
                      rounded=True,
                      ax=ax,
                      impurity=show_values,
                      proportion=False)
            st.pyplot(fig)
            st.markdown(get_image_download_link(
                fig, "arbol_decision", "📥 Descargar imagen"), unsafe_allow_html=True)

        # Explicación de la visualización
        with st.expander("Cómo interpretar el árbol"):
            st.markdown("""
            - **Nodos**: Representan decisiones basadas en características.
            - **Flechas**: La rama izquierda indica que la condición se cumple, la derecha que no.
            - **Colores**: Indican a qué clase pertenecen las muestras en ese nodo (en clasificación).
            - **Impureza**: Gini o Entropy en clasificación, indican qué tan homogéneas son las muestras.
            - **Samples**: Cuántas muestras llegan a ese nodo.
            - **Value**: Distribución de clases en ese nodo.
            """)

    with tab2:
        st.markdown(
            "<h2 class='sub-header'>Importancia de las Características</h2>", unsafe_allow_html=True)

        # Crear dataframe con importancias
        importance = pd.DataFrame({
            'Característica': feature_names,
            'Importancia': tree_model.feature_importances_
        }).sort_values('Importancia', ascending=False)

        # Mostrar gráfico de barras
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance, x='Importancia',
                    y='Característica', ax=ax_imp)
        ax_imp.set_title('Importancia de las Características')

        col_imp1, col_imp2 = st.columns([2, 1])

        with col_imp1:
            st.pyplot(fig_imp)
            st.markdown(get_image_download_link(
                fig_imp, "importancia_caracteristicas", "📥 Descargar imagen"), unsafe_allow_html=True)

        with col_imp2:
            st.dataframe(importance.style.format(
                {'Importancia': '{:.4f}'}), height=300)

            with st.expander("¿Qué significa la importancia?"):
                st.markdown("""
                La **importancia de características** indica cuánto contribuye cada característica a la predicción.
                
                Se calcula en base a cuánto mejora cada característica la pureza de los nodos (o reduce el error) cuando se usa para dividir los datos.
                
                Las características con mayor importancia son las más relevantes para el modelo y tienen mayor poder predictivo.
                """)

    with tab3:
        st.markdown("<h2 class='sub-header'>Evaluación Detallada</h2>",
                    unsafe_allow_html=True)

        # Mostrar evaluación detallada
        show_detailed_evaluation(y_test, y_pred, class_names, tree_type)

    with tab4:
        st.markdown(
            "<h2 class='sub-header'>Inferencia con Nuevos Datos</h2>", unsafe_allow_html=True)

        st.info(
            "Introduce valores para las características y haz predicciones con el modelo entrenado.")

        # Crear formulario para entrada de datos
        col1, col2 = st.columns([3, 1])

        with col1:
            # Crear inputs para cada característica
            feature_inputs = {}

            # Calcular valores mínimos y máximos para cada característica
            X_min = X.min()
            X_max = X.max()
            X_mean = X.mean()

            num_features = len(feature_names)
            cols_per_row = 3

            for i in range(0, num_features, cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < num_features:
                        feature_idx = i + j
                        feature_name = feature_names[feature_idx]
                        min_val = float(X_min[feature_idx])
                        max_val = float(X_max[feature_idx])
                        mean_val = float(X_mean[feature_idx])

                        # Mostrar slider para la característica
                        feature_inputs[feature_name] = cols[j].slider(
                            f"{feature_name}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100,
                            format="%.4f"
                        )

            # Convertir inputs a formato para predicción
            X_new = [feature_inputs[name] for name in feature_names]

            # Botón para predecir
            predict_button = st.button("Realizar Predicción")

        with col2:
            if predict_button:
                # Realizar predicción
                prediction = tree_model.predict([X_new])

                st.markdown("<div class='metric-card'>",
                            unsafe_allow_html=True)
                if tree_type == "Clasificación":
                    st.metric("Clase Predicha",
                              f"{class_names[prediction[0]]}")

                    # Probabilidades de clase si el modelo lo soporta
                    if hasattr(tree_model, "predict_proba"):
                        proba = tree_model.predict_proba([X_new])[0]

                        # Mostrar probabilidades para cada clase
                        st.markdown("### Probabilidades por clase:")
                        proba_df = pd.DataFrame({
                            'Clase': class_names,
                            'Probabilidad': proba
                        })
                        st.dataframe(proba_df.style.format(
                            {"Probabilidad": "{:.4f}"}))

                        # Gráfico de barras para probabilidades
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.bar(class_names, proba)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Probabilidad')
                        ax.set_title('Probabilidad por clase')
                        st.pyplot(fig)
                else:
                    st.metric("Valor Predicho", f"{prediction[0]:.4f}")

                st.markdown("</div>", unsafe_allow_html=True)

                # Explicación de la decisión
                st.markdown("### Camino de decisión:")

                # Recorrer el árbol para la muestra
                feature_idx = tree_model.tree_.feature
                threshold = tree_model.tree_.threshold

                # Construir el camino de decisión
                node_indicator = tree_model.decision_path([X_new])
                leaf_id = tree_model.apply([X_new])

                # Obtener los nodos en el camino
                node_index = node_indicator.indices[node_indicator.indptr[0]
                    :node_indicator.indptr[1]]

                path_explanation = []
                for node_id in node_index:
                    # Detener si es un nodo hoja
                    if leaf_id[0] == node_id:
                        continue

                    # Obtener la característica y el umbral de la decisión
                    feature_id = feature_idx[node_id]
                    feature_name = feature_names[feature_id]
                    threshold_value = threshold[node_id]

                    # Comprobar si la muestra va por la izquierda o derecha
                    if X_new[feature_id] <= threshold_value:
                        path_explanation.append(
                            f"- {feature_name} = {X_new[feature_id]:.4f} ≤ {threshold_value:.4f} ✅")
                    else:
                        path_explanation.append(
                            f"- {feature_name} = {X_new[feature_id]:.4f} > {threshold_value:.4f} ✅")

                # Mostrar el camino de decisión
                st.markdown("\n".join(path_explanation))

    with tab5:
        st.markdown("<h2 class='sub-header'>Exportar Modelo</h2>",
                    unsafe_allow_html=True)

        st.info(
            "Descarga el modelo entrenado en diferentes formatos para utilizarlo en tus aplicaciones.")

        col1, col2 = st.columns(2)

        with col1:
            # Código Python
            st.markdown("### Código Python")
            model_code = generate_model_code(
                tree_model,
                tree_type,
                max_depth,
                min_samples_split,
                criterion,
                feature_names,
                class_names
            )

            st.download_button(
                "📥 Descargar código Python",
                model_code,
                file_name="modelo_arbol_decision.py",
                mime="text/plain",
                help="Código Python para recrear y utilizar el modelo entrenado"
            )

            # Formato pickle
            pickle_data = export_model_pickle(tree_model)
            st.download_button(
                "📥 Descargar modelo (Pickle)",
                pickle_data,
                file_name="modelo_arbol_decision.pkl",
                mime="application/octet-stream",
                help="Archivo binario para cargar el modelo directamente en Python"
            )

        with col2:
            # Visualización del árbol en texto
            st.markdown("### Representación de Texto")
            tree_text = export_text(
                tree_model,
                feature_names=list(feature_names),
                spacing=2,
                decimals=4
            )

            st.text_area(
                "Árbol en formato de texto:",
                tree_text,
                height=300,
                help="Representación textual del árbol de decisión"
            )

            st.download_button(
                "📥 Descargar representación de texto",
                tree_text,
                file_name="arbol_decision_texto.txt",
                mime="text/plain",
                help="Representación textual del árbol para referencia"
            )

            # ONNX (si está disponible)
            if ONNX_AVAILABLE:
                onnx_data = export_model_onnx(tree_model, len(feature_names))
                if onnx_data:
                    st.download_button(
                        "📥 Descargar modelo (ONNX)",
                        onnx_data,
                        file_name="modelo_arbol_decision.onnx",
                        mime="application/octet-stream",
                        help="Formato ONNX para despliegue en múltiples plataformas"
                    )

    with tab6:
        st.markdown(
            "<h2 class='sub-header'>Explicación Educativa</h2>", unsafe_allow_html=True)

        st.markdown("""
        ## ¿Qué son los árboles de decisión?

        Los árboles de decisión son modelos de aprendizaje automático que toman decisiones secuenciales basadas en características de los datos, similar a un diagrama de flujo. Son populares por su interpretabilidad y simplicidad visual.

        ### Estructura de un árbol de decisión:

        - **Nodo raíz**: El punto de inicio donde se evalúa toda la población.
        - **Nodos internos**: Representan pruebas sobre atributos específicos.
        - **Ramas**: Muestran los resultados de estas pruebas.
        - **Nodos hoja**: Muestran la decisión final o predicción.

        ### ¿Cómo se construye un árbol?

        El algoritmo sigue estos pasos:
        1. Busca la mejor característica para dividir los datos
        2. Divide los datos en subconjuntos basados en esa característica
        3. Repite el proceso en cada subconjunto hasta que:
           - Se alcance la profundidad máxima
           - No haya suficientes muestras para dividir
           - Todos los datos en el nodo pertenezcan a la misma clase

        ### Ventajas de los árboles de decisión

        - **Fácil interpretación**: Se pueden visualizar y explicar fácilmente.
        - **No requieren preparación de datos**: Funcionan bien con datos categóricos y numéricos.
        - **Capturan relaciones no lineales**: Pueden modelar interacciones complejas.
        - **Identifican características importantes**: Muestran qué variables tienen mayor impacto.

        ### Limitaciones

        - **Tendencia al sobreajuste**: Especialmente con árboles muy profundos.
        - **Inestables**: Pequeños cambios en los datos pueden generar árboles muy diferentes.
        - **Sesgados hacia características con más niveles**: Pueden favorecer variables con más valores posibles.

        ### Aplicaciones prácticas

        - Diagnóstico médico
        - Análisis de riesgo crediticio
        - Sistemas de recomendación
        - Detección de fraudes
        - Análisis de sentimientos
        """)

        # Conceptos interactivos
        with st.expander("Conceptos clave de árboles de decisión"):
            concept = st.selectbox(
                "Selecciona un concepto para aprender más:",
                ["Gini vs Entropy", "Overfitting y Underfitting",
                    "Poda de árboles", "Random Forest"]
            )

            if concept == "Gini vs Entropy":
                st.markdown("""
                ### Gini vs Entropy
                
                Ambos son métodos para medir la impureza o heterogeneidad en un nodo.
                
                **Índice Gini**:
                - Mide la probabilidad de clasificar incorrectamente un elemento.
                - Varía de 0 (todos los elementos pertenecen a una sola clase) a 0.5 (distribución uniforme).
                - Fórmula: Gini = 1 - Σ(pi²) donde pi es la proporción de la clase i.
                - Generalmente más rápido computacionalmente.
                
                **Entropía**:
                - Mide la aleatoriedad o incertidumbre en los datos.
                - Varía de 0 (homogéneo) a 1 (distribución uniforme).
                - Fórmula: Entropy = -Σ(pi * log2(pi))
                - Tiende a crear árboles más balanceados.
                
                En la práctica, ambos métodos suelen dar resultados similares.
                """)
            elif concept == "Overfitting y Underfitting":
                st.markdown("""
                ### Overfitting y Underfitting
                
                **Overfitting (Sobreajuste)**:
                - El modelo aprende demasiado de los datos de entrenamiento, incluyendo el ruido.
                - Alta precisión en entrenamiento, baja en datos nuevos.
                - Síntomas: Árbol muy profundo, muchas ramas.
                - Soluciones: Limitar profundidad, aumentar muestras mínimas, poda.
                
                **Underfitting (Subajuste)**:
                - El modelo es demasiado simple y no captura patrones importantes.
                - Baja precisión tanto en entrenamiento como en pruebas.
                - Síntomas: Árbol muy pequeño, pocas divisiones.
                - Soluciones: Aumentar profundidad, reducir muestras mínimas.
                
                El objetivo es encontrar el equilibrio óptimo entre estos dos extremos.
                """)
            elif concept == "Poda de árboles":
                st.markdown("""
                ### Poda de árboles (Pruning)
                
                La poda es una técnica para reducir el tamaño del árbol y evitar el sobreajuste.
                
                **Tipos de poda**:
                
                1. **Pre-poda**: Detener el crecimiento del árbol antes de que se sobreajuste.
                   - Limitar profundidad máxima
                   - Establecer número mínimo de muestras
                   - Establecer ganancia mínima de información
                
                2. **Post-poda**: Construir el árbol completo y luego remover ramas.
                   - Reducción del error de costo-complejidad
                   - Reemplazar subárboles con nodos hoja
                   - Evaluar rendimiento en un conjunto de validación
                
                La poda adecuada mejora la generalización y reduce la complejidad del modelo.
                """)
            elif concept == "Random Forest":
                st.markdown("""
                ### Random Forest
                
                Random Forest es una extensión de los árboles de decisión que combina múltiples árboles para mejorar el rendimiento.
                
                **Funcionamiento**:
                1. Crea múltiples árboles de decisión (típicamente cientos).
                2. Cada árbol se entrena con una muestra aleatoria de los datos (bootstrap).
                3. En cada división, solo considera un subconjunto aleatorio de características.
                4. La predicción final es el promedio (regresión) o voto mayoritario (clasificación).
                
                **Ventajas sobre árboles individuales**:
                - Menos propenso al sobreajuste
                - Mayor precisión
                - Más estable frente a variaciones en los datos
                - Proporciona importancia de características más robusta
                
                Random Forest es uno de los algoritmos más potentes y versátiles en machine learning.
                """)
        st.info(
            "👈 Configura los parámetros y presiona 'Entrenar Árbol' para visualizar los resultados")

        # Mostrar ejemplo de árbol
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dtc_002.png",
                 caption="Ejemplo de un árbol de decisión para el dataset Iris")

        # Información general sobre árboles de decisión
        with st.expander("Introducción a los Árboles de Decisión"):
            st.markdown("""
        # ¿Qué son los Árboles de Decisión?

        Los árboles de decisión son herramientas de aprendizaje automático que permiten tomar decisiones a través de una serie de reglas simples. Se representan como un árbol donde:

        - Cada **nodo interno** representa una pregunta sobre una característica
        - Cada **rama** representa la respuesta a esa pregunta
        - Cada **hoja** representa una predicción o decisión final

        Son muy populares porque son fáciles de entender, visualizar y explicar a personas sin conocimientos técnicos.

        ## Aplicaciones de los Árboles de Decisión

        - **Medicina**: Diagnóstico de enfermedades basado en síntomas
        - **Finanzas**: Evaluación de riesgo crediticio
        - **Marketing**: Segmentación de clientes
        - **Biología**: Clasificación de especies
        - **Sistemas de recomendación**: Sugerir productos o contenido

        ## ¿Por qué usar Árboles de Decisión?

        - **Interpretabilidad**: Fáciles de entender y explicar
        - **Versatilidad**: Pueden trabajar con datos numéricos y categóricos
        - **Mínima preparación de datos**: No requieren normalización
        - **Capturan relaciones no lineales** entre variables
        - **Identifican automáticamente las características más importantes**
        """)
else:
    # Contenido principal para algoritmos no implementados
    st.markdown(f"""
    <div style="background-color: #E1F5FE; padding: 30px; border-radius: 10px; border-left: 5px solid #03A9F4; margin: 20px 0;">
        <h2 style="color: #0288D1; margin-top: 0;">Bienvenido a MLTutor</h2>
        <p style="font-size: 18px;">MLTutor es una plataforma educativa para aprender Machine Learning de forma interactiva.</p>
        <p style="font-size: 16px;">Actualmente estamos en desarrollo. Por favor, selecciona "Árboles de Decisión" en el menú lateral para explorar la funcionalidad completa disponible.</p>
        <h3>Próximamente:</h3>
        <ul>
            <li>Regresión Logística</li>
            <li>K-Nearest Neighbors (KNN)</li>
            <li>Redes Neuronales</li>
        </ul>
    </div>
    
    <div style="text-align: center; margin-top: 40px;">
        <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png" 
             alt="Comparación de algoritmos de ML" 
             style="max-width: 80%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <p style="margin-top: 10px; color: #666; font-style: italic;">Comparación visual de diferentes algoritmos de Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# Agregar pie de página
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2025 MLTutor | Desarrollado por Javier Palanca | Universitat Politècnica de València</p>
</div>
""", unsafe_allow_html=True)
