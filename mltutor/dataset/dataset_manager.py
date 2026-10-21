"""
Este módulo contiene funciones para la carga y gestión de datos en la aplicación MLTutor.
Incluye funciones para cargar conjuntos de datos integrados, cargar archivos CSV y procesar datos.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from mltutor.desktop import ui as st
import os
import datetime

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mltutor.dataset.additional_datasets import load_additional_dataset


def reset_moons_dataset(n_samples=500, noise=0.2):
    random_step_based_on_current_time = int(
        datetime.datetime.now().timestamp())
    X_arr, y_arr = make_moons(n_samples=n_samples, noise=noise,
                              random_state=random_step_based_on_current_time)
    st.session_state['moons_X'] = X_arr
    st.session_state['moons_y'] = y_arr


def load_builtin_dataset(dataset_name):
    """
    Carga un conjunto de datos integrado según el nombre.

    Parameters:
    -----------
    dataset_name : str
        Nombre del conjunto de datos ('Iris', 'Vino', 'Cáncer', 'Titanic', 'Propinas', 'Viviendas California', 'Pingüinos')

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases
    dataset_info : str
        Información sobre el conjunto de datos
    """
    if "Iris" in dataset_name:
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="species")
        feature_names = data.feature_names.tolist() if hasattr(
            data.feature_names, 'tolist') else list(data.feature_names)
        class_names = data.target_names.tolist()
        dataset_info = {
            'description': "Dataset Iris: 150 muestras, 4 características, 3 clases de flores",
            'target': 'species',
            'task_type': 'Clasificación',
            'samples': 150,
            'features': 4,
            'classes': 3
        }
        task_type = "Clasificación"

    elif "Vino" in dataset_name:
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="wine_type")
        feature_names = data.feature_names.tolist() if hasattr(
            data.feature_names, 'tolist') else list(data.feature_names)
        class_names = data.target_names.tolist()
        dataset_info = {
            'description': "Dataset Vino: 178 muestras, 13 características, 3 tipos de vino",
            'target': 'wine_type',
            'task_type': 'Clasificación',
            'samples': 178,
            'features': 13,
            'classes': 3
        }
        task_type = "Clasificación"

    elif "Cáncer" in dataset_name:
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="malignant")
        feature_names = data.feature_names.tolist() if hasattr(
            data.feature_names, 'tolist') else list(data.feature_names)
        class_names = data.target_names.tolist()
        dataset_info = {
            'description': "Dataset Cáncer: 569 muestras, 30 características, diagnóstico binario",
            'target': 'diagnosis',
            'task_type': 'Clasificación',
            'samples': 569,
            'features': 30,
            'classes': 2
        }
        task_type = "Clasificación"

    elif "Titanic" in dataset_name:
        X, y, feature_names, class_names, dataset_info, task_type = load_additional_dataset(
            dataset_name)

    elif "Propinas" in dataset_name:
        X, y, feature_names, class_names, dataset_info, task_type = load_additional_dataset(
            dataset_name)

    elif "Viviendas California" in dataset_name:
        X, y, feature_names, class_names, dataset_info, task_type = load_additional_dataset(
            dataset_name)

    elif "Pingüinos" in dataset_name:
        X, y, feature_names, class_names, dataset_info, task_type = load_additional_dataset(
            dataset_name)

    # Nuevos datasets
    elif "Dígitos" in dataset_name:
        data = load_digits()
        X = pd.DataFrame(data.data)
        y = pd.Series(data.target, name="digit")
        feature_names = [f"p{i}" for i in range(X.shape[1])]
        class_names = [str(c) for c in sorted(np.unique(data.target))]
        dataset_info = {
            'description': f"Dígitos manuscritos: {X.shape[0]} muestras, {X.shape[1]} píxeles (8x8) aplanados, 10 clases",
            'target': 'digit',
            'task_type': 'Clasificación',
            'samples': X.shape[0],
            'features': X.shape[1],
            'classes': 10
        }
        task_type = "Clasificación"

    elif "Diabetes" in dataset_name:
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="disease_progression")
        feature_names = list(data.feature_names)
        class_names = None
        dataset_info = {
            'description': f"Diabetes: {X.shape[0]} muestras, {X.shape[1]} características; regresión de progresión de la enfermedad",
            'target': 'disease_progression',
            'task_type': 'Regresión',
            'samples': X.shape[0],
            'features': X.shape[1],
            'classes': None
        }
        task_type = "Regresión"

    elif "Diamantes" in dataset_name:
        df = sns.load_dataset("diamonds")
        # Codificar categóricas
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        # Muestreo para rendimiento
        if len(df) > 8000:
            df = df.sample(8000, random_state=42)
        y = df['price']
        X = df.drop(columns=['price'])
        feature_names = X.columns.tolist()
        class_names = None
        dataset_info = {
            'description': f"Diamantes (muestreo): {len(df)} muestras, {X.shape[1]} características, predicción de precio",
            'target': 'price',
            'task_type': 'Regresión',
            'samples': len(df),
            'features': X.shape[1],
            'classes': None
        }
        task_type = "Regresión"

    elif "MPG" in dataset_name:
        df = sns.load_dataset("mpg").dropna()
        y = df['mpg']
        X = df.drop(columns=['mpg'])
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        feature_names = X.columns.tolist()
        class_names = None
        dataset_info = {
            'description': f"MPG: {len(df)} muestras limpias, {X.shape[1]} características, consumo de combustible (regresión)",
            'target': 'mpg',
            'task_type': 'Regresión',
            'samples': len(df),
            'features': X.shape[1],
            'classes': None
        }
        task_type = "Regresión"

    elif "Moons" in dataset_name or "🌙" in dataset_name:
        # Dataset sintético Two Moons (clasificación no lineal)
        if st.session_state.get('moons_X', None) is None:
            reset_moons_dataset()
        X_arr = st.session_state.get('moons_X', None)
        y_arr = st.session_state.get('moons_y', None)
        X = pd.DataFrame(X_arr, columns=['x1', 'x2'])
        y = pd.Series(y_arr, name='label')
        feature_names = ['x1', 'x2']
        class_names = ['class_0', 'class_1']
        dataset_info = {
            'description': f"Moons sintético: {X.shape[0]} muestras, 2 características, 2 clases (no lineal).",
            'target': 'label',
            'task_type': 'Clasificación',
            'samples': X.shape[0],
            'features': X.shape[1],
            'classes': 2
        }
        task_type = "Clasificación"

    else:
        raise ValueError(f"Conjunto de datos '{dataset_name}' no reconocido")

    return X, y, feature_names, class_names, dataset_info, task_type


def load_dataset_from_file(file_path, target_column=None, task_type="auto"):
    """
    Carga un conjunto de datos desde un archivo CSV.

    Parameters:
    -----------
    file_path : str
        Ruta al archivo CSV
    target_column : str, optional
        Nombre de la columna objetivo. Si es None, se usa la última columna.
    task_type : str, default="auto"
        Tipo de tarea ('Clasificación', 'Regresión', 'auto')

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    feature_names : list
        Nombres de las características
    class_names : list or None
        Nombres de las clases (solo para clasificación)
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea ('Clasificación' o 'Regresión')
    """
    # Cargar el dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")

    # Determinar la columna objetivo
    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        raise ValueError(
            f"Columna objetivo '{target_column}' no encontrada en el dataset")

    # Separar características y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Codificar automáticamente características categóricas (string/object)
    categorical_features = X.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Codificar variable objetivo si es categórica
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(
            y.astype(str)), name=target_column)

    # Obtener nombres de características
    feature_names = X.columns.tolist()

    # Determinar tipo de tarea si es automático
    if task_type == "auto":
        # Si y tiene pocos valores únicos (menos del 5% del total), asumimos clasificación
        if len(y.unique()) < max(5, len(y) * 0.05):
            task_type = "Clasificación"
        else:
            task_type = "Regresión"

    # Para clasificación, determinar nombres de clases
    if task_type == "Clasificación":
        class_values = sorted(y.unique())
        # Intentar usar etiquetas si están disponibles
        if hasattr(y, 'cat') and hasattr(y.cat, 'categories'):
            class_names = y.cat.categories.tolist()
        else:
            # Si son números enteros, usar nombres genéricos
            if all(isinstance(val, (int, np.integer)) for val in class_values):
                class_names = [f"Clase {i}" for i in class_values]
            else:
                class_names = [str(val) for val in class_values]
    else:
        class_names = None

    # Generar información del dataset
    n_samples, n_features = X.shape
    if task_type == "Clasificación":
        n_classes = len(class_names)
        dataset_info = f"Dataset personalizado: {n_samples} muestras, {n_features} características, {n_classes} clases"
    else:
        dataset_info = f"Dataset personalizado: {n_samples} muestras, {n_features} características, regresión"

    return X, y, feature_names, class_names, dataset_info, task_type


def list_sample_datasets():
    """
    Lista los conjuntos de datos de ejemplo disponibles en el directorio 'data/sample_datasets'.

    Returns:
    --------
    list
        Lista de rutas a los archivos CSV de ejemplo
    """
    sample_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data", "sample_datasets")
    if not os.path.exists(sample_dir):
        return []

    return [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith('.csv')
    ]


def preprocess_data(X, y, test_size=0.3, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : Series o array
        Variable objetivo
    test_size : float, default=0.3
        Proporción de datos para prueba
    random_state : int, default=42
        Semilla para reproducibilidad

    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Conjuntos de entrenamiento y prueba
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_data(dataset_option):
    """
    Carga un conjunto de datos según la opción seleccionada.

    Parameters:
    -----------
    dataset_option : str
        Nombre del conjunto de datos a cargar

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    feature_names : list
        Nombres de las características
    class_names : list o None
        Nombres de las clases (para clasificación)
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Lista de datasets integrados (tanto formato antiguo como nuevo)
    dataset_mapping = {
        "Iris (clasificación de flores)": "🌸 Iris - Clasificación de flores",
        "Vino (clasificación de vinos)": "🍷 Vino - Clasificación de vinos",
        "Cáncer de mama (diagnóstico)": "🔬 Cáncer - Diagnóstico binario",
        "🌸 Iris - Clasificación de flores": "🌸 Iris - Clasificación de flores",
        "🍷 Vino - Clasificación de vinos": "🍷 Vino - Clasificación de vinos",
        "🔬 Cáncer - Diagnóstico binario": "🔬 Cáncer - Diagnóstico binario",
        "🚢 Titanic - Supervivencia": "🚢 Titanic - Supervivencia",
        "💰 Propinas - Predicción de propinas": "💰 Propinas - Predicción de propinas",
        "🏠 Viviendas California - Precios": "🏠 Viviendas California - Precios",
        "🐧 Pingüinos - Clasificación de especies": "🐧 Pingüinos - Clasificación de especies",
        "🔢 Dígitos - Clasificación de dígitos": "🔢 Dígitos - Clasificación de dígitos",
        "🩺 Diabetes - Progresión (regresión)": "🩺 Diabetes - Progresión (regresión)",
        "🌙 Moons - Clasificación sintética": "🌙 Moons - Clasificación sintética",
        "💎 Diamantes - Precio (regresión)": "💎 Diamantes - Precio (regresión)",
        "⛽ MPG - Consumo combustible (regresión)": "⛽ MPG - Consumo combustible (regresión)"
    }

    # Comprobar si es un dataset CSV personalizado cargado
    from mltutor.desktop import ui as st
    if hasattr(st, 'session_state') and 'csv_datasets' in st.session_state and dataset_option in st.session_state.csv_datasets:
        csv_info = st.session_state.csv_datasets[dataset_option]
        return load_dataset_from_file(
            csv_info['file_path'],
            csv_info['target_col'],
            csv_info['task_type']
        )

    # Normalizar el nombre del dataset
    if dataset_option in dataset_mapping:
        normalized_name = dataset_mapping[dataset_option]
        return load_builtin_dataset(normalized_name)

    # Comprobar si es un conjunto de datos personalizado (CSV) - compatibilidad con formato anterior
    if dataset_option.endswith('.csv'):
        file_path = dataset_option
        if not os.path.exists(file_path):
            # Comprobar si está en el directorio de muestras
            sample_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "data", "sample_datasets")
            sample_path = os.path.join(sample_dir, os.path.basename(file_path))
            if os.path.exists(sample_path):
                file_path = sample_path

        return load_dataset_from_file(file_path)

    # Si llegamos aquí, el conjunto de datos no se reconoce
    raise ValueError(f"Conjunto de datos '{dataset_option}' no reconocido")


def create_dataset_selector(show_predefined=True):
    """
    Crea un selector de conjuntos de datos mejorado para la interfaz de usuario.

    Parameters:
    -----------
    show_predefined : bool, default=True
        Si mostrar la opción de datasets predefinidos

    Returns:
    --------
    str or tuple
        Si se carga un archivo CSV: tuple (ruta_archivo, columna_objetivo, tipo_tarea)
        Si no: nombre del dataset seleccionado
    """
    st.subheader("📊 Selección de Dataset")

    # Opciones de datasets expandidas
    builtin_options = [
        "🌸 Iris - Clasificación de flores",
        "🍷 Vino - Clasificación de vinos",
        "🔬 Cáncer - Diagnóstico binario",
        "🔢 Dígitos - Clasificación de dígitos",
        "🩺 Diabetes - Progresión (regresión)",
        "💎 Diamantes - Precio (regresión)",
        "⛽ MPG - Consumo combustible (regresión)",
        "🚢 Titanic - Supervivencia",
        "💰 Propinas - Predicción de propinas",
        "🏠 Viviendas California - Precios",
        "🐧 Pingüinos - Clasificación de especies"
    ]

    # Método de carga
    if show_predefined:
        data_source = st.radio(
            "Fuente de datos:",
            ["Datasets predefinidos", "Cargar archivo CSV"],
            help="Selecciona si quieres usar un dataset incluido o cargar tu propio archivo CSV"
        )
    else:
        data_source = "Cargar archivo CSV"
        st.info(
            "💡 En esta sección puedes cargar tu propio archivo CSV para análisis personalizado")

    if data_source == "Cargar archivo CSV":
        st.markdown("### 📁 Cargar Archivo CSV")

        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV:",
            type=['csv'],
            help="El archivo debe estar en formato CSV con encabezados"
        )

        if uploaded_file is not None:
            try:
                # Leer el archivo
                df = pd.read_csv(uploaded_file)

                # Validaciones básicas
                if df.empty:
                    st.error("❌ El archivo está vacío")
                    return None

                if len(df.columns) < 2:
                    st.error(
                        "❌ El archivo debe tener al menos 2 columnas (características + objetivo)")
                    return None

                # Mostrar vista previa
                st.success(
                    f"✅ Archivo cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")

                with st.expander("👀 Vista previa del dataset", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)

                    # Información del dataset
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Filas", df.shape[0])
                    with col2:
                        st.metric("Columnas", df.shape[1])
                    with col3:
                        missing_values = df.isnull().sum().sum()
                        st.metric("Valores faltantes", missing_values)

                # Configuración
                st.markdown("### ⚙️ Configuración")

                col1, col2 = st.columns(2)

                with col1:
                    target_col = st.selectbox(
                        "Columna objetivo:",
                        df.columns.tolist(),
                        index=len(df.columns)-1,
                        help="Selecciona la columna que contiene la variable a predecir"
                    )

                with col2:
                    # Detectar tipo de tarea automáticamente
                    if target_col:
                        unique_values = df[target_col].nunique()
                        total_values = len(df[target_col])

                        if unique_values <= 20 and unique_values < total_values * 0.1:
                            suggested_task = "Clasificación"
                        else:
                            suggested_task = "Regresión"
                    else:
                        suggested_task = "Clasificación"

                    task_options = ["auto", "Clasificación", "Regresión"]
                    default_idx = task_options.index(
                        suggested_task) if suggested_task in task_options else 0

                    task_type = st.selectbox(
                        "Tipo de tarea:",
                        task_options,
                        index=default_idx,
                        help="'auto' detecta automáticamente, o selecciona manualmente"
                    )

                # Información adicional sobre la columna objetivo
                if target_col:
                    st.markdown("### 🎯 Información de la Variable Objetivo")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Columna:** {target_col}")
                        st.write(
                            f"**Valores únicos:** {df[target_col].nunique()}")
                        st.write(f"**Tipo de datos:** {df[target_col].dtype}")

                    with col2:
                        if df[target_col].nunique() <= 10:
                            st.write("**Distribución de valores:**")
                            value_counts = df[target_col].value_counts()
                            for val, count in value_counts.items():
                                st.write(
                                    f"• {val}: {count} ({count/len(df)*100:.1f}%)")

                # Guardar en archivo temporal
                csv_path = st.store_csv(df)

                # Añadir dataset a la lista de datasets disponibles en session_state
                dataset_name = f"📄 {uploaded_file.name}"
                if 'csv_datasets' not in st.session_state:
                    st.session_state.csv_datasets = {}

                st.session_state.csv_datasets[dataset_name] = {
                    'file_path': csv_path,
                    'target_col': target_col,
                    'task_type': task_type,
                    'original_name': uploaded_file.name
                }

                # Actualizar el dataset seleccionado para usar el nuevo CSV
                st.session_state.selected_dataset = dataset_name
                st.session_state.unified_dataset_selector = dataset_name

                st.success(
                    f"✅ Dataset '{uploaded_file.name}' añadido a la lista de datasets disponibles")

                return csv_path, target_col, task_type

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
                return None

        else:
            st.info("👆 Por favor, carga un archivo CSV para continuar")
            return None

    else:  # Datasets predefinidos
        st.markdown("### 🎯 Datasets Disponibles")
        dataset_option = st.selectbox(
            "Selecciona un dataset:",
            builtin_options,
            help="Estos datasets están listos para usar y son perfectos para aprender Machine Learning"
        )

        # Mostrar información del dataset seleccionado
        dataset_info = {
            "🌸 Iris - Clasificación de flores": "150 muestras • 4 características • 3 especies de iris • Clasificación",
            "🍷 Vino - Clasificación de vinos": "178 muestras • 13 características • 3 clases de vino • Clasificación",
            "🔬 Cáncer - Diagnóstico binario": "569 muestras • 30 características • Benigno/Maligno • Clasificación",
            "🔢 Dígitos - Clasificación de dígitos": "1797 muestras • 64 píxeles • 10 clases • Clasificación",
            "🩺 Diabetes - Progresión (regresión)": "442 muestras • 10 características • Progresión enfermedad • Regresión",
            "💎 Diamantes - Precio (regresión)": "≈54k (muestreo ≤8000) • 9 características • Precio • Regresión",
            "⛽ MPG - Consumo combustible (regresión)": "392 muestras (limpias) • ~7-8 características • MPG • Regresión",
            "🚢 Titanic - Supervivencia": "891 pasajeros • 11 características • Supervivencia • Clasificación",
            "💰 Propinas - Predicción de propinas": "244 cuentas • 6 características • Propina • Regresión",
            "🏠 Viviendas California - Precios": "20,640 distritos • 8 características • Precio vivienda • Regresión",
            "🐧 Pingüinos - Clasificación de especies": "333 pingüinos • 6 características • 3 especies • Clasificación"
        }

        if dataset_option in dataset_info:
            st.info(f"ℹ️ {dataset_info[dataset_option]}")

        return dataset_option
