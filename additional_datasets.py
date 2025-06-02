"""
Datasets adicionales para MLTutor.
Incluye datasets populares desde seaborn y sklearn.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder


def _validate_dataset(X, y, dataset_name):
    """
    Valida que los datos no sean None o vacíos.

    Parameters:
    -----------
    X : DataFrame
        Características
    y : Series
        Variable objetivo
    dataset_name : str
        Nombre del dataset para mensajes de error
    """
    if X is None or X.empty:
        raise ValueError(f"Error: X es None o está vacío en {dataset_name}")
    if y is None or y.empty:
        raise ValueError(f"Error: y es None o está vacío en {dataset_name}")
    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"Error: Datos vacíos en {dataset_name}")
    if len(X) != len(y):
        raise ValueError(
            f"Error: X y y tienen diferente longitud en {dataset_name}")


def _load_seaborn_with_fallback(dataset_name, fallback_filename=None):
    """
    Carga un dataset desde seaborn con fallback a archivo local.

    Parameters:
    -----------
    dataset_name : str
        Nombre del dataset en seaborn
    fallback_filename : str, optional
        Nombre del archivo de fallback

    Returns:
    --------
    DataFrame : Los datos cargados
    """
    try:
        data = sns.load_dataset(dataset_name)
        if data is None or data.empty:
            raise ValueError(
                f"No se pudo cargar el dataset {dataset_name} desde seaborn")
        return data
    except Exception as e:
        print(
            f"Advertencia: No se pudo cargar {dataset_name} desde seaborn: {e}")
        if fallback_filename:
            # Fallback a archivo local
            file_path = os.path.join(os.path.dirname(
                __file__), 'data', 'sample_datasets', fallback_filename)
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                print(f"Usando archivo local: {file_path}")
                return data
        raise ValueError(
            f"No se pudo cargar el dataset {dataset_name} ni desde seaborn ni desde archivo local")


def load_titanic_dataset():
    """
    Carga el dataset del Titanic desde seaborn.

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo (survived)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Cargar dataset del Titanic desde seaborn con fallback
    titanic = _load_seaborn_with_fallback('titanic', 'titanic.csv')

    # Manejar valores faltantes
    titanic['age'] = titanic['age'].fillna(titanic['age'].median())
    titanic['embarked'] = titanic['embarked'].fillna('S')
    titanic['fare'] = titanic['fare'].fillna(titanic['fare'].median())

    # Codificar variables categóricas
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_class = LabelEncoder()
    le_who = LabelEncoder()

    titanic['sex_encoded'] = le_sex.fit_transform(titanic['sex'])
    titanic['embarked_encoded'] = le_embarked.fit_transform(
        titanic['embarked'])
    titanic['class_encoded'] = le_class.fit_transform(titanic['class'])
    titanic['who_encoded'] = le_who.fit_transform(titanic['who'])

    # Usar todas las características disponibles sin eliminar columnas a priori
    numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    encoded_features = ['sex_encoded', 'embarked_encoded',
                        'class_encoded', 'who_encoded']
    boolean_features = ['adult_male', 'alone']

    feature_columns = numeric_features + encoded_features + boolean_features

    # Asegurar que X no sea None
    X = titanic[feature_columns].copy()
    y = titanic['survived'].copy()

    # Validar datos
    _validate_dataset(X, y, "load_titanic_dataset")

    feature_names = [
        'Clase_Pasajero', 'Edad', 'Hermanos_Conyuges', 'Padres_Hijos', 'Tarifa',
        'Sexo', 'Puerto_Embarque', 'Clase_Social', 'Categoria_Persona', 'Hombre_Adulto', 'Solo'
    ]

    class_names = ['No_Sobrevivió', 'Sobrevivió']

    dataset_info = f"Dataset Titanic: {len(X)} pasajeros, {len(feature_names)} características, supervivencia (binario)"
    task_type = "Clasificación"

    return X, y, feature_names, class_names, dataset_info, task_type


def load_tips_dataset():
    """
    Carga el dataset de propinas desde seaborn.

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo (tip)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases (None para regresión)
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Cargar dataset de propinas desde seaborn con fallback
    tips = _load_seaborn_with_fallback('tips', 'tips.csv')

    # Codificar variables categóricas
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_day = LabelEncoder()
    le_time = LabelEncoder()

    tips['sex_encoded'] = le_sex.fit_transform(tips['sex'])
    tips['smoker_encoded'] = le_smoker.fit_transform(tips['smoker'])
    tips['day_encoded'] = le_day.fit_transform(tips['day'])
    tips['time_encoded'] = le_time.fit_transform(tips['time'])

    # Usar todas las características disponibles sin eliminar columnas a priori
    numeric_features = ['total_bill', 'size']
    encoded_features = ['sex_encoded',
                        'smoker_encoded', 'day_encoded', 'time_encoded']

    feature_columns = numeric_features + encoded_features

    # Asegurar que X no sea None
    X = tips[feature_columns].copy()
    y = tips['tip'].copy()

    # Validar datos
    _validate_dataset(X, y, "load_tips_dataset")

    feature_names = ['Cuenta_Total', 'Tamaño_Grupo',
                     'Sexo', 'Fumador', 'Día_Semana', 'Comida']

    class_names = None  # Es regresión

    dataset_info = f"Dataset Propinas: {len(X)} comidas, {len(feature_names)} características, predicción de propina (regresión)"
    task_type = "Regresión"

    return X, y, feature_names, class_names, dataset_info, task_type


def load_california_housing_dataset():
    """
    Carga el dataset de viviendas de California desde sklearn.

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo (precio)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases (None para regresión)
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Cargar dataset desde sklearn
    housing = fetch_california_housing()

    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="price")

    # Validar datos
    _validate_dataset(X, y, "load_california_housing_dataset")

    feature_names = [
        'Ingresos_Medios', 'Edad_Casa', 'Habitaciones_Promedio', 'Dormitorios_Promedio',
        'Población', 'Ocupación_Promedio', 'Latitud', 'Longitud'
    ]

    class_names = None  # Es regresión

    dataset_info = f"Dataset Viviendas California: {len(X)} distritos, {len(feature_names)} características, precio medio (regresión)"
    task_type = "Regresión"

    return X, y, feature_names, class_names, dataset_info, task_type


def load_penguins_dataset():
    """
    Carga el dataset de pingüinos desde seaborn.

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo (species)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Cargar dataset de pingüinos desde seaborn con fallback
    penguins = _load_seaborn_with_fallback('penguins', 'penguins.csv')

    # Eliminar filas con valores faltantes
    penguins = penguins.dropna()

    # Codificar variables categóricas
    le_island = LabelEncoder()
    le_sex = LabelEncoder()

    penguins['island_encoded'] = le_island.fit_transform(penguins['island'])
    penguins['sex_encoded'] = le_sex.fit_transform(penguins['sex'])

    # Usar todas las características disponibles sin eliminar columnas a priori
    numeric_features = ['bill_length_mm', 'bill_depth_mm',
                        'flipper_length_mm', 'body_mass_g']
    encoded_features = ['island_encoded', 'sex_encoded']

    feature_columns = numeric_features + encoded_features

    # Asegurar que X no sea None
    X = penguins[feature_columns].copy()

    # Codificar la variable objetivo
    le_species = LabelEncoder()
    y = pd.Series(le_species.fit_transform(
        penguins['species']), name="species")

    # Validar datos
    _validate_dataset(X, y, "load_penguins_dataset")

    feature_names = ['Longitud_Pico', 'Profundidad_Pico',
                     'Longitud_Aleta', 'Masa_Corporal', 'Isla', 'Sexo']
    class_names = ['Adelie', 'Chinstrap', 'Gentoo']

    dataset_info = f"Dataset Pingüinos: {len(X)} pingüinos, {len(feature_names)} características, 3 especies"
    task_type = "Clasificación"

    return X, y, feature_names, class_names, dataset_info, task_type


def load_iris_dataset():
    """
    Carga el dataset de iris desde seaborn.

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo (species)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Cargar dataset de iris desde seaborn con fallback
    iris = _load_seaborn_with_fallback('iris', 'iris.csv')

    # Usar todas las características numéricas sin eliminar columnas a priori
    feature_columns = ['sepal_length',
                       'sepal_width', 'petal_length', 'petal_width']

    # Asegurar que X no sea None
    X = iris[feature_columns].copy()

    # Codificar la variable objetivo
    le_species = LabelEncoder()
    y = pd.Series(le_species.fit_transform(iris['species']), name="species")

    # Validar datos
    _validate_dataset(X, y, "load_iris_dataset")

    feature_names = ['Longitud_Sépalo', 'Ancho_Sépalo',
                     'Longitud_Pétalo', 'Ancho_Pétalo']
    class_names = ['Setosa', 'Versicolor', 'Virginica']

    dataset_info = f"Dataset Iris: {len(X)} flores, {len(feature_names)} características, 3 especies"
    task_type = "Clasificación"

    return X, y, feature_names, class_names, dataset_info, task_type


def load_mpg_dataset():
    """
    Carga el dataset de millas por galón desde seaborn.

    Returns:
    --------
    X : DataFrame
        Características
    y : Series
        Variable objetivo (mpg)
    feature_names : list
        Nombres de las características
    class_names : list
        Nombres de las clases (None para regresión)
    dataset_info : str
        Información sobre el conjunto de datos
    task_type : str
        Tipo de tarea
    """
    # Cargar dataset de mpg desde seaborn
    mpg = _load_seaborn_with_fallback('mpg')

    # Eliminar filas con valores faltantes
    mpg = mpg.dropna()

    # Codificar variables categóricas
    le_origin = LabelEncoder()
    mpg['origin_encoded'] = le_origin.fit_transform(mpg['origin'])

    # Usar todas las características numéricas y codificadas sin eliminar columnas a priori
    numeric_features = ['cylinders', 'displacement',
                        'horsepower', 'weight', 'acceleration', 'model_year']
    encoded_features = ['origin_encoded']

    feature_columns = numeric_features + encoded_features

    # Asegurar que X no sea None
    X = mpg[feature_columns].copy()
    y = mpg['mpg'].copy()

    # Validar datos
    _validate_dataset(X, y, "load_mpg_dataset")

    feature_names = ['Cilindros', 'Cilindrada', 'Caballos_Fuerza',
                     'Peso', 'Aceleración', 'Año_Modelo', 'Origen']

    class_names = None  # Es regresión

    dataset_info = f"Dataset MPG: {len(X)} automóviles, {len(feature_names)} características, millas por galón (regresión)"
    task_type = "Regresión"

    return X, y, feature_names, class_names, dataset_info, task_type


def load_additional_dataset(dataset_name):
    """
    Función principal para cargar datasets adicionales.

    Parameters:
    -----------
    dataset_name : str
        Nombre del dataset a cargar

    Returns:
    --------
    X, y, feature_names, class_names, dataset_info, task_type
    """
    try:
        if "Titanic" in dataset_name:
            return load_titanic_dataset()
        elif "Propinas" in dataset_name or "Tips" in dataset_name:
            return load_tips_dataset()
        elif "Viviendas" in dataset_name or "California" in dataset_name:
            return load_california_housing_dataset()
        elif "Pingüinos" in dataset_name or "Penguins" in dataset_name:
            return load_penguins_dataset()
        elif "Iris" in dataset_name:
            return load_iris_dataset()
        elif "MPG" in dataset_name or "Automóviles" in dataset_name:
            return load_mpg_dataset()
        else:
            raise ValueError(
                f"Dataset adicional '{dataset_name}' no reconocido")
    except Exception as e:
        print(f"Error cargando dataset {dataset_name}: {e}")
        raise
