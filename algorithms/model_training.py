"""
Este módulo contiene funciones para entrenar modelos de árboles de decisión, modelos lineales y K-Nearest Neighbors.
Incluye funciones para crear, entrenar y evaluar modelos de clasificación y regresión.
"""

import streamlit as st
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from dataset.dataset_manager import preprocess_data
from algorithms.model_evaluation import evaluate_classification_model, evaluate_regression_model
from viz.nn import safe_get_output_size


def train_decision_tree_classifier(X, y, max_depth, min_samples_split, criterion):
    """
    Crea y entrena un modelo de árbol de decisión para clasificación.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio para medir la calidad de una división

    Returns:
    --------
        modelo entrenado
    """

    # Crear y entrenar el modelo
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )
    model.fit(X, y)

    return model


def train_decision_tree_regressor(X, y, max_depth, min_samples_split, criterion):
    """
    Crea y entrena un modelo de árbol de decisión para regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio para medir la calidad de una división

    Returns:
    --------
        modelo entrenado
    """

    # Crear y entrenar el modelo
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )
    model.fit(X, y)

    return model


def train_decision_tree(X, y, tree_type, max_depth, min_samples_split, criterion):
    """
    Crea y entrena un modelo de árbol de decisión para clasificación o regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    tree_type : str
        Tipo de árbol ('Clasificación' o 'Regresión')
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Número mínimo de muestras para dividir un nodo
    criterion : str
        Criterio para medir la calidad de una división

    Returns:
    --------
        modelo entrenado
    """
    if tree_type == "Clasificación":
        return train_decision_tree_classifier(
            X, y, max_depth, min_samples_split, criterion)
    else:
        return train_decision_tree_regressor(
            X, y, max_depth, min_samples_split, criterion)


def predict_sample(model, X_new):
    """
    Realiza una predicción para una nueva muestra.

    Parameters:
    -----------
    model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo entrenado
    X_new : array
        Características de la muestra a predecir

    Returns:
    --------
    prediction : array
        Predicción del modelo
    proba : array, optional
        Probabilidades de cada clase (solo para clasificación)
    """
    # Realizar predicción
    prediction = model.predict([X_new])

    # Si es un clasificador y soporta probabilidades, calcular probabilidades
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([X_new])[0]
        return prediction, proba

    return prediction, None


def train_linear_regression(X, y):
    """
    Crea y entrena un modelo de regresión lineal.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo

    Returns:
    --------
        modelo entrenado
    """
    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)

    return model


def train_logistic_regression(X, y, max_iter=1000):
    """
    Crea y entrena un modelo de regresión logística.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    max_iter : int, default=1000
        Número máximo de iteraciones para el algoritmo de optimización

    Returns:
    --------
        modelo entrenado
    """
    # Crear y entrenar el modelo
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X, y)

    return model


def train_linear_model(X, y, model_type="Linear", max_iter=1000):
    """
    Crea y entrena un modelo lineal (regresión lineal o logística).

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    model_type : str, default="Linear"
        Tipo de modelo ("Linear" para regresión lineal, "Logistic" para regresión logística)
    max_iter : int, default=1000
        Número máximo de iteraciones (solo para regresión logística)

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    if model_type == "Linear":
        return train_linear_regression(X, y)
    else:  # Logistic
        return train_logistic_regression(X, y, max_iter)


def train_knn_classifier(X, y, n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    Crea y entrena un modelo de K-Nearest Neighbors para clasificación.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    n_neighbors : int, default=5
        Número de vecinos cercanos a considerar
    weights : str, default='uniform'
        Función de peso ('uniform', 'distance')
    metric : str, default='minkowski'
        Métrica de distancia ('minkowski', 'euclidean', 'manhattan')

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """

    # Crear y entrenar el modelo
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    model.fit(X, y)

    return model


def train_knn_regressor(X, y, n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    Crea y entrena un modelo de K-Nearest Neighbors para regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    n_neighbors : int, default=5
        Número de vecinos cercanos a considerar
    weights : str, default='uniform'
        Función de peso ('uniform', 'distance')
    metric : str, default='minkowski'
        Métrica de distancia ('minkowski', 'euclidean', 'manhattan')

    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """

    # Crear y entrenar el modelo
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    model.fit(X, y)

    # Devolver resultados
    return model


def train_knn_model(X, y, task_type="Clasificación", n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    Crea y entrena un modelo de K-Nearest Neighbors para clasificación o regresión.

    Parameters:
    -----------
    X : DataFrame o array
        Características
    y : array
        Variable objetivo
    task_type : str, default="Clasificación"
        Tipo de tarea ('Clasificación' o 'Regresión')
    n_neighbors : int, default=5
        Número de vecinos cercanos a considerar
    weights : str, default='uniform'
        Función de peso ('uniform', 'distance')
    metric : str, default='minkowski'
        Métrica de distancia ('minkowski', 'euclidean', 'manhattan')
    Returns:
    --------
    dict
        Diccionario con el modelo entrenado, datos de entrenamiento/prueba y métricas
    """
    if task_type == "Clasificación":
        return train_knn_classifier(
            X, y, n_neighbors, weights, metric)
    else:
        return train_knn_regressor(
            X, y, n_neighbors, weights, metric)


def train_neural_network(df, target_col, config, learning_rate, epochs, validation_split,
                         early_stopping, patience, reduce_lr, lr_factor, progress_callback=None):
    """
    Entrena una red neuronal con la configuración especificada.
    """
    try:
        # Importar TensorFlow/Keras
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        import time

        # Paso 1: Preparar datos (ya mostrado)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Preprocesamiento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42,
            stratify=y if config['task_type'] == 'Clasificación' else None
        )

        # Paso 2: Construyendo la red neuronal
        if progress_callback:
            progress_callback(
                2, "Construyendo arquitectura de red neuronal con capas y neuronas...")
        time.sleep(0.8)  # Pausa para que se vea el paso

        # Procesar variable objetivo
        label_encoder = None
        if config['task_type'] == 'Clasificación':
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            # Decisión de one-hot encoding basada en función de activación y número de clases
            output_size = safe_get_output_size(config)
            if config['output_activation'] == 'softmax' or (output_size > 1 and config['output_activation'] != 'sigmoid'):
                # Para softmax multiclase o funciones no-estándar multiclase
                y_train_encoded = keras.utils.to_categorical(y_train_encoded)
                y_test_encoded = keras.utils.to_categorical(y_test_encoded)
            # Para sigmoid (binaria o multiclase) mantener encoding simple
        else:
            y_train_encoded = y_train.values
            y_test_encoded = y_test.values

        # 🔧 CONSTRUCCIÓN ROBUSTA DEL MODELO - MÉTODO MEJORADO
        # Opción 1: Usar Functional API (más robusta)
        if True:  # Cambiar a False para usar Sequential
            # FUNCTIONAL API - GARANTIZA model.input
            input_layer = keras.layers.Input(
                shape=(config['input_size'],), name='input_layer')
            x = input_layer

            # Primera capa
            x = keras.layers.Dense(
                config['architecture'][1],
                activation=config['activation'],
                name='dense_1')(x)
            x = keras.layers.Dropout(
                config['dropout_rate'], name='dropout_1')(x)

            # Capas ocultas
            for i, layer_size in enumerate(config['architecture'][2:-1], start=2):
                x = keras.layers.Dense(
                    layer_size, activation=config['activation'],
                    name=f'dense_{i}')(x)
                x = keras.layers.Dropout(
                    config['dropout_rate'], name=f'dropout_{i}')(x)

            # Capa de salida
            output_layer = keras.layers.Dense(
                config['output_size'],
                activation=config['output_activation'],
                name='output_layer'
            )(x)

            # Crear modelo funcional
            model = keras.Model(inputs=input_layer,
                                outputs=output_layer, name='neural_network')

            # ✅ model.input GARANTIZADO aquí

        else:

            # Construir modelo con Input layer (mejores prácticas)
            model = keras.Sequential(name='neural_network')

            # Capa de entrada explícita (elimina warnings)
            model.add(keras.layers.Input(shape=(config['input_size'],)))

            # Primera capa densa (sin input_shape)
            model.add(keras.layers.Dense(
                config['architecture'][1],
                activation=config['activation'],
                name="dense_1"
            ))
            model.add(keras.layers.Dropout(config['dropout_rate']))

            # Capas ocultas
            for i, layer_size in enumerate(config['architecture'][2:-1], start=2):
                model.add(keras.layers.Dense(
                    layer_size,
                    activation=config['activation'],
                    name=f'dense_{i}'
                ))
                model.add(keras.layers.Dropout(config['dropout_rate']))

            # Capa de salida
            model.add(keras.layers.Dense(
                config['output_size'],
                activation=config['output_activation'],
                name="output_layer"
            ))

        # Paso 3: Compilando el modelo
        if progress_callback:
            progress_callback(
                3, "Compilando modelo con optimizadores y funciones de pérdida...")
        time.sleep(0.8)

        # Compilar modelo - Función de pérdida inteligente según activación
        if config['task_type'] == 'Clasificación':
            # Selección inteligente de función de pérdida
            output_size = safe_get_output_size(config)
            if config['output_activation'] == 'sigmoid':
                if output_size == 1:
                    loss = 'binary_crossentropy'  # Estándar para binaria con sigmoid
                else:
                    # Sigmoid multiclase (multi-label)
                    loss = 'binary_crossentropy'
                metrics = ['accuracy']
            elif config['output_activation'] == 'softmax':
                if output_size == 1:
                    # Softmax con 1 neurona es problemático, pero manejar el caso
                    loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
                    st.warning(
                        "⚠️ Softmax con 1 neurona detectada. Puede causar problemas.")
                else:
                    loss = 'categorical_crossentropy'  # Estándar para multiclase con softmax
                    metrics = ['accuracy']
            elif config['output_activation'] == 'linear':
                # Linear para clasificación - usar sparse categorical
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
                st.warning(
                    "⚠️ Función linear detectada en clasificación. Rendimiento puede ser subóptimo.")
            elif config['output_activation'] == 'tanh':
                # Tanh para clasificación - tratar como regresión pero con accuracy
                loss = 'mse'
                metrics = ['accuracy']
                st.warning(
                    "⚠️ Función tanh detectada en clasificación. Comportamiento no estándar.")
            else:
                # Fallback
                loss = 'categorical_crossentropy' if output_size > 1 else 'binary_crossentropy'
                metrics = ['accuracy']
        else:
            # Para regresión
            if config['output_activation'] == 'linear':
                loss = 'mse'  # Estándar para regresión
                metrics = ['mae']
            elif config['output_activation'] in ['sigmoid', 'tanh']:
                loss = 'mse'  # MSE también funciona con activaciones acotadas
                metrics = ['mae']
                if config['output_activation'] == 'sigmoid':
                    st.info(
                        "ℹ️ Sigmoid limitará las salidas a [0,1]. Asegúrate de que tus datos objetivo estén normalizados.")
                else:  # tanh
                    st.info(
                        "ℹ️ Tanh limitará las salidas a [-1,1]. Asegúrate de que tus datos objetivo estén normalizados.")
            elif config['output_activation'] == 'softmax':
                loss = 'mse'
                metrics = ['mae']
                st.error(
                    "⚠️ Softmax en regresión: las salidas sumarán 1. Esto raramente es lo deseado.")
            else:
                loss = 'mse'
                metrics = ['mae']

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        if config['optimizer'] == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif config['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Callbacks
        callbacks = []

        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            )
            callbacks.append(early_stop)

        if reduce_lr:
            reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=lr_factor, patience=patience//2, min_lr=1e-7
            )
            callbacks.append(reduce_lr_callback)

        # Paso 4: Iniciando entrenamiento
        if progress_callback:
            progress_callback(
                4, f"Entrenando red neuronal ({epochs} épocas máximo)... ¡Puede tardar unos minutos!")
        time.sleep(1.0)  # Pausa más larga antes del entrenamiento

        # Entrenar modelo
        history = model.fit(
            X_train, y_train_encoded,
            epochs=epochs,
            batch_size=config['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        # PASO 5: INICIALIZACIÓN COMPLETA DEL MODELO PARA VISUALIZACIONES
        if progress_callback:
            progress_callback(5, "Preparando modelo para visualizaciones...")

        # Forzar construcción completa del modelo
        try:
            # Asegurar que el modelo esté completamente construido
            sample_data = X_test[:1].astype(np.float32)
            _ = model.predict(sample_data, verbose=0)

            # Verificar que model.input esté definido
            if model.input is None:
                # Forzar definición de input si es necesario
                model.build(input_shape=(None, config['input_size']))
                _ = model(sample_data)

            # Crear modelo de activaciones para análisis de capas
            if len(model.layers) > 2:  # Al menos Input + Hidden + Output
                intermediate_layers = []
                for i, layer in enumerate(model.layers):
                    # Excluir la primera capa (Input) y la última (Output)
                    if i > 0 and i < len(model.layers) - 1:
                        if hasattr(layer, 'output') and layer.output is not None:
                            intermediate_layers.append(layer.output)

                if intermediate_layers:
                    import tensorflow as tf
                    activation_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=intermediate_layers
                    )
                    # Verificar que funcione
                    _ = activation_model.predict(sample_data, verbose=0)

                    # Marcar que el modelo de activaciones está listo
                    model._activation_model_ready = activation_model

            # Marcar el modelo como completamente inicializado
            model._fully_initialized = True

        except Exception as init_error:
            # Si falla la inicialización, al menos el modelo base funciona
            if progress_callback:
                progress_callback(
                    5, f"Advertencia en inicialización: {str(init_error)}")
            model._fully_initialized = False

        return model, history, X_test, y_test_encoded, scaler, label_encoder

    except ImportError:
        st.error(
            "❌ TensorFlow no está instalado. Las redes neuronales requieren TensorFlow.")
        st.info("Instala TensorFlow con: `pip install tensorflow`")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {str(e)}")
        return None, None, None, None, None, None
