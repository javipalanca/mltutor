from datetime import datetime
import json

SCATTERPLOT_MATRIX = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar tus datos (reemplaza esto con tu método de carga)
# df = pd.read_csv('tu_archivo.csv')

# Separar características y objetivo
X = df.iloc[:, :-1]  # Todas las columnas excepto la última
y = df.iloc[:, -1]   # Última columna como objetivo

# Estadísticas descriptivas
print(df.describe())

# Distribución del objetivo
fig, ax = plt.subplots(figsize=(10, 6))

# Para clasificación:
if len(np.unique(y)) <= 10:  # Si hay pocas clases únicas
    value_counts = y.value_counts().sort_index()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    ax.set_title("Distribución de Clases")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")
else:  # Para regresión
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title("Distribución de Valores Objetivo")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# Matriz de correlación
corr = X.corr()
# Máscara para triángulo superior
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
           square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Matriz de Correlación de Características")

plt.tight_layout()
plt.show()

# Matriz de dispersión (Scatterplot Matrix)
# Seleccionar características específicas para visualizar
# Reemplaza con tus características de interés
selected_features = ['feature1', 'feature2', 'feature3']
max_features = min(6, len(selected_features))

# Crear el dataframe para la visualización
plot_df = X[selected_features].copy()
plot_df['target'] = y  # Añadir la variable objetivo para colorear

# Generar el pairplot
pair_plot = sns.pairplot(
    plot_df,
    hue='target',
    diag_kind='kde',  # Opciones: 'hist' para histograma o 'kde' para densidad
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
    diag_kws={'alpha': 0.5},
    height=2.5
)
pair_plot.fig.suptitle(
    "Matriz de Dispersión de Características", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
"""

LOAD_NN = """
import pickle
import numpy as np

# Cargar modelo completo
with open('neural_network_complete.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

model = loaded_model['model']
scaler = loaded_model['scaler']
label_encoder = loaded_model['label_encoder']
config = loaded_model['config']

# Hacer predicción con nuevos datos
def predecir(nuevos_datos):
    # nuevos_datos debe ser una lista con valores para cada característica
    datos_escalados = scaler.transform([nuevos_datos])
    prediccion = model.predict(datos_escalados)
    
    if config['task_type'] == 'Clasificación':
        output_size = safe_get_output_size(config)
        if output_size > 2:
            clase_idx = np.argmax(prediccion[0])
            if label_encoder:
                clase = label_encoder.inverse_transform([clase_idx])[0]
            else:
                clase = f"Clase {clase_idx}"
            confianza = prediccion[0][clase_idx]
            return clase, confianza
        else:
            probabilidad = prediccion[0][0]
            clase_idx = 1 if probabilidad > 0.5 else 0
            if label_encoder:
                clase = label_encoder.inverse_transform([clase_idx])[0]
            else:
                clase = f"Clase {clase_idx}"
            return clase, probabilidad
    else:
        return prediccion[0][0]

# Ejemplo de uso:
# resultado = predecir([valor1, valor2, valor3, ...])
# print(resultado)
"""

DECISION_BOUNDARY_CODE = """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

def plot_decision_boundary(model, X, y, feature_names=None, class_names=None):
    \"\"\"
    Visualiza la frontera de decisión para un modelo con 2 características.

    Parameters:
    -----------
    model : Modelo de scikit-learn
        Modelo entrenado con método predict
    X : array-like
        Datos de características (solo se usan las primeras 2 columnas)
    y : array-like
        Etiquetas de clase
    feature_names : list, opcional
        Nombres de las características
    class_names : list, opcional
        Nombres de las clases

    Returns:
    --------
    fig : Figura de matplotlib
    \"\"\"
    # Asegurar que solo usamos 2 características
    X_plot = X[:, :2] if X.shape[1] > 2 else X

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crear objeto de visualización de frontera
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X_plot,
        alpha=0.5,
        ax=ax,
        response_method="predict"
    )

    # Colorear los puntos según su clase
    scatter = ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=y,
        edgecolor="k",
        s=50
    )

    # Configurar etiquetas
    if feature_names and len(feature_names) >= 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("Característica 1")
        ax.set_ylabel("Característica 2")

    # Configurar leyenda
    if class_names:
        legend_labels = class_names
    else:
        legend_labels = [f"Clase {i}" for i in range(len(np.unique(y)))]

    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=legend_labels,
        title="Clases"
    )

    ax.add_artist(legend)
    ax.set_title("Frontera de Decisión")

    return fig

# Para usar:
# fig = plot_decision_boundary(model, X, y, feature_names, class_names)
# plt.show()
"""

VIZ_TREE_CODE = """
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Suponiendo que ya tienes un modelo entrenado (tree_model)
# y los nombres de las características (feature_names) y clases (class_names)

fig, ax = plt.subplots(figsize=(14, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=class_names,  # Solo para clasificación
    filled=True,
    rounded=True,
    ax=ax,
    proportion=True,
    impurity=True
)

plt.tight_layout()
plt.show()

# Para guardar a un archivo:
# plt.savefig('arbol_decision.png', dpi=300, bbox_inches='tight')
"""

TEXT_TREE_CODE = """
from sklearn.tree import export_text

def get_tree_text(model, feature_names, show_class_name=True):
    \"\"\"
    Obtiene una representación de texto de un árbol de decisión.

    Parameters:
    -----------
    model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol entrenado
    feature_names : list
        Nombres de las características
    show_class_name : bool
        Si es True, muestra los nombres de las clases (para clasificación)

    Returns:
    --------
    str
        Representación de texto del árbol
    \"\"\"
    return export_text(
        model,
        feature_names=feature_names,
        show_weights=True
    )

# Ejemplo de uso:
tree_text = get_tree_text(tree_model, feature_names)
print(tree_text)

# Para guardar a un archivo:
# with open('arbol_texto.txt', 'w') as f:
#     f.write(tree_text)
"""

CONFUSION_MATRIX_CODE = """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names)
ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusión')

plt.tight_layout()
plt.show()
"""

PRECISION_CODE = """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd

# Obtener el reporte de clasificación
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

# Extraer precisión por clase
prec_by_class = {
    class_name: report[class_name]['precision'] for class_name in class_names
}

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=list(prec_by_class.keys()), y=list(prec_by_class.values()), ax=ax)
ax.set_ylim(0, 1)
ax.set_title('Precisión por clase')
ax.set_ylabel('Precisión')
ax.set_xlabel('Clase')

plt.tight_layout()
plt.show()
"""

PRED_VS_REAL_CODE = """
import matplotlib.pyplot as plt
import numpy as np

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(y_test, y_pred, alpha=0.5,
                    c=np.abs(y_test - y_pred), cmap='viridis')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Valores reales')
ax.set_ylabel('Predicciones')
ax.set_title('Predicciones vs Valores reales')
plt.colorbar(scatter, ax=ax, label='Error absoluto')

plt.tight_layout()
plt.show()
"""

ERROR_DISTRIBUTION_CODE = """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calcular los errores
errors = y_test - y_pred

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(errors, kde=True, ax=ax)
ax.axvline(x=0, color='r', linestyle='--')
ax.set_title('Distribución de errores')
ax.set_xlabel('Error (Real - Predicción)')

plt.tight_layout()
plt.show()
"""

DECISION_PATH_CODE = """
import numpy as np

def mostrar_camino_decision(tree_model, X_nuevo, feature_names, class_names=None):
    \"\"\"
    Muestra el camino de decisión para un ejemplo específico.
    
    Parameters:
    -----------
    tree_model : DecisionTreeClassifier o DecisionTreeRegressor
        Modelo de árbol de decisión entrenado
    X_nuevo : array
        Ejemplo para predecir (debe ser un solo ejemplo)
    feature_names : list
        Nombres de las características
    class_names : list, optional
        Nombres de las clases (solo para clasificación)
    \"\"\"
    # Asegurar que X_nuevo sea un array numpy 2D con una sola fila
    X_nuevo = np.asarray(X_nuevo).reshape(1, -1)
    
    # Obtener información del árbol
    feature_idx = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    
    # Construir el camino de decisión
    node_indicator = tree_model.decision_path(X_nuevo)
    leaf_id = tree_model.apply(X_nuevo)
    
    # Obtener los nodos en el camino
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    # Mostrar el camino paso a paso
    print("Camino de decisión:")
    for node_id in node_index:
        # Detener si es un nodo hoja
        if leaf_id[0] == node_id:
            continue
            
        # Obtener la característica y el umbral de la decisión
        feature_id = feature_idx[node_id]
        feature_name = feature_names[feature_id]
        threshold_value = threshold[node_id]
        
        # Comprobar si la muestra va por la izquierda o derecha
        if X_nuevo[0, feature_id] <= threshold_value:
            print(f"- {feature_name} = {X_nuevo[0, feature_id]:.4f} ≤ {threshold_value:.4f}")
        else:
            print(f"- {feature_name} = {X_nuevo[0, feature_id]:.4f} > {threshold_value:.4f}")
    
    # Mostrar la predicción final
    prediccion = tree_model.predict(X_nuevo)[0]
    if hasattr(tree_model, 'classes_') and class_names:
        print(f"Predicción final: {class_names[prediccion]}")
    else:
        print(f"Predicción final: {prediccion:.4f}")

# Ejemplo de uso:
# mostrar_camino_decision(tree_model, X_nuevo, feature_names, class_names)
"""


def generate_regression_code():
    """Generate complete Python code for logistic and linear regression."""

    code = f"""
import pickle

# Guardar modelo
with open('modelo_regresion_logistica.pkl', 'wb') as f:
    pickle.dump(model, f)

# Cargar modelo guardado
with open('modelo_regresion_logistica.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

# Ejemplo de uso para nuevas predicciones:
nuevo_ejemplo = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
nuevo_ejemplo = np.array(nuevo_ejemplo).reshape(1, -1)
prediccion = modelo_cargado.predict(nuevo_ejemplo)
"""
    return code


def generate_neural_network_architecture_code(architecture, activation, output_activation,
                                              dropout_rate, optimizer, batch_size, task_type, feature_names):
    """Genera código Python completo para la arquitectura de red neuronal."""

    feature_names_str = str(
        feature_names) if feature_names else "['feature_1', 'feature_2', ...]"

    # Determinar loss y metrics según el tipo de tarea
    if task_type == "Clasificación":
        if architecture[-1] == 1:  # Clasificación binaria
            loss = "binary_crossentropy"
            metrics = "['accuracy']"
            output_processing = """
# Para clasificación binaria
y_pred_classes = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")
"""
        else:  # Clasificación multiclase
            if output_activation == "softmax":
                loss = "sparse_categorical_crossentropy"
            else:
                loss = "categorical_crossentropy"
            metrics = "['accuracy']"
            output_processing = """
# Para clasificación multiclase
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")
print("\\nReporte de clasificación:")
print(classification_report(y_test, y_pred_classes))
"""
    else:  # Regresión
        loss = "mse"
        metrics = "['mae']"
        output_processing = """
# Para regresión
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
"""

    # Generar código de las capas
    layers_code = []
    for i, neurons in enumerate(architecture[1:-1], 1):
        if i == 1:  # Primera capa oculta
            layers_code.append(f"""
# Capa oculta {i}
model.add(Dense({neurons}, activation='{activation}', input_shape=({architecture[0]},)))
model.add(Dropout({dropout_rate}))""")
        else:
            layers_code.append(f"""
# Capa oculta {i}
model.add(Dense({neurons}, activation='{activation}'))
model.add(Dropout({dropout_rate}))""")

    layers_code_str = "".join(layers_code)

    code = f"""# Código completo para Red Neuronal - {task_type}
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. CARGAR Y PREPARAR LOS DATOS
# Reemplaza esta sección con tu método de carga de datos
# df = pd.read_csv('tu_archivo.csv')  # Cargar desde CSV

# Características y variable objetivo
feature_names = {feature_names_str}
# X = df[feature_names]  # Características
# y = df['target']  # Variable objetivo

# 2. PREPROCESAMIENTO
# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
    stratify=y if '{task_type}' == 'Clasificación' else None
)

# Normalizar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Procesar variable objetivo
{"# Para clasificación multiclase con softmax, usar sparse_categorical_crossentropy" if task_type == "Clasificación" and architecture[-1] > 1 and output_activation == "softmax" else ""}
{"# Para clasificación binaria, mantener y como está" if task_type == "Clasificación" and architecture[-1] == 1 else ""}
{"# Para regresión, mantener y como está" if task_type == "Regresión" else ""}

# 3. CONSTRUIR EL MODELO
model = Sequential()
{layers_code_str}

# Capa de salida
model.add(Dense({architecture[-1]}, activation='{output_activation}'))

# 4. COMPILAR EL MODELO
# Seleccionar optimizador
if '{optimizer}' == 'adam':
    optimizer = Adam()
elif '{optimizer}' == 'sgd':
    optimizer = SGD()
elif '{optimizer}' == 'rmsprop':
    optimizer = RMSprop()

model.compile(
    optimizer=optimizer,
    loss='{loss}',
    metrics={metrics}
)

# 5. MOSTRAR RESUMEN DE LA ARQUITECTURA
print("=== ARQUITECTURA DE LA RED NEURONAL ===")
model.summary()

# Información detallada
total_params = model.count_params()
print(f"\\nTotal de parámetros: {{total_params:,}}")
print(f"Arquitectura: {architecture}")
print(f"Funciones de activación: {activation} (ocultas), {output_activation} (salida)")
print(f"Dropout: {dropout_rate}")
print(f"Optimizador: {optimizer}")
print(f"Batch size: {batch_size}")

# 6. ENTRENAR EL MODELO
print("\\n=== INICIANDO ENTRENAMIENTO ===")

# Callbacks opcionales
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7
    )
]

# Entrenar
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,  # Ajusta según necesites
    batch_size={batch_size},
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 7. EVALUAR EL MODELO
print("\\n=== EVALUACIÓN DEL MODELO ===")

# Predicciones
y_pred = model.predict(X_test_scaled)
{output_processing}

# 8. VISUALIZAR HISTORIAL DE ENTRENAMIENTO
plt.figure(figsize=(12, 4))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Métrica principal
plt.subplot(1, 2, 2)
metric_key = list(history.history.keys())[1]  # Primera métrica después de loss
plt.plot(history.history[metric_key], label='Entrenamiento')
plt.plot(history.history[f'val_{{metric_key}}'], label='Validación')
plt.title(f'{{metric_key.title()}} durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel(metric_key.title())
plt.legend()

plt.tight_layout()
plt.show()

# 9. FUNCIÓN PARA NUEVAS PREDICCIONES
def predecir_nueva_muestra(nueva_muestra):
    \"\"\"
    Función para hacer predicciones con nuevos datos.
    
    Parámetros:
    nueva_muestra: lista con valores para cada característica
                  en el orden: {feature_names_str}
    
    Retorna:
    prediccion: resultado de la predicción
    \"\"\"
    # Convertir a array y normalizar
    nueva_muestra = np.array(nueva_muestra).reshape(1, -1)
    nueva_muestra_scaled = scaler.transform(nueva_muestra)
    
    # Predecir
    prediccion = model.predict(nueva_muestra_scaled)
    
    {"# Para clasificación, convertir a clase" if task_type == "Clasificación" else "# Para regresión, devolver valor directo"}
    {"if prediccion[0][0] > 0.5: return 'Clase 1' else return 'Clase 0'  # Binaria" if task_type == "Clasificación" and architecture[-1] == 1 else ""}
    {"return np.argmax(prediccion[0])  # Multiclase" if task_type == "Clasificación" and architecture[-1] > 1 else ""}
    {"return prediccion[0][0]  # Regresión" if task_type == "Regresión" else ""}

# Ejemplo de uso:
# nueva_muestra = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
# resultado = predecir_nueva_muestra(nueva_muestra)
# print(f"Predicción: {{resultado}}")

# 10. GUARDAR EL MODELO
print("\\n=== GUARDANDO MODELO ===")

# Guardar modelo completo
model.save('modelo_red_neuronal.h5')
print("Modelo guardado como 'modelo_red_neuronal.h5'")

# Guardar scaler por separado
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler guardado como 'scaler.pkl'")

# Código para cargar el modelo guardado:
# modelo_cargado = keras.models.load_model('modelo_red_neuronal.h5')
# with open('scaler.pkl', 'rb') as f:
#     scaler_cargado = pickle.load(f)

print("\\n✅ ¡Entrenamiento completado!")
print("Tu red neuronal está lista para hacer predicciones.")
"""

    return code


def generate_neural_network_evaluation_code(config, feature_names, class_names=None):
    """Genera código Python para evaluación de red neuronal."""

    feature_names_str = str(
        feature_names) if feature_names else "['feature_1', 'feature_2', ...]"
    class_names_str = str(
        class_names) if class_names else "['Clase_0', 'Clase_1', ...]"

    if config['task_type'] == "Clasificación":
        if config['output_size'] == 1:  # Clasificación binaria
            evaluation_metrics = """
# Evaluación para clasificación binaria
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_test_flat = y_test.flatten()

# Métricas principales
accuracy = accuracy_score(y_test_flat, y_pred_classes)
precision = precision_score(y_test_flat, y_pred_classes)
recall = recall_score(y_test_flat, y_pred_classes)
f1 = f1_score(y_test_flat, y_pred_classes)

print("=== MÉTRICAS DE CLASIFICACIÓN BINARIA ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test_flat, y_pred_classes)
print("\\nMatriz de Confusión:")
print(cm)
"""
        else:  # Clasificación multiclase
            evaluation_metrics = """
# Evaluación para clasificación multiclase
y_pred_classes = np.argmax(y_pred, axis=1)
if len(y_test.shape) > 1:
    y_test_classes = np.argmax(y_test, axis=1)
else:
    y_test_classes = y_test.flatten()

# Métricas principales
accuracy = accuracy_score(y_test_classes, y_pred_classes)

print("=== MÉTRICAS DE CLASIFICACIÓN MULTICLASE ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Reporte detallado
class_names = """ + class_names_str + """
print("\\nReporte de Clasificación:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_test_classes, y_pred_classes)
print("\\nMatriz de Confusión:")
print(cm)
"""
    else:  # Regresión
        evaluation_metrics = """
# Evaluación para regresión
y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()

# Métricas principales
mse = mean_squared_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print("=== MÉTRICAS DE REGRESIÓN ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Análisis de residuos
residuos = y_test_flat - y_pred_flat
print(f"\\nAnálisis de Residuos:")
print(f"Media de residuos: {np.mean(residuos):.6f}")
print(f"Desviación estándar de residuos: {np.std(residuos):.4f}")
"""

    visualization_code = """
# Visualizaciones
plt.figure(figsize=(15, 10))

# Historial de entrenamiento
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Métrica principal (accuracy o mae)
plt.subplot(2, 3, 2)
metric_key = list(history.history.keys())[1]  # Primera métrica después de loss
plt.plot(history.history[metric_key], label='Entrenamiento')
if f'val_{metric_key}' in history.history:
    plt.plot(history.history[f'val_{metric_key}'], label='Validación')
plt.title(f'{metric_key.title()} durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel(metric_key.title())
plt.legend()
"""

    if config['task_type'] == "Clasificación":
        specific_viz = """
# Matriz de confusión visualizada
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')

# Distribución de confianza
plt.subplot(2, 3, 4)
if """ + str(config['output_size']) + """ == 1:
    confidence = np.maximum(y_pred.flatten(), 1 - y_pred.flatten())
else:
    confidence = np.max(y_pred, axis=1)
plt.hist(confidence, bins=20, alpha=0.7)
plt.title('Distribución de Confianza de Predicciones')
plt.xlabel('Confianza')
plt.ylabel('Frecuencia')
"""
    else:
        specific_viz = """
# Predicciones vs Valores Reales
plt.subplot(2, 3, 3)
plt.scatter(y_test_flat, y_pred_flat, alpha=0.6)
plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')

# Distribución de residuos
plt.subplot(2, 3, 4)
plt.hist(residuos, bins=20, alpha=0.7)
plt.title('Distribución de Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')

# Q-Q plot de residuos
plt.subplot(2, 3, 5)
from scipy import stats
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuos')
"""

    code = f"""# Código completo para Evaluación de Red Neuronal - {config['task_type']}
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGAR MODELO Y DATOS
# Asumiendo que ya tienes:
# - model: tu modelo entrenado
# - X_test, y_test: datos de prueba
# - scaler: preprocessor para normalizar datos
# - history: historial de entrenamiento

print("=== EVALUACIÓN DE RED NEURONAL ===")
print("Tipo de tarea: {config['task_type']}")
print("Arquitectura: {config['architecture']}")

# 2. HACER PREDICCIONES
print("\\nHaciendo predicciones...")
y_pred = model.predict(X_test, verbose=0)

# 3. CALCULAR MÉTRICAS
{evaluation_metrics}

# 4. VISUALIZACIONES
{visualization_code}
{specific_viz}

plt.tight_layout()
plt.show()

# 5. ANÁLISIS DETALLADO DEL MODELO
print("\\n=== INFORMACIÓN DEL MODELO ===")
model.summary()

# Contar parámetros
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\\nParámetros totales: {{total_params:,}}")
print(f"Parámetros entrenables: {{trainable_params:,}}")
print(f"Parámetros no entrenables: {{non_trainable_params:,}}")

# 6. FUNCIÓN PARA NUEVAS PREDICCIONES CON MÉTRICAS
def evaluar_nueva_muestra(nueva_muestra, valor_real=None):
    \"\"\"
    Evalúa una nueva muestra y opcionalmente compara con valor real.
    
    Parámetros:
    nueva_muestra: lista con valores para cada característica
    valor_real: valor real para comparar (opcional)
    \"\"\"
    # Normalizar
    nueva_muestra = np.array(nueva_muestra).reshape(1, -1)
    nueva_muestra_scaled = scaler.transform(nueva_muestra)
    
    # Predecir
    prediccion = model.predict(nueva_muestra_scaled, verbose=0)
    
    print(f"\\n=== PREDICCIÓN INDIVIDUAL ===")
    print(f"Entrada: {{nueva_muestra[0]}}")
    
    {"# Clasificación" if config['task_type'] == "Clasificación" else "# Regresión"}
    {"if prediccion[0][0] > 0.5:" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    clase_pred = 1" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    confianza = prediccion[0][0]" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"else:" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    clase_pred = 0" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"    confianza = 1 - prediccion[0][0]" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"print(f'Clase predicha: {clase_pred}')" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    {"print(f'Confianza: {confianza:.4f} ({confianza*100:.2f}%)')" if config['task_type'] == "Clasificación" and config['output_size'] == 1 else ""}
    
    {"clase_pred = np.argmax(prediccion[0])" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"confianza = np.max(prediccion[0])" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"print(f'Clase predicha: {clase_pred}')" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"print(f'Confianza: {confianza:.4f} ({confianza*100:.2f}%)')" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    {"print(f'Probabilidades por clase: {prediccion[0]}')" if config['task_type'] == "Clasificación" and config['output_size'] > 1 else ""}
    
    {"valor_pred = prediccion[0][0]" if config['task_type'] == "Regresión" else ""}
    {"print(f'Valor predicho: {valor_pred:.4f}')" if config['task_type'] == "Regresión" else ""}
    
    if valor_real is not None:
        {"error = abs(valor_real - clase_pred)" if config['task_type'] == "Clasificación" else "error = abs(valor_real - valor_pred)"}
        print(f"Valor real: {{valor_real}}")
        print(f"Error: {{error:.4f}}")
    
    {"return clase_pred, confianza" if config['task_type'] == "Clasificación" else "return valor_pred"}

# Ejemplo de uso:
# nueva_muestra = [valor1, valor2, valor3, ...]  # Reemplaza con tus valores
# resultado = evaluar_nueva_muestra(nueva_muestra)
# print(f"Resultado: {{resultado}}")

print("\\n✅ Evaluación completada!")
"""

    return code


def generate_neural_network_visualization_code(config):
    """Genera código Python para visualizaciones de red neuronal."""

    code = f"""# Código completo para Visualizaciones de Red Neuronal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Configuración de matplotlib
plt.style.use('default')
sns.set_palette("husl")

print("=== VISUALIZACIONES DE RED NEURONAL ===")
print("Tipo de tarea: {config['task_type']}")
print("Arquitectura: {config['architecture']}")

# 1. HISTORIAL DE ENTRENAMIENTO
def plot_training_history_detailed(history):
    \"\"\"Crea gráficas detalladas del historial de entrenamiento.\"\"\"
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pérdida
    axes[0, 0].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validación', linewidth=2)
    axes[0, 0].set_title('Pérdida durante el entrenamiento', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Métrica principal
    metric_key = list(history.history.keys())[1]  # Primera métrica después de loss
    axes[0, 1].plot(history.history[metric_key], label='Entrenamiento', linewidth=2)
    if f'val_{{metric_key}}' in history.history:
        axes[0, 1].plot(history.history[f'val_{{metric_key}}'], label='Validación', linewidth=2)
    axes[0, 1].set_title(f'{{metric_key.title()}} durante el entrenamiento', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel(metric_key.title())
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (si disponible)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], color='red', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\\nno disponible', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Mejora por época
    loss_improvement = np.diff(history.history['loss'])
    axes[1, 1].plot(loss_improvement, color='purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Mejora por Época (Pérdida)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Cambio en Pérdida')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 2. ANÁLISIS DE PESOS Y SESGOS
def analyze_weights_and_biases(model):
    \"\"\"Analiza la distribución de pesos y sesgos en todas las capas.\"\"\"
    
    layer_weights = []
    layer_biases = []
    
    # Extraer pesos y sesgos
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            if len(weights) >= 2:
                layer_weights.append(weights[0])
                layer_biases.append(weights[1])
    
    if not layer_weights:
        print("No se encontraron capas con pesos")
        return
    
    num_layers = len(layer_weights)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4*num_layers))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, (weights, biases) in enumerate(zip(layer_weights, layer_biases)):
        # Histograma de pesos
        axes[i, 0].hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[i, 0].set_title(f'Distribución de Pesos - Capa {{i+1}}', fontweight='bold')
        axes[i, 0].set_xlabel('Valor de Peso')
        axes[i, 0].set_ylabel('Frecuencia')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Estadísticas de pesos
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        axes[i, 0].axvline(mean_w, color='red', linestyle='--', label=f'Media: {{mean_w:.4f}}')
        axes[i, 0].axvline(mean_w + std_w, color='orange', linestyle=':', label=f'±1σ: {{std_w:.4f}}')
        axes[i, 0].axvline(mean_w - std_w, color='orange', linestyle=':')
        axes[i, 0].legend()
        
        # Histograma de sesgos
        axes[i, 1].hist(biases.flatten(), bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[i, 1].set_title(f'Distribución de Sesgos - Capa {{i+1}}', fontweight='bold')
        axes[i, 1].set_xlabel('Valor de Sesgo')
        axes[i, 1].set_ylabel('Frecuencia')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Estadísticas de sesgos
        mean_b = np.mean(biases)
        std_b = np.std(biases)
        axes[i, 1].axvline(mean_b, color='red', linestyle='--', label=f'Media: {{mean_b:.4f}}')
        axes[i, 1].axvline(mean_b + std_b, color='blue', linestyle=':', label=f'±1σ: {{std_b:.4f}}')
        axes[i, 1].axvline(mean_b - std_b, color='blue', linestyle=':')
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas generales
    all_weights = np.concatenate([w.flatten() for w in layer_weights])
    all_biases = np.concatenate([b.flatten() for b in layer_biases])
    
    print("\\n=== ESTADÍSTICAS GENERALES ===")
    print(f"Número total de pesos: {{len(all_weights):,}}")
    print(f"Número total de sesgos: {{len(all_biases):,}}")
    print(f"\\nPesos - Media: {{np.mean(all_weights):.6f}}, Std: {{np.std(all_weights):.6f}}")
    print(f"Pesos - Min: {{np.min(all_weights):.6f}}, Max: {{np.max(all_weights):.6f}}")
    print(f"\\nSesgos - Media: {{np.mean(all_biases):.6f}}, Std: {{np.std(all_biases):.6f}}")
    print(f"Sesgos - Min: {{np.min(all_biases):.6f}}, Max: {{np.max(all_biases):.6f}}")
    
    # Detección de problemas
    dead_weights = np.sum(np.abs(all_weights) < 1e-6)
    if dead_weights > len(all_weights) * 0.1:
        print(f"\\n⚠️ ADVERTENCIA: {{dead_weights}} pesos muy cerca de cero ({{dead_weights/len(all_weights)*100:.1f}}%)")
    
    if np.std(all_weights) < 0.01:
        print("\\n🚨 PROBLEMA: Pesos muy pequeños, la red puede no haber aprendido correctamente")
    elif np.std(all_weights) > 2:
        print("\\n⚠️ ATENCIÓN: Pesos muy grandes, posible inestabilidad")

# 3. ANÁLISIS DE ACTIVACIONES
def analyze_layer_activations(model, X_sample):
    \"\"\"Analiza las activaciones de cada capa con datos de muestra.\"\"\"
    
    # Crear modelo para extraer activaciones
    layer_outputs = [layer.output for layer in model.layers[:-1]]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Obtener activaciones
    activations = activation_model.predict(X_sample, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]
    
    # Analizar cada capa
    print("\\n=== ANÁLISIS DE ACTIVACIONES ===")
    for i, activation in enumerate(activations):
        print(f"\\nCapa {{i+1}}:")
        print(f"  Forma: {{activation.shape}}")
        print(f"  Media: {{np.mean(activation):.4f}}")
        print(f"  Desviación estándar: {{np.std(activation):.4f}}")
        print(f"  Min: {{np.min(activation):.4f}}, Max: {{np.max(activation):.4f}}")
        
        # Neuronas muertas (siempre 0)
        dead_neurons = np.mean(activation == 0, axis=0)
        dead_ratio = np.mean(dead_neurons > 0.95) * 100
        print(f"  Neuronas muertas: {{dead_ratio:.1f}}%")
        
        # Neuronas saturadas (siempre cerca del máximo)
        if activation.max() > 0:
            saturated_neurons = np.mean(activation >= 0.99 * activation.max(), axis=0)
            saturated_ratio = np.mean(saturated_neurons > 0.95) * 100
            print(f"  Neuronas saturadas: {{saturated_ratio:.1f}}%")
        
        # Visualizar distribución de activaciones
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(activation.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'Distribución de Activaciones - Capa {{i+1}}')
        plt.xlabel('Valor de Activación')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(activation.T)
        plt.title(f'Box Plot por Neurona - Capa {{i+1}}')
        plt.xlabel('Neurona')
        plt.ylabel('Activación')
        plt.xticks(range(1, min(21, activation.shape[1]+1)))  # Máximo 20 neuronas
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 4. FUNCIÓN PRINCIPAL DE VISUALIZACIÓN
def visualize_neural_network_complete(model, history, X_sample=None):
    \"\"\"Ejecuta todas las visualizaciones de la red neuronal.\"\"\"
    
    print("Generando visualizaciones completas...")
    
    # 1. Historial de entrenamiento
    print("\\n1. Analizando historial de entrenamiento...")
    plot_training_history_detailed(history)
    
    # 2. Pesos y sesgos
    print("\\n2. Analizando pesos y sesgos...")
    analyze_weights_and_biases(model)
    
    # 3. Activaciones (si se proporcionan datos)
    if X_sample is not None:
        print("\\n3. Analizando activaciones...")
        analyze_layer_activations(model, X_sample)
    else:
        print("\\n3. Análisis de activaciones omitido (no se proporcionaron datos)")
    
    print("\\n✅ Visualizaciones completadas!")

# EJEMPLO DE USO:
# Asumiendo que tienes:
# - model: tu modelo entrenado
# - history: historial de entrenamiento
# - X_test: datos de prueba para análisis de activaciones

# Ejecutar todas las visualizaciones
# visualize_neural_network_complete(model, history, X_test[:100])

# O ejecutar individualmente:
# plot_training_history_detailed(history)
# analyze_weights_and_biases(model)
# analyze_layer_activations(model, X_test[:100])
"""

    return code


def generate_neural_network_code(config, label_encoder):
    return f"""
# Código generado automáticamente para Red Neuronal
# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NeuralNetworkPredictor:
    def __init__(self):
        # Configuración del modelo
        self.config = {json.dumps(config, indent=8)}

        # Inicializar preprocesadores
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if {bool(label_encoder)} else None

        # Configurar preprocesadores (reemplaza con tus datos de entrenamiento)
        # self.scaler.fit(X_train)  # X_train son tus datos de entrenamiento
        # if self.label_encoder:
        #     self.label_encoder.fit(y_train)  # y_train son tus etiquetas

    def activation_function(self, x, activation):
        \"\"\"Implementa funciones de activación\"\"\"
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return x  # linear

    def predict(self, X):
        \"\"\"Hace predicciones con la red neuronal\"\"\"
        # Normalizar entrada
        X_scaled = self.scaler.transform(X)

        # Forward pass a través de las capas
        # NOTA: Debes implementar los pesos específicos de tu modelo entrenado

        # Ejemplo de estructura (reemplaza con tus pesos reales):
        # layer_1 = self.activation_function(np.dot(X_scaled, weights_1) + bias_1, '{config['activation']}')
        # layer_2 = self.activation_function(np.dot(layer_1, weights_2) + bias_2, '{config['activation']}')
        # output = self.activation_function(np.dot(layer_2, weights_out) + bias_out, '{config['output_activation']}')

        # Placeholder para la implementación
        print("⚠️ Implementa los pesos específicos del modelo en este método")
        return np.zeros((X.shape[0], {config['output_size']}))

    def predict_class(self, X):
        \"\"\"Predicción para clasificación\"\"\"
        predictions = self.predict(X)

        if self.config['task_type'] == 'Clasificación':
            output_size = safe_get_output_size(self.config)
            if output_size > 2:
                class_indices = np.argmax(predictions, axis=1)
                if self.label_encoder:
                    return self.label_encoder.inverse_transform(class_indices)
                else:
                    return [f"Clase {{i}}" for i in class_indices]
            else:
                class_indices = (predictions > 0.5).astype(int).flatten()
                if self.label_encoder:
                    return self.label_encoder.inverse_transform(class_indices)
                else:
                    return [f"Clase {{i}}" for i in class_indices]
        else:
            return predictions.flatten()

# Uso del modelo:
# predictor = NeuralNetworkPredictor()
# 
# # Configura los preprocesadores con tus datos de entrenamiento
# # predictor.scaler.fit(X_train)
# # if predictor.label_encoder:
# #     predictor.label_encoder.fit(y_train)
# 
# # Hacer predicciones
# # nuevos_datos = [[valor1, valor2, valor3, ...]]
# # resultado = predictor.predict_class(nuevos_datos)
# # print(resultado)
"""


def generate_neural_network_complete_code(config, feature_names, class_names=None):
    """Genera código Python completo para entrenar y usar la red neuronal."""
    # Esta función se puede expandir para generar código más completo
    # incluyendo carga de datos, entrenamiento completo, etc.
    pass


def generate_decision_boundary_code(feature_names_boundary, class_names):
    return f"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# Datos de entrenamiento (solo las primeras 2 características)
X_2d = X_train[:, [0, 1]]  # Usar las características seleccionadas
y_train = y_train


# Entrenar el modelo simplificado
model_2d.fit(X_2d, y_train)

# Crear figura
fig, ax = plt.subplots(figsize=(14, 10))

# Visualizar frontera de decisión usando sklearn
disp = DecisionBoundaryDisplay.from_estimator(
    model_2d,
    X_2d,
    response_method="predict",
    alpha=0.6,
    ax=ax,
    grid_resolution=200
)

# Añadir puntos de datos sobre la frontera
scatter = ax.scatter(
    X_2d[:, 0], 
    X_2d[:, 1], 
    c=y_train, 
    edgecolors='black', 
    s=60,
    alpha=0.8
)

# Configurar etiquetas de los ejes
feature_names = {feature_names_boundary}
if len(feature_names) >= 2:
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
else:
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")

# Añadir leyenda de clases
class_names = {class_names}
if class_names:
    legend_labels = class_names
else:
    legend_labels = [f"Clase {{i}}" for i in range(len(np.unique(y_train)))]

# Crear leyenda para las clases
handles, _ = scatter.legend_elements()
ax.legend(handles, legend_labels, title="Clases", loc="best")

ax.set_title("Frontera de Decisión")
plt.tight_layout()
plt.show()

# Para guardar a un archivo:
# plt.savefig('frontera_decision.png', dpi=300, bbox_inches='tight')
"""


def generate_tree_model_export_code(feature_names):
    return f"""
# Código para usar el modelo de árbol de decisión
import pickle
import numpy as np

# Cargar el modelo
with open('decision_tree_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
class_names = model_data['class_names']
task_type = model_data['task_type']

# Función para hacer predicciones
def predecir(valores_caracteristicas):
    \"\"\"
    Hace una predicción con el modelo de árbol de decisión.
    
    Args:
        valores_caracteristicas (list): Lista con valores para cada característica
                                      en el orden: {feature_names}
    
    Returns:
        Predicción del modelo
    \"\"\"
    # Convertir a array numpy
    X = np.array(valores_caracteristicas).reshape(1, -1)
    
    # Hacer predicción
    prediction = model.predict(X)[0]
    
    if task_type == "Clasificación":
        if class_names:
            return class_names[prediction]
        else:
            return f"Clase {{prediction}}"
    else:
        return prediction

# Ejemplo de uso:
# resultado = predecir([valor1, valor2, valor3, ...])
# print(f"Predicción: {{resultado}}")

# Para obtener probabilidades (solo clasificación):
if task_type == "Clasificación" and hasattr(model, 'predict_proba'):
    def predecir_con_probabilidades(valores_caracteristicas):
        X = np.array(valores_caracteristicas).reshape(1, -1)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        resultado = {{
            'prediccion': class_names[prediction] if class_names else f"Clase {{prediction}}",
            'probabilidades': {{
                (class_names[i] if class_names else f"Clase {{i}}"): prob 
                for i, prob in enumerate(probabilities)
            }}
        }}
        
        return resultado
"""
