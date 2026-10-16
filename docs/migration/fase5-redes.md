# Fase 5 — 🧠 Redes Neuronales

Plan de port de `migration-porter` (2026-10-15). Cubre INVENTARIO §6 (🧠 Redes Neuronales), la parte de §0
(progreso, cancelación y diálogos de exportación), §7 (código Keras generado) y §9 de estas pantallas.
Referencia: rama `codex/desktop-migration-study`, la versión Qt. Plantilla: [fase4-knn.md](fase4-knn.md).

La lección es la más grande (XL) y la única con un entrenamiento largo e iterativo. **Reutiliza** C1–C3
(evaluación e interpretación), C8 (rejilla 2D), C9 (formulario), C11/C12 (exportación y código) y
`modelSession`, y añade nueve componentes nuevos (C21–C29, §3). Se propone dividirla en **cuatro subfases**
(F5a–F5d, §9) que se pueden cerrar y revisar por separado.

**Hallazgo principal (D69):** la paridad con Keras puede ser **exacta dentro del redondeo f32** si se quitan
las dos fuentes de azar que Rust no puede reproducir: los pesos iniciales (se toman los de Keras del fixture)
y el orden de los batches (sin barajar). `tools/nn_reference.py` (NumPy, ~430 líneas) reproduce el `fit` de
Keras 3.12 —pérdidas desde los logits, Adam/SGD/RMSprop con las fórmulas de Keras, medias ponderadas por
el tamaño del batch, validación con las últimas filas, EarlyStopping y ReduceLROnPlateau— y coincide con
Keras en **17 de 18 casos durante todas las épocas** (pesos finales a ≤ 3,5e-4, casi siempre ≤ 5e-6). El
único que diverge es una configuración caótica a propósito (RMSprop con LR 0,3). Rust debe seguir esa
especificación. Para el entrenamiento real (con barajado, dropout e inicialización propios) solo se puede
exigir paridad **estadística** (nivel b): bandas sobre 5 semillas.

**Hallazgo de entorno:** con **tensorflow-metal 1.2.0** (GPU de Apple), `model.predict` tras `set_weights`
devuelve salidas **erróneas** (`model(x)` es correcto). La app oculta la GPU (`USE_GPU=0`); los fixtures se
generan en CPU con `enable_op_determinism` y un hilo, y las plantillas de código fuerzan la CPU (D93).

**El contrato IPC (§5) es la frontera.** `rust-ml-engineer` trabaja en `crates/mltutor-nn`, `crates/mltutor-core`
y `crates/mltutor-export`; `tauri-ui-engineer`, en `src-tauri` y `ui`; `ml-lesson-author`, en `content/`.

Índice: [0 Capturas](#0-capturas-de-referencia-qt) · [1 Estructura](#1-estructura-de-la-lección) ·
[2 Pestañas](#2-pestañas) · [3 Componentes](#3-componentes-reutilizados-y-nuevos) ·
[4 Casos límite](#4-casos-límite-y-comportamientos-de-python) · [5 Contrato IPC](#5-contrato-ipc) ·
[6 Fixtures](#6-fixtures-de-paridad) · [7 Contenido](#7-contenido-extraído) ·
[8 Mapeo a Rust](#8-mapeo-a-rust) · [9 Reparto y checklist](#9-subfases-reparto-de-tareas-y-checklist) ·
[10 Decisiones](#10-decisiones-para-el-usuario) · [11 Riesgos](#11-riesgos) ·
[12 Hallazgos](#12-hallazgos-de-la-versión-python) · [13 Cambios propuestos](#13-cambios-propuestos-en-otros-documentos)

## 0. Capturas de referencia (Qt)

En `docs/migration/capturas/qt/fase5/`, generadas con

```sh
uv run python tools/capture_qt.py docs/migration/capturas/qt/fase5 --phase 5 --quantize
```

(ventana real, no `offscreen`; unos 2 min 10 s; 51 pantallas, 89 PNG, 9,8 MB ya cuantizados). Solo 02, 13 y 21
tienen además `NNN_ventana.png`. Como en F2–F4, los expanders abiertos con `expand_all` siguen abiertos en las
capturas siguientes.

**Las vistas WebEngine salen en blanco en Qt** (MEMORY): aquí son casi todas las gráficas (Plotly) y los dos
lienzos HTML (diagrama de la arquitectura y mapa de neuronas muertas). La fase 5 de `capture_qt.py` las
renderiza además con **Chrome headless** desde el HTML que escribe la app: `NNN_web<k>.png` (hasta 8 por
pantalla, en orden de aparición). Las superficies de decisión son matplotlib y sí salen en la captura Qt.
Los entrenamientos de las capturas no tienen semilla (como la app): los números cambian entre ejecuciones.

| Archivo | Pantalla |
|---|---|
| `00_entrenamiento_sin_configurar` … `00e_exportar_sin_entrenar` | ⚙️ sin haber pasado por 🏗️ ("⚠️ Primero debes configurar la arquitectura de la red." + botón) y 📈/🎯/🔮/💾 sin modelo ("⚠️ Primero debes entrenar un modelo.") |
| `01_datos_iris` | 📊 Datos: header propio + `st.info` de consejos + selector y exploración de F1 |
| `02_arquitectura_iris(_ventana, _web1)`, `03_…_expandido(_web1)` | 🏗️ Iris: info entrada/salida, controles, diagrama (`web1`), configuración adicional, resumen, tabla por capa (con el error de D76), complejidad |
| `04_arquitectura_salida_linear`, `05_…_sigmoid_multiclase`, `06_…_oculta_sigmoid` | Mensajes de las combinaciones (`st.error`, `st.warning`) |
| `07_arquitectura_1_capa`, `08_arquitectura_5_capas` | Captions y sliders por capa; diagrama con 1 y 5 capas |
| `09_arquitectura_codigo` | "📝 Generar Código de la Arquitectura" (código literal, D74) |
| `10_entrenamiento_iris`, `11_…_avanzado`, `12_entrenamiento_sin_callbacks` | ⚙️: LR, épocas, validación con la distribución errónea (D73), avanzado con y sin callbacks |
| `13_entrenado_iris(_ventana)` | Tras entrenar: "✅ **¡Entrenamiento completado!**…" y "🎉 ¡Red neuronal entrenada exitosamente!" (los pasos 1–4 no se llegan a ver en Qt) |
| `14_evaluacion_iris(_web1)`, `15_…_expandido` | 📈: resultados (Plotly del historial en `web1`), C1 + C3 y diagnóstico completo |
| `16_historial_iris(_web1)` | 🎯 📊 Historial: 2 subgráficas + 4 métricas |
| `17_pesos_iris(_web1…6)`, `18_…_expandido` | 🧠 Pesos: resumen, métricas globales, 2 histogramas por capa (Plotly) |
| `19_superficie_iris`, `20_superficie_petalos_iris` | 🎯 Superficie (matplotlib, RdYlBu) en el espacio escalado |
| `21_activaciones_iris(_ventana, _web1)`, `22_…_expandido` | 📉 Activaciones: tabla, aviso, expanders por capa y el mapa (`web1`) |
| `23_prediccion_formulario_iris`, `24_prediccion_resultado_iris` | 🔮 C9 con la tarjeta y la tabla de probabilidades |
| `25_exportar_tensorflow` … `29_exportar_metadatos` | 💾 los 4 paneles de Python (SavedModel exportado en 26) |
| `30_arquitectura_cancer_softmax_heredado`, `30b_error_softmax_heredado_cancer` | **Hallazgo 12.4**: al pasar de Iris a Cáncer el selector de salida conserva "softmax" y el entrenamiento falla con un error de TensorFlow en inglés |
| `30c_arquitectura_cancer` … `34_prediccion_resultado_cancer` | Binaria tras elegir sigmoid: entrenamiento, evaluación, superficie, predicción con "Probabilidad clase positiva" |
| `35_arquitectura_diabetes` … `40_prediccion_resultado_diabetes` | Regresión: info "valor numérico", evaluación (R² muy negativo con los valores por defecto, D84), historial MSE/MAE, superficie viridis, activaciones, predicción |
| `41_superficie_moons` | 2 características: superficie sin selectores |
| `42_error_salida_sigmoid_iris`, `43_evaluacion_tras_error` | Multiclase + sigmoid: error al entrenar y **página rota** en 📈 (`AttributeError`, la app marca el modelo como entrenado): D71 |

No se ha capturado: los diálogos nativos, la barra de progreso durante el entrenamiento (el spinner de Qt
dice "Época n/N · pérdida: x") ni las exportaciones HDF5/TFLite (están en el fixture `breast_cancer_default`).

---

## 1. Estructura de la lección

`mltutor/apps/neural_network_app.py::run_neural_networks_app`.

1. Cabecera (F1, `content/inicio/navegacion.toml [lessons.redes]`): "🧠 Redes Neuronales", subtítulo y el
   expander "ℹ️ ¿Qué son las Redes Neuronales?" (`content/redes/intro.md`, ya en F1).
2. 7 pestañas (botones `tab_nn_0..6`, estado `active_tab_nn`) y `---`: 📊 Datos · 🏗️ Arquitectura ·
   ⚙️ Entrenamiento · 📈 Evaluación · 🎯 Visualización · 🔮 Predicciones · 💾 Exportar.
3. En las pestañas 1–6: `st.info("📊 **Dataset actual:** {selected_dataset}")` + `---` (aunque los datos de
   la red sean de otro dataset: §12.6).
4. Navbar: 0 (—, "Continuar a Arquitectura"); 1 ("Volver a Datos", "Continuar a Entrenamiento" + nota "**¿Listo
   para entrenar?** ¡Tu arquitectura está configurada!"); 2 ("Volver a Arquitectura", "Ver Evaluación" solo si
   `model_trained_nn`); 3 ("Volver a Entrenamiento", "Ver Visualización", solo con modelo); 4 ("Volver a
   Evaluación", "Ver Predicciones", siempre); **5 y 6 sin navbar**.

### 1.1 Estado de sesión (Python)

| Clave | Escribe | Contenido |
|---|---|---|
| `active_tab_nn` | pestañas, navbar | 0..6 |
| `nn_df`, `nn_target_col` ("target"), `nn_feature_names`, `nn_class_names`, `nn_task_type`, `nn_dataset_info` | **cada render de 🏗️** (`load_data`) | dataset completo + columna `target` |
| `nn_architecture`, `nn_config` | cada render de 🏗️ | `{architecture, activation, output_activation, dropout_rate, batch_size, optimizer, task_type, input_size, output_size}`; **solo existe si se ha visitado 🏗️** (D90) |
| `layer_{i}` | sliders de neuronas | únicos controles con key de 🏗️ |
| `nn_model`, `nn_history`, `nn_test_data` (X_test escalado, y_test codificado), `nn_scaler`, `nn_label_encoder`, `model_trained_nn` | "🧠 Entrenar Red Neuronal" | `model_trained_nn = True` **aunque el entrenamiento falle** (`nn_model = None`): D71 |
| `nn_y_pred` | 📈 | se escribe y no se usa |
| `viz_type` | panel de 🎯 y de 💾 | global (D13) |
| `bins_weights`, `bins_biases`, `viz_f1/viz_f2`, `viz_f1_reg/viz_f2_reg`, `hist_act_{i}`, `means_{i}`, `nn_export_format` | subvistas | — |
| `feature_{i}` | formulario | compartidos con las otras lecciones |

Los controles **sin key** (activaciones, salida, dropout, batch, optimizador, LR, épocas, validación, callbacks)
conservan su valor por posición en la página: al cambiar de dataset, la salida "softmax" de Iris pasa a Cáncer
(§12.4).

### 1.2 Flujo de entrenamiento (`algorithms/model_training.py::train_neural_network`)

1. `X = nn_df.drop(target)`, `y = nn_df[target]`; **`StandardScaler().fit_transform(X)` con todas las filas**
   (antes de dividir: D72).
2. `train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y si clasificación)`: **20 % fijo**
   (no hay control), **estratificado** en clasificación (a diferencia de D7) y sin estratificar en regresión
   (= el split de F1 con 0,2).
3. Clasificación: `LabelEncoder` sobre y_train; **one-hot** (`to_categorical`) si la salida es softmax o si hay
   más de una neurona y no es sigmoid.
4. Modelo **funcional**: `Input(name='input_layer')` → por cada capa oculta `Dense(n, activation, name='dense_i')`
   + `Dropout(rate, name='dropout_i')` → `Dense(salida, output_activation, name='output_layer')`; Glorot uniform y
   sesgos a cero (por defecto de Keras). `model.layers` = `[input_layer, dense_1, dropout_1, dense_2, dropout_2,
   output_layer]`.
5. Pérdida y métrica según la activación de salida (tabla §2.3.1); `Adam/SGD/RMSprop(learning_rate)` con el resto
   por defecto.
6. Callbacks: `EarlyStopping(monitor='val_loss', patience, restore_best_weights=True)` y
   `ReduceLROnPlateau(monitor='val_loss', factor, patience=max(1, (patience or 10)//2), min_lr=1e-7)`, más el de
   progreso (`report_progress` por época, `check_cancelled` por batch).
7. `fit(X_train, y_train, epochs, batch_size, validation_split=v/100, shuffle=True)`: la validación son las
   **últimas** `n − floor(n·(1 − v))` filas de X_train, sin barajar; el entrenamiento baraja cada época.
8. Pausas artificiales `time.sleep(0.8 + 0.8 + 1.0)` entre pasos (D82) y un "paso 5/4" de preparación.

---

## 2. Pestañas

Todos los literales están en `content/redes/mensajes.toml` (con la clave entre corchetes) y en los `.md`.

### 2.1 📊 Datos (pestaña 0)

`st.header("📊 Selección y Preparación de Datos")` + `st.info(datos-consejos.md)` + selector y exploración de F1
(`DatasetTab.svelte`, sin la descripción de la pestaña de entrenamiento de las otras lecciones).

### 2.2 🏗️ Arquitectura (pestaña 1)

Orden: header → `### 🎛️ Configuración de la Red Neuronal` → `st.info` entrada/salida → expander
"💡 ¿Cómo decidir el tamaño de la red?" (`arquitectura-tamano.md`) → 2 columnas (controles | diagrama) →
`### ⚙️ Configuración Adicional` (3 columnas) → expander "💡 Tips…" (`arquitectura-consejos.md`) → resumen →
tabla → complejidad → código → navbar.

| Control | Tipo | Etiqueta | Rango / opciones | Por defecto | key | Efecto |
|---|---|---|---|---|---|---|
| Capas ocultas | slider entero | `Número de capas ocultas` | 1 … 5 | 2 | — | caption 🟦/🟨/🟥 |
| Neuronas (×capas) | slider entero | `Neuronas en capa oculta {i}` | 1 … 256 | `min(sug, 64)`, `sug = max(10, entrada // i)`, en multiclase `max(sug, 2·clases)` | `layer_{i−1}` | help con la sugerencia |
| Activación oculta | selectbox | `Función de activación` | relu, tanh, sigmoid | relu | — | success/info/warning |
| Activación salida | selectbox | `Función de activación de salida` | clasificación: sigmoid, softmax, linear, tanh; regresión: linear, sigmoid, tanh, softmax | sigmoid (binaria), softmax (multiclase), linear (regresión) | — | mensajes de §2.2.1 |
| Dropout | slider real | `Tasa de Dropout` | 0,0 … 0,8 paso 0,1 | 0,2 | — | caption 🔴/🟢/🟡/🟠 |
| Batch | selectbox | `Tamaño de Batch` | 16, 32, 64, 128, 256 | 64 | — | caption según `n // 4` y 32 |
| Optimizador | selectbox | `Optimizador` | adam, sgd, rmsprop | adam | — | caption |
| Código | botón | `📝 Generar Código de la Arquitectura` | — | — | — | `st.code` + descarga `red_neuronal_arquitectura_{tarea}.py` |

Valores por defecto por dataset (`nn/controls.json`): Iris [4, 10, 10, 3], Vino [13, 13, 10, 3], Cáncer
[30, 30, 15, 1], Dígitos [64, 64, 32, 10], Diabetes [10, 10, 10, 1], California [8, 10, 10, 1], Moons
[2, 10, 10, 1], Titanic [11, 11, 10, 1], Pingüinos [6, 10, 10, 3], MPG [8, 10, 10, 1] en Python (con `name`; 7
entradas en Rust por D8), CSV de 2 columnas [2, 10, 10, k].

#### 2.2.1 Combinaciones de salida

| Tarea | Salida | Mensaje (🏗️) | Al entrenar (Python) | Rust |
|---|---|---|---|---|
| Binaria | sigmoid | ✅ success | BCE + BinaryAccuracy | igual |
| Binaria | softmax | `st.error` | **falla** (rank) + "⚠️ Softmax con 1 neurona…" | D71: bloqueado |
| Binaria | linear | `st.error` | **falla** (`SparseSoftmaxCrossEntropyWithLogits`, etiqueta fuera de rango) | D71 |
| Binaria | tanh | `st.error` | MSE + BinaryAccuracy (umbral 0,5) + "⚠️ Función tanh…" | igual |
| Multiclase | softmax | ✅ | CCE (one-hot) + CategoricalAccuracy | igual |
| Multiclase | sigmoid | `st.warning` | **falla** (`target (None,)` frente a `(None, 3)`) | D71 |
| Multiclase | linear | `st.error` | **falla** (one-hot con pérdida sparse) | D71 |
| Multiclase | tanh | `st.error` | MSE (one-hot) + CategoricalAccuracy | igual |
| Regresión | linear | ✅ | MSE + MAE | igual |
| Regresión | sigmoid / tanh | `st.warning` | MSE + MAE + `st.info` al entrenar | igual |
| Regresión | softmax | `st.error` | MSE + MAE, **la salida vale siempre 1** (gradiente nulo: no aprende) + `st.error` al entrenar | igual |

#### 2.2.2 Diagrama (`viz/nn.py::create_neural_network_visualization`)

HTML con `<canvas>` (`components.html`, alto 400): caja azul "🧠 Arquitectura de Red Neuronal" / "Visualización
dinámica de la estructura de la red para {tarea}", lienzo 2:1 (mínimo 300 px) con fondo degradado
`#f8f9fa → #e9ecef`, conexiones `#BDC3C7` con opacidad 0,3 entre **todas** las neuronas, nodos de radio
`min(20, ancho/(capas·8), max(8, (alto−80)/(máx_neuronas·3)))` con borde `#2C3E50`: entrada `#4ECDC4`, ocultas
`#45B7D1`, salida `#FF6B6B`; etiquetas "Entrada\n(4)", "Oculta i\n(n)", "Salida\n(k)" abajo; leyenda con
"Capas Ocultas (RELU)" y "Capa de Salida (SOFTMAX)". **Sin interactividad ni número de parámetros** (el inventario
lo daba por hecho). Con 256 neuronas dibuja 256 nodos apelotonados y 65 536 líneas. D80: componente nativo.

#### 2.2.3 Resumen

- `🔢 Total de Parámetros` = Σ (nᵢ·nᵢ₊₁ + nᵢ₊₁) (`calculate_network_parameters`, igual a `count_params()` de
  Keras), `📚 Capas Totales` = nº de capas incluida la entrada, `⚖️ Complejidad` (ratio parámetros/filas: Baja
  < 0,1, Media < 1, Alta) y `🧠 Tipo de Red` "Perceptrón Multicapa" en la misma columna.
- `#### 📊 Detalles por Capa`: **tabla errónea** (D76). Con [4, 10, 10, 3] muestra Entrada 4/0, "Oculta 1" con
  110 parámetros (los de la capa 2) y "Salida" con 10 neuronas y 33 parámetros; falta una capa y la suma (143) no
  es el total (193).
- `#### 🔍 Análisis de Complejidad`: success/warning/error con el ratio `{:.3f}`.

### 2.3 ⚙️ Entrenamiento (pestaña 2)

Sin `nn_config`: warning + "🔙 Ir a Arquitectura" y nada más. Con configuración: `st.info(entrenamiento-conceptos.md)`
→ `### 🎛️ Parámetros de Entrenamiento` → 3 columnas → expander avanzado (cerrado) → `### 🚀 Iniciar Entrenamiento`.

| Control | Tipo | Etiqueta | Rango | Por defecto | Efecto |
|---|---|---|---|---|---|
| LR | selectbox | `Tasa de Aprendizaje` | 0,001, 0,01, 0,1, 0,3 | 0,001 | caption |
| Épocas | slider | `Épocas` | 10 … 500 paso 10 | `min(100, min(200, max(50, n // 10)))` (Iris 50, Cáncer **56**, California 100, Titanic 89) | caption <50 / ≤150 / más |
| Validación | slider | `% Datos de Validación` | 10 … 40 | 20 | caption "📊 **Distribución**: Train=…, Val=…, Test=…" **errónea** (D73) |
| Early stopping | checkbox | `Activar Parada Temprana` | — | ✓ | success / warning |
| Paciencia | slider | `Paciencia (épocas)` | 5 … 50 | 10 | solo con ✓; caption ≤5 / ≤15 / más |
| Reducción de LR | checkbox | `Reducir Learning Rate Automáticamente` | — | ✓ | success / info |
| Factor | slider real | `Factor de Reducción` | 0,1 … 0,9 (paso 0,01 en Qt) | 0,5 | solo con ✓; caption ≤0,3 / ≤0,7 / más |
| Entrenar | botón primary | `🧠 Entrenar Red Neuronal` | — | — | §1.2 |

Progreso (Python): spinner + contenedor con "🔄 **Paso 1/4**…" a "🚀 **Paso 4/4**…" y "Paso 5/4" que Qt nunca llega a
pintar; la barra de estado muestra "Época {e}/{N} · pérdida: {x:.4f}" y el porcentaje; **Cancelar** detiene en el
siguiente batch (excepción `Cancelled`, que no es `Exception`) y deja el modelo anterior. Resultado:
"✅ **¡Entrenamiento completado!**…" + "🎉 ¡Red neuronal entrenada exitosamente!", o los errores de
`[entrenamiento]`.

#### 2.3.1 Tamaños reales (Keras)

`n_test = ceil(0,2·n)`, `n_train = n − n_test`, `n_fit = floor(n_train·(1 − v/100))`, `n_val = n_train − n_fit`.
Iris con 20 %: 96 / 24 / 30 (la caption dice 96 / 30 / 24). Cáncer: 364 / 91 / 114 (dice 364 / 113 / 92).

### 2.4 📈 Evaluación (pestaña 3)

1. `evaluate_nn`: `#### 📊 Resultados del Entrenamiento` + métricas de `model.evaluate` (clasificación:
   "🎯 Precisión en Test" = accuracy y "📉 Pérdida en Test", `{:.3f}`; regresión: "📉 Error en Test" = MSE) +
   `### 📈 Progreso del Entrenamiento` + gráfica del historial (C27, 1×2, altura 400).
2. `show_detailed_evaluation` (**C1/C2 + C3** de F2, con "Tu modelo de regresión lineal explica…"): clases con
   argmax o `> 0.5`.
3. Diagnóstico (**C23**): `### 🔍 Diagnóstico del Entrenamiento` → `### 🎯 Estado General` (checklist de 4
   comprobaciones + puntuación + métrica de validación) → `### 📊 Tendencias` (`### 📈 Análisis de Tendencias por
   Fase` con expander, 3 métricas de pendiente, análisis comparativo y consistencia, o "historial muy corto") →
   `### 🔄 Patrones de Comportamiento Detectados` (2 columnas) → expander de contexto → `### ⚠️ Alertas` →
   `### 💡 Recomendaciones` (un aviso y el expander "🔧 Acciones Sugeridas"). Reglas exactas en C23.

`show_neural_network_evaluation` (con "🎲 Confianza Promedio", matriz Plotly y "📝 Generar Código de Evaluación")
es **código muerto** (§12.13).

### 2.5 🎯 Visualización (pestaña 4)

`st.info(visualizacion-intro.md)` + panel de 4 botones (`Historial`, `Pesos`, `Superficie`, `Activaciones`; sin
selección por defecto). Navbar siempre.

- **📊 Historial** (`show_training_history_tab`): expander `historial-interpretar.md`, la gráfica de C27 y 4
  métricas: "🔴 Pérdida Final" (`{:.6f}`, delta `-{mejora:.1f}%`), "🟡 Pérdida Validación" (delta
  `Gap: {gap:.6f} ({pct:+.1f}%)`, color inverso si |pct| ≥ 10), "⏱️ Épocas", "{🎯|📊|📈} Estabilidad"
  (CV de las últimas `min(5, épocas // 2)` pérdidas: < 1 Estable, < 5 Moderado, si no Variable).
- **🧠 Análisis de Pesos** (`show_weights_analysis_tab`): expanders "💡 Cómo interpretar…" y "⚙️ Opciones" (Bins
  pesos 10–120/50, Bins sesgos 5–60/20, escala log); una caption por capa densa ("✅ Capa 2: (4, 10) pesos, (10,)
  sesgos", con la numeración de `model.layers`: D77); tabla resumen (8 columnas); 4 métricas globales y un aviso
  según la desviación típica global (< 0,01, > 2); por capa ("#### 📊 Capa 1…", otra numeración) dos histogramas
  Plotly (pesos `#1f77b4`, sesgos `#ff7f0e`, `nbinsx`, línea gris en 0 y verde discontinua en la media,
  altura 320, `bargap` 0,02) con su caption; `### 🩺 Evaluación General` con los insights.
- **🎯 Superficie de Decisión** (`show_decision_surface_tab`): con más de 2 características, 2 selects (keys
  `viz_f1/viz_f2` o `_reg`, índices 0 y 1; si coinciden, warning y nada más). Rejilla **en el espacio escalado**
  con paso 0,02 y margen 0,5 sobre el rango de **X_test**, el resto de características en la media de X_test;
  clasificación: argmax o `> 0.5`, `contourf` RdYlBu 50 niveles + puntos de prueba; regresión: `contourf` viridis
  100 niveles + colorbar "Valor predicho". Iris: 226 × 246 = 55 596 celdas; en California, con AveOccup, millones
  (D78). El "surrogate" KNN de regresión solo se usa si `predict` falla: inalcanzable (D79).
- **📉 Análisis de Activaciones** (`show_layer_activations_tab`): expander "💡 ¿Qué verás aquí?"; "⚙️ Parámetros de
  Análisis" con 10 controles (tabla de `[activaciones]`); con las **primeras** `muestras` filas de prueba,
  por capa oculta: muertas = todas las activaciones ≤ umbral; constantes = no muertas con std < umbral; saturación
  = % de activaciones ≥ 0,99 si el máximo ≤ 1,2. Tabla, aviso global (> 50 % error, > 20 % warning), expander por
  capa con 4 métricas y 2 checkboxes (histograma de 50 bins, barras de medias), el **mapa** HTML (alto 460,
  ancho `min(1600, max(700, 260 + 180·(capas−1)))`; opciones de dispersión determinista, % de conexiones con un
  hash, radio adaptativo, agrupación en subcolumnas; 💀/😴; tooltip con bias, media, σ y |w|max de cada neurona)
  y el expander final. Con tanh/sigmoid, "muerta" significa "siempre ≤ 0" (§12.10).

### 2.6 🔮 Predicciones (pestaña 5)

`st.header("🔮 Predicciones")` + **C9** con los datos **sin escalar** de todo el dataset (`nn_df` sin la columna
objetivo) y los metadatos del dataset; al predecir, entrada float32 → `nn_scaler.transform` → red. Binaria: tarjeta
"Clase: {nombre}" + "Probabilidad clase positiva: {p:.4f}" (umbral `>= 0.5`); multiclase: tarjeta + tabla
Clase/Probabilidad `{:.4f}`; regresión: "Valor: {v:.4f}". **No hay predicción por lotes ni exploración**: existen en
`show_neural_network_predictions`, que nunca se llama (D81).

### 2.7 💾 Exportar (pestaña 6)

`show_neural_network_export`: `ℹ️ Información del Modelo` (2 `st.info`, con la fecha **actual** como "Fecha
entrenamiento"), `📁 Opciones de Exportación` con 4 botones y `💡 Información Adicional` (`exportar-recomendaciones.md`).

| Panel | Contenido (Python) | Rust (D75) |
|---|---|---|
| 🤖 Modelo TensorFlow | radio SavedModel (.pb) / HDF5 (.h5) / TensorFlow Lite (.tflite), "Modo seguro", "💾 Exportar Modelo TensorFlow" → descarga | **📦 Paquete Keras (.zip)**: `modelo.json` + `reconstruir_modelo.py` (genera .keras, .h5, SavedModel, .tflite, scaler.pkl y modelo_completo.pkl; verificado) + `predecir_numpy.py` + `LEEME.txt` |
| 📊 Modelo Completo | pickle con modelo, scaler, label encoder y config + `LOAD_NN` (usa `safe_get_output_size`, no definida) | incluido en el paquete Keras |
| 📝 Código Python | plantilla que devuelve ceros ("implementa los pesos") | predictor NumPy con los pesos reales (`red_predictor_numpy.py.j2`) + script de entrenamiento (D74) |
| 📋 Metadatos | `st.json` + descarga `neural_network_metadata.json` (`num_classes` = 1 en binaria) | igual, corregido, con la fecha del entrenamiento |
| — | — | **💾 ONNX** (C28): escalado + Gemm + activaciones |

---

## 3. Componentes reutilizados y nuevos

| Comp. | Estado tras F4 | Uso en Redes | Qué falta |
|---|---|---|---|
| Trait `Predictor` | hecho | `Mlp` lo implementa (`predict`, `predict_proba` = salida de la red, `classes`) | `refit` no aplica (la superficie usa el modelo completo con medias, D78) |
| C1/C2 + C3 | hecho | igual | nada |
| C8 rejilla 2D | hecho | modo "resto de características en la media" | variante `fixed = media de X_test` en espacio escalado (D78) |
| C9 formulario | hecho | rangos de todo el dataset, escalado antes de predecir | `PredictionResult::Class.proba` (= salida), "Probabilidad clase positiva" |
| C11/C12 export y codegen | hecho | ONNX MLP, paquete Keras, NumPy, metadatos; 5 plantillas | ramas nuevas |
| `modelSession`, `viewLoader` | hecho | igual | tareas largas con eventos por época |
| **C21** Motor MLP | — | todo | **nuevo** (`mltutor-nn`) |
| **C22** Preparación | — | split estratificado, escalado, validación | **nuevo** (core) |
| **C23** Diagnóstico | — | 📈 | **nuevo** (core, lógica pura) |
| **C24** Diagrama de red | — | 🏗️ y mapa de 📉 | **nuevo** (Svelte, SVG) |
| **C25** Pesos | — | 🧠 | **nuevo** (core + Svelte) |
| **C26** Activaciones | — | 📉 | **nuevo** (nn + Svelte) |
| **C27** Historial en vivo | — | ⚙️, 📈, 📊 | **nuevo** (Svelte + eventos) |
| **C28** Exportación de la red | — | 💾 | **nuevo** (export) |
| **C29** Exploración y lotes | — | 🔮 (D81) | **nuevo** |

### C21. Motor MLP (`mltutor-nn`)

Especificación ejecutable: **`tools/nn_reference.py`** (D69). Reglas (Keras 3.12, verificadas en `nn/exact`):

1. **Arquitectura**: `Dense` = `a·W + b` con `W` de forma (entradas, neuronas) y `b`; activaciones relu
   (gradiente 0 en z = 0), tanh, sigmoid, softmax (por filas, restando el máximo) y linear. Dropout **después de
   cada capa oculta**, invertido: `a·m/(1−rate)` con `m ~ Bernoulli(1−rate)` por muestra y neurona; identidad en
   inferencia (`nn/exact/iris_dropout_masks.json`: valores {0, 2} con rate 0,5).
2. **Inicialización**: Glorot uniforme `U(−l, l)`, `l = sqrt(6/(fan_in + fan_out))`, sesgos 0, con RNG propio
   sembrado (D70). En los tests de nivel (a) se cargan los pesos iniciales del fixture.
3. **Pérdidas** (media sobre el batch; `sum_over_batch_size`): con salida sigmoid + BCE, **desde los logits**
   (`max(z,0) − z·y + log1p(e^−|z|)`; Keras detecta la sigmoid y no recorta con ε); con softmax + CCE, desde los
   logits (`logsumexp`); MSE = media por fila de `(p − y)²` y media del batch. Etiqueta binaria como columna
   (n, 1).
4. **Métricas**: BinaryAccuracy (`p > 0.5` frente a y, también con salida tanh), CategoricalAccuracy (argmax
   frente a argmax del one-hot), MAE. Se calculan con las predicciones **antes** del paso de cada batch.
5. **Optimizadores** (fórmulas de `keras/src/optimizers`, en f32, `t` = nº de pasos acumulado):
   - Adam: `m += (g − m)(1 − β1)`, `v += (g² − v)(1 − β2)`, `α = lr·sqrt(1 − β2ᵗ)/(1 − β1ᵗ)`,
     `w −= m·α/(sqrt(v) + 1e-7)` (ε **sin** corregir: no es la fórmula de burn).
   - RMSprop: `v = 0.9·v + 0.1·g²`, `w −= lr·g/sqrt(v + 1e-7)` (ε **dentro** de la raíz; burn lo suma fuera).
   - SGD: `w −= lr·g` (momentum 0).
6. **Bucle `fit`**: `n_fit = floor(n_train·(1 − v))` filas de entrenamiento y las **últimas** `n_val` de
   validación; batches consecutivos (el último puede ser menor); en la versión real (nivel b), permutación propia
   de las filas de entrenamiento en cada época (D70). Logs de la época: `loss` y métrica = medias **ponderadas por el
   tamaño del batch** durante la época (no la pérdida al final); `val_*` con los pesos del final de la época, por
   batches del mismo tamaño.
7. **Callbacks** (en este orden, al final de cada época):
   - EarlyStopping: si nunca se guardaron pesos, guarda los actuales; `wait += 1`; si `val_loss < best` (estricto,
     `min_delta` 0): `best`, `best_epoch`, pesos y `wait = 0`; si no, para cuando `wait >= patience` y `epoch > 0`.
     **Al terminar restaura SIEMPRE los mejores pesos** (aunque no haya parado antes): en 13 de 18 casos el modelo
     final es el de la mejor época, no el de la última.
   - ReduceLROnPlateau: registra `learning_rate` (antes de reducir) en el historial; mejora si
     `val_loss + 1e-4 < best`; si no, `wait += 1` y con `wait >= paciencia_rlr` (`max(1, paciencia // 2)`, o 5 sin
     early stopping) y `lr > 1e-7` (en f32): `lr = max(lr·factor, 1e-7)`, `wait = 0`; cooldown 0.
8. **Historial**: `{accuracy|mae, loss, val_accuracy|val_mae, val_loss, learning_rate?}` (Keras ordena las claves
   alfabéticamente); `learning_rate` solo con ReduceLROnPlateau.
9. **Progreso y cancelación**: evento por época (época, pérdidas, métricas, LR) y comprobación del `AtomicBool` en
   cada batch; al cancelar se descarta el entrenamiento y queda el modelo anterior (D83).
10. **Determinismo**: un `Mutex` global para el backend de burn (MEMORY) y todo el azar (inicialización, máscaras,
    permutaciones) con un `RandomState` MT19937 propio, en orden fijo (D68 resuelto: MLP propio sin burn ni Mutex; ver §8.4).
11. **Rendimiento**: Python tarda ~1 s en Iris (50 épocas, + 2,6 s de pausas) y ~30 s en California
    (100 épocas, 207 batches por época). Presupuesto Rust: Iris < 150 ms, California < 10 s, sin bloquear la UI.

API propuesta:

```rust
pub enum Activation { Relu, Tanh, Sigmoid, Softmax, Linear }
pub enum OptimizerKind { Adam, Sgd, RmsProp }
pub enum LossKind { BinaryCrossentropy, CategoricalCrossentropy, Mse }   // C22 resuelve cuál
pub enum MetricKind { BinaryAccuracy, CategoricalAccuracy, Mae }
pub struct MlpConfig { pub layers: Vec<usize> /* [entrada, ocultas.., salida] */, pub hidden: Activation,
                       pub output: Activation, pub dropout: f32 }
pub struct TrainConfig { pub optimizer: OptimizerKind, pub learning_rate: f32, pub batch_size: usize,
                         pub epochs: usize, pub validation_split: f64, pub early_stopping: Option<usize>,
                         pub reduce_lr: Option<f32>, pub shuffle: bool, pub seed: u64 }
pub struct Mlp { config: MlpConfig, weights: Vec<(Array2<f32>, Array1<f32>)> }   // exportable sin burn
pub struct EpochLog { pub epoch: usize, pub loss: f64, pub metric: f64, pub val_loss: f64, pub val_metric: f64,
                      pub learning_rate: Option<f64> }
pub struct History { pub epochs: Vec<EpochLog>, pub best_epoch: Option<usize>, pub stopped_epoch: Option<usize> }
pub fn train(cfg: &MlpConfig, tc: &TrainConfig, loss: LossKind, metric: MetricKind, x: &[f32], y: &[f32],
             init: Option<Vec<(Array2<f32>, Array1<f32>)>>, on_epoch: &mut dyn FnMut(&EpochLog),
             cancel: &AtomicBool) -> Result<(Mlp, History), NnError>;
impl Mlp { pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32>;
           pub fn hidden_activations(&self, x: ArrayView2<f32>) -> Vec<Array2<f32>>; }
```

(`Array*` de ndarray 0.17 en `mltutor-nn` o `Vec<f32>` en la frontera con core, que usa ndarray 0.16: MEMORY.)

### C22. Preparación de datos (`mltutor-core::nn_data`)

- `stratified_split(codes, test_size, seed) -> (train, test)`: `tools/nn_reference.py::stratified_split`
  (`_approximate_mode` + `RandomState.permutation`, que Rust ya reproduce con `rand_mt`). Verificado en 8 datasets
  (`nn/split_stratified.json`). Ojo: `continuous = counts/total·n_draws` en f64 con ese orden de operaciones, los
  empates del resto se comparan con `==` exacto y `rng.choice(..., replace=False)` consume una permutación completa
  aunque se elijan todos.
- Regresión: el split de F1 con `test_size = 0,2`.
- `StandardScaler` sobre **todas** las filas (D72): media y desviación poblacional; escala 1 en columnas constantes.
  Tolerancia 1e-12 (numpy suma por pares según la disposición en memoria: 2e-15 entre un DataFrame y otro).
- `keras_validation_split(n, v) = floor(n·(1 − v))` en f64.
- `resolve_training(tarea, salida, n_salida) -> {one_hot, loss, metric}` o `NnError::UntrainableOutput` (D71),
  con la tabla §2.2.1 (`nn_reference.resolve_training`).
- Sugerencias de la UI: neuronas por capa, épocas por defecto, caption de distribución real (D73), complejidad.

### C23. Diagnóstico del entrenamiento (`mltutor-core::nn_diagnostics`)

Lógica pura sobre el historial (`model_evaluation.py` líneas 513–1180), salida = la lista de bloques de la vista
con sus claves de `[diagnostico]`. Detalles que hay que copiar:

- `convergence_rate`: con ≥ 10 épocas `(mean(loss[:n//4]) − mean(loss[-n//4:]))/mean(loss[:n//4])`; **ojo**:
  `loss[-n//4:]` es `loss[(-n)//4:]` en Python, las últimas `ceil(n/4)` épocas (10 épocas → 2 al principio y 3 al
  final). Con < 10, `(loss[0] − loss[−1])/loss[0]`. `loss_converged = rate > 0.01`.
- `overfitting_gap`: ventana `min(5, n // 2)` (0 con 1 época → media vacía: NaN en Python), `(mean(val) −
  mean(train))/mean(train)`, sobreajuste si > 0,15.
- `stability`: con ≥ 5 épocas `std/mean` de las 5 últimas (std poblacional), estable si < 0,05; si no, `inf`
  (se imprime "inf%").
- Clasificación: `accuracy`/`val_accuracy` finales, buenos si > 0,7, excelente si val > 0,85, sobreajuste de
  accuracy si train − val > 0,15. Regresión: `good_mae = val_mae < mean(mae[:5])` (**solo relativo**: §12.9).
- Calidad: excellent/good/moderate/poor; `converged = loss_converged ∧ calidad ∈ {excellent, good}`;
  `is_stable = loss_stable ∧ calidad ≠ poor`.
- Puntuación = aciertos de 4 comprobaciones; textos según calidad y porcentaje (tabla `[diagnostico]`).
- Tendencias con ≥ 10 épocas: pendiente de mínimos cuadrados de cada tercio (`[:n//3]`, `[n//3:2n//3]`,
  `[2n//3:]`); umbrales de §`[diagnostico]`; "fase más activa" = primer máximo de |pendiente|; consistencia =
  std poblacional de las 3 pendientes.
- Patrones: siempre uno de convergencia (> 0,5 / > 0,1 / > 0,01 / resto) + sobreajuste + inestabilidad; el de
  plateau y "sin patrones" son inalcanzables (no se portan, D84).
- Recomendaciones: tabla de `generate_training_recommendations`; acción extra al principio si la calidad es poor.
- D84: además, el aviso NUEVO de R² negativo y el texto crítico corregido en regresión.

Fixture: `nn/diagnostics.json` (14 historiales sintéticos con todas las ramas) y `expected.diagnostics` de cada
caso `nn/exact` (historiales reales). Tolerancia: textos exactos; números 1e-12.

### C24. Diagrama de la red (`NetworkDiagram.svelte`, D80)

Un componente SVG para 🏗️ y para el mapa de 📉, sin HTML incrustado.

- Modo arquitectura: colores, etiquetas y leyenda de §2.2.2; **mejoras**: tooltip por capa con neuronas y
  parámetros (`[arquitectura] diagrama_tooltip_capa`), el total en el pie; con más de 24 neuronas en una capa se
  dibujan 24 (las 12 primeras, "⋮" y las 12 últimas) con la nota "… y {n} neuronas más"; conexiones con opacidad
  decreciente según su número (con 256 × 256 Python dibujaba 65 536 líneas).
- Modo activaciones: el mapa de Python con todas sus opciones (dispersión con el mismo hash determinista, % de
  conexiones con el mismo hash, radio adaptativo `min(14, max(5, alto/(n+6)))` o 10, agrupación en subcolumnas
  con `floor(alto_útil/espaciado)` por subcolumna y como máximo `máx_subcolumnas`, separación `min(32,
  hueco·0,25)`), colores (entrada `#4ECDC4`, ocultas `#1976d2`, salida `#FF6B6B`), opacidades (0,25 muerta, 0,45
  constante), 💀/😴, etiquetas arriba "Entrada (4)", leyenda y tooltip con bias, w̄, σ, |w|max y estado
  (`formatVal`: `toExponential(2)` si |v| < 1e-4, si no 4 decimales).
- Datos: `NetworkMapView` (§5.3), calculado en Rust.

### C25. Pesos (`mltutor-core::nn_stats::weights`)

Por capa densa: forma, media, std (poblacional), % |w| < 1e-3 (la tabla lo redondea a 1 decimal), media y std de
sesgos, nº de parámetros, mínimos y máximos; globales sobre todos los pesos (sin sesgos); los avisos e insights.
Histogramas en Plotly (`nbinsx` de Plotly: la UI le pasa los valores y `nbinsx`, como Python; no se calculan bins
en Rust). Nombres de capa coherentes (D77). Fixture: `expected.weights` (`nn_reference.weight_stats` coincide).

### C26. Activaciones (`mltutor-nn::Mlp::hidden_activations` + `core::nn_stats::activations`)

Primeras `muestras` filas de X_test (en f32), salida de cada capa oculta **antes del dropout**; muertas, constantes,
saturación, medias por neurona; el histograma de 50 bins lo hace Plotly con los valores. Fixture:
`expected.activations_default` (50 muestras, umbrales 0 y 1e-3) y `activations_options` (200, 0,05, 0,01), con
`histogram_values`; `iris_relu_3layers_lr03` tiene 15 neuronas muertas.

### C27. Historial en vivo (`TrainingHistoryChart.svelte`)

La gráfica 1×2 de Python (títulos, ejes, azul/rojo, altura 400) reutilizada en ⚙️ (en vivo, actualizándose con cada
evento de época, como mucho a 20 fps), en 📈 y en 📊 Historial (con las 4 métricas de §2.5). En regresión,
"Pérdida (MSE)" y "Error Absoluto Medio". Marca la mejor época si EarlyStopping restauró sus pesos (NUEVO) y
las reducciones de LR (NUEVO, del `learning_rate` del historial).

### C28. Exportación de la red (`mltutor-export::{onnx::mlp, package::keras}`)

- **ONNX** (`onnx_proposal_model` en `tools/golden_nn.py`): entrada `float_input` [N, F] f32 **sin escalar** →
  `Sub(scaler_mean)` → `Div(scaler_scale)` → por capa `Gemm(W, B)` + `Relu|Tanh|Sigmoid|Softmax(axis=1)|Identity`;
  salidas `label` (int64: `ArgMax(axis=1, keepdims=0)` o `Greater(0.5) → Cast → Squeeze`) y `probabilities`, o
  `variable` en regresión. ir_version 8, opset 15, nombres de nodo únicos. Verificado con `onnx.checker`
  (full_check) y `onnx.reference`: error ≤ 3,8e-6 frente a Keras (3,1e-5 en Diabetes con LR 0,1, por la magnitud de
  la salida) y 0 etiquetas distintas en los 18 casos. onnxruntime, en el job de F6.
- **Paquete Keras ZIP**: `modelo.json` (formato `mltutor-red-neuronal/1`: tarea, nombres, clases, arquitectura,
  activaciones, dropout, pérdida, métrica, optimizador con la LR final, escalador con `n_samples`, capas con
  `name/units/activation/kernel/bias`, `config` de la app y `check` con 5 filas sin escalar y su salida) +
  `reconstruir_modelo.py` + `predecir_numpy.py` + `LEEME.txt`. Ejecutado en los casos `full`: pasa la comprobación
  (error < 1e-4) y genera los 6 archivos.
- **Metadatos** JSON (§2.7) y nombres de archivo: `red_neuronal.onnx`, `modelo_red.zip`,
  `red_entrenamiento.py`, `red_predictor_numpy.py`, `red_neuronal_metadatos.json`.

### C29. Exploración y predicción por lotes (D81)

Subvistas nuevas de 🔮 (panel "🔮 Predicción individual" · "📊 Predicción por lotes" · "🎲 Exploración
interactiva"), con los textos de `[lotes]` y `[exploracion]` y los `.md`. Lotes: diálogo nativo para abrir un CSV,
comprobación de columnas (faltan → error; sobran → aviso), vista previa, predicciones con "Predicción",
"Probabilidad"/"Confianza", y "📥 Descargar Resultados" (`predicciones_neural_network.csv`). Exploración: muestra base
(índice del dataset), su predicción, característica a explorar, 50 valores de mín a máx, gráfica Plotly y el
análisis (rango, volatilidad o impacto relativo) con sus umbrales; NUEVO en multiclase: el análisis sobre la
probabilidad de la clase base. Sin fixture de Python (código muerto): tests unitarios en core con la red de
`nn/exact/iris_default` (pesos finales).

---

## 4. Casos límite y comportamientos de Python

| Caso | Python | Rust (propuesta) |
|---|---|---|
| Salida no entrenable (multiclase sigmoid/linear, binaria softmax/linear) | Error de TF en inglés; `model_trained_nn = True`; 📈 y 🎯 se rompen | D71: bloquear con mensaje en español |
| Clase con 1 muestra (CSV `one_class_per_row`) | "The least populated class in y has only 1 member…" y páginas rotas | `NnError::TooFewPerClass` en español (el split estratificado lo exige) |
| Cambiar de dataset tras usar Iris | El selector de salida sigue en "softmax" (Cáncer falla) | D85: valor recomendado al cambiar de tarea/nº de clases |
| No visitar 🏗️ antes de ⚙️ | "⚠️ Primero debes configurar…" | D90: configuración por defecto siempre |
| Cambiar 🏗️ tras entrenar | 📈 sigue con el modelo viejo; el banner dice el dataset actual | D91: banner de modelo obsoleto |
| Regresión con softmax | Salida siempre 1, pesos sin cambiar | Igual (paridad; `diabetes_softmax_output`) |
| Diabetes con valores por defecto | R² ≈ −3,4 y "✅ Error de regresión aceptable" | D84 (aviso) y D88 |
| Pérdida NaN (LR alto) | Diagnóstico con NaN | Igual + aviso; el historial admite NaN (JSON `"nan"`) |
| EarlyStopping sin parar | Restaura los pesos de la mejor época al final | Igual (C21.7) |
| Superficie con muchas celdas (California + AveOccup) | Millones de predicciones; se cuelga | D78: rejilla limitada |
| Superficie con 1 característica (CSV) | `IndexError` | Mensaje "se necesitan 2 características" |
| Activaciones tanh/sigmoid | "Muerta" = siempre ≤ 0 | Igual + nota (§12.10) |
| Umbral binario | Evaluación `> 0.5`, formulario `>= 0.5` | D87 |
| Batch mayor que el train (CSV de 30 filas) | Un batch por época | Igual |
| Cancelar | Excepción en el batch; modelo anterior intacto | D83 |
| tensorflow-metal en el equipo del estudiante | `predict` erróneo tras `set_weights` | Plantillas con CPU forzada (D93) |

---

## 5. Contrato IPC

### 5.1 Convenciones

Las de F1–F4 (§5.1 de F2): comandos `async` + `spawn_blocking`, `build.rs` + `capabilities/default.json`, ts-rs,
`CommandError` en español, `model_id` (`ModelStale`), `seq` en `train_model`, evento `task-progress`. Nuevo:
evento **`nn-epoch`** por época y cancelación por `task_id` con el comando existente.

### 5.2 Estado

`LessonId::Redes` (nuevo). `TrainedModel` guarda `Mlp`, `History`, el escalador, el split, la configuración y la fecha
del entrenamiento. El estado de interfaz (pestaña, configuración de 🏗️ y ⚙️, subvistas D13, pares de la superficie,
opciones de pesos y activaciones, valores del formulario, estado de la exploración y del lote) vive en
`lessons/redes/state.svelte.ts`. La configuración se valida en Rust al entrenar.

### 5.3 Tipos (nuevos o ampliados)

```rust
#[serde(rename_all = "snake_case")] pub enum Activation { Relu, Tanh, Sigmoid, Softmax, Linear }
#[serde(rename_all = "snake_case")] pub enum Optimizer { Adam, Sgd, Rmsprop }
pub struct NnArchitecture { pub hidden: Vec<u16> /* 1..=5 capas, 1..=256 */, pub activation: Activation,
                            pub output_activation: Activation, pub dropout: f32 /* 0..=0.8 */,
                            pub batch_size: u16, pub optimizer: Optimizer }
pub struct NnTraining { pub learning_rate: f32, pub epochs: u16 /* 10..=500 */, pub validation_pct: u8 /* 10..=40 */,
                        pub patience: Option<u8>, pub lr_factor: Option<f32> }
pub enum ModelConfig { /* F2–F4… */ NeuralNet { arch: NnArchitecture, training: NnTraining } }

// 🏗️ (síncrono y barato; se llama al cambiar un control, debounce 50 ms)
pub struct ArchitectureRequest { pub arch: NnArchitecture }        // dataset = el seleccionado
pub struct ArchitectureView { pub input: u32, pub output: u32, pub task: TaskKind, pub n_classes: Option<u32>,
                              pub n_rows: u32, pub suggested_neurons: Vec<u16>, pub layers: Vec<LayerRow>,
                              pub total_params: u64, pub complexity_ratio: f64, pub complexity: Complexity,
                              pub output_check: OutputCheck /* ok|warning|error|untrainable + clave */,
                              pub output_options: Vec<Activation>, pub recommended_output: Activation,
                              pub default_epochs: u16, pub batch_hint: BatchHint, pub size_tips: SizeTips }
pub struct LayerRow { pub name: String /* "Entrada" | "Oculta i" | "Salida" */, pub neurons: u32,
                      pub params: u64, pub activation: Option<Activation> }
pub struct SplitSizes { pub train: u32, pub val: u32, pub test: u32 }   // D73, dentro de TrainingDefaults
pub struct TrainingDefaults { pub epochs: u16, pub suggested_epochs: u16, pub split: SplitSizes }

// ⚙️ entrenamiento: train_model (F2) con ModelConfig::NeuralNet; eventos
pub struct NnEpochEvent { pub task_id: u32, pub epoch: u16, pub epochs: u16, pub loss: f64, pub val_loss: f64,
                          pub metric: f64, pub val_metric: f64, pub learning_rate: Option<f64>,
                          pub phase: TrainPhase /* preparing|building|compiling|training */ }
pub struct NnSummary { pub arch: NnArchitecture, pub training: NnTraining, pub layers: Vec<u32>, pub loss: String,
                       pub metric: String, pub epochs_trained: u16, pub best_epoch: Option<u16>,
                       pub stopped_early: bool, pub n_fit: u32, pub n_val: u32, pub n_test: u32,
                       pub trained_at: String /* fecha local, D75 */ }
// TrainedModelInfo añade `nn: Option<NnSummary>`. CommandError añade UntrainableOutput { key, n_classes },
// TooFewPerClass { class_name }.

pub struct NnEvaluationView { pub test_loss: f64, pub test_metric: f64, pub metric: String,
                              pub evaluation: EvaluationView /* C1/C2 de F2 */, pub r2_negative: Option<f64> }
pub struct HistoryView { pub epochs: Vec<EpochLogDto>, pub best_epoch: Option<u16>, pub lr_drops: Vec<u16>,
                         pub cards: HistoryCards /* 4 métricas de §2.5 */ }
pub struct DiagnosticsView { pub blocks: Vec<DiagBlock> }   // { kind: notice|metric|text|list|expander, key, args }
pub struct WeightsView { pub layers: Vec<LayerWeights /* name, shape, stats, weights: Vec<f32>, biases: Vec<f32> */>,
                         pub global: GlobalWeights, pub verdict: String, pub insights: Vec<String> }
pub struct ActivationsRequest { pub model_id: u32, pub sample_size: u16, pub dead_threshold: f32,
                                pub constant_std: f32 }
pub struct ActivationsView { pub rows_used: u32, pub layers: Vec<LayerActivations /* name, units, dead, constant,
                             saturation_pct, mean_act, std_act, neuron_means, values (para el histograma) */>,
                             pub dead_pct: f64, pub map: NetworkMapView }
pub struct NetworkMapView { pub layers: Vec<u32>, pub neurons: Vec<Vec<NeuronInfo /* bias, mean_w, std_w,
                            max_abs_w, state */>> }
// La superficie usa decision_grid (F2) con GridMode::NnMeans (D78); la exploración y el lote:
pub struct ExploreRequest { pub model_id: u32, pub row: u32, pub feature: u32 }
pub struct ExploreView { pub base: Vec<f64>, pub base_prediction: PredictionResult, pub xs: Vec<f64>,
                         pub ys: Vec<f64>, pub classes: Option<Vec<u32>>, pub stats: ExploreStats, pub verdict: String }
pub struct BatchPredictView { pub rows: u32, pub missing: Vec<String>, pub extra: Vec<String>,
                              pub preview: Table, pub results: Option<Table> }

pub enum ExportFormat { /* … */ Onnx, KerasPackage, NnNumpyPredictor, NnMetadata }
pub enum CodeTemplate { /* … */ NnTraining }
```

### 5.4 Comandos

| Comando | Entrada | Salida | Errores | Notas |
|---|---|---|---|---|
| `nn_architecture` | `ArchitectureRequest` | `ArchitectureView` | `NoDataset` | < 5 ms; incluye `TrainingDefaults` |
| `train_model` (F2) | `TrainRequest` con `ModelConfig::NeuralNet` | `TrainedModelInfo` (+ `nn`) | `UntrainableOutput`, `TooFewPerClass`, `NonFinite`, `Cancelled` | Tarea larga: `task-progress` (fases) + `nn-epoch`; cancelación por batch |
| `evaluate_model` (F2) | `model_id` | `NnEvaluationView` | `ModelStale` | C1/C2 + pérdida/métrica de test |
| `nn_history` | `model_id` | `HistoryView` | `ModelStale` | |
| `nn_diagnostics` | `model_id` | `DiagnosticsView` | `ModelStale` | C23 |
| `nn_weights` | `model_id` | `WeightsView` | `ModelStale` | |
| `nn_activations` | `ActivationsRequest` | `ActivationsView` | `ModelStale` | < 20 ms |
| `decision_grid` (F2) | `GridRequest` | `DecisionGrid` | `InvalidSelection` | D78 |
| `prediction_form`, `predict_one` (F2) | — | — | — | escalado dentro; `proba` = salida |
| `nn_explore` | `ExploreRequest` | `ExploreView` | `ModelStale` | D81 |
| `nn_predict_batch` | `model_id` + ruta | `BatchPredictView` | `Csv*`, `MissingColumns` | D81; guardar con `save_file` |
| `generate_code(model_id, NnTraining)` (F2) | — | `String` | — | `red_entrenamiento.py.j2` (también sin modelo, con la configuración actual: D74) |
| `export_model(model_id, format)` (F2) | `Onnx`, `KerasPackage`, `NnNumpyPredictor`, `NnMetadata` | `Option<String>` | `Export`, `FileSave` | C28 |

Presupuesto: todas las vistas < 100 ms con California salvo el entrenamiento (< 10 s, en segundo plano).

---

## 6. Fixtures de paridad

`uv run python tools/export_golden.py --only nn --only code/nn_training` y `--check` (exit 0 verificado). Código en
`tools/golden_nn.py` (registrado desde `export_golden.py`; reutiliza sus helpers y `golden_trees.dumps_compact`) y
`tools/nn_reference.py`. Versiones en `meta.tensorflow`: TensorFlow 2.16.2 (tensorflow-macos), Keras 3.12.0,
tensorflow-metal 1.2.0 instalado pero **la GPU se oculta** (`tf.config.set_visible_devices([], 'GPU')`, como la app),
`enable_op_determinism`, 1 hilo (`TF_NUM_INTRAOP/INTEROP_THREADS=1`), `keras.utils.set_random_seed`. 35 archivos, 10 MB.
Generación completa: unos 6 min (el nivel b, 2,5 min; California, 1 min).

| Ruta | Qué ejercita |
|---|---|
| `nn/exact/<caso>.json` (18 + `iris_dropout_masks`) | Nivel (a) (§6.1). Casos `full` (Iris, Cáncer, Diabetes por defecto): todas las vistas, las exportaciones TF de Python y el paquete Keras ejecutado; casos `views`: las vistas y el paquete sin ejecutar |
| `nn/stat/<caso>.json` (11) | Nivel (b): Iris (adam, sgd 0,01, rmsprop), Cáncer, Vino, Diabetes (0,001 y 0,1), California (100 épocas), Moons y 2 CSV |
| `nn/diagnostics.json` | 14 historiales sintéticos (ideal, sobreajuste, pobre, moderado, inestable, subida final, 3, 8, 10, 11 y 20 épocas planas, 3 de regresión) |
| `nn/controls.json` | 12 datasets por defecto; variaciones de Iris/Cáncer/Diabetes (salida ×4, oculta ×3, capas 1/3/5, dropout 0/0,2/0,3/0,6, batch ×5, optimizador ×3, 256 neuronas); ⚙️ (LR ×4, épocas 10/50/160/500, validación, paciencia, factor, sin callbacks) y la tabla `distribucion_real_vs_caption` para 10–40 %; 6 códigos de arquitectura (literal = plantilla; binaria no compila); pestañas sin modelo |
| `nn/split_stratified.json` | Iris, Vino, Cáncer, Dígitos, Titanic, Pingüinos, Moons y un CSV |
| `nn/errors.json` | 6 combinaciones que fallan y lo que muestran las pestañas 3–6 después |
| `code/nn_training.json` | `red_entrenamiento.py.j2` ejecutado en 6 casos |

Contenido de cada caso `nn/exact` (`expected`): `split` (índices, comprobación de la referencia), `scaler`, `data`
(`x_fit`/`y_fit` en f32 —lista o base64 por encima de 1 000 filas—, `split_at`, batches), `compile` (pérdida, métrica
que resolvió Keras, optimizador), `initial_weights`, `first_batch` (gradientes de Keras y diferencia con la
referencia), `steps` (pesos tras los pasos 1–3), `epoch0_batch_logs_running_mean`, `history`, `callbacks` (mejor época,
época de parada, márgenes relativos de `val_loss`), `final_weights`, `reference` (§6.1), `test` (salida cruda, y_pred,
`evaluate`), `weights`, `activations_default`, `surface_default` (ejes, medias, 400 celdas al azar con su salida y
el hash de toda la rejilla), `prediction_form`, `onnx_proposal_D75`, `export_package`, `diagnostics` y `views`
(todas las pestañas sin Qt: textos, tablas, controles, figuras matplotlib y Plotly resumidas, lienzos HTML
resumidos en sus constantes JS; fechas como `<fecha>`).

### 6.1 Estrategia de paridad en dos niveles (D69)

**Nivel (a), determinista.** Lo que Rust no puede reproducir de Keras es el azar: el generador de los pesos
iniciales, las máscaras de dropout y la permutación de cada época. Se elimina así:

1. **Pesos iniciales**: se exportan los que Keras sorteó (`initial_weights`) y Rust arranca de ellos (`init` de C21).
2. **Orden de los batches**: `fit(shuffle=False)`; los batches son filas consecutivas de `x_fit` (el fixture da las
   filas exactas en f32).
3. **Dropout**: rate 0 en el nivel (a); la fórmula del dropout se verifica aparte con **máscaras exportadas**
   (`iris_dropout_masks.json`: 5 pasos de Adam con máscaras de `RandomState(42)`, diferencia ≤ 4,2e-7).
4. **Kernels**: f32 en ambos lados; solo cambia el orden de las sumas.

Con eso, `nn_reference.py` (y Rust) deben reproducir: gradientes del primer batch (≤ 3,1e-5 en Diabetes por la escala
del objetivo; ≤ 6e-8 en el resto), pesos tras los primeros pasos (≤ 2,1e-6), el historial completo, la época de parada,
la mejor época, las reducciones de LR y los pesos finales. Resumen (columnas: épocas entrenadas/máximas, mejor época
restaurada, épocas en que la referencia coincide con Keras a 1e-3 relativo, diferencia máxima de los pesos finales,
error del ONNX propuesto frente a Keras, neuronas muertas con 50 muestras, tamaño):

| Caso | Red | Optimizador | Pérdida · métrica | Épocas | Mejor | Acuerdo | Δ pesos | ONNX | 💀 | KB |
|---|---|---|---|---|---|---|---|---|---|---|
| `iris_default` | [4, 10, 10, 3] relu/softmax | adam 0.001 b64 | CCE · CategoricalAccuracy | 50/50 | 49 | 50/50 | 1.1e-06 | 6.0e-08 | 1 | 686 |
| `iris_sgd_tanh_lr01_b16` | [4, 10, 10, 3] tanh/softmax | sgd 0.1 b16 | CCE · CategoricalAccuracy | 30/30 | — | 30/30 | 1.8e-07 | 2.4e-07 | — | 84 |
| `iris_rmsprop_sigmoid_1layer` | [4, 8, 3] sigmoid/softmax | rmsprop 0.01 b32 | CCE · CategoricalAccuracy | 60/60 | 59 | 60/60 | 9.5e-07 | 1.2e-07 | — | 78 |
| `iris_tanh_output` | [4, 10, 10, 3] relu/tanh | adam 0.01 b16 | MSE · CategoricalAccuracy | 40/40 | 39 | 40/40 | 4.8e-06 | 3.0e-07 | — | 88 |
| `iris_relu_3layers_lr03` | [4, 16, 12, 8, 3] relu/softmax | adam 0.3 b16 | CCE · CategoricalAccuracy | 11/40 | 5 | 11/11 | 3.5e-04 | 4.7e-10 | 15 | 509 |
| `breast_cancer_default` | [30, 30, 15, 1] relu/sigmoid | adam 0.001 b64 | BCE · BinaryAccuracy | 56/56 | 55 | 56/56 | 1.7e-06 | 1.8e-07 | 0 | 1429 |
| `breast_cancer_sgd_tanh_output` | [30, 30, 15, 1] relu/tanh | sgd 0.1 b32 | MSE · BinaryAccuracy | 30/30 | — | 30/30 | 1.5e-07 | 2.2e-07 | — | 586 |
| `wine_default` | [13, 13, 10, 3] relu/softmax | adam 0.001 b64 | CCE · CategoricalAccuracy | 50/50 | 49 | 50/50 | 1.3e-06 | 1.2e-07 | 1 | 543 |
| `wine_rmsprop_lr03_chaotic` | [13, 13, 10, 3] relu/softmax | rmsprop 0.3 b16 | CCE · CategoricalAccuracy | 18/40 | 7 | **1/18** | 2.7 | 6.0e-08 | — | 136 |
| `diabetes_default` | [10, 10, 10, 1] relu/linear | adam 0.001 b64 | MSE · MAE | 50/50 | 49 | 50/50 | 3.4e-06 | 1.9e-06 | 1 | 828 |
| `diabetes_adam_lr01_b32` | [10, 10, 10, 1] relu/linear | adam 0.1 b32 | MSE · MAE | 18/50 | 7 | 18/18 | 3.3e-05 | 3.1e-05 | — | 167 |
| `diabetes_sgd_lr001_tanh` | [10, 10, 10, 1] tanh/linear | sgd 0.001 b64 | MSE · MAE | 20/20 | — | 20/20 | 2.4e-04 | 1.5e-05 | — | 167 |
| `diabetes_softmax_output` | [10, 10, 10, 1] relu/softmax | adam 0.001 b64 | MSE · MAE | 10/10 | 0 | 10/10 | 0 | 0 | — | 157 |
| `diabetes_sigmoid_output` | [10, 10, 10, 1] relu/sigmoid | adam 0.01 b64 | MSE · MAE | 20/20 | 19 | 20/20 | 2.2e-06 | 6.0e-08 | — | 168 |
| `california_housing_default_e20` | [8, 10, 10, 1] relu/linear | adam 0.001 b64 | MSE · MAE | 20/20 | 19 | 20/20 | 1.5e-02 | 3.8e-06 | 2 | 1598 |
| `moons_default` | [2, 10, 10, 1] relu/sigmoid | adam 0.001 b64 | BCE · BinaryAccuracy | 50/50 | 49 | 50/50 | 2.4e-06 | 1.2e-07 | 1 | 478 |
| `csv_string_target_default` | [2, 10, 10, 3] relu/softmax | adam 0.001 b64 | CCE · CategoricalAccuracy | 11/50 | 0 | 11/11 | 6.0e-08 | 6.0e-08 | 2 | 375 |
| `csv_numeric_regression_default` | [2, 10, 10, 1] relu/linear | adam 0.001 b64 | MSE · MAE | 50/50 | 49 | 50/50 | 6.0e-07 | 6.0e-08 | 3 | 392 |

Lecturas: (1) la referencia coincide con Keras en todas las épocas de 17 casos, con la misma época de parada y la
misma mejor época; (2) `wine_rmsprop_lr03_chaotic` es caótico a propósito: diverge en la época 2 y los tests solo
comparan la primera (y que Rust también se vuelve inestable); (3) California acumula 4 140 pasos y los pesos finales
derivan 1,5e-2 aunque las pérdidas coinciden a 1e-3: allí los pesos finales no son normativos; (4) EarlyStopping
restaura la mejor época aunque no pare (13 casos terminan con la época 49 de 50, por ejemplo).

**Nivel (b), estadístico.** El entrenamiento real de Rust (con su RNG) se compara con la distribución de 5
entrenamientos de la app (semillas 0–4, dropout 0,2, barajado): por métrica, `bands` (mín, máx, media, std) y
`acceptance = [mín − m, máx + m]` con `m = max(3·std, suelo)` (accuracy 0,03; R² 0,05; errores 10 % de la media).
**La media de Rust sobre 5 semillas** debe caer dentro y **cada semilla** dentro de `[mín − 2m, máx + 2m]`.

| Caso | Épocas | Métrica de prueba (mín–máx en 5 semillas) |
|---|---|---|
| `iris_default` | 50 | accuracy 0,633–0,767 |
| `iris_sgd_lr001` / `iris_rmsprop` | 50 | 0,600–0,767 / 0,633–0,767 |
| `breast_cancer_default` | 56 | 0,939–0,974 |
| `wine_default` | 50 | 0,750–0,944 |
| `diabetes_default` | 50 | R² −3,77 a −2,83 (MSE 20 307–25 295) |
| `diabetes_adam_lr01` | 21–33 | R² 0,417–0,448 |
| `california_housing_default` | 60–99 | R² 0,699–0,721 |
| `moons_default` | 50 | 0,860–0,880 |
| `csv_string_target_default` / `csv_numeric_regression_default` | 11–50 / 50 | 0,167–0,500 / R² −0,63 a −0,11 (bandas anchas: datos diminutos) |

### 6.2 Tolerancias (`meta.tolerance` de cada caso)

| Qué | Tolerancia | Por qué |
|---|---|---|
| Split estratificado y de regresión | **exacto** | `nn_reference.stratified_split` / D7 |
| Media y escala del StandardScaler | 1e-12 relativo | sumas por pares de numpy según la disposición en memoria |
| `x_fit` (f32) | 1 ulp | escalado en f64 y conversión a f32 |
| Gradientes del primer batch, pesos tras 1–3 pasos | 1e-6 absoluto (5e-5 en Diabetes, objetivo sin escalar) | orden de las sumas en f32 |
| Historial | 1e-4 relativo en las épocas `< reference.agree_epochs_rel_1e-3` | idem, acumulado |
| Épocas, mejor época, reducciones de LR | exactas salvo `callbacks.val_loss_relative_margins[i] < 1e-5` | decisiones con `<` estricto |
| Pesos finales | 1e-4 absoluto si `reference.final_weights_max_abs_diff < 1e-4`; si no, no se comparan | trayectoria caótica o larga |
| Vistas desde los pesos finales (salidas, pesos, activaciones, superficie, formulario) | 1e-5 absoluto; clases exactas salvo |p − 0,5| < 1e-5 o empates de argmax | forward f32 |
| Dropout con máscaras | 1e-6 | fórmula invertida |
| Diagnóstico | textos exactos; números 1e-12 | lógica pura |
| ONNX | 1e-5 absoluto frente a `test.raw_output`; etiquetas exactas | Sub/Div en f32 |
| Nivel (b) | `acceptance` | otro RNG |
| Textos | exactos (salvo `<fecha>`) | literales |

Si Rust no reproduce un caso de nivel (a) dentro de tolerancia, el test debe fallar y anotarse en MEMORY.md; no se
relaja a "misma accuracy".

---

## 7. Contenido extraído

Literal, con comentario de origen (copiado con el mismo `textwrap.dedent().strip()` que aplica `ui.markdown`):

```
content/redes/     intro.md (F1), mensajes.toml (NUEVO: comun, datos, arquitectura, entrenamiento, evaluacion,
                   diagnostico, visualizacion, historial, pesos, superficie, activaciones, predicciones, lotes,
                   exploracion, exportar, exportar_rust; marcas NUEVO/CORREGIDO de D71–D90 e inalcanzables),
                   datos-consejos.md, arquitectura-tamano.md ({n_muestras}, {n_caracteristicas}, {mitad},
                   {doble}, {cuadruple}, {max_parametros}), arquitectura-consejos.md ({n_muestras},
                   {batch_recomendado}), entrenamiento-conceptos.md, entrenamiento-tecnicas.md,
                   visualizacion-intro.md, historial-interpretar.md, pesos-interpretar.md,
                   activaciones-que-veras.md, activaciones-muertas.md, diagnostico-tendencias.md,
                   diagnostico-historial-corto.md, diagnostico-contexto.md, exportar-recomendaciones.md,
                   exploracion-que-es.md y exploracion-consejos.md ({caracteristica}; inalcanzables, D81)
content/code/      red_arquitectura.py.j2 (LITERAL del código de 🏗️, verificado en 6 configuraciones),
                   red_predictor_plantilla.py.j2 y red_cargar_completo.py.j2 (LITERALES de 💾),
                   red_entrenamiento.py.j2 (NUEVO, D74, ejecutado: code/nn_training.json),
                   paquete_modelo_red.py.j2, paquete_leeme_red.txt.j2, red_predictor_numpy.py.j2
                   (NUEVOS, D75, ejecutados en export_package)
```

Pendiente para `ml-lesson-author`: `content/redes/exportar.md` ("ℹ️ ¿Qué formato elijo?", como F2–F4), un texto corto
sobre por qué la red entrena distinto que en la versión anterior (semilla fija, D70) y revisar los NUEVO.

---

## 8. Mapeo a Rust

### 8.1 Cálculos

| Cálculo | Python | Rust | Huecos |
|---|---|---|---|
| Split estratificado | `train_test_split(stratify)` | `core::nn_data::stratified_split` (C22) | — (`rand_mt` ya existe) |
| Escalado | `StandardScaler` | `core::nn_data::Scaler` | — |
| Red, pérdidas, optimizadores, callbacks | Keras 3.12 / TF 2.16 | `mltutor-nn` (C21): burn para tensores y autodiff + optimizadores, inicialización, máscaras y callbacks propios (D68) | Adam/RMSprop de Keras ≠ burn; ReduceLROnPlateau y EarlyStopping propios |
| Métricas y evaluación | sklearn + `model.evaluate` | C1/C2 (F2) + pérdida de test | — |
| Diagnóstico | Python | `core::nn_diagnostics` (C23) | — |
| Pesos y activaciones | NumPy | `core::nn_stats` + `Mlp::hidden_activations` | — |
| Superficie | bucle Python + matplotlib | `grid::decision_grid` modo medias (D78) | rejilla limitada |
| Exploración, lotes | — (inalcanzable) | C29 | — |
| ONNX | — | `mltutor-export::onnx::mlp` | `Squeeze` con `axes` como entrada (opset 13+) |
| Paquete Keras | `model.save`, TFLite, pickle | `mltutor-export::package::keras` | — |
| Código | `code_examples.py` | `codegen` con 5 plantillas | — |

### 8.2 Vistas

| Vista | Comando | Componente Svelte |
|---|---|---|
| 📊 Datos | (F1) | `DatasetTab` + `Notice` |
| 🏗️ Arquitectura | `nn_architecture`, `generate_code(NnTraining)` | `NnArchitectureTab.svelte` (`Slider`, `Select`, `Notice`, `MetricGrid`, `DataTable`, `Expander`, `ModelCode`) + `NetworkDiagram` (C24) |
| ⚙️ Entrenamiento | `train_model`, eventos `nn-epoch` | `NnTrainTab.svelte` + `TrainingProgress.svelte` (pasos 1–4, barra, Cancelar) + `TrainingHistoryChart` (C27, en vivo) |
| 📈 Evaluación | `evaluate_model`, `nn_history`, `nn_diagnostics` | `ClassificationEvaluation`/`RegressionEvaluation` + `TrainingHistoryChart` + `NnDiagnostics.svelte` |
| 📊 Historial | `nn_history` | `TrainingHistoryChart` + `MetricGrid` |
| 🧠 Pesos | `nn_weights` | `NnWeights.svelte` (histogramas Plotly) |
| 🎯 Superficie | `decision_grid` | `FeaturePairSelector` + `DecisionGridView` (F2) |
| 📉 Activaciones | `nn_activations` | `NnActivations.svelte` + `NetworkDiagram` (modo mapa) |
| 🔮 Predicciones | `prediction_form`, `predict_one`, `nn_predict_batch`, `nn_explore` | `PredictionForm`, `PredictionResultCard` + `NnBatchPredict.svelte`, `NnExplore.svelte` |
| 💾 Exportar | `export_model`, `generate_code` | `ExportPanel` (4 botones) + `ModelCode` + `JsonView` |

### 8.3 Detalles numéricos y de formato que Rust debe copiar

- Todo el entrenamiento en **f32** (Keras); métricas y diagnóstico en f64 a partir de los logs f32.
- Logits: BCE y CCE desde z, no desde p (Keras); MSE a través de la activación de salida (también softmax).
- `floor(n·(1 − v))` y `ceil(0,2·n)` en f64; `v = pct / 100`.
- Glorot con `fan_in`, `fan_out` de cada `Dense`; sesgos a 0.
- `learning_rate` registrado antes de reducir; `lr·factor` en f64 y guardado en f32; comparación `lr > 1e-7` en f32.
- `{:,}` de Python = separador de miles con coma ("1,411"), también en la UI en español (paridad de textos).
- `{x:+.1f}` con signo; "inf%" cuando la estabilidad es infinita.

### 8.4 Implementación de F5a (2026-10-16, `rust-ml-engineer`)

**D68 resuelto: MLP propio, sin burn.** `mltutor-nn` no depende de burn: capas densas, activaciones,
pérdidas desde los logits, retropropagación manual, Adam/SGD/RMSprop de Keras, Glorot, máscaras, permutaciones,
EarlyStopping y ReduceLROnPlateau (unas 900 líneas con documentación). Motivos medidos:
- La especificación (`nn_reference.py`) ya es una retropropagación manual; con burn había que reescribir igualmente
  optimizadores, inicialización, RNG, pérdidas y callbacks, y quedaba un `Mutex` global por el RNG del backend.
- Iris, 100 épocas: **5 ms** (burn en el spike de F0: unos 90 ms). California, 100 épocas sin parada temprana:
  **0,66 s** (Python: unos 30 s). Los 55 entrenamientos del nivel (b): 2 s en total.
- −4 MB de binario y sin estado global: el mismo `seed` da el mismo historial en cualquier hilo.
- Los productos acumulan en **f64** y redondean una vez a f32 (orden fijo): el error es medio ulp, así que quedamos lo
  más cerca posible de Keras sea cual sea el orden de sus sumas. Con acumulación f32 Diabetes superaba la tolerancia de
  los gradientes (9e-5 > 5e-5); con f64, 3,05e-5, igual que la referencia NumPy.
- RNG (D70): un solo `RandomState(seed)` de NumPy (MT19937, `mltutor_core::sampling`) en este orden: Glorot capa a capa
  (kernel por filas, `uniform(-l, l)`), y en cada época la permutación de las filas y, por batch y capa oculta, las
  máscaras `random_sample() >= rate`. Es reproducible también en NumPy.

**API para F5b–F5d** (documentada en los `lib.rs`): `mltutor_nn::train_network(x, &target, &NnArchitecture,
&NnTraining, shuffle, init, &FitControl { on_epoch, cancel }) -> TrainedNetwork`; `NeuralNetwork: Predictor` (escala
dentro; `predict_proba` = salida, `[1 − p, p]` con una neurona); vistas en `mltutor_core::{nn_data, nn_diagnostics,
nn_stats, grid::mean_grid, view::ViewNode}`; exportación en `mltutor_export::{onnx::mlp_model,
package::keras::keras_package}` y `codegen::NnTrainingContext`. Desviaciones del contrato §5.3:
- `NnTraining` lleva `seed: u32` (serde por defecto 42).
- `EpochLog.epoch` empieza en 0, como Keras; los textos de progreso muestran `epoch + 1`.
- `DiagnosticsView.blocks` es un árbol `ViewNode` con los textos ya resueltos y la misma forma que `expected.views`
  de los fixtures (`text`, `notice`, `metric`, `row`/`column`, `expander`), en vez de claves + argumentos.
- Errores en `mltutor_core::nn_data::NnError` (no en `ModelError`, para no romper los `match` de `src-tauri`), con
  `UntrainableOutput { output, n_classes, key, reason }`, `TooFewPerClass`, `SplitTooSmall` y `ValidationTooSmall`.
- La red no implementa `refit` (D78): la superficie usa `grid::mean_grid`.
- Los metadatos JSON (📋, D75) y C29 (exploración y lotes) quedan para F5d.

**Desviaciones de C23 aprobadas (coordinador, 2026-10-16)** y diferencias conocidas:
- Con `loss_stable` y calidad `poor`, Python decía "Entrenamiento inestable (CV 0,3 %)" y recomendaba bajar el LR. Rust
  usa `estable_rendimiento_bajo`, `patron_rendimiento_bajo(_rec)` y `alerta_rendimiento_bajo(_accion)`; la puntuación
  no cambia (`csv_string_target_default`, `iris_relu_3layers_lr03`).
- `nota_validacion_pequena` (info) sigue a "❌ Sobreajuste en accuracy" si hay menos de `SMALL_VALIDATION = 30` filas de
  validación (el comentario del TOML dice 100; se siguió la indicación del coordinador).
- Los textos marcados CORREGIDO/NUEVO en `mensajes.toml` y los `.md` sustituyen a los de Python; el test reemplaza en
  el fixture el literal de Python por el texto actual y exige igualdad en todo lo demás.
- `np.polyfit` devuelve ruido de LAPACK (±1e-17) en un tercio exactamente plano y desempata al azar dos pendientes
  iguales; Rust usa la fórmula cerrada (0 exacto). En 4 historiales (`cls_flat_20`, `cls_rising_late_30`, `reg_bad_30`,
  `diabetes_softmax_output`) la fila de tendencias y la del análisis comparativo no se comparan; el resto, sí.
- Pendientes: 1e-12 relativo a la escala de la pérdida (no a la pendiente).

**Fixture corregido:** `nn/exact/iris_dropout_masks.json` guarda como `initial_weights` los pesos **después** de los
5 pasos (`_pairs` no copia y el optimizador de referencia actualiza `w0` en sitio). Se añadió el caso
`iris_dropout_masks_w0` (mismo código con copia) y el test usa ese; el original no cambia.

**Tolerancias aplicadas en el nivel (a):** historial 1e-4 relativo en las épocas en que la referencia está a ≤ 1e-5 de
Keras y 1e-3 en las demás épocas acordadas (California); pasos 1–3 a `max(1e-6, 4 × diferencia de la referencia)` (Adam
con LR 0,1–0,3 mueve cada peso ≈ ±lr y la referencia ya difiere 4e-5).


#### 8.4.1 Revisión Rust de F5a y cálculo de F5c (2026-10-16, `rust-ml-engineer`)

**Interfaces acordadas (K1–K6)**, ya en `crates/`:
- **K1** `History.diverged_epoch: Option<usize>` (`#[serde(default)]`): primera época con la pérdida de algún batch o
  la de validación NaN/∞. El entrenamiento **sigue**, como Keras sin `TerminateOnNaN`. Los `f64` de `EpochLog` se
  serializan con `mltutor_core::serde_nonfinite` (`"nan"`, `"inf"`, `"-inf"`; también lee `null` como NaN), así que el
  historial divergido hace ida y vuelta. `NeuralNetwork::is_valid()` es `false` si algún peso, sesgo o valor del
  escalador no es finito (se calcula en `new` y al deserializar); entonces toda predicción devuelve
  `ModelError::Numerical` y las exportaciones `ExportError::InvalidModel`. California, SGD, LR 0,3, [64, 64]: diverge
  en la época 0.
- **K2** Histogramas en Rust con el algoritmo de Plotly para `go.Histogram(x, nbinsx)`
  (`core::stats::histogram_plotly`: `autoBin` + `autoShiftNumericBins` + `Lib.increment` + recorte de los bins vacíos de
  los extremos). Se ha comparado con plotly.js 4.1.2 ejecutado en jsdom en 140 casos (pesos, sesgos y activaciones de
  los 18 `nn/exact`, normales aleatorias, ReLU y valores únicos): mismos recuentos en todos; 54 de ellos quedan en
  `crates/mltutor-core/tests/data/plotly_histogram.json`. `nn_stats::weights_view(&views, bins_w = 50, bins_b = 20) ->
  WeightsView { stats, histograms: [{ weights, biases }] }` y `nn_stats::activations_view(..) -> ActivationsView {
  rows_used, layers: [{ stats, histogram }], dead_pct, map }`. Por neurona solo viajan `neuron_means` y `neuron_stds`.
  Con nbinsx = 50 Plotly dibuja entre 20 y 50 barras (redondea el ancho hacia arriba a 2, 5 o 10·10ᵏ).
- **K3** `NeuralNetwork::hidden_activations(x_original, sample_size) -> Result<Activations { rows_used, layers },
  ModelError>`: recorta a `min(sample_size, filas)` y escala dentro. Envoltorios: `NeuralNetwork::activations_view(x_test,
  &ActivationParams)` y `TrainedNetwork::test_activations(x_dataset, &params)`, que solo copia las primeras filas de test.
- **K4** `FitControl::from(&TrainControl)` y `TrainControl::from(&FitControl)` (la bandera; el callback por época no
  tiene equivalente en el core). La cancelación se comprueba antes de cada batch de entrenamiento, de validación y del
  `evaluate` final.
- **K5** `train_network(x, &y, &arch, &training, TrainOptions { shuffle, init }, &FitControl)`; la app usa
  `TrainOptions::default()` (`shuffle = true`, `init = None`).
- **K6** `nn_data::training_cost(total_params, n_fit, epochs) -> TrainingCost { work = params × n_fit × epochs,
  estimated_seconds = work / 2e9, level }`, con `CostLevel::Low` (< 10 s estimados), `High` (10–60 s) y `VeryHigh`
  (> 1 min). Rendimiento medido en un Mac M (12 núcleos, release, batch 32, California): 5 × 256 → 3,9e9 work/s;
  3 × 128 → 1,9e9; [64, 64] → 1,7e9; [10, 10] → 5e8 (dominado por el coste fijo, pero tarda 5 ms por época). Con 100
  épocas en California: [64, 64] Low (3,7 s reales), 3 × 128 High (24 s), 5 × 256 VeryHigh (unos 90 s).

**Hallazgos corregidos:**
- Productos en paralelo (rayon) por filas de salida (`matmul`, `matmul_a_bt`) y por bloques de 16 filas de `kk`
  (`matmul_at_b`). Cada elemento conserva la misma suma en el mismo orden; `matmul_a_bt` transpone `w` una vez y
  queda contiguo. Umbral: 2¹⁹ multiplicaciones (con 2¹⁵, [64, 64] iba al doble de lento). Los 18 entrenamientos del
  nivel (a) y 25 del nivel (b) dan los mismos bits que antes con 1, 3 y 8 hilos. California 5 × 256, batch 32:
  **3,18 s → 0,92 s por época** (1,62 s con un hilo); California por defecto, 100 épocas: 0,66 → 0,51 s; Iris: 4,3 ms.
- **ReLU con NaN:** `f32::max(NaN, 0)` devuelve 0, así que la ReLU anterior convertía NaN en 0 y ocultaba la
  divergencia (`tf.nn.relu(NaN)` es NaN). Ahora `if v < 0 { 0 }`, sin cambiar ningún bit de la paridad.
- `Scaler::fit` usa `_is_constant_feature` de sklearn (`var <= n·ε·var + (n·media·ε)²`, o `escala < 10ε`) → escala 1.
- `Mlp::new`/`validate` y deserialización validada; `forward`, `gradients` y `hidden_activations` devuelven
  `Result<_, ModelError>`; `Optimizer::step` devuelve error si los gradientes o el estado no cuadran.
- `NeuralNetwork` con campos privados: `new`, `mlp()`, `scaler()`, `class_codes()`, `resolved()`, `is_valid()`,
  `output_is_probability()` (`false` con salida tanh: las vistas de probabilidad deben ocultarse o avisar),
  `mean_grid(x_test, &y_test, fx, fy, &TrainControl)`. `grid::mean_grid` recibe `&Scaler`.
- Escalado por índices de fila (`Scaler::transform_rows_f32`) en lugar de `x.select` de train y test.
- `modelo.json` compacto (`package::keras::model_json_text`): sangría de `json.dumps(indent=2)` salvo los arrays de
  escalares, en una línea (una por fila del kernel), y pesos con el decimal f32 más corto, comprobado para que se lea
  el mismo f32. 5 × 256: 3,2 MB (antes unos 10 MB). Se rechazan pesos, escalador o salidas de `check` no finitos.
- ONNX: `label` = código de la clase (`Gather(class_labels)` sobre `ArgMax`, o sobre `Greater → Cast → Squeeze`) y, en
  binaria, `probabilities` = `Concat(1 − p, p)` [N, 2] (`positive_probability`, `negative_probability`). `class_values`
  no se usa para la red: la etiqueta es el código, como dice el LEEME. `tools/golden_nn.py::onnx_proposal_model` hace el
  mismo grafo y los 18 `nn/exact` se han regenerado (solo cambian `onnx_proposal_D75` y `export_package`).
- **No aplicado (hallazgo 7):** en Keras 3.12 `ReduceLROnPlateau` y `EarlyStopping` son `MonitorCallback` con
  `best = baseline = None`, y `_is_improvement(_, None)` es `True`: la primera época siempre es la mejor, también con
  NaN (el `np.inf` era de Keras 2). Comprobado con Keras 3.12.0 (pesos NaN, paciencia 1: `learning_rate [0.1, 0.1,
  0.05]`, para en la época 2); `engine.rs::callbacks_with_nan_from_the_start_follow_keras_3` reproduce ese caso.

**F5c (cálculo):** 📈 usa `NeuralNetwork: Predictor` con C1/C2 de F2, `test_loss`/`test_metric` y C23; 🎯 usa el
historial (K1), `weights_view` (K2), `mean_grid` (clasificación y regresión, D78) y `activations_view` con el mapa
(K2, K3). **Desviación aprobada (C23, 2026-10-16):** en regresión, el diagnóstico usa `convergio_regresion`,
`salud_bueno_regresion`, `estado_bueno_regresion`, `rec_bueno_regresion` y `rec_bueno_otro_regresion` en lugar de las
claves de clasificación (la calidad "good" de regresión solo dice que el MAE de validación ha bajado); el resto sigue
siendo exacto respecto a Python, y el test sustituye esos literales en los fixtures.

**Fixtures nuevos:** `code/nn_training.json` añade `csv_iris_classes_1_2_3` y `csv_cancer_classes_1_2` (Iris y Cáncer
con `target + 1` en una columna `clase`; clases "Clase 1/2/3"; banda de `iris_default`/`breast_cancer_default`) y guarda
el `input` de cada caso.

---

## 9. Subfases, reparto de tareas y checklist

Orden en cada subfase: core → paridad → comandos → UI → contenido → smoke test. Cada subfase cierra con CI verde,
`rust-reviewer` (y `ml-correctness-reviewer` en F5a y F5d) y sus casillas de INVENTARIO §6 marcadas.

### F5a — Motor y paridad (`rust-ml-engineer`; sin UI)

1. [ ] `core::nn_data`: `stratified_split`, `Scaler`, `keras_validation_split`, `resolve_training`. Tests:
   `nn/split_stratified.json` exacto; `nn/exact/*.split`, `scaler`, `data.x_fit`; `nn/errors.json` (las 4
   combinaciones dan `UntrainableOutput`, el CSV `TooFewPerClass`).
2. [ ] `mltutor-nn`: capas, activaciones, pérdidas desde los logits, métricas, Adam/SGD/RMSprop de Keras, dropout con
   máscaras, bucle `fit` y callbacks (C21). Tests: `first_batch`, `steps`, `history` (épocas acordadas), callbacks,
   `final_weights`, `test.raw_output` de los 18 casos con `initial_weights` y `shuffle = false`;
   `iris_dropout_masks`. Benchmark de California (`#[ignore]`) < 10 s.
3. [ ] Nivel (b): `#[ignore]` + job semanal o manual: 5 semillas por caso de `nn/stat`, media dentro de `acceptance`.
4. [ ] `core::nn_diagnostics` (C23). Tests: `nn/diagnostics.json` (14 casos) y `expected.diagnostics` de los 18.
5. [ ] `core::nn_stats` (C25, C26) y `Mlp::hidden_activations`. Tests: `weights`, `activations_*`.
6. [ ] Determinismo: el mismo `seed` da el mismo historial en dos ejecuciones y con otros hilos.

### F5b — Arquitectura y entrenamiento (`tauri-ui-engineer`, `ml-lesson-author`)

1. [ ] Backend: `ModelConfig::NeuralNet`, `LessonId::Redes`, `nn_architecture`, `train_model` con eventos `nn-epoch`,
   cancelación, `CommandError::{UntrainableOutput, TooFewPerClass}`; `build.rs` + capabilities.
2. [ ] Store `lessons/redes/state.svelte.ts` (D85, D90, D91, D13).
3. [ ] Pestañas 0–2 con todos los controles, captions y avisos de §2.2–2.3; `NetworkDiagram` (C24, modo arquitectura);
   `TrainingProgress` + `TrainingHistoryChart` en vivo (C27); D71, D73, D82, D83.
4. [ ] `generate_code(NnTraining)` con la configuración actual (D74). Test: `code/nn_training.json` (texto exacto).
5. [ ] Contenido: cargar `content/redes/*` sin reescribirlo; tests de textos contra `nn/controls.json`.
6. [ ] `--smoke-test`: Iris, Cáncer, Diabetes y California con la red por defecto (10 épocas) y una cancelación.

### F5c — Evaluación y visualización (`tauri-ui-engineer`, `rust-ml-engineer` para la rejilla)

1. [ ] `evaluate_model` para la red (+ D84, D86, D87), `nn_history`, `nn_diagnostics`, `nn_weights`, `nn_activations`,
   `decision_grid` modo medias (D78).
2. [ ] `NnDiagnostics`, `NnWeights`, `NnActivations` + `NetworkDiagram` modo mapa, superficie con `DecisionGridView`.
   Tests de vitest con `views.*` de los casos `full`.
3. [ ] Capturas Rust de 14–22, 32–33, 37–39b y 41, comparadas con las de Qt y sus `_web`.

### F5d — Predicciones y exportación (`rust-ml-engineer` + `tauri-ui-engineer`)

1. [ ] `predict_one` con escalado; `nn_explore`, `nn_predict_batch` (C29, D81).
2. [ ] `mltutor-export`: `onnx::mlp` (estructura frente a `onnx_proposal_D75.nodes`, salidas frente a
   `test.raw_output` a 1e-5), `package::keras` (`modelo.json` frente a `export_package.model_json` salvo flotantes;
   scripts iguales a `export_package.script`/`numpy_script`; LEEME), metadatos (D75).
3. [ ] `ExportPanel` con 4 botones; diálogos nativos.
4. [ ] Capturas Rust de todas las pantallas restantes y `capturas/comparacion-fase5.md`.
5. [ ] INVENTARIO §6 y §7 marcados; revisión final de `ml-correctness-reviewer` (paridad a/b, ONNX, paquete).

### 9.1 Reparto resumido

| Agente | F5a | F5b | F5c | F5d |
|---|---|---|---|---|
| `rust-ml-engineer` | todo | — | rejilla modo medias | export, exploración, lotes |
| `tauri-ui-engineer` | — | comandos, store, 🏗️, ⚙️, progreso | 📈, 🎯 | 🔮, 💾, capturas |
| `ml-lesson-author` | — | contenido, textos NUEVO | revisión de diagnóstico | `exportar.md` |
| revisión | rust + ml | rust | rust | rust + ml |

---

## 10. Decisiones para el usuario

Se numeran a continuación de D67 (D68–D93). Si no se dice lo contrario, se aplica la recomendación.

| # | Decisión | Recomendación |
|---|---|---|
| D68 | **RESUELTO (2026-10-16): MLP propio sin burn** (ver §8.4). Propuesta original: **Motor**: burn 0.22 (aprobado en F0) frente a un MLP propio sobre ndarray | Mantener **burn para tensores y autodiff**, pero con todo lo que fija la paridad escrito en `mltutor-nn`: inicialización Glorot, máscaras de dropout y permutaciones con `StdRng` propio (no el RNG global de burn), Adam y RMSprop con las fórmulas de Keras (las de burn ponen ε en otro sitio), pérdidas desde los logits, EarlyStopping y ReduceLROnPlateau. Alternativa: un MLP propio sin burn (unas 400 líneas, −4 MB, sin Mutex), igual de válido para esta lección |
| D69 | **Paridad con Keras** (concreción de D6) | **Dos niveles**: (a) exacta dentro del redondeo f32 con los pesos iniciales de Keras, sin barajar ni dropout (especificación `tools/nn_reference.py`, verificada en 17 de 18 casos durante todas las épocas) y el dropout con máscaras dadas; (b) estadística: la media de 5 semillas de Rust dentro de la banda de 5 semillas de Keras |
| D70 | **Semilla**: Python no fija ninguna (cada entrenamiento da otro resultado) | Semilla 42 fija (regla 4): mismo resultado al repetir. Un control "🎲 Semilla" (1–9999, 42) en el expander avanzado para que el estudiante vea la variabilidad (NUEVO) |
| D71 | **Activaciones de salida que no se pueden entrenar** (multiclase sigmoid/linear, binaria softmax/linear): Python entrena, falla en inglés, marca el modelo como entrenado y rompe 📈 y 🎯 | Mantener las 4 opciones (son didácticas), mostrar el aviso de Python y además un `st.error` NUEVO que explica por qué no se puede, y **deshabilitar** "🧠 Entrenar Red Neuronal". El modelo anterior se conserva |
| D72 | **Escalado con todas las filas** antes de dividir (fuga de información de la prueba) | Paridad: igual que Python, y el script de Keras lo explica en un comentario. Alternativa: ajustar el escalador solo con el train (cambian ligeramente todas las cifras) |
| D73 | **"📊 Distribución"** de ⚙️ con números falsos (Iris: Val=30, Test=24; reales 24 y 30) | Mostrar los tamaños reales de Keras (§2.3.1) |
| D74 | **Código de la arquitectura**: `X` sin definir, SyntaxError en binaria, LR/épocas/callbacks fijos, escala con el train | Sustituirlo por `red_entrenamiento.py.j2` (ejecutado y verificado) con la configuración actual de 🏗️ y ⚙️; el mismo script en 💾. El literal queda solo para los tests |
| D75 | **Exportar** (concreción de D1/D28): SavedModel, HDF5, TFLite y pickle no se pueden generar sin TensorFlow; el "📝 Código Python" de Python es una plantilla que devuelve ceros; la fecha es la actual; `num_classes` = 1 en binaria | Cuatro botones: **📦 Paquete Keras (.zip)** (genera .keras, .h5, SavedModel, .tflite, scaler.pkl y modelo_completo.pkl al ejecutarlo; verificado), **💾 ONNX** (escalado + Gemm + activaciones; verificado), **📝 Código Python** (predictor NumPy con los pesos reales + el script de D74) y **📋 Metadatos** corregidos con la fecha del entrenamiento |
| D76 | **Tabla "Detalles por Capa"** desplazada (falta una capa y los parámetros no suman el total) | Una fila por capa con sus neuronas y parámetros (la suma es el total) |
| D77 | **Numeración de capas** en 🧠/📉: "Capa 2, 4, 6" (índices de `model.layers`, contando Input y Dropout) y "Capa 1, 2, 3" en la misma pantalla | Nombres de la arquitectura en todas partes ("Oculta 1", "Oculta 2", "Salida"), con el nombre de Keras (`dense_1`) entre paréntesis en el expander |
| D78 | **Superficie**: rejilla de paso 0,02 en el espacio escalado (55 596 celdas en Iris; millones con AveOccup de California, Python se cuelga), ejes en unidades escaladas, matplotlib | Rejilla de C8 (≤ 200 × 200) sobre el mismo rango, el resto de características en la media de X_test (como Python), Plotly, **ejes en unidades originales** y los puntos de prueba; caption NUEVA |
| D79 | Surrogate KNN de la superficie de regresión (solo si `predict` falla) | No portarlo (inalcanzable) |
| D80 | **Diagrama y mapa de la red**: dos lienzos HTML sin interactividad (el de 🏗️) o con tooltip (el de 📉); con 256 neuronas, 65 536 líneas; el inventario esperaba el nº de parámetros | Un componente SVG nativo para los dos, con todas las opciones de dibujo de Python, tooltip con neuronas y parámetros por capa y resumen de capas grandes |
| D81 | **Predicción por lotes y Exploración interactiva**: listadas en el inventario, pero `show_neural_network_predictions` no se llama nunca | Portarlas como subvistas NUEVAS de 🔮 con sus textos de Python (son didácticas y baratas). Alternativa: no portarlas (D5) y quitar las casillas |
| D82 | **Progreso**: pausas de 2,6 s, pasos que Qt no llega a mostrar ("Paso 5/4") | Pasos 1–4 en vivo sin pausas, barra por época, la gráfica del historial actualizándose en vivo y "Época n/N · pérdida" |
| D83 | **Cancelar** | Cancelación por batch; el modelo anterior queda intacto y se muestra `[entrenamiento] cancelado` |
| D84 | **Diagnóstico**: con los valores por defecto Diabetes acaba con R² ≈ −3,4 y dice "✅ Error de regresión aceptable" y "👍 Bueno: 100%" (compara con las primeras épocas); el texto crítico habla de accuracy en regresión | Portar la lógica exacta (paridad) sin las ramas inalcanzables (plateau, "sin patrones"), añadir un aviso NUEVO si el R² de prueba es negativo y corregir el texto crítico en regresión |
| D85 | **Selectores sin key**: la salida "softmax" de Iris pasa a Cáncer y el entrenamiento falla | Al cambiar la tarea o el número de clases, la salida vuelve a la recomendada; el resto de la configuración se conserva |
| D86 | "🎯 Precisión en Test" y "Precisión durante el Entrenamiento" son la exactitud (accuracy) | "🎯 Exactitud (accuracy) en Test" y "Exactitud durante el Entrenamiento", como en C1 |
| D87 | Umbral binario: `> 0.5` en evaluación, `>= 0.5` en el formulario | `> 0.5` en todas partes (como `BinaryAccuracy` de Keras) |
| D88 | **Valores por defecto de Python que entrenan poco** (LR 0,001 con 2 pasos por época en Iris: 63–77 %; Diabetes con R² negativo) | Mantenerlos (paridad); D84 avisa. Alternativa: LR por defecto 0,01 |
| D89 | **Capturas Qt de F5** (9,8 MB, con 38 PNG de Chrome) y **fixtures** (10 MB) | Versionarlos, como en F2–F4 |
| D90 | **⚙️ sin haber pasado por 🏗️**: aviso y nada más | La configuración por defecto existe siempre; el aviso se elimina |
| D91 | **Configuración o dataset cambiados después de entrenar**: las pestañas siguen con la red vieja sin decirlo | Banner de modelo obsoleto (D32/D45 de F2–F4) |
| D92 | **Subfases** | F5a (motor y paridad), F5b (🏗️ y ⚙️), F5c (📈 y 🎯), F5d (🔮 y 💾), cada una con CI y revisión |
| D93 | **tensorflow-metal**: `model.predict` erróneo tras `set_weights` en la GPU de Apple | Fixtures en CPU; las plantillas ocultan la GPU (`tf.config.set_visible_devices([], "GPU")`) con un comentario |

---

## 11. Riesgos

| Riesgo | Mitigación |
|---|---|
| burn 0.22: optimizadores con otra fórmula, RNG global, API cambiante | D68: optimizadores y RNG propios; burn solo para autodiff (alternativa: MLP propio) |
| Paridad (a) frágil en trayectorias caóticas o largas | `agree_epochs` y márgenes por caso; los pesos finales solo se exigen si la referencia los reproduce |
| Rendimiento: California 100 épocas × 207 batches en un portátil de laboratorio | Presupuesto < 10 s; `opt-level` 2–3 en dev (MEMORY); progreso y cancelación |
| Eventos por época demasiado frecuentes (500 épocas en Iris en < 1 s) | Agrupar a ≤ 20 eventos/s; el último siempre |
| Nivel (b) con datasets diminutos (CSV): bandas inútiles | Solo informativo en los CSV |
| Versión de TensorFlow: los mensajes de error y algunos detalles cambian | Fixtures fijados a TF 2.16.2 / Keras 3.12; los mensajes de error no son normativos en Rust (D71) |
| Diagrama SVG con 5 × 256 neuronas en WebKitGTK | Resumen de capas grandes (D80) y conexiones agregadas |
| Paquete Keras: el estudiante con otra versión de TF | Guardado formato a formato con aviso si alguno falla; comprobación numérica |
| `Squeeze` del ONNX binario con `axes` como entrada (opset 13+) mal soportado por otros runtimes | Validar con onnxruntime en F6; alternativa `Reshape` |
| Alcance XL | Subfases F5a–F5d (D92) |

---

## 12. Hallazgos de la versión Python

1. El split de la red es **estratificado y con test 0,2 fijo** (no el D7 de las otras lecciones), y el escalador se
   ajusta con todas las filas.
2. `validation_split` de Keras usa las **últimas** filas de X_train, sin barajar.
3. EarlyStopping de Keras 3.12 restaura los mejores pesos **al final del entrenamiento aunque no pare**.
4. Los selectores de 🏗️ y ⚙️ no tienen key: al cambiar de Iris a Cáncer la salida sigue en "softmax" y el entrenamiento
   falla (captura `30b`).
5. Cuatro combinaciones de salida no se pueden entrenar; la app marca el modelo como entrenado igualmente y 📈/🎯 se
   rompen (`AttributeError`, captura `43`).
6. Las pestañas dicen "📊 Dataset actual: …" con el dataset seleccionado aunque la red se haya configurado o entrenado
   con otro (`nn_df` solo se actualiza al pasar por 🏗️).
7. La caption "📊 Distribución" de ⚙️ calcula otros tamaños que los reales (intercambia validación y prueba con 20 %).
8. La tabla "Detalles por Capa" está desplazada una fila y su suma no es el total de parámetros.
9. El diagnóstico de regresión solo compara el MAE de validación con el de las 5 primeras épocas: con R² = −3,4 da
   "✅ Error de regresión aceptable" y "👍 Bueno: 100%"; y el texto crítico menciona la accuracy.
10. En 📉, "neurona muerta" = todas sus activaciones ≤ umbral (0): con tanh o sigmoid significa "siempre negativa" o
    nunca ocurre; la saturación solo se mide si el máximo es ≤ 1,2.
11. Las capas se numeran con su índice en `model.layers` (+1, contando Input y Dropout) en unos sitios y 1, 2, 3 en
    otros de la misma pantalla.
12. La superficie trabaja en el espacio escalado (ejes en desviaciones típicas) con paso 0,02: Iris 55 596 celdas;
    California se cuelga con las columnas de valores extremos.
13. Código muerto confirmado: `viz/nn_extra.py` entero (nadie lo importa), `show_neural_network_evaluation`,
    `show_neural_network_visualizations`, `show_neural_network_predictions` (lotes y exploración), `initialize_model_safely`,
    `generate_neural_network_evaluation_code`, `generate_neural_network_visualization_code`,
    `generate_neural_network_complete_code` (devuelve None), el surrogate KNN, `nn_y_pred`, `nn_architecture`,
    `model._activation_model_ready`, los imports sin usar (`px`, `sns`, `plt`, `tf` en la app) y las ramas
    "plateau" y "sin patrones" del diagnóstico.
14. El código de la arquitectura no se puede ejecutar (`X` sin definir; SyntaxError en binaria) y usa otros
    hiperparámetros que la app; su gráfica de "métrica" usa `history.keys()[1]`, que con Keras 3 es `loss`.
15. "📝 Código Python" de 💾 es una plantilla sin pesos; `LOAD_NN` llama a `safe_get_output_size`, no definida.
16. Los metadatos ponen `num_classes = 1` en binaria y "Fecha entrenamiento" es la hora actual.
17. Con tensorflow-metal 1.2.0, `model.predict` tras `set_weights` da salidas erróneas en la GPU de Apple;
    `model(x)` es correcto. La app no lo sufre porque oculta la GPU.
18. Keras usa la pérdida desde los logits cuando la salida es sigmoid o softmax; con RMSprop pone ε dentro de la raíz.
19. Con los valores por defecto la red apenas aprende: Iris 63–77 % (2 pasos por época, 50 épocas, LR 0,001),
    Diabetes R² negativo (objetivo sin escalar).
20. `{:,}` produce "1,411" (separador inglés) en los textos en español.

---

## 13. Cambios propuestos en otros documentos

No se han editado (los aplica quien cierre la fase).

**INVENTARIO.md §6**: Arquitectura: "… diagrama interactivo de la red con el nº de parámetros" → "diagrama nativo con
tooltip y nº de parámetros por capa (D80; Python no los tenía)" y "Incluye el código Keras de la arquitectura con
descarga" → "… (script corregido, D74)". Entrenamiento: añadir "split estratificado con test 0,2 fijo y escalado
(D72); semilla (D70); combinaciones de salida no entrenables bloqueadas (D71)". Evaluación: quitar "probabilidades por
clase" y "el código Python de evaluación" (solo existían en `show_neural_network_evaluation`, muerta). Visualización:
"🎲 Exploración interactiva" pasa a 🔮 (D81). Predicciones: "individual (formulario), por lotes y exploración (D81;
inalcanzables en Python)". Exportar: "📦 paquete Keras ZIP, 💾 ONNX, 📝 predictor NumPy + script, 📋 metadatos (D75)".
§7: "evaluación y visualización de la red" → quitar (código muerto); añadir "script de entrenamiento Keras (D74)".
§8: añadir `show_neural_network_evaluation`, `show_neural_network_visualizations`, `initialize_model_safely`,
`generate_neural_network_{evaluation,visualization,complete}_code`, el surrogate KNN y las ramas plateau del
diagnóstico; aclarar que `viz/nn_extra.py` está muerto **entero**.

**PLAN.md, Fase 5**: enlazar este documento y `capturas/qt/fase5/`; añadir D68–D93 (con enlace a §10); sustituir
"Paridad estadística, no exacta" por "Paridad en dos niveles (D69)"; añadir las subfases F5a–F5d.

**tests/parity/README.md**: ya actualizado con `nn/` y `code/nn_training.json`.

**MEMORY.md**: ver las líneas propuestas en el informe de `migration-porter` (2026-10-15).
