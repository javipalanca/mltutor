# Fase 1 — Shell de la app y datos

Plan de port de `migration-porter` (2026-10-13). Cubre INVENTARIO §0 (estructura común),
§1 (📊 Datos), §2 (📁 Cargar CSV Personalizado) y la parte de §9 que toca a esas pantallas.
Referencia: rama `codex/desktop-migration-study`, la versión Qt.

**El contrato IPC (§5) es la frontera.** `rust-ml-engineer` trabaja en `crates/mltutor-core`
y en los tipos del contrato. `tauri-ui-engineer` trabaja en `src-tauri` y `ui`. Ninguno de
los dos toca los archivos del otro. Las listas de tareas están en §8.

Índice: [1 Shell](#1-shell-estructura-común) · [2 Datos](#2-pestaña--datos) ·
[3 CSV](#3-página--cargar-csv-personalizado) · [4 Casos límite del CSV](#4-casos-límite-del-csv) ·
[5 Contrato IPC](#5-contrato-ipc) · [6 Fixtures](#6-fixtures-de-paridad) ·
[7 Contenido](#7-contenido-extraído) · [8 Reparto de tareas](#8-reparto-de-tareas) ·
[9 Decisiones](#9-decisiones-para-el-usuario) · [10 Riesgos](#10-riesgos) ·
[11 Hallazgos](#11-hallazgos-de-la-versión-python)

## 0. Capturas de referencia (Qt)

Están en `docs/migration/capturas/qt/` y se generan con `uv run python tools/capture_qt.py
docs/migration/capturas/qt`. El script reutiliza `MainWindow` como `--smoke-test`, navega
cambiando `session.state` y pulsa los controles despachando acciones sobre el árbol de
nodos. Tarda aproximadamente un minuto y abre una ventana real: con `QT_QPA_PLATFORM=offscreen`
no funciona. `NNN.png` es la página completa (el widget de contenido entero) y
`NNN_ventana.png` es la ventana tal como se ve (1400×920), con la barra lateral.

| Archivo | Pantalla |
|---|---|
| `01_inicio(_ventana).png` | Inicio con la barra lateral, las tarjetas y el pie |
| `02_datos_iris(_ventana).png` | 📊 Regresión › 📊 Datos con Iris (estado inicial) |
| `03_datos_iris_expandido.png` | Lo mismo con los expanders abiertos ("¿Qué significan…?", "Ver código") |
| `04_datos_correlacion_iris.png` | 🔗 Matriz de Correlación |
| `05_datos_pairplot_opciones.png` | Opciones del pairplot antes de generar, con el expander de consejos |
| `06_datos_pairplot_kde_iris.png` | Pairplot con diagonal KDE (valor por defecto) |
| `07_datos_pairplot_hist_iris.png` | Pairplot con diagonal de histograma |
| `08_datos_diabetes.png`, `09_…expandido.png` | Regresión: histograma del objetivo y expander de estadísticas |
| `10_datos_pairplot_diabetes.png` | Pairplot de regresión: hue numérico (ver §9, D11) |
| `11_datos_correlacion_digitos.png` | Correlación con 65 columnas y 3 columnas constantes (NaN) |
| `12_datos_titanic_desbalance.png` | Titanic (ratio 1,6: "bien balanceado"; describe sin las columnas booleanas) |
| `13_datos_moons.png` | Moons, con el aviso "Dataset generado aleatoriamente." |
| `14_datos_redes_tips.png` | 🧠 Redes › 📊 Datos con el `st.info` de consejos propio de redes |
| `15_regresion_pestana_evaluacion_dataset_actual.png` | Pestaña ≥ 2: solo "📊 Dataset actual" |
| `16_csv_vacio(_ventana).png` | 📁 Cargar CSV sin archivo (ayuda abierta y ejemplos) |
| `17_csv_cargado_clasificacion.png`, `18_…expandido.png` | `tips.csv` con el objetivo por defecto (`size`, sugerida Clasificación) |
| `19_csv_cargado_regresion_expandido.png` | Objetivo `tip` (Regresión): estadísticas e histograma |

Todas las vistas de estas pantallas son imágenes de matplotlib (`QLabel`), así que las
capturas las incluyen. **No se ha capturado:** el diálogo nativo de abrir archivo y el de
guardar imagen, el estado intermedio de progreso del pairplot (la barra solo se ve en la
barra de estado mientras trabaja el worker) ni la página de error del worker. Pesan unos
9,7 MB. Si interesa, se pueden pasar por `pngquant` antes del commit.

---

## 1. Shell (estructura común)

### 1.1 Ventana (Qt, `desktop/window.py`)

- Título "MLTutor — Aprende Machine Learning", 1400×920, mínimo 1000×700. La barra lateral
  ocupa 250 px (en la UI Rust ya mide 260) y el divisor no se puede plegar.
- Barra de estado: texto ("Preparando MLTutor…", "Actualizando…", "Listo", "Operación
  cancelada", "Cancelando al terminar el paso actual…", "No se pudo completar la operación",
  "Guardado: {nombre}"), una barra de progreso de 170 px (indeterminada; pasa a 0–100 si hay
  porcentaje) y el botón "Cancelar". Mientras trabaja, la interfaz se desactiva.
- Si falla, se muestra un aviso de error con el texto "No se pudo completar la operación. Puedes
  volver a intentarlo o elegir otra sección." y la traza en un bloque de código.
- Tipografía: Inter, Segoe UI o Arial de 14 px, en color `#243247`. La paleta completa está
  en `STYLE` y ya está portada en `ui/src/styles/theme.css`.

### 1.2 Barra lateral (`app.py::main`)

| Control | Etiqueta exacta | key | Efecto |
|---|---|---|---|
| Texto | `### Navegación:` | — | — |
| Botón | `🏠 Inicio` | `nav_home` | `navigation = "🏠 Inicio"` |
| Botón | `🌲 Árboles de Decisión` | `nav_trees` | lección de árboles |
| Botón | `📊 Regresión` | `nav_linear` | lección de regresión |
| Botón | `🔍 K-Nearest Neighbors` | `nav_knn` | lección de KNN |
| Botón | `🧠 Redes Neuronales` | `nav_nn` | lección de redes |
| Separador + texto | `---`, `### 🔧 Herramientas:` | — | — |
| Botón | `📁 Cargar CSV Personalizado` | `nav_csv` | página CSV |

El botón activo es `primary` (#1e88e5 con texto blanco). **Excepción:** en Python el botón CSV
nunca se resalta. Se recomienda resaltarlo también (ver §9, D14). Al cambiar de página se
conserva la pestaña activa de cada lección. El scroll vuelve arriba si cambian la página o
la pestaña y se mantiene si no cambian (`present()`). El resto del estado de sesión también
se conserva (§5.2).

### 1.3 Cabecera común (`ui.py::setup_page`) y pie (`show_footer`)

Todas las páginas empiezan con "🎓 MLTutor" (h1 centrado, #1e88e5, 30–32 px) y "Plataforma de
aprendizaje de Machine Learning" (#0d47a1, 20 px). Todas terminan con el pie: tres frases
separadas por " • " (en Qt salen en tres líneas, 10 px, #888). Contenido en
`content/inicio/cabecera.md` y `pie.md`.

### 1.4 Página de inicio (`ui.py::show_welcome_page`)

Orden: cabecera común → `st.header("🌟 Bienvenido a 🎓 MLTutor")` → `### Una plataforma
interactiva para aprender Machine Learning` → tarjeta "¿Qué es el Machine Learning?" (fondo
#E1F5FE, h3 #0288D1, radio 10, padding 20) → `## 📚 Algoritmos disponibles` → "Selecciona un
algoritmo para comenzar a aprender:" → 4 columnas iguales, cada una con su tarjeta (alto
200 px, borde izquierdo de 5 px, colores en `content/inicio/tarjetas.toml`) y debajo un
botón a todo el ancho "✨ Explorar …" que navega a la lección. Solo la tarjeta de Árboles
lleva "**Estado:** Disponible" (en verde #2E7D32) → `## 📝 Cómo usar MLTutor` → tarjeta "Guía
de uso" (fondo #F5F5F5) con una lista numerada de 5 pasos → pie.

### 1.5 Lecciones: cabecera, pestañas y navbar

Cada lección empieza con `st.header(título)`, un subtítulo y el expander "ℹ️ ¿Qué es/son …?"
(cerrado; textos en `content/<lección>/intro.md`). **Los subtítulos de `ui/src/lib/routes.ts`
no coinciden con los de Python** y hay que corregirlos con los de
`content/inicio/navegacion.toml` (`[lessons.*].subtitle`).

**Pestañas.** No son un `QTabWidget`: son 7 botones de igual ancho en una fila (`st.columns(7)`),
con la pestaña activa como `primary`, seguidos de un separador `---`. El estado se guarda en
una clave por lección: `active_tab` (árboles), `active_tab_lr`, `active_tab_knn` y
`active_tab_nn`, con 0 por defecto. Las keys de los botones son `tab_{i}`, `tab_lr_{i}`,
`tab_knn_{i}` y `tab_nn_{i}`. Las etiquetas están en `navegacion.toml`.

**Navbar** (`apps/navbar.py::navbar(state_key, prev_msg, next_msg, prev_note, next_note)`),
al final de cada pestaña: `---` → `### 🧭 Navegación` → 2 columnas. A la izquierda, si
`prev_msg`, el botón `🔙 {prev_msg}` (secundario, a todo el ancho, desactivado en la pestaña 0)
y la nota en Markdown. A la derecha, si `next_msg`, el botón `🔜 {next_msg}` (primary) y la
nota. Los botones restan o suman 1 a la pestaña. Mensajes:

| Lección | Pestaña 0 → … → 6 |
|---|---|
| Árboles / KNN | (—, Continuar a Entrenamiento) · (Volver a Datos, Continuar a Evaluación) · (Volver a Entrenamiento, Continuar a Visualización) · (Volver a Evaluación, Continuar a Características) · (Volver a Visualización, Continuar a Predicciones) · (Volver a Características, Continuar a Exportar) · (Volver a Predicciones, —) |
| Regresión | igual, cambiando Características → **Coeficientes** |
| Redes | (—, Continuar a Arquitectura) · (Volver a Datos, Continuar a Entrenamiento, next_note "**¿Listo para entrenar?** ¡Tu arquitectura está configurada!") · (Volver a Arquitectura, "Ver Evaluación" solo si está entrenada, o —) · (Volver a Entrenamiento, Ver Visualización) · (Volver a Evaluación, Ver Predicciones) · … (se detalla en F5) |

**Panel de subvistas** (`ui.py::create_button_panel(buttons)`): N columnas iguales con un
botón cada una (`(label, value, key)`). El activo es `primary`. **Trampa:** el estado es una
única clave global `viz_type`, compartida por todas las lecciones. Si el valor guardado no
está entre las opciones del panel, se elige la primera; si está (p. ej. "Frontera" o "ROC"),
se conserva al cambiar de lección. Ver §9, D13.

### 1.6 Componentes de presentación (renderizador Qt)

| Componente | Aspecto en Qt (referencia visual) |
|---|---|
| Aviso `info/warning/error/success` | Fondo #e3f2fd / #fff8e1 / #ffebee / #e8f5e9, radio 6, cuerpo en Markdown |
| Métrica | Tarjeta #f3f7fc con borde #e1e9f5 y radio 10; etiqueta normal; valor de 26 px en negrita #0d47a1; `delta` opcional debajo; `help` en el tooltip |
| Expander | Botón conmutable a todo el ancho con la etiqueta; el cuerpo se muestra u oculta; el estado se recuerda por id (`view_state`) durante la sesión |
| Tabla | Cabecera #eaf0f9 (texto #355174, negrita, alto mínimo 44); filas de 36 px alternando con #f5f8fd; índice visible en #f3f6fb/#64748b (salvo `hide_index`); números alineados a la derecha con formato `.6g` y el valor completo en el tooltip; columnas de 100 a 320 px; altura total sin scroll interno; selección + Ctrl/⌘C copia en TSV los valores originales; leyenda "{n:,} filas · {m} columnas  ·  Selecciona y copia con Ctrl+C / ⌘C". **No se puede ordenar** (el inventario dice "ordenables": es una mejora) |
| Bloque de código | Panel #152238 con borde #263954 y radio 10; etiqueta del lenguaje en mayúsculas (#b8c9e0, 12 px, negrita); botón "Copiar código" → "✓ Copiado"; Menlo/Consolas 14 px #e4edf8; colores del resaltado: keyword #c4a7ff, builtin #7dd3fc, name #82e2bf, string #a8db8f, comment #98a8c2, number #f6bd79, operator #89c9ef |
| `show_code_with_download(code, title, file)` | `#### {title}` + expander "Ver código" (cerrado) con el bloque, y debajo 2 columnas: botón "📥 Descargar código" y un `st.info` "Usa el botón de copia en la esquina superior derecha del bloque de código para copiarlo" |
| Imagen (matplotlib) | Escalada al ancho, con el botón "Guardar imagen…" (PNG a 120 dpi). Los enlaces `get_image_download_link` se convierten en botón (PNG a 300 dpi) |
| Selectores | Combo #f8fafd con radio 10; radio (vertical u horizontal); multiselect de chips (`MultiSelect`, con resumen "{k} de {n} seleccionadas · Máximo {m}"); slider + spinbox a la par; el slider solo emite al soltar |
| Archivo | Etiqueta + botón "Seleccionar archivo CSV…" (o el nombre del archivo) + "Quitar archivo" |

---

## 2. Pestaña 📊 Datos

Código: `dataset/dataset_tab.py::run_dataset_tab(active_tab, tips=None)`, que llaman
árboles, regresión y KNN. Redes llama directamente a `run_select_dataset()` +
`run_explore_dataset_tab()`, y antes muestra su `st.info(tips)` (`content/datos/tips-redes.md`).

### 2.1 Qué se muestra en cada pestaña

| Pestaña | Árboles / Regresión / KNN | Redes |
|---|---|---|
| 0 (📊 Datos) | Header "📊 Selección y Preparación de Datos" + selector + exploración completa | Lo mismo, más el info de consejos |
| 1 (🏋️ Entrenamiento) | Header + **selector** (sin exploración) y después la configuración de la lección | "📊 **Dataset actual:** …" |
| ≥ 2 | `st.info("📊 **Dataset actual:** {selected_dataset}")` + `---` | igual |

### 2.2 Selector (`run_select_dataset`)

| Control | Tipo | Etiqueta | Opciones | Por defecto | key | Efecto |
|---|---|---|---|---|---|---|
| Dataset | selectbox | `Dataset:` | 12 integrados, en este orden: 🌸 Iris - Clasificación de flores · 🍷 Vino - Clasificación de vinos · 🔬 Cáncer - Diagnóstico binario · 🔢 Dígitos - Clasificación de dígitos · 🩺 Diabetes - Progresión (regresión) · 💎 Diamantes - Precio (regresión) · ⛽ MPG - Consumo combustible (regresión) · 🌙 Moons - Clasificación sintética · 🚢 Titanic - Supervivencia · 💰 Propinas - Predicción de propinas · 🏠 Viviendas California - Precios · 🐧 Pingüinos - Clasificación de especies; después los CSV cargados ("📄 {archivo}") | `selected_dataset`, o Iris | `unified_dataset_selector` | Escribe `session_state.selected_dataset` (compartido por todas las lecciones) |

Ayuda: "El dataset seleccionado se mantendrá entre las pestañas de Exploración y
Entrenamiento". Debajo, una línea según el dataset (`selector_blurb` de
`content/datasets/<id>.toml`). **Dígitos, Diabetes, Diamantes, MPG y los CSV no tienen
línea.** Con Moons se llama a `reset_moons_dataset()` (500 muestras, ruido 0,2,
**semilla = timestamp UNIX**) **en cada renderizado** y se muestra `st.info("Dataset generado
aleatoriamente.")` (ver §9, D9). La interfaz no tiene controles de nº de muestras ni de ruido:
el inventario §1 está mal en ese punto. Termina con `---`.

### 2.3 Información (`show_dataset_info`)

`st.success("✅ Dataset seleccionado: **{nombre}**")`, `### 📊 Información del Dataset` y 4
métricas en 4 columnas:

| Métrica | Valor | help |
|---|---|---|
| 📏 Filas | n.º de filas | Número total de ejemplos para entrenar |
| 📊 Columnas | n_features + 1 | Características + variable objetivo |
| 🎯 Variable Objetivo | siempre `target` | Lo que la red va a predecir |
| 🏷️/📈 Tipo de Tarea | Clasificación / Regresión | Clasificación o Regresión |

La descripción de `dataset_info` **no** aparece en esta pestaña: `create_info_box` está
comentado. Sí aparece en la de Entrenamiento de cada lección (fase de cada lección). Con D8,
en MPG "Columnas" vale 8 (7 características + objetivo) y `name` no cuenta.

### 2.4 Exploración (`run_explore_dataset_tab`)

Datos: `df_display` = X con los **nombres legibles** (`feature_names`, mapeados desde
`X.columns` cuando las longitudes coinciden) + la columna `target`. En clasificación,
`target` contiene los **nombres de clase**; en regresión, el valor numérico. **Rust alinea
X e y por posición** (Python lo hace por índice: ver el fallo de Pingüinos en §11).

1. `st.header("Exploración de Datos")`.
2. `### 👀 Vista previa de datos`: tabla `df_display.head(10)`, con índice (en Pingüinos el
   índice original tiene huecos; en Rust, 0..9). Con D8, MPG muestra `name` (texto) antes de
   `target`.
3. Expander "💡 ¿Qué significan estos datos?" (cerrado), `content/datos/que-significan.md`
   con `{target_col}` = `target`.
4. `### Estadísticas Descriptivas`: `df_display.describe()`. Solo columnas numéricas: **las
   booleanas (Titanic `Hombre_Adulto`, `Solo`) y el objetivo textual de clasificación quedan
   fuera**. Filas: count, mean, std (ddof=1), min, 25%, 50%, 75% y max (cuantiles lineales).
   Celdas con formato `.6g`.
5. `### 🎯 Distribución del Objetivo`:
   - **Clasificación**: un aviso según el ratio = máx/mín de `value_counts`. Si es > 3,
     `st.warning("⚠️ **Dataset desbalanceado**: La clase más frecuente tiene {ratio:.1f}x más
     ejemplos que la menos frecuente")` + `st.info("💡 Los modelos de IA funcionan mejor con
     clases balanceadas. Considera técnicas de balanceo si es necesario.")`; si no,
     `st.success("✅ **Dataset bien balanceado**: Las clases tienen cantidad similar de
     ejemplos")`. Gráfico: barras `sns.barplot` de `value_counts().sort_index()` (**orden
     alfabético del nombre de clase**: en Cáncer, benign antes que malignant), todas del mismo
     color (C0 de seaborn con saturación 0,75 ≈ `#3274a1`), título "Distribución de Clases",
     ejes "Clase" / "Cantidad", etiquetas giradas 45° (alineadas a la derecha) si hay más de 3
     clases. Figura de 10×6 en la columna central de `[1, 3, 1]`.
   - **Regresión**: histograma de 30 bins iguales entre el mínimo y el máximo (np.histogram),
     alpha 0,7, borde negro, color C0 `#1f77b4`; título "Distribución de Valores Objetivo",
     ejes "Valor" / "Frecuencia". Antes del gráfico, el expander "📊 Estadísticas de la
     Variable Objetivo" (cerrado) con 4 métricas en `{:.2f}`: 🎯 Media, 📏 Desv. Estándar
     (ddof=1), 📉 Mínimo y 📈 Máximo.
6. `### 📊 Análisis Avanzado de Datos`, `#### Elige el tipo de análisis que quieres realizar:`
   y 2 botones a todo el ancho que funcionan como selector excluyente (estado
   `active_analysis` ∈ {None, "correlation", "pairplot"}, inicialmente None; el activo es
   primary):
   - `🔗 Matriz de Correlación`, con help "Analiza las relaciones lineales entre características";
   - `📈 Matriz de Dispersión (Pairplot)`, con help "Visualiza relaciones entre todas las parejas de características".
7. Análisis activo (§2.5 / §2.6).
8. `show_code_with_download(SCATTERPLOT_MATRIX, "Código para análisis exploratorio",
   "analisis_exploratorio.py")`, con la plantilla en `content/code/analisis_exploratorio.py.j2`.
9. Navbar (§1.5).

Errores: `st.error("Error al cargar el dataset: {e}")` + `st.info("Por favor, selecciona un
dataset válido para continuar con la exploración.")`.

### 2.5 Matriz de correlación

`#### Matriz de Correlación` + `st.info` (`content/datos/correlacion.md`).

- Datos: `concat([X con sus columnas ORIGINALES, y numérico como 'target']).corr()` (Pearson,
  observaciones completas por pares; NaN si una columna es constante: Dígitos `0`, `32` y `39`).
  **Las etiquetas son `X.columns`, no los nombres legibles** (en Titanic se ve `sex_encoded`;
  en Dígitos, `0..63`). Recomendación: mantenerlo así, para que la tabla coincida con el
  código que se enseña.
- Vista: heatmap `coolwarm`, con anotaciones `.2f`, celdas cuadradas, líneas blancas de 0,5,
  colorbar al 80 %. **Solo se muestra el triángulo inferior estricto** (máscara `np.triu`
  con k=0, así que la diagonal queda oculta). El rango de color va del mínimo al máximo de
  las celdas visibles (en Iris, de -0,43 a 0,96); no es simétrico. Ver §9, D12. Título
  "Matriz de Correlación de Características". Figura de 20×16 en la columna central (las
  fuentes quedan diminutas en Qt: en Plotly debe verse mejor).
- Botón "📥 Descargar matriz de correlación" → `matriz_correlacion.png`.

### 2.6 Pairplot

`#### Matriz de Dispersión (Pairplot)` + `st.info` (`content/datos/pairplot.md`) y
`##### Opciones de visualización` en 2 columnas:

| Control | Tipo | Etiqueta | Opciones / rango | Por defecto | key | Efecto |
|---|---|---|---|---|---|---|
| Diagonal | radio horizontal | `Tipo de gráfico en la diagonal:` | Histograma · KDE (Estimación de Densidad) | índice 1 (KDE) | sin key (se recuerda por sitio de llamada) | `hist` o `kde` |
| Máx. características | slider | `Número máximo de características:` | 2 .. min(6, n), paso 1 | min(4, n) | sin key | Límite y valor por defecto del multiselect. Si n ≤ 2 no hay slider y aparece `st.info("Sólo hay {n} características disponibles; usando las {n} para el pairplot.")` |
| Características | multiselect | `Características a incluir en la matriz de dispersión:` | nombres legibles | las primeras `max` | sin key | Columnas del pairplot; `max_selections = max` |

Help del slider: "Un número mayor de características puede hacer que el gráfico sea más
difícil de interpretar." Help del multiselect: "Selecciona hasta {max} características para
incluir en la visualización." Si la selección está vacía, se usan las primeras `max` y aparece
`st.info("No se seleccionaron características. Usando las primeras {max} por defecto.")`.
**Ocurre siempre al cambiar de dataset**, porque los nombres anteriores no existen y la
selección filtrada queda vacía (captura 10).

Avisos: con más de 25 gráficos (k²), "⚠️ **Cuidado**: {k} características = {k²} gráficos.
Puede ser muy lento."; con más de 16, "ℹ️ **Nota**: … Tiempo estimado: ~{k²·filas/50000:.1f}s".
Botones en `[3, 1]`: `🚀 Generar Matriz de Dispersión` (primary) y `🛑 Reiniciar` (help: "Usa
esto si la generación se queda colgada"; solo hace rerun). Hasta generar se muestra `st.info("👆
Haz clic en el botón…")` y el expander "💡 Consejos para evitar que se cuelgue la generación"
(`pairplot-consejos.md`).

Al generar:

- `plot_df` = las características elegidas con nombres legibles + `target` (nombres de clase o
  valor).
- **Muestreo** si hay más de 2000 filas. Clasificación: `groupby(target).sample(min(len,
  2000 // n_clases), random_state=42)` (semilla 42 nueva en cada grupo). Regresión:
  `sample(n=2000, random_state=42)` (= `RandomState(42).permutation(n)[:2000]`). Con
  muestreo aparece "📊 Usando {n} puntos de {total} totales para optimizar velocidad" y el
  título añade " (Muestra: {n}/{total} puntos)". Afecta a California (20 640) y Diamantes
  (8000). El aviso "límite de seguridad" es inalcanzable.
- Estilo: con más de 2000 filas (alta calidad): puntos alpha 0,7, s=40, borde blanco de 0,5,
  diagonal alpha 0,8 y celdas de 2,5 in. Si no: alpha 0,6, s=20 sin borde, diagonal alpha 0,6
  y celdas de 2,0 in.
- `sns.pairplot(hue='target', diag_kind, corner=False)`: rejilla completa k×k, dispersión
  fuera de la diagonal y leyenda "target" a la derecha. Título "Matriz de Dispersión de
  Características" (14 pt).
- **Colores (clasificación):** niveles en el orden `categorical_order` (orden de aparición
  en los datos: Iris setosa, versicolor, virginica). Hasta 10 niveles, paleta `tab10` de
  matplotlib: `#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd #8c564b #e377c2 #7f7f7f #bcbd22 #17becf`.
  Con más, `husl`.
- **KDE (seaborn 0.13 `kdeplot`, el que usa el pairplot):** `scipy.stats.gaussian_kde` con
  `bw_method="scott"` (factor = n^(-1/5)) y `bw_adjust=1`. La desviación del núcleo es
  factor·std(ddof=1). Hay **una KDE por nivel**, cada una con su rejilla (`common_grid=False`):
  `linspace(min − 3·bw, max + 3·bw, 200)` de ese nivel. `common_norm=True` multiplica la
  densidad por n_nivel/n_total. Si el nivel tiene n < 2 o varianza 0, se omite (Dígitos `p0`).
  Relleno (`fill=True`) con alpha 0,6/0,8. Verificado: el fixture coincide con los polígonos
  que dibuja `sns.pairplot` (12 decimales).
- **Histograma de la diagonal:** `histplot` con bordes `np.histogram_bin_edges(todos los
  valores, "auto")` comunes a todos los niveles, una serie por nivel superpuesta
  (`multiple="layer"`) y recuento.
- **Regresión:** el hue es el valor numérico. seaborn colorea la dispersión con un mapa
  continuo cubehelix (de rosa claro a morado) con 5 marcas en la leyenda, y en la diagonal
  dibuja **una KDE o un histograma por cada valor único** del objetivo, lo que resulta
  ilegible (captura 10). Ver §9, D11.
- Botón "📥 Descargar matriz de dispersión" → `matriz_dispersion.png`. Progreso simulado (10,
  20, 30, 50, 80, 90 y 100 %, con `time.sleep(1)`) y los mensajes de `mensajes.toml
  [pairplot]`. En Rust basta con un spinner, porque el cálculo tarda milisegundos.

---

## 3. Página 📁 Cargar CSV Personalizado

Código: `app.py::run_csv_loader_app` + `dataset_manager.py::create_dataset_selector(show_predefined=False)`.
Textos en `content/csv/`.

1. `st.header("📁 Cargar CSV Personalizado")` + subtítulo.
2. Expander "ℹ️ ¿Cómo usar esta herramienta?" (**abierto**), `content/csv/ayuda.md`. `---`.
3. `st.subheader("📊 Selección de Dataset")`, el info "💡 En esta sección puedes cargar…" y
   `### 📁 Cargar Archivo CSV`.
4. File uploader `Selecciona un archivo CSV:` (`type=['csv']`, help "El archivo debe estar en
   formato CSV con encabezados"). En Qt abre un diálogo nativo con el filtro `Archivos (*.csv)`.
5. Sin archivo: `st.info("👆 Por favor, carga un archivo CSV para continuar")`, y la página
   añade `### 💡 Ejemplos de Datasets que puedes cargar:` con 4 ejemplos
   (`content/csv/ejemplos.toml`; columnas [1, 10], icono en `##`, "**nombre** (tarea)" y la
   descripción en cursiva, con `---` tras cada uno).
6. Con archivo:
   - `pd.read_csv(file)`: UTF-8, separador `,` y cabecera obligatoria. Validaciones: vacío →
     `❌ El archivo está vacío`; menos de 2 columnas → `❌ El archivo debe tener al menos 2
     columnas (características + objetivo)`; excepción → `❌ Error al procesar el archivo: {e}`.
   - `st.success("✅ Archivo cargado exitosamente: {filas} filas, {cols} columnas")`.
   - Expander "👀 Vista previa del dataset" (**abierto**): `df.head()` (5 filas) y 3 métricas:
     Filas, Columnas y Valores faltantes (`isnull().sum().sum()`).
   - `### ⚙️ Configuración` en 2 columnas:

     | Control | Tipo | Etiqueta | Opciones | Por defecto | Efecto |
     |---|---|---|---|---|---|
     | Objetivo | selectbox | `Columna objetivo:` | todas las columnas | la última | Columna y |
     | Tarea | selectbox | `Tipo de tarea:` | auto · Clasificación · Regresión | la **sugerida**: `nunique ≤ 20 y nunique < 0,1·n` → Clasificación, si no Regresión | Se guarda tal cual, incluido "auto" |

     Help: "Selecciona la columna que contiene la variable a predecir" y "'auto' detecta
     automáticamente, o selecciona manualmente". Ninguno tiene key: se recuerdan por sitio
     de llamada.
   - `### 🎯 Información de la Variable Objetivo`, en 2 columnas: "**Columna:** …", "**Valores
     únicos:** …", "**Tipo de datos:** {dtype de pandas}" y, si nunique ≤ 10, "**Distribución
     de valores:**" con `• {v}: {n} ({pct:.1f}%)` en el orden de `value_counts` (de mayor a
     menor frecuencia).
   - Registro: `store_csv(df)` (reescribe el CSV en UTF-8 en un directorio temporal de la
     sesión) y `csv_datasets["📄 {archivo}"] = {file_path, target_col, task_type,
     original_name}`. Después hace **`selected_dataset = unified_dataset_selector = "📄
     {archivo}"`**: el CSV pasa a ser el dataset de todas las lecciones.
     `st.success("✅ Dataset '{archivo}' añadido a la lista de datasets disponibles")`.
7. Resultado (`run_csv_loader_app`): `---`, `st.success("✅ ¡Dataset CSV cargado
   exitosamente!")`, `### 📊 Información del Dataset Cargado:` y 4 métricas (📏 Filas, 📊
   Columnas, 🎯 Variable Objetivo, {🏷️|📈} Tipo de Tarea, que puede mostrar "auto").
   Después dos expanders cerrados:
   - "📈 Estadísticas Descriptivas": `df.describe()` del CSV **crudo** (solo las columnas
     numéricas; incluye el objetivo si es numérico).
   - "🎯 Distribución de la Variable Objetivo". Clasificación: 2 columnas, con la tabla
     `value_counts().to_frame("Cantidad")` (orden por frecuencia) y barras `pandas.plot.bar`
     (C0 `#1f77b4`, título "Distribución de {col}", eje y "Cantidad", etiquetas a 45°, 8×6).
     Regresión (o "auto"): "**Estadísticas:**" con "• Mínimo/Máximo/Media/Mediana/Desv.
     Estándar: {:.4f}" e histograma `Series.hist(bins=30, alpha=0.7)` (con rejilla, eje x
     {col}, eje y "Frecuencia").
   - Error: `st.warning("No se pudieron cargar las estadísticas adicionales: {e}")`.
8. "Quitar archivo" vacía el uploader, pero **el CSV sigue registrado** en `csv_datasets` y
   seleccionado.

Al cargar el CSV en una lección (`load_dataset_from_file(path, target, task)`), el objetivo
se separa del resto; **las columnas `object` de X pasan por LabelEncoder** (orden alfabético
de `astype(str)`, así que NaN se convierte en la categoría "nan"); el objetivo `object` también
se codifica. Con "auto", es Clasificación si `nunique < max(5, 0,05·n)`; **esta regla es
distinta de la sugerencia del selector**. Las clases se llaman "Clase {i}" si los valores son
enteros (**se pierden los nombres originales**) y `str(v)` en otro caso (bool → "False"/"True").
`dataset_info` es un *str*: "Dataset personalizado: {n} muestras, {f} características, {k}
clases" o "…, regresión".

---

## 4. Casos límite del CSV

Fixtures en `tests/parity/csv/*.json`: el archivo de entrada (texto y base64) y lo que hace
Python con él. "Rust" indica la propuesta.

| Caso (fixture) | Python | Rust |
|---|---|---|
| Codificación no UTF-8 (`latin1_encoding`) | `UnicodeDecodeError`, que llega al estudiante en inglés | UTF-8 (con o sin BOM) y, si falla, Windows-1252 con el aviso "Se ha leído el archivo como Windows-1252/Latin-1" (D10). Mensaje en español |
| BOM UTF-8 (`utf8_bom`) | Se ignora | Igual |
| Separador `;` (Excel en español) (`semicolon_separator`) | 1 columna → "❌ … al menos 2 columnas" | Detectar `,` `;` `\t` en la primera línea (D10). Si no, el mismo mensaje |
| Coma decimal entre comillas (`decimal_comma_quoted`) | Columna `object` → LabelEncoder (¡"1,5" pasa a ser una categoría!) | Igual que Python salvo que se decida otra cosa (D10: con `;` como separador, aceptar la coma decimal) |
| Columnas no numéricas en X (`categorical_and_nan`, `string_target`) | LabelEncoder alfabético por columna | Igual: inferencia de tipos (bool, entero, real, texto) y codificación ordenada por la representación de texto; se guarda el mapa código → texto para la UI |
| NaN en X numérica | Se mantiene NaN: describe lo ignora y el entrenamiento falla después en sklearn | Se mantiene `f64::NAN` y se informa con "Valores faltantes". Las lecciones (F2+) deciden: aviso y eliminar filas. Nunca un panic |
| NaN en X de texto | Pasa a la categoría "nan" | Igual (paridad), o categoría propia "(vacío)" (D10) |
| NaN en el objetivo | Texto: clase "nan"; numérico: NaN | Igual en F1; las lecciones deben avisar |
| Objetivo de texto (`string_target`) | Clases "Clase 0..k" | **Nombres originales** (`rosa`, `lirio`…) (D10, recomendado) |
| Objetivo booleano (`bool_target`) | dtype bool → clases "False"/"True" | Igual |
| Pocas clases o 1 muestra por clase (`one_class_per_row`) | Clasificación con 3 clases de 1 muestra; el split posterior no estratifica | Igual; avisar en Entrenamiento (F2) |
| Reglas de tarea distintas (`seven_levels_100_rows`) | Sugerencia "Clasificación", pero "auto" → Regresión | Las dos reglas, tal cual, con sus nombres (`suggest_task` y `auto_task`) |
| Objetivo entero con muchos valores (`int_target_many_values`) | Regresión | Igual |
| Solo cabecera (`header_only`) | "❌ El archivo está vacío" | Igual |
| Archivo vacío (`empty_file`) | "❌ Error al procesar el archivo: No columns to parse from file" | `CsvError::Empty` → "❌ El archivo está vacío" |
| 1 columna (`single_column`) | "❌ … al menos 2 columnas" | Igual |
| Nombres de columna duplicados | pandas los renombra (`a`, `a.1`) | Igual (`a`, `a.1`) |
| Filas con distinto nº de campos | `ParserError` | `CsvError::Malformed { line }` |
| Tamaño | Sin límite (la UI de Qt muestrea 100 filas para medir columnas) | Límite de 100 MB (D10); por encima, error. La vista previa siempre se recorta; el pairplot muestrea 2000 filas |

---

## 5. Contrato IPC

### 5.1 Convenciones

- Todos los comandos son `async` y el trabajo con datos va en
  `tauri::async_runtime::spawn_blocking` (MEMORY: los panics en el hilo principal abortan la
  app). Cada uno se declara en `src-tauri/build.rs` y en `capabilities/default.json`.
- Tipos con `serde` + `ts-rs` (`#[ts(export)]`), en snake_case. Los enums van etiquetados con
  `#[serde(tag = "kind", rename_all = "snake_case")]`, como `LaunchMode`.
- **Matrices:** `Matrix { rows: u32, cols: u32, values: Vec<Option<f64>> }` en row-major.
  `Option` porque serde_json escribe NaN como `null` pero no lo vuelve a leer. Las series
  largas (densidades, columnas) van como `Vec<f64>`. **Nunca se envían imágenes:** las
  descargas PNG las genera Plotly en el frontend (`Plotly.toImage`) y se guardan con
  `save_file` (diálogo y escritura en Rust; la UI no tiene `plugin-fs`).
- Los tipos de dominio viven en `mltutor-core` (sin Tauri). Los DTO del IPC viven en
  `src-tauri/src/ipc/` (módulo nuevo) y se convierten desde los tipos de core con `From`. Así
  core no depende de ts-rs. Alternativa: `ts-rs` como dependencia opcional de core con la
  feature `ts`. Recomendación: DTO en `src-tauri` y que el frontend reciba solo esos.
- Los errores de core (`thiserror`) se traducen a `CommandError` con un mensaje en español
  listo para `Notice`.

```rust
// src-tauri/src/ipc/error.rs
#[derive(Serialize, TS)] #[serde(tag = "kind", rename_all = "snake_case")] #[ts(export)]
pub enum CommandError {
    DatasetNotFound { message: String },                 // "Conjunto de datos '{x}' no reconocido"
    CsvEmpty { message: String },                        // "❌ El archivo está vacío"
    CsvTooFewColumns { message: String },                // "❌ El archivo debe tener al menos 2 columnas (características + objetivo)"
    CsvEncoding { message: String },                     // "❌ El archivo no es UTF-8 ni Windows-1252"
    CsvMalformed { line: u64, message: String },         // "❌ La línea {line} tiene {n} campos y la cabecera {m}"
    CsvTooLarge { bytes: u64, limit: u64, message: String },
    CsvIo { message: String },                           // "❌ Error al procesar el archivo: {e}"
    UnknownColumn { column: String, message: String },   // "Columna objetivo '{c}' no encontrada en el dataset"
    InvalidSelection { message: String },                // características inexistentes, etc.
    Internal { message: String },                        // "No se pudo completar la operación. …"
}
```

### 5.2 Estado por sesión (`src-tauri/src/session.rs`, `Mutex<Session>` en `AppState`)

En Python, `selected_dataset` es **una única clave de `session_state` compartida por las cuatro
lecciones y por la página CSV** (comprobado: la escriben `run_select_dataset`, en
cualquier lección, y `create_dataset_selector`, al cargar un CSV; la leen las 4 lecciones y
`create_prediction_interface`). Cada lección guarda aparte el último que cargó
(`dataset_option`, `dataset_option_lr`, `dataset_option_knn`) para saber si debe reentrenar.

```rust
pub struct Session {
    pub selected: DatasetRef,                     // por defecto Builtin(Iris)
    selection_seq: u64, csv_seq: u64,             // último seq aplicado (selección / registro CSV)
    csv: IndexMap<u32, CsvEntry>,                 // como mucho uno: otro archivo libera el anterior
    draft: Option<(u32, CsvDraftHandle)>,         // archivo leído aún sin objetivo confirmado
    cache: HashMap<DatasetId, DatasetHandle>,     // integrados ya parseados; Moons fijo (D9)
}
pub struct CsvEntry { entry: DatasetEntry /* label "📄 {archivo}" */, file_name: String,
                      target: String, task: TaskChoice,
                      dataset: DatasetHandle /* describe y distribución en OnceLock */ }
```

Los métodos que liberan un borrador o un CSV (`clear_draft`, `discard_draft`, `store_draft`,
`register_csv`) lo devuelven, y `DataState` lo suelta después de liberar el lock.

El estado **de interfaz** se queda en el frontend, en un store de Svelte por sesión: pestaña
activa de cada lección (`activeTab[lessonId]`), `active_analysis`, tipo de diagonal,
características del pairplot, panel de subvistas (D13) y expanders abiertos. El backend no
lo necesita para calcular, y así no hace falta una ida y vuelta por cada clic.

### 5.3 Tipos

```rust
#[serde(rename_all = "snake_case")]
pub enum DatasetId { Iris, Wine, BreastCancer, Digits, Diabetes, Diamonds, Mpg, Moons,
                     Titanic, Tips, CaliforniaHousing, Penguins }  // orden del selector
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetRef { Builtin { id: DatasetId }, Csv { id: u32 } }
#[serde(rename_all = "snake_case")] pub enum TaskType { Classification, Regression }
#[serde(rename_all = "snake_case")] pub enum TaskChoice { Auto, Classification, Regression }

pub struct DatasetEntry {            // una opción del selector
    pub dataset: DatasetRef,
    pub label: String,               // "🌸 Iris - Clasificación de flores" | "📄 ventas.csv"
    pub blurb: Option<String>,       // Markdown bajo el selector (selector_blurb)
    pub task: TaskType,
    pub synthetic: bool,             // Moons → muestra "Dataset generado aleatoriamente."
}

pub struct DatasetSummary {
    pub dataset: DatasetRef,
    pub label: String,
    pub task: TaskType,
    pub n_rows: u32,
    pub n_features: u32,             // métrica "📊 Columnas" = n_features + 1
    pub target_column: String,       // siempre "target" en la UI
    pub feature_columns: Vec<String>,  // X.columns originales (correlación, código)
    pub feature_display: Vec<String>,  // nombres legibles (vista previa, pairplot)
    pub label_columns: Vec<String>,    // D8: ["name"] en MPG
    pub class_names: Option<Vec<String>>,
    pub description: String,         // dataset_info.description (se muestra en Entrenamiento)
}

#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Cell { Number { v: f64 }, Text { v: String }, Bool { v: bool }, Missing }
pub struct TablePreview {
    pub columns: Vec<String>,
    pub index: Vec<String>,          // "0".."9"
    pub rows: Vec<Vec<Cell>>,        // row-major
    pub total_rows: u32,
}

pub struct DescribeTable {           // pandas describe()
    pub columns: Vec<String>,
    pub stats: Vec<String>,          // ["count","mean","std","min","25%","50%","75%","max"]
    pub values: Matrix,              // stats × columns
    pub excluded_columns: Vec<String>, // no numéricas (bool, texto, objetivo de clase)
}

#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetDistribution {
    Bar { labels: Vec<String>, counts: Vec<u32>, balance_ratio: f64, balanced: bool },
    Histogram { edges: Vec<f64>, counts: Vec<u32>, mean: f64, std: f64, min: f64, max: f64 },
}

pub struct CorrelationMatrix {
    pub labels: Vec<String>,         // feature_columns + "target"
    pub values: Matrix,              // n × n, simétrica, None si NaN
    pub visible_min: Option<f64>, pub visible_max: Option<f64>,  // triángulo inferior estricto
}

#[serde(rename_all = "snake_case")] pub enum DiagKind { Kde, Hist }
pub struct PairplotRequest { pub dataset: DatasetRef, pub features: Vec<String> /* display */,
                             pub diag: DiagKind }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Hue {
    Categorical { levels: Vec<String>, codes: Vec<u16> },   // orden de aparición, paleta tab10
    Continuous { values: Vec<f64>, min: f64, max: f64 },    // regresión (D11)
}
pub struct KdeCurve { pub level: String, pub n: u32, pub bandwidth: Option<f64>,
                      pub grid_min: f64, pub grid_max: f64, pub density: Vec<f64> /* 200; vacío si se omite */ }
pub struct HistSeries { pub level: String, pub counts: Vec<u32> }
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiagPanel { Kde { curves: Vec<KdeCurve> }, Hist { edges: Vec<f64>, series: Vec<HistSeries> } }
pub struct PairplotData {
    pub features: Vec<String>,
    pub n_points: u32, pub n_total: u32, pub sampled: bool,
    pub points: Matrix,              // n_points × k, row-major (sin NaN de los datasets integrados)
    pub hue: Hue,
    pub tooltips: Option<Vec<String>>, // D8: nombre del coche en MPG
    pub diagonal: Vec<DiagPanel>,      // una por característica
    pub high_quality: bool,            // estilo de §2.6
    pub title: String,                 // con " (Muestra: n/N puntos)" si hubo muestreo
}

pub struct CsvColumnInfo { pub name: String, pub dtype: String /* int64|float64|bool|object */,
                           pub missing: u32, pub nunique: u32 }
pub struct TargetInfo { pub column: String, pub nunique: u32, pub dtype: String,
                        pub value_counts: Option<Vec<(String, u32)>> /* si nunique ≤ 10, por frecuencia */,
                        pub suggested_task: TaskType }
pub struct CsvPreview {
    pub draft_id: u32, pub file_name: String,
    pub n_rows: u32, pub n_cols: u32, pub missing_values: u32,
    pub columns: Vec<CsvColumnInfo>,
    pub head: TablePreview,          // 5 filas
    pub default_target: String,      // la última columna
    pub target: TargetInfo,          // de default_target
    pub notices: Vec<String>,        // p. ej. codificación o separador detectados (D10)
}
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CsvTargetStats {
    Classes { value_counts: Vec<(String, u32)> },        // por frecuencia (tabla + barras)
    Numeric { min: f64, max: f64, mean: f64, median: f64, std: f64,
              hist_edges: Vec<f64>, hist_counts: Vec<u32> }  // 30 bins
}
pub struct CsvLoaded {
    pub dataset: DatasetEntry,       // ya registrado y seleccionado
    pub task_choice: TaskChoice,     // lo que eligió el usuario ("auto" se muestra tal cual)
    pub resolved_task: TaskType,     // regla auto_task de load_dataset_from_file
    pub n_rows: u32, pub n_cols: u32, pub target: String,
    pub describe: DescribeTable,     // del CSV crudo
    pub target_stats: CsvTargetStats,
}
```

### 5.4 Comandos

| Comando | Entrada | Salida | Errores | Notas |
|---|---|---|---|---|
| `list_datasets()` | — | `Vec<DatasetEntry>` | — | 12 integrados en el orden del selector + el CSV registrado (como mucho uno) |
| `get_selected_dataset()` | — | `SelectedDataset { dataset, last_seq }` | — | Lo lee la UI al arrancar (o al recargar la página). `last_seq` es el `seq` más alto aplicado; la UI sigue con `max(Date.now()·1000, last_seq + 1)` aunque el reloj haya retrocedido |
| `select_dataset(dataset, seq)` | `DatasetRef`, `u64` | `DatasetSummary` | `DatasetNotFound` | Aplica la selección bajo el lock **antes** de calcular el resumen; gana el `seq` más alto (uno más viejo no la cambia, pero devuelve su resumen). Moons se genera una vez por sesión (D9) |
| `dataset_summary(dataset)` | `DatasetRef` | `DatasetSummary` | `DatasetNotFound` | Para las métricas de §2.3 |
| `dataset_preview(dataset, rows)` | `DatasetRef`, `u32` (10; máx. 1000) | `TablePreview` | `DatasetNotFound` | `df_display.head(rows)`, con D8. NaN va como `missing` y ±inf como texto `inf`/`-inf`, como pandas |
| `dataset_describe(dataset)` | `DatasetRef` | `DescribeTable` | ídem | En el CSV registrado se calcula una vez (`OnceLock` en la entrada) |
| `target_distribution(dataset)` | `DatasetRef` | `TargetDistribution` | ídem | ídem |
| `correlation_matrix(dataset, task_id)` | `DatasetRef`, `u32` | `CorrelationMatrix` | ídem, `Cancelled` | Cancelable |
| `pairplot_data(req)` | `PairplotRequest` | `PairplotData` | `InvalidSelection` (0 o > 6 características, o nombres desconocidos), `DatasetNotFound` | Muestreo de §2.6 |
| `cancel_task(task_id)` | `u32` | `bool` | — | Activa el token de la operación `task_id`; `false` si ya terminó. La operación cancelada falla con `Cancelled` (la UI no lo muestra como error) |
| `pick_and_load_csv(filter_name, task_id)` | `String` (etiqueta del filtro, máx. 80), `u32` | `Option<CsvPreview>` | `CsvEmpty`, `CsvTooFewColumns`, `CsvEncoding`, `CsvMalformed`, `CsvTooLarge` (> 100 MiB, D10), `CsvIo`, `Cancelled` | Abre el diálogo nativo "Abrir" y lee el archivo en Rust: **la ruta nunca pasa por JS**. `None` si se cierra el diálogo. Libera antes el borrador anterior y guarda uno nuevo, que **no** se registra todavía. Solo archivos normales (FIFO y dispositivos se rechazan) |
| `load_scenario_csv(task_id)` | `u32` | `CsvPreview` | ídem, `InvalidSelection` | Solo con `--screenshot --scenario <nombre>:<ruta.csv>`: la ruta sale de la línea de órdenes |
| `csv_target_info(draft_id, target)` | `u32`, `String` | `TargetInfo` | `UnknownColumn`, `CsvIo` (borrador ya sustituido) | Al cambiar "Columna objetivo:" |
| `set_csv_target(draft_id, target, task, seq, task_id)` | `u32`, `String`, `TaskChoice`, `u64`, `u32` | `CsvLoaded` | `UnknownColumn`, `CsvIo` (borrador sustituido o descartado), `Cancelled` (cancelado o superado por un registro con `seq` mayor) | Registra o actualiza `"📄 {archivo}"` (el mismo nombre reutiliza el id, como en Python; otro archivo libera el anterior) y lo selecciona salvo que haya una selección más nueva. Si se cancela, la página CSV conserva la vista previa y ofrece "Registrar de nuevo" |
| `discard_csv(draft_id)` | `u32` | `bool` | — | "Quitar archivo": libera el borrador (en un worker, fuera del lock). El CSV registrado sigue en el selector, como en Python |
| `save_file` | cuerpo binario raw + cabeceras `x-mltutor-file-name` (`encodeURIComponent`) y `x-mltutor-file-kind` (`png` \| `python`) | `Option<String>` (nombre guardado) | `FileSave` | Diálogo nativo "Guardar" con el nombre y el filtro de la extensión; escribe en Rust (la UI no tiene permisos de FS). Si el usuario quita la extensión y `nombre.ext` existe, pide confirmación nativa antes de reemplazarlo. Máx. 64 MiB (se comprueba con el cuerpo ya en memoria) |

Los comandos que cambian la selección llevan un `seq` creciente de la ventana (microsegundos
del reloj + contador). Los que pueden tardar llevan un `task_id` que la barra de estado usa
para "Cancelar" (`cancel_task`). Los borradores y CSV liberados se sueltan fuera del lock de
la sesión. `regenerate_moons` no existe: D9 fija Moons por sesión.

En Python, todas las vistas de §2.4 se calculan cada vez. En Rust, el frontend pide los 4
primeros comandos en paralelo al montar la pestaña, y correlación y pairplot solo bajo
demanda. Presupuesto: < 100 ms con California (20 640 × 8).

### 5.5 Qué módulo de `mltutor-core` implementa cada cosa

| Módulo de core | Implementa | Lo usan |
|---|---|---|
| `dataset::catalog` | `DatasetId`, CSV embebidos (`include_bytes!` desde `tests/parity/datasets/data/`), etiquetas, orden | `list_datasets`, `select_dataset` |
| `dataset::table` | `Dataset { features: Array2<f64>, feature_columns, feature_display, labels: Vec<Vec<String>>, target: Target, task, class_names }`; parser del CSV embebido usando `columns`/`label_columns` del fixture | todos |
| `dataset::metadata` | Lee `content/datasets/<id>.toml` (`include_str!` + `toml`): blurb, descripción, nombres legibles, valores categóricos y rangos | selector, F2+ (formulario de predicción) |
| `dataset::moons` | `make_moons(n=500, noise=0.2, seed)` | Moons (D9) |
| `csv` | Lector (UTF-8/BOM; D10: Windows-1252 y separador), inferencia de tipos al estilo de pandas (int64/float64/bool/object), `missing`, `LabelEncoder` ordenado, `suggest_task` (selector) y `auto_task` (`load_dataset_from_file`), nombres de clase y `dataset_info` | `load_csv`, `csv_target_info`, `set_csv_target` |
| `stats::describe` | count/mean/std(ddof=1)/min/cuantiles lineales/max, ignorando NaN | `dataset_describe`, CSV |
| `stats::value_counts` | Recuento con orden por índice (alfabético) o por frecuencia (estable como pandas) | distribución, CSV |
| `stats::histogram` | Bins iguales (`np.histogram`, el último cerrado) y `histogram_bin_edges("auto")` (como numpy: anchura = mín(anchura de Sturges, anchura de FD); si la de FD es 0, la de Sturges; nº de bins = ceil(rango/anchura)) | distribución, diagonal |
| `stats::correlation` | Pearson por pares con observaciones completas; NaN si la varianza es 0 | `correlation_matrix` |
| `stats::kde` | `gaussian_kde` univariante (Scott), rejilla de seaborn, `common_norm` | `pairplot_data` |
| `sampling` | `RandomState(seed).permutation` con rand_mt (compartido con el split D7), `DataFrame.sample` y `groupby().sample` | pairplot, split |
| `split` (D7) | `train_test_split` sin estratificar, exacto | F2+, pero se hace en F1 (PLAN) |

#### 5.5.1 API implementada y desviaciones (`rust-ml-engineer`, 2026-10-13)

La lista estable de funciones está en el doc-comment de `crates/mltutor-core/src/lib.rs`.
Diferencias con §5.1–5.3 que debe absorber la conversión a DTO en `src-tauri`:

- **Sin conceptos de sesión en core:** `DatasetRef`, `CsvId`, `draft_id` y `file_name` (de
  `CsvPreview`) los añade `src-tauri`. `DatasetEntry` lleva `id: DatasetId` en lugar de
  `dataset: DatasetRef`, y `DatasetSummary` no lleva `dataset`. `CsvLoaded` no existe en
  core: se compone con `RawTable::to_dataset` (con `resolved_task = Dataset::task()`),
  `RawTable::describe` y `RawTable::target_stats`.
- **Matrices:** `ndarray::Array2<f64>` con NaN (`DescribeTable.values`,
  `CorrelationMatrix.values`, `PairplotData.points`). El DTO las pasa a `Matrix` con `None`.
- **Recuentos y tamaños** en `usize` (el DTO los convierte a `u32`).
- **Campos añadidos:** `CorrelationMatrix.nan_columns`; `PairplotData.rows` (filas
  posicionales de la muestra); `KdeCurve.scott_factor: Option<f64>`. En un nivel omitido,
  `grid_min`/`grid_max` valen el mínimo y el máximo de sus datos (0 si no hay).
- **Función nueva:** `explore::pairplot_defaults(&Dataset) -> PairplotDefaults`
  (`n_features`, `slider: Option<{min, max, default}>`, `default_selected`).
- **Errores:** `CsvError::{Empty, TooFewColumns { found }, Encoding, Malformed { line,
  found, expected }, TooLarge { bytes, limit }, Io { message }, UnknownColumn { column }}`
  y `DatasetError::{UnknownDataset, UnknownColumn, InvalidSelection { reason }, Metadata,
  Data, Csv}`. Su `Display` ya está en español, sin el prefijo "❌ ".
- **Avisos del CSV:** `CsvPreview.notices: Vec<CsvNotice>` (`Windows1252`, `Separator`,
  `DecimalComma { columns }`) con `CsvNotice::message()` en español, en lugar de `Vec<String>`.
- **Comportamiento:** una fila con menos campos que la cabecera también es `Malformed` (pandas
  rellena con NaN). Con separador `;` se acepta la coma decimal (D10). Un objetivo de texto
  o booleano siempre da `CsvTargetStats::Classes`, aunque la tarea sea "auto" o Regresión
  (Python fallaba). `pairplot_data` acepta de 1 a 6 características, distintas y existentes.
- **Moons:** `load_builtin(DatasetId::Moons)` es el CSV embebido (semilla 42, D9). Para
  `regenerate_moons`, `moons_dataset(seed)`; `make_moons(500, 0.2, 42)` reproduce el CSV
  bit a bit.

---

## 6. Fixtures de paridad

`uv run python tools/export_golden.py` genera los fixtures y `--check` los verifica (0 cambios, exit 0).

| Ruta | Contenido nuevo | Tolerancia |
|---|---|---|
| `datasets/*.json` | Ahora con `label_columns` y `csv_columns`. **MPG (D8):** `columns`, `shape` (392×7), `feature_names`, `dataset_info` y `data_sha256_f64` sin `name`; el CSV conserva `name` **con su texto original** (p. ej. "chevrolet chevelle malibu") antes de `mpg` | exacta |
| `stats/<slug>.json` (12) | `display_columns`, `column_mapping`, `preview_rows` (10), `describe` (+ excluidas), `target_distribution` (barras ordenadas con ratio y aviso, o histograma de 30 bins con mean/std/min/max), `correlation` (row-major, `null` = NaN, `nan_columns`, `visible_min/max`), `pairplot` (slider, selección por defecto, `hue_order`, muestreo y `sample_rows` en California y Diamantes, estilo, **KDE** por característica × nivel con `scott_factor`, `bandwidth`, `grid_min/max`, `weight` y 200 densidades, e **histograma** de la diagonal) | describe 1e-12 rel; correlación 1e-12 abs; KDE 1e-9 rel en densidad y 1e-12 en el ancho de banda; histogramas y muestreo exactos |
| `csv/<caso>.json` (14) | Entrada (texto y base64) y salida de Python: error de lectura o de validación, shape, dtypes, faltantes, sugerencia, value_counts, describe y `load_dataset_from_file` con auto, Clasificación y Regresión | exacta |

KDE: método `seaborn._statistics.KDE` (el mismo objeto que usa `kdeplot`), con
`bw_method="scott"`, `bw_adjust=1`, `gridsize=200`, `cut=3` y `clip=None`. Ancho de banda =
`n^(-1/5) · std(ddof=1)` por nivel. En regresión, el fixture da la KDE sin hue (propuesta D11) y
lo indica en `input.regression_hue`. En Pingüinos, `input.python_bug` documenta el desalineado.

Antes de embeber California, Diamantes y MPG, se comprueba `csv_sha256` en un test de core.

---

## 7. Contenido extraído

Literal, sin reescribir (conserva el orden y los emojis). Los comentarios `<!-- -->` dicen
de dónde sale cada texto y cómo se presenta.

```
content/inicio/   cabecera.md, bienvenida.md, que-es-ml.md, algoritmos.md, tarjetas.toml,
                  como-usar.md, pie.md, navegacion.toml (barra lateral, título/subtítulo,
                  pestañas y state key de cada lección)
content/datos/    mensajes.toml (todos los textos cortos con {placeholders}), que-significan.md,
                  correlacion.md, pairplot.md, pairplot-consejos.md, tips-redes.md
content/csv/      ayuda.md, ejemplos.toml, mensajes.toml
content/code/     analisis_exploratorio.py.j2 (SCATTERPLOT_MATRIX)
content/{arboles,regresion,knn,redes}/intro.md   expander "ℹ️ ¿Qué es…?" de cada lección
content/datasets/<id>.toml (12)   etiqueta, emoji, tarea, objetivo, descripción, blurb,
                  línea de información, origen, clases, label_columns (MPG), generador (Moons)
                  y una tabla [[features]] por columna: column, display, description, kind
                  (binary/categorical/continuous con la regla del formulario), min, max, mean
                  y values (código → texto) o levels
```

Correcciones en los TOML respecto a `dataset_metadata.py` (anotadas en el propio TOML):

- Propinas `day_encoded`: Python dice 0=Jueves…3=Domingo, pero el LabelEncoder da 0=Fri,
  1=Sat, 2=Sun y 3=Thur. En el TOML queda 0=Viernes, 1=Sábado, 2=Domingo, 3=Jueves (D15).
- Iris: las descripciones estaban con nombres de seaborn (`sepal_length`), pero la app carga
  las columnas de sklearn (`sepal length (cm)`), así que en Python nunca se mostraban. En el
  TOML se han asociado a las columnas reales.

Los textos con placeholders usan la sintaxis de `format` de Python (`{ratio:.1f}`). Para la UI,
basta con un pequeño `fmt(template, vars)` con soporte de `.Nf`.

---

## 8. Reparto de tareas

### 8.1 `rust-ml-engineer` (`crates/mltutor-core`, y los tipos de dominio que piden los DTO)

1. [ ] Dependencias: `ndarray 0.16`, `thiserror`, `serde`, `csv` (crate), `encoding_rs`
   (D10), `toml`, `rand_mt 6.1` e `indexmap`. Sin BLAS.
2. [ ] `dataset::catalog` + `dataset::table`: embeber los 12 CSV de `tests/parity/datasets/data`
   y parsearlos según `columns`/`label_columns`/`target_name` del fixture. Test: para cada
   dataset, `data_sha256_f64`, `csv_sha256`, `shape`, `feature_names`, `class_names`,
   `target_summary` y `head` frente a `datasets/*.json`.
3. [ ] `dataset::metadata`: cargar `content/datasets/*.toml`. Test: los `display` coinciden con
   `feature_names` del fixture, en el mismo orden.
4. [ ] `dataset::moons` según D9. Si se aprueba la paridad exacta: `make_moons` con MT19937 y
   `gauss` legacy de numpy (polar) frente a `datasets/data/moons.csv`.
5. [ ] `sampling`: `permutation`, `sample(n, seed)` y `groupby_sample`. Test: `stats/california_housing.json` y
   `stats/diamonds.json` → `pairplot.sample_rows` exactos. D7: `split` frente a `split/iris.json`.
6. [ ] `stats::{describe, value_counts, histogram, correlation, kde}`. Tests: `stats/*.json`
   con sus tolerancias, incluidos los NaN de Dígitos, los niveles omitidos y la exclusión de
   columnas bool en describe.
7. [ ] Frame de exploración alineado por posición (no reproducir el fallo de Pingüinos) con
   la vista previa, el ratio de balance y los nombres de clase.
8. [ ] `csv`: lector, inferencia, codificación y heurísticas. Tests: los 14 `csv/*.json`
   (error esperado o resultado de `load_dataset_from_file` para los tres `task`). Fuzzing
   básico (`cargo fuzz` o proptest con bytes aleatorios: nunca panic), según el criterio de
   estabilidad del PLAN.
9. [ ] `pairplot` (en `stats` o en un módulo `explore`): construir `PairplotData` en core
   (puntos, hue, diagonales) para que Tauri solo lo convierta. Validación de selección.
10. [ ] Errores con `thiserror`: `DatasetError` y `CsvError`. Sin `unwrap` en producción.
11. [ ] Benchmark rápido (test `#[ignore]` o criterion): California completo < 100 ms.

### 8.2 `tauri-ui-engineer` (`src-tauri` + `ui`)

Backend (`src-tauri`):

1. [ ] `src/ipc/` con los DTO y `CommandError` de §5 (ts-rs → `ui/src/lib/bindings`).
   `src/session.rs` con `Session`.
2. [ ] Los comandos de §5.4: `async` + `spawn_blocking`, declarados en `build.rs` y en
   `capabilities/default.json`. Tests unitarios de la conversión de errores y del estado
   (CSV registrado → seleccionado; mismo nombre → reemplazo).
3. [ ] `tauri-plugin-dialog` (abrir CSV, guardar PNG/código) y `tauri-plugin-fs` limitados a
   lo que eligió el usuario. Hoy no están en `src-tauri/Cargo.toml`, aunque el PLAN de F0 los
   menciona.
4. [ ] CSP: `style-src 'unsafe-inline'` y `dangerousDisableAssetCspModification: ["style-src"]`
   para Plotly (MEMORY). Revisar KaTeX.
5. [ ] `--smoke-test`: además de las 6 rutas, abrir 📊 Datos en una lección y pedir correlación
   y pairplot de Iris.

Frontend (`ui`):

6. [ ] Corregir `routes.ts` con los títulos y subtítulos de `content/inicio/navegacion.toml`.
   Añadir la cabecera común y el pie a todas las páginas.
7. [ ] Carga de contenido: `content/**/*.md?raw` y TOML en build (con `smol-toml` en un plugin
   de Vite). Renderizado Markdown + KaTeX + resaltado de Python, todo local.
8. [ ] Componentes reutilizables (§8.3), con tests de vitest.
9. [ ] Página Inicio (§1.4) y cabecera de lección + pestañas + navbar (§1.5), con el estado por
   lección en un store.
10. [ ] `DatasetTab.svelte` (§2): selector, información, vista previa, describe, distribución,
    análisis avanzado (correlación y pairplot con sus controles y avisos) y código. Mientras
    no existan las lecciones, se monta en la pestaña 0 de las 4 (y en la 1, con solo el
    selector, en árboles, regresión y KNN).
11. [ ] `CsvPage.svelte` (§3), con los estados vacío, previsualización y cargado, y con los
    errores de `CommandError` en `Notice`.
12. [ ] Plotly: barras, histograma, heatmap y pairplot (subplots k×k con `scattergl` si hay
    más de 1000 puntos; diagonal como `scatter` con `fill='tozeroy'` o `bar`), con las paletas
    de §2.6 y la descarga de PNG.
13. [ ] Tema claro fiel a `theme.css` y tema oscuro derivado (PLAN F1).
14. [ ] Capturas Rust de las mismas pantallas que §0 y comparación (Definición de terminado).

### 8.3 Componentes UI que se crean ya

| Componente | Props principales | Equivale a |
|---|---|---|
| `LessonTabs` | `tabs: string[]`, `active: number`, `onchange` | Fila de botones de pestaña |
| `LessonNavbar` | `prev?: string`, `next?: string`, `prevNote?`, `nextNote?`, `disabledPrev` | `apps/navbar.py` (título "🧭 Navegación", 🔙/🔜) |
| `ButtonPanel` | `options: {label, value}[]`, `value`, `onchange` | `create_button_panel` (y el par correlación/pairplot) |
| `Notice` | `severity: info\|warning\|error\|success`, `markdown` | `st.info/warning/error/success` |
| `Metric` | `label`, `value`, `delta?`, `help?` | `st.metric` |
| `Expander` | `label`, `open` (por defecto), `id` (recordar estado) | `st.expander` |
| `DataTable` | `columns`, `rows: Cell[][]`, `index?`, `hideIndex?`, `sortable?` | `st.dataframe` (formato .6g, copiar TSV, leyenda) |
| `CodeBlock` | `code`, `language`, `downloadName?`, `title?` | `st.code` + `show_code_with_download` |
| `PlotlyChart` | `data`, `layout`, `config`, `downloadName?` | `st.pyplot` / `st.plotly_chart` |
| Además | `Markdown`, `Card` (tarjetas de inicio), `Select`, `Radio`, `Slider` (con su número), `MultiSelect` (chips), `FilePicker`, `Spinner/Progress` | — |

---

## 9. Decisiones para el usuario

Se numeran a continuación de D1–D8. Cada una tiene una recomendación que se aplica por defecto si no se dice otra cosa.

| # | Decisión | Recomendación |
|---|---|---|
| D9 | **Moons**: Python lo regenera con la hora como semilla **en cada renderizado** de las pestañas Datos y Entrenamiento, así que los datos cambian con cada clic | Generarlo una vez por sesión con semilla 42 (= fixture) y mantenerlo fijo. Se conserva el aviso "Dataset generado aleatoriamente.". Opcional: un botón "🔄 Regenerar" con semilla nueva (sería contenido nuevo) |
| D10 | **CSV más tolerante** que Python: Windows-1252 si no es UTF-8; separador `;` o tabulador detectado; nombres originales de las clases en vez de "Clase i"; límite de 100 MB | Sí a las cuatro cosas, avisando con un `Notice`. NaN en texto: mantener "nan" como en Python (paridad) |
| D11 | **Pairplot de regresión**: Python usa el objetivo numérico como hue (una KDE o un histograma por valor único: ilegible) | En la dispersión, color continuo (cubehelix, como seaborn) con barra de color; en la diagonal, una sola KDE o histograma sin hue |
| D12 | **Escala de color de la correlación**: Python usa el rango de los datos visibles (en Iris, de -0,43 a 0,96) | Escala fija de -1 a 1 centrada en 0 (`coolwarm`): más correcta para enseñar. Se mantienen la máscara y las anotaciones |
| D13 | **Estado del panel de subvistas**: global y compartido entre lecciones (`viz_type`) | Un estado por lección y por panel |
| D14 | Resaltar "📁 Cargar CSV Personalizado" en la barra lateral cuando está activo (Python no lo hace) | Sí |
| D15 | Corregir el mapa de días de Propinas en los metadatos (hoy etiqueta mal el formulario de predicción) | Sí (ya está en el TOML) |
| D16 | Texto de "💡 Consejos para evitar que se cuelgue la generación": habla de Ctrl+F5 y de tiempos de matplotlib que no aplican en Rust | Conservar el expander, pero ocultarlo o adaptar los tiempos con `ml-lesson-author`. Por ahora se extrae literal |
| D17 | Ubicación de los CSV embebidos | `include_bytes!` desde `tests/parity/datasets/data/` (una sola copia, ya verificada con hash) en lugar de duplicarlos en el crate |

---

## 10. Riesgos

| Riesgo | Mitigación |
|---|---|
| El pairplot en Plotly con 6×6 subplots y 2000 puntos va lento en WebKitGTK | `scattergl` por encima de 1000 puntos; probarlo en el AppImage con render por software (CI de Linux) |
| Inferencia de tipos distinta de la de pandas (enteros con NaN → float64, "True"/"False", espacios, notación científica) | Fixtures `csv/*` y fuzzing; documentar en MEMORY las diferencias aceptadas |
| `histogram_bin_edges("auto")` y los cuantiles de numpy tienen detalles finos (FD con IQR 0, último bin cerrado) | Tests con los fixtures de las 12 diagonales |
| Paridad exacta de `make_moons` (gauss legacy de numpy) | Si no se consigue, embeber `moons.csv` (semilla 42) y generar los demás con semilla propia (D9) |
| CSP más laxa para Plotly | Solo `style-src 'unsafe-inline'`; `script-src` sigue estricta |
| Tamaño del contenido duplicado entre TOML y fixtures (nombres legibles) | El test de metadata compara TOML y fixture |
| Capturas Qt de 9,7 MB en el repo | Comprimir con pngquant o guardarlas como artefacto |

---

## 11. Hallazgos de la versión Python

Detalles no obvios. Se proponen para MEMORY.md (ver el informe):

1. **Pingüinos se ve mal en la exploración:** `X` conserva el índice con huecos de `dropna()`
   y `y` es un `RangeIndex` nuevo. `pd.concat(axis=1)` alinea por índice: 342 filas, 9 NaN y
   etiquetas desplazadas en la vista previa, describe, la distribución, la correlación y el
   pairplot. El entrenamiento no se ve afectado (split posicional).
2. `selected_dataset` es global; el selector aparece también en Entrenamiento (salvo en
   redes); al cargar un CSV, este pasa a ser el dataset de todas las lecciones.
3. Moons se regenera con la hora en cada renderizado del selector.
4. La correlación usa `X.columns` (nombres originales) y no los nombres legibles, oculta
   también la diagonal y el rango de color va del mínimo al máximo visible.
5. El objetivo de clasificación se ordena alfabéticamente por nombre en las barras y por
   aparición en el pairplot.
6. describe excluye las columnas bool (Titanic).
7. El selector del CSV y `load_dataset_from_file` usan reglas de tarea distintas, y "auto" se
   muestra literal en la métrica.
8. `load_dataset_from_file` pierde los nombres de las clases de texto ("Clase 0…").
9. `create_button_panel` comparte `viz_type` entre todas las lecciones.
10. El mapa de días de Propinas en `dataset_metadata.py` es incorrecto, y las descripciones
    de Iris nunca se muestran.
11. Las tablas de Qt no se pueden ordenar, y la descripción del dataset no aparece en la
    pestaña Datos.
12. `seaborn.pairplot` con hue numérico: dispersión continua y diagonal con un nivel por cada valor.
