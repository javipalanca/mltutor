# Comparación visual Qt ↔ Rust — Fase 1 (shell y 📊 Datos)

Fecha: 2026-10-13. Capturas Qt en [`qt/`](qt/) (`tools/capture_qt.py`) y Rust en
[`rust/`](rust/), con los mismos nombres. Las Rust se generan con el binario de release:

```sh
target/release/mltutor --screenshot rust/06_datos_pairplot_kde_iris.png \
    --scenario datos_pairplot_kde_iris --full-page
# CSV: --scenario csv_cargado_clasificacion:tools/data/seaborn/tips.csv
```

Están cuantizadas a 256 colores con Pillow (2,7 MB las 22). Diferencia de formato: las
Qt `NNN.png` muestran solo el contenido (1135 px de ancho) y las Rust la ventana entera
a 1400 px con la barra lateral, porque `--full-page` amplía el `WKWebView` hasta la altura
de la página. Las `_ventana` son en ambos casos la ventana tal y como se ve (Rust: 1400×888
sin la barra de título; Qt: 1400×920 con la barra de estado). Las capturas Rust son a 1×;
las Qt, también.

Los datos los calcula `mltutor-core` (RealFacade). El tema oscuro no se ha capturado
(la captura sigue el aspecto del sistema, que estaba en claro).

## Resumen

| | Coincide | Mejora | Falta o difiere |
|---|---|---|---|
| Shell | Barra lateral, orden, emojis, botón activo azul, cabecera "🎓 MLTutor", pie, barra de estado "Listo" | CSV resaltado al estar activo (D14); pie en una línea; tipografía Inter con mejor antialiasing; tema oscuro derivado | Sin botón "Cancelar" en la barra de estado (no hay tareas largas en F1) |
| Tablas | Cabecera #eaf0f9, filas alternas, índice gris, números a la derecha con `.6g`, leyenda "N filas · M columnas · Selecciona y copia…" | Ordenables por columna en la vista previa; Ctrl/⌘+A | — |
| Gráficos | Colores de Python (#3274a1, C0, tab10, coolwarm, cubehelix), títulos y ejes | Interactivos (zoom, hover con valores), nítidos a cualquier tamaño, fuentes legibles; PNG a 2× con diálogo nativo | El pairplot no muestra las marcas del eje y en la primera fila (seaborn las pone en la KDE) |
| Contenido | Todos los textos, avisos, expanders y su estado inicial, navbar | — | Pestañas ≥ 1 de cada lección con un aviso "🚧 se está portando" (F2–F5) |

## Pantalla por pantalla

**01 Inicio (`01_inicio`, `01_inicio_ventana`).** Coincide la estructura completa:
cabecera, "🌟 Bienvenido", tarjeta azul "¿Qué es el Machine Learning?", las 4 tarjetas con
borde izquierdo de color y altura de 200 px, "Estado: Disponible" en verde solo en Árboles,
los botones "✨ Explorar …" y la "Guía de uso" gris. Mejora: espaciado más compacto (en la
ventana se ven las tarjetas y sus botones sin hacer scroll, en Qt quedan cortados) y la lista
con viñetas reales. Diferencia menor: el subtítulo "Plataforma de aprendizaje…" queda algo
más cerca del título.

**02 Datos Iris (`02_datos_iris`, `_ventana`).** Coinciden cabecera de lección, subtítulo
(ahora el de `navegacion.toml`, como en Python), expander "ℹ️ ¿Qué es la Regresión?", la fila
de 7 pestañas con "📊 Datos" en azul, separador, "Selección y Preparación de Datos", selector,
línea "🌸 Iris: Perfecto para empezar…", el aviso verde, las 4 métricas, vista previa (10
filas con índice), describe y la distribución de clases. Mejora: el gráfico de barras ocupa
la columna central con fuentes legibles y hover; en Qt es una imagen de matplotlib pequeña.

**03 Expandido.** Igual que Qt: intro de la regresión, "¿Qué significan estos datos?" y "Ver
código" abiertos. El bloque de código mantiene el panel oscuro #152238, la etiqueta
"PYTHON", "Copiar código" y los colores del resaltado; abajo "📥 Descargar código" y el aviso
de copia. Mejora: el resaltado es más preciso (highlight.js) y el texto se puede seleccionar.

**04 Correlación Iris.** Coinciden la máscara (triángulo inferior estricto, diagonal oculta),
anotaciones `.2f`, coolwarm y líneas blancas. Mejoras: escala fija de -1 a 1 (D12) con barra
de color, celdas cuadradas grandes y legibles (en Qt la figura 20×16 deja las fuentes
diminutas), hover con 3 decimales. "📥 Descargar matriz de correlación" guarda un PNG.

**05 Opciones del pairplot.** Coinciden radio horizontal "Histograma / KDE" con KDE marcado,
slider + número (2..4), multiselect de chips con "4 de 4 seleccionadas · Máximo 4", botones
"🚀 Generar…" (3/4) y "🛑 Reiniciar" (1/4), el aviso "👆 Haz clic…" y el expander de consejos
abierto. El texto de los consejos se mantiene literal (D16 pendiente con `ml-lesson-author`).

**06 / 07 Pairplot KDE e histograma (Iris).** Coinciden la rejilla 4×4 completa, tab10 por
orden de aparición (setosa azul, versicolor naranja, virginica verde), leyenda "target" a la
derecha, KDE por nivel con relleno (mismos valores que seaborn, verificados con los
fixtures) e histogramas superpuestos con bordes comunes. Mejoras: `scattergl` (fluido al
hacer zoom), hover por punto, ejes compartidos por columna/fila con zoom enlazado. Difiere:
la primera fila no tiene números en el eje y (en seaborn salen sobre la KDE, con la escala
de la fila); se añade "✅ Matriz generada exitosamente en …s", que en Python también existe.

**08 / 09 Diabetes.** Coinciden el histograma de 30 bins (C0, alpha 0,7, borde negro), título
y ejes, y el expander "📊 Estadísticas de la Variable Objetivo" con las 4 métricas `.2f`.
Sin línea bajo el selector, como en Python.

**10 Pairplot de regresión (Diabetes).** Mejora clara (D11): color continuo cubehelix con
barra de color "target" y una sola KDE en la diagonal. En Qt la diagonal dibuja una curva por
cada valor del objetivo y es ilegible.

**11 Correlación de Dígitos (65 columnas).** Coinciden los huecos de las columnas
constantes (0, 32, 39: NaN). Mejora: con más de 20 columnas se omiten las anotaciones (en Qt
son ilegibles) y el hover da el valor; la figura usa todo el ancho.

**12 Titanic.** Coinciden "✅ Dataset bien balanceado" (ratio 1,6), describe sin
`Hombre_Adulto` ni `Solo` (bool) y los nombres legibles. Las tablas anchas tienen scroll
horizontal propio (en Qt también).

**13 Moons.** Coinciden la línea del selector y el aviso "Dataset generado aleatoriamente.".
Difiere a propósito (D9): los datos son fijos por sesión (semilla 42).

**14 Redes con Propinas.** Coinciden el aviso "🎓 Tips para Redes Neuronales" bajo la
cabecera y las pestañas de redes (Datos, Arquitectura…). Los nombres legibles de Propinas
(Cuenta_Total, Tamaño_Grupo…) salen igual.

**15 Pestaña ≥ 2.** Coincide "📊 Dataset actual: …" + separador y la navbar con "🔙 Volver a
Entrenamiento" / "🔜 Continuar a Visualización". Falta el contenido de Evaluación (Fase 2):
aviso "🚧".

**16 CSV vacío.** Coinciden el expander de ayuda abierto, "📊 Selección de Dataset", el
aviso azul, "📁 Cargar Archivo CSV", el botón "Seleccionar archivo CSV…" (diálogo nativo con
filtro CSV), "👆 Por favor, carga…" y los 4 ejemplos con icono grande y separador. El botón
"📁 Cargar CSV Personalizado" sale resaltado (D14).

**17 / 18 / 19 CSV `tips.csv`.** Coinciden "✅ Archivo cargado exitosamente: 244 filas, 7
columnas", el expander de vista previa (5 filas, 3 métricas), "Columna objetivo: size" y
"Tipo de tarea: Clasificación" (sugerida), la información del objetivo con la distribución
`• 2: 156 (63.9%)…` en el mismo orden, "✅ Dataset 'tips.csv' añadido…", el bloque de
resultado con las 4 métricas, describe del CSV crudo, y la tabla "Cantidad" + barras C0. En
19 (objetivo `tip`, Regresión): las 5 estadísticas `.4f` y el histograma de 30 bins.
Mejora: el CSV queda seleccionado en todas las lecciones y aparece en el selector como
"📄 tips.csv"; los errores (vacío, 1 columna, filas mal formadas, codificación) salen en un
aviso rojo en español.

## Pendiente

- Marcas del eje y en la primera fila del pairplot.
- Capturas en modo oscuro y en Windows/Linux (`--screenshot` solo existe en macOS).
- Estados intermedios no capturados (tampoco en Qt): diálogo de archivo, spinner del
  pairplot (dura milisegundos) y errores.
