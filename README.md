# 🎓 MLTutor

Aplicación de escritorio para **aprender Machine Learning** experimentando: eliges
un conjunto de datos, ajustas los parámetros, entrenas el modelo y ves qué ocurre
por dentro, con explicaciones en español, gráficos interactivos y el código Python
equivalente para repetirlo fuera de la app.

Funciona **sin conexión, sin instalador y sin permisos de administrador**: se
descarga, se abre y listo. Pensada para estudiantes y para usarla en clase o en
laboratorios.

> **Versión 1.0.0**, escrita en Rust (Tauri). La versión Python anterior sigue
> disponible (ver [Versiones anteriores](#versiones-anteriores)).

## 📚 Lecciones

Árboles, Regresión y KNN se organizan en las mismas pestañas: 📊 Datos,
🏋️ Entrenamiento, 📈 Evaluación, 📉 Visualización, 🔍 Características (o
Coeficientes), 🔮 Predicciones y 💾 Exportar. Redes neuronales añade
🏗️ Arquitectura antes de ⚙️ Entrenamiento.

- 🌲 **Árboles de decisión:** clasificación y regresión, criterios (gini, entropía,
  error cuadrático…), diagrama del árbol con zoom, reglas en texto, importancia de
  las características (también por permutación), fronteras de decisión y el camino
  que sigue cada predicción.
- 📊 **Regresión:** lineal y logística, coeficientes y su interpretación, residuos,
  probabilidades, umbrales, curvas ROC y precisión-recall, y matriz de confusión.
- 🔍 **K-Nearest Neighbors:** clasificación y regresión, vecinos de cada
  predicción, métricas de distancia, efecto de K, análisis de distancias entre
  clases y un lienzo interactivo donde se arrastran puntos y se ve cómo cambia la
  predicción.
- 🧠 **Redes neuronales:** 🏗️ arquitectura con diagrama interactivo, activaciones,
  optimizadores, dropout, early stopping, ⚙️ entrenamiento en directo con la curva
  de pérdida (y botón para cancelar), diagnóstico del entrenamiento, pesos,
  superficies de decisión, neuronas muertas, exploración de cómo responde la red a
  cada característica y predicción individual o por lotes desde un CSV.
- 📁 **Datos:** 12 conjuntos incluidos (🌸 Iris, 🍷 Vino, 🔬 Cáncer, 🩺 Diabetes,
  🏠 Viviendas California, 🔢 Dígitos, 🚢 Titanic, 💰 Propinas, 🐧 Pingüinos,
  💎 Diamantes, ⛽ MPG y 🌙 Moons), carga de tus propios CSV, estadísticas,
  correlación y matriz de dispersión (pairplot).
- 💾 **Exportar:** código Python (scikit-learn o Keras) que repite el entrenamiento,
  el modelo en **ONNX** y un **paquete Python** que lo reconstruye en
  scikit-learn o Keras (ver [Qué cambia](#-qué-cambia-respecto-a-la-versión-python)).

## ⬇️ Descargar y abrir

Las descargas están en la página de
[Releases](https://github.com/javipalanca/mltutor/releases). Elige la de tu
sistema:

| Sistema | Archivo | Tamaño aproximado |
|---|---|---:|
| Windows 10/11 (x86_64) | `mltutor-windows-x86_64.exe` | 22 MB |
| macOS con Apple Silicon (M1 o posterior) | `mltutor-macos-arm64.zip` | 9 MB |
| Linux x86_64 (Ubuntu 22.04 o compatible) | `mltutor-linux-x86_64.AppImage` | 86 MB |

Junto a ellas está `SHA256SUMS.txt`, con las sumas de comprobación. Las releases
`v0.3.x` y anteriores son de la versión Python; la versión Rust empieza en la
`v1.0.0`.

Todavía no hay firma de desarrollador en Windows ni en macOS, así que la primera
vez el sistema avisa de que no conoce la aplicación. Es normal; abajo se explica
cómo abrirla.

### 🪟 Windows

1. Descarga `mltutor-windows-x86_64.exe` y ábrelo con doble clic. No se instala:
   puedes guardarlo en cualquier carpeta (Escritorio, Documentos, un USB…).
2. Si aparece **«Windows protegió su PC»** (SmartScreen), pulsa **Más información**
   y después **Ejecutar de todas formas**. Solo hace falta la primera vez.

MLTutor usa **Microsoft Edge WebView2** para dibujar su ventana, que ya viene de
serie con Windows 10 y Windows 11. Si en algún equipo faltara, la app no se cierra
sin más: muestra un aviso con el enlace de descarga. Para solucionarlo, **sin ser
administrador**:

1. Abre la [página de WebView2 de Microsoft](https://developer.microsoft.com/microsoft-edge/webview2/).
2. Descarga el instalador **«Evergreen Bootstrapper»** y ejecútalo. Sin permisos de
   administrador se instala solo para tu usuario.
3. Vuelve a abrir MLTutor.

Si en el equipo no se puede instalar nada (algunos laboratorios), descarga en esa
misma página la versión **«Fixed Version» (x64)**, descomprímela en una carpeta
llamada `WebView2` junto a `mltutor.exe` y vuelve a abrir MLTutor: la usará
automáticamente.

### 🍎 macOS (Apple Silicon)

1. Descarga `mltutor-macos-arm64.zip` y descomprímelo (Safari suele hacerlo solo).
2. Abre `MLTutor.app` con doble clic. No hace falta moverla a Aplicaciones.
3. La primera vez macOS dirá que **no puede verificar** la aplicación. Pulsa
   **OK** (o **Listo**), ve a **Ajustes del Sistema → Privacidad y seguridad**,
   baja hasta el mensaje sobre MLTutor y pulsa **Abrir igualmente**. Confirma y se
   abrirá; las siguientes veces ya no pregunta.

Alternativa desde Terminal, en la carpeta donde está la app:

```sh
xattr -dr com.apple.quarantine MLTutor.app
```

La app lleva una firma *ad-hoc* (sin Developer ID ni notarización de Apple). Por
eso aparece ese aviso y no el de «está dañada». No hay versión para Mac con
procesador Intel.

### 🐧 Linux

1. Descarga `mltutor-linux-x86_64.AppImage`.
2. Dale permiso de ejecución, desde las propiedades del archivo o con:

   ```sh
   chmod +x mltutor-linux-x86_64.AppImage
   ./mltutor-linux-x86_64.AppImage
   ```

3. Si no arranca y menciona **FUSE** o `libfuse.so.2` (Ubuntu 22.04 no trae
   `libfuse2` de serie), ábrelo así, sin instalar nada:

   ```sh
   ./mltutor-linux-x86_64.AppImage --appimage-extract-and-run
   ```

El AppImage incluye su propio motor web (WebKitGTK), por eso es la descarga más
grande. Se valida en Ubuntu 22.04; las distribuciones más recientes suelen
funcionar, pero no es un binario universal para cualquier Linux.

Si la ventana sale en blanco (pasa en algunas máquinas virtuales y tarjetas
gráficas), prueba a abrirlo con
`WEBKIT_DISABLE_DMABUF_RENDERER=1 ./mltutor-linux-x86_64.AppImage`.

## 💻 Requisitos

| Sistema | Requisitos |
|---|---|
| Windows | Windows 10 u 11 de 64 bits (x86_64) con WebView2 (incluido de serie) |
| macOS | macOS 11 Big Sur o posterior, con Apple Silicon |
| Linux | x86_64, Ubuntu 22.04 o una distribución compatible |

No hace falta instalar Python, TensorFlow ni nada más, ni tener conexión a
Internet: los datos, los gráficos y las fórmulas van dentro de la app. Basta con
una cuenta de usuario normal, aunque las políticas de cada centro (por ejemplo,
las que bloquean ejecutables descargados) siguen aplicándose.

### Si algo falla

MLTutor guarda un registro en **`~/.mltutor/mltutor.log`** (en Windows,
`C:\Users\<tu usuario>\.mltutor\mltutor.log`). Cuando crece mucho, el anterior se
guarda como `mltutor.log.1`. Si informas de un problema, adjunta ese archivo e
indica tu sistema y qué estabas haciendo.

## 🔄 Qué cambia respecto a la versión Python

Las lecciones, las pestañas, las explicaciones, los colores y los emojis son los
mismos, y los resultados coinciden con los de scikit-learn y Keras (se comprueba
automáticamente). Lo que cambia:

- **Descargas mucho más pequeñas.** La versión Python incluía Python, Qt y
  TensorFlow:

  | Sistema | Python/Qt (`v0.3.0-rc.4`) | Rust |
  |---|---:|---:|
  | Windows | 676 MB | 22 MB |
  | macOS (zip) | 477 MB | 8,7 MB |
  | Linux | 881 MB | 86 MB |

  Tamaños de las descargas publicadas de la `v0.3.0-rc.4` y de las compilaciones
  de prueba de la versión Rust del 2026-10-20.

- **Más rápida.** Arranca en unos 0,6 s (Mac con Apple Silicon) y los cálculos
  interactivos tardan milisegundos. Algunos ejemplos medidos en el mismo Mac:

  | Tarea | Python | Rust |
  |---|---:|---:|
  | Red neuronal por defecto, Viviendas California, 100 épocas | unos 30 s (en la app) | 0,66 s |
  | Red 10×10, Iris, 100 épocas | 2,4 s (Keras, solo `fit`) | 7,5 ms |
  | Regresión logística, Cáncer | 484 ms (sin converger) | 0,93 ms |
  | Árbol de profundidad 10, Viviendas California | 58 ms | 51 ms |
  | KNN: predecir el conjunto de prueba de Viviendas California | 9,9 ms | 1,6 ms |

  Más detalles en [docs/migration/benchmarks.md](docs/migration/benchmarks.md).

- **Exportaciones en ONNX y paquetes Python.** La app ya no lleva Python dentro,
  así que no puede guardar directamente archivos Pickle (`.pkl`), Keras (`.keras`,
  `.h5`), SavedModel ni TensorFlow Lite. En su lugar ofrece:
  - **💾 ONNX:** el modelo entrenado en un formato abierto que se usa desde Python
    (`onnxruntime`), C#, Java o JavaScript sin depender de MLTutor.
  - **📦 Paquete Python (.zip):** los parámetros aprendidos y un script que
    reconstruye **el mismo modelo** en scikit-learn o Keras, comprueba que predice
    lo mismo que MLTutor y lo guarda como `.pkl` (o, en las redes, como `.keras`,
    `.h5`, SavedModel y `.tflite`). Es un paso más, pero fuera de la app.
  - **📄 Código Python:** el script que repite el entrenamiento con los mismos
    datos y el mismo reparto entrenamiento/prueba.
- **Sin conexión de verdad.** Viviendas California y Diamantes ya no se descargan
  de Internet: vienen incluidos.
- **Más estable.** Los entrenamientos largos se pueden cancelar y un error en un
  cálculo no cierra la app.

## Versiones anteriores

- **Python/Qt** (PySide6 + scikit-learn + TensorFlow), releases `v0.3.0-rc.N`:
  rama [`codex/desktop-migration-study`](https://github.com/javipalanca/mltutor/tree/codex/desktop-migration-study).
- **Streamlit** (versión web original y su despliegue con Docker), releases
  `v0.2.x`: rama [`main`](https://github.com/javipalanca/mltutor/tree/main), hasta
  que se publique la versión Rust estable.

Cuando la versión Rust pase a `main`, la última versión Python quedará guardada en
el tag `python-final`.

## 🛠️ Desarrollo

Cómo compilar, probar y publicar la app: [BUILD_EXECUTABLES.md](BUILD_EXECUTABLES.md).
Normas para contribuir y para agentes de código: [AGENTS.md](AGENTS.md).

## 📄 Licencia

MLTutor se distribuye bajo la [licencia MIT](LICENSE): puedes usarlo, copiarlo,
modificarlo y redistribuirlo libremente, también en clase, siempre que conserves el aviso
de copyright.

---

Desarrollado por Javi Palanca · Departament de Sistemes Informàtics i Computació ·
Universitat Politècnica de València
