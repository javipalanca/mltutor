# Plan de migración de MLTutor a Rust

Rama: `feature/rust-migration` · Inicio: 2026-10-13 · Estado: **Fases 0 y 1 completadas (2026-10-14, revisadas por rust-reviewer), Fase 2 completada (2026-10-14, revisada por rust-reviewer y ml-correctness-reviewer), Fase 3 en curso**

## Objetivo

Reescribir MLTutor en Rust (Tauri 2 + Svelte + linfa + burn) cumpliendo tres requisitos:

1. **Mantener las mismas funciones**: todo lo listado en [INVENTARIO.md](INVENTARIO.md),
   con la misma vistosidad o más.
2. **Seguir siendo multiplataforma**: Windows x86_64, macOS Apple Silicon y Linux x86_64.
3. **Seguir publicando en GitHub Releases** las descargas de los tres sistemas.

Y además ganar velocidad, estabilidad y una descarga más ligera.

## Criterios de aceptación globales

| Criterio | Cómo se mide |
|---|---|
| Paridad funcional | Todas las casillas de INVENTARIO.md marcadas |
| Paridad numérica | Tests de `tests/parity` en verde, con tolerancias justificadas por algoritmo |
| Paridad visual | Capturas Rust junto a capturas Qt (`--screenshot`) revisadas por pantalla, sin pérdidas |
| Rendimiento | Arranque < 2 s en frío; recálculos interactivos < 100 ms con los datasets incluidos; entrenar la red por defecto, más rápido que en Python |
| Tamaño | Cada descarga < 60 MB (la versión actual con TensorFlow ronda varios cientos) |
| Estabilidad | Ningún panic llega al usuario; fuzzing básico del cargador de CSV; CI verde en los 3 sistemas |
| Releases | Un tag `v*` publica las 3 descargas y `SHA256SUMS.txt`; un `-rc.N` publica una prerelease |
| Sin red | La app funciona sin conexión (CSP estricta, todo empaquetado) |

## Arquitectura

```
Cargo.toml (workspace) · rust-toolchain.toml
crates/
  mltutor-core/     datasets embebidos, CSV, preprocesado, split, estadísticas,
                    árboles, lineal/logística, KNN, métricas, ROC/PR, importancia,
                    rejillas para fronteras y superficies, generación de código Python
  mltutor-nn/       MLP con burn 0.22 (backend flex, CPU): capas, activaciones, dropout,
                    optimizadores, early stopping, reducción del LR, historial,
                    pesos y activaciones, diagnóstico del entrenamiento
  mltutor-export/   ONNX nativo (protobuf ai.onnx/ai.onnx.ml) y paquetes de exportación
src-tauri/          comandos IPC, estado por lección, tareas con progreso y cancelación,
                    diálogos, registro (log), --smoke-test y --screenshot
ui/                 Svelte 5 + TS + Vite: shell, navegación, pestañas, controles,
                    gráficos (Plotly), Markdown + KaTeX, resaltado de código
content/            textos pedagógicos (.md) y plantillas de código Python (.py.j2)
tests/parity/       fixtures JSON generados desde la versión Python
tools/              export_golden.py, scripts de capturas
docs/migration/     este plan, el inventario y los planes por módulo
```

Flujo de una interacción: un control de la UI cambia, se espera el debounce y se llama
a `invoke("comando", args)`. El comando lanza la tarea en `spawn_blocking`, que usa
`mltutor-core` o `mltutor-nn`, emite eventos de progreso y devuelve datos serializables.
La UI dibuja con Plotly o SVG. Nunca se envían imágenes rasterizadas.

## Decisiones (D1–D4 aprobadas por el usuario el 2026-10-13; D5–D6 aplicadas por defecto)

| # | Decisión | Recomendación |
|---|---|---|
| D1 | **Exportaciones de formato Python** (Pickle, HDF5, SavedModel, TFLite): Rust no puede generarlas sin incluir Python | Exportar **ONNX nativo** para todos los modelos (cargable con `onnxruntime` en Python y convertible a TFLite con `onnx2tf`) y además un **"paquete Python"**: un ZIP con los parámetros en JSON y un script que reconstruye el objeto sklearn/Keras y lo guarda como `.pkl`, `.h5`, SavedModel o `.tflite`. Las mismas opciones de la UI, con un paso más fuera de la app. |
| D2 | Formato de Linux | **AppImage** (`mltutor-linux-x86_64.AppImage`): portable, con WebKitGTK incluido y validado en Ubuntu 22.04 |
| D3 | WebView2 en Windows | Depender del WebView2 *evergreen*, que viene de serie en Windows 10/11. Si falta, la app muestra un aviso con el enlace. No se usa la versión fija (+150 MB). |
| D4 | Versionado | La primera versión Rust será la **0.4.0**, con prereleases `0.4.0-rc.N` desde la rama de migración. La versión vive en `Cargo.toml` del workspace, sincronizada con `tauri.conf.json`. |
| D5 | Código muerto de Python (sección 8 del inventario) | No portarlo |
| D6 | Paridad de árboles y regresión logística | Tolerancia, no igualdad: aceptar diferencias por desempates y solver, siempre documentadas |
| D7 | Split train/test | ✅ Aprobada: **idéntico al de Python**, sin estratificar, con `rand_mt` (MT19937) para reproducir `RandomState(42).permutation`. Mismos datos de train y test que la versión actual. La estratificación queda como mejora futura opcional |
| D18 | `freezePrototype: false` en Tauri (con `true`, Plotly falla con "readonly property") | Aplicado. Riesgo bajo: no se carga contenido remoto y `script-src 'self'` sigue estricto. Revisable |
| D19–D35 | Detalles de la Fase 2 (paridad de la logística frente a la referencia convergida, umbrales 0,3/0,5/0,7, R² real, colores coherentes en la frontera, coeficientes corregidos, exportación con código + ONNX + paquete ZIP, script de "Generar Código" correcto, entre otros) | Recomendaciones aplicadas por defecto (2026-10-14). Detalle en [fase2-regresion.md §10](fase2-regresion.md) |
| D36–D51 | Detalles de la Fase 3 (CART bit a bit, preselección de la tarea, importancia nativa + permutación, nombres de clase en el texto, diagrama SVG con zoom, PNG/SVG, código Python del árbol corregido, umbral ONNX como el mayor float32 ≤ t, entre otros) | Recomendaciones aplicadas por defecto (2026-10-14). Detalle en [fase3-arboles.md §10](fase3-arboles.md) |
| D9–D17 | Detalles de la Fase 1 (Moons fijo por sesión, CSV más tolerante, pairplot de regresión legible, correlación de -1 a 1, estado de subvistas por lección, CSV resaltado en la barra lateral, días de Propinas corregidos, consejos del pairplot, CSV embebidos desde `tests/parity`) | Recomendaciones aplicadas por defecto (2026-10-13). Detalle en [fase1-shell-datos.md §9](fase1-shell-datos.md). Revisables por el usuario |
| D8 | MPG: columna `name` | ✅ Aprobada: **no se usa como característica** del modelo (301 modelos de coche codificados como número no tienen sentido). Se conserva en la exploración de datos: vista previa, tooltip en el pairplot y gráficos de dispersión |

## Fases

Cada fase termina con CI verde en los 3 sistemas. **No se publican releases intermedias**
(decisión del usuario, 2026-10-13, para no generar versiones que sean ruido). Para probar
en equipos reales se usan los artefactos de cada ejecución del CI (Actions → artifacts,
14 días). La primera release (`v0.4.0-rc.1`) se publica cuando la funcionalidad esté
completa o casi completa, previsiblemente tras la F6. El orden reduce primero el riesgo
(publicación y arquitectura) y empieza por la lección más sencilla, que sirve de piloto.

### Fase 0 — Cimientos y canal de publicación  *(riesgo: alto · tamaño: M)*

Objetivo: una app Tauri vacía que se compila, se prueba y **se publica en las tres
plataformas** antes de escribir lógica.

- [x] `rust-toolchain.toml` (stable, rustfmt, clippy) y el workspace con los crates vacíos.
- [x] Spike de dependencias (resultados en MEMORY.md): versiones de `linfa-*`, `burn` y `tauri`, y qué cubre
      cada una (¿linfa-trees hace regresión? ¿linfa-logistic hace multinomial?). Se
      anota en MEMORY.md.
- [x] Esqueleto de Tauri 2 + Svelte 5 + Vite + TS, con CSP estricta, sin red,
      `tauri-plugin-dialog`/`fs` y el icono actual (`assets/`).
- [x] Flags del binario (la captura de Windows y Linux, pendiente): `--smoke-test`, que navega por todas las rutas y sale con 0, y
      `--screenshot <png>`. Además, registro en `~/.mltutor/mltutor.log` y hook de panic.
- [x] (verde en los 3 sistemas desde la ejecución 38430755760) Workflow `.github/workflows/rust.yml`: fmt, clippy, test, npm check/test, build
      Tauri, smoke test (xvfb en Linux) y empaquetado
      `mltutor-windows-x86_64.exe` / `mltutor-macos-arm64.zip` /
      `mltutor-linux-x86_64.AppImage`. Release con tags y `SHA256SUMS.txt`, con la
      misma lógica de prerelease que el workflow actual.
- [x] `release.sh` adaptado para leer y escribir la versión en Cargo.toml y
      tauri.conf.json, conservando las reglas de main y de `-rc`.
- [ ] D3: aviso con enlace en Windows si falta el WebView2 Runtime (`tauri-ui-engineer`).
- [x] `tools/export_golden.py`, con la infraestructura y un primer fixture (Iris). También están los 12 datasets, el split y los fixtures de lineal y logística.
- [x] **Hito:** CI en verde en los 3 sistemas, con artefactos descargables que se abren y
      pasan el smoke test. Sin release.

Agentes: `packaging-release-engineer` (CI y releases), `tauri-ui-engineer` (esqueleto),
`rust-ml-engineer` (spike de crates), `migration-porter` (golden tooling),
`rust-reviewer` (revisión).

### Fase 1 — Shell de la app y datos  *(riesgo: medio · tamaño: L)*

Plan detallado, contrato IPC y reparto: [fase1-shell-datos.md](fase1-shell-datos.md).
Capturas de referencia: `docs/migration/capturas/qt/`.

- [x] Shell: barra lateral, página de inicio y pie, sistema de pestañas por lección con
      navbar, panel de subvistas, avisos, métricas, expanders, tablas y bloques de
      código. Tema claro/oscuro fiel a los colores actuales.
- [x] Renderizado de `content/*.md` con Markdown, KaTeX y resaltado de Python.
- [x] `mltutor-core::dataset`: los 11 datasets embebidos más Moons, con los ficheros de
      California y Diamantes incluidos. Cargador de CSV con inferencia de tipos,
      detección de la tarea y codificación de etiquetas. Split sin estratificar (D7) y
      estadísticas descriptivas, correlación y KDE (fixtures en `tests/parity/stats`).
- [x] Pestaña 📊 Datos común y página 📁 Cargar CSV Personalizado (inventario §1–2),
      con heatmap de correlación y pairplot en Plotly.
- [x] Metadatos de datasets portados como datos (`content/datasets/*.toml`).
- [x] D8: ajustar `tools/export_golden.py` para que en MPG `name` sea una columna de
      etiqueta y no una característica (el fixture de datos sigue incluyéndola en el CSV).
- [x] D7: split con `rand_mt`, verificado exactamente contra `tests/parity/split/*.json`.
- [x] Fixtures de paridad: splits, estadísticas y codificaciones (todo exacto o ≤ 4e-13).

Agentes: `migration-porter`, luego `rust-ml-engineer` y `tauri-ui-engineer` en
paralelo, y `ml-lesson-author` (contenido).

### Fase 2 — Lección piloto: 📊 Regresión  *(riesgo: medio · tamaño: L)*

Plan detallado, contrato y reparto: [fase2-regresion.md](fase2-regresion.md). Capturas: `docs/migration/capturas/qt/fase2/`.

Primero porque linfa la cubre bien. Establece los **componentes compartidos** que
reutilizarán el resto de lecciones.

- [x] Lineal (`linfa-linear`) y logística (solver Newton/L-BFGS propio; ver MEMORY) binaria y multiclase (`linfa-logistic`), con
      paridad frente a sklearn (coeficientes, predicciones y probabilidades).
- [x] Componentes compartidos: métricas de clasificación y regresión, matriz de
      confusión, informe por clase, ROC/PR (macro y micro; weighted solo en core, D20), residuos,
      frontera de decisión y superficie 2D, formulario de predicción e importancia.
- [x] Coeficientes, predicciones y exportación (ONNX + paquete Python, según D1),
      junto con el código Python generado.
- [x] Inventario §4 completo, con la captura comparativa revisada.
- [x] Script de entrenamiento con CSV propio: hoy pasa `csv_sep`, `csv_decimal` y `csv_encoding` a `None`.
- [x] `content/code/paquete_leeme.txt.j2` numera las clases con `loop.index0`; con un CSV de clases numéricas (p. ej. 3 y 6) hay que mostrar los valores reales (`class_values`).

#### Pendientes de la verificación de F2 (se hacen con la F3)

- [ ] (parcial: con R bien condicionada ya se usa sustitución hacia atrás; 3000×1500 pasa de 14,4 s a 1,4 s; queda el caso de rango deficiente: 1500×1500 tarda 12,7 s sin poder cancelar) OLS: la SVD de R (linfa-linalg) cuesta O(p³) y no se puede cancelar (1500 columnas: 11 s).
      Resolver por sustitución hacia atrás cuando R esté bien condicionada y dejar la SVD solo
      para el rango deficiente, o limitar p con un error claro.
- [x] OLS: con valores de ±1e308 devuelve `Ok` con el intercepto NaN. Comprobar la finitud de los
      offsets, de R y c y del intercepto, y devolver `values_too_large()`.
- [x] `last_seq` por lección: tras recargar con el reloj atrasado, una lección puede quedar sin
      entrenar. Exponer el máximo de TODAS las seq aceptadas en `get_selected_dataset`.
- [x] Newton: `Objective::hessian` reserva matrices n×(p+1) en cada iteración; reutilizar los búferes.

### Fase 3 — 🌲 Árboles de Decisión  *(riesgo: alto · tamaño: L)*

Plan detallado, contrato y reparto: [fase3-arboles.md](fase3-arboles.md). Capturas: `docs/migration/capturas/qt/fase3/`.
Incluye la permutation importance (C10) y el paquete Python ZIP.

- [ ] CART propio con **paridad exacta** con sklearn (D36, especificación ejecutable en `tools/cart_reference.py`), clasificación y regresión, con
      los criterios del inventario, paridad de estructura y predicciones frente a sklearn.
- [ ] Diagrama del árbol en SVG interactivo (zoom, nodos coloreados por clase o valor,
      ancho y alto) y reglas en texto.
- [ ] Camino de decisión en las predicciones, importancias (Gini y permutación) y
      exportaciones (ONNX TreeEnsemble, código Python del árbol, reglas .txt e imagen).
- [ ] Inventario §3 completo.

### Fase 4 — 🔍 KNN  *(riesgo: medio · tamaño: M)*

- [ ] KNN de clasificación y regresión sobre `linfa-nn` (kd-tree o fuerza bruta), con
      pesos uniformes o por distancia y métricas minkowski, euclídea y manhattan.
- [ ] Visualización interactiva con el punto de prueba editable y la animación de
      vecinos en la web (sin HTML incrustado), análisis de distancias y separabilidad.
- [ ] Inventario §5 completo.

### Fase 5 — 🧠 Redes Neuronales  *(riesgo: alto · tamaño: XL)*

- [ ] `mltutor-nn` con burn: MLP configurable, inicialización Glorot como en Keras,
      dropout, adam/sgd/rmsprop, batch, validación, early stopping y ReduceLROnPlateau
      propios, progreso por época y cancelación por batch.
- [ ] Paridad estadística, no exacta, con Keras: métricas finales dentro de una banda
      sobre varias semillas.
- [ ] Diagrama interactivo de la arquitectura con el nº de parámetros, historial, pesos,
      superficie, activaciones y neuronas muertas, y exploración.
- [ ] Diagnóstico del entrenamiento (estado, tendencias, patrones, alertas y
      recomendaciones), portado como lógica pura y probado con historiales de fixture.
- [ ] Predicción individual y por lotes. Exportación ONNX de la red (Gemm + activaciones)
      y paquete Keras (D1), más el código Keras generado.
- [ ] Inventario §6 completo.

### Fase 6 — Exportación y código: cierre  *(riesgo: medio · tamaño: M)*

- [ ] Validación cruzada de exportaciones en CI: un job de Python carga cada ONNX con
      `onnxruntime` y ejecuta cada paquete Python, y comprueba que las predicciones
      coinciden con las de Rust. Sustituye a `--verify-models`.
- [ ] Cada plantilla de `content/code/*.py.j2` se ejecuta en CI con sklearn/Keras.

### Fase 7 — Pulido y validación  *(riesgo: bajo · tamaño: M)*

- [ ] Revisión visual pantalla por pantalla frente a la versión Qt, accesibilidad
      (teclado, contraste), HiDPI y tamaños de ventana.
- [ ] Benchmarks (`criterion`) y presupuesto de tamaño del binario.
- [ ] Validación en equipos limpios con cuentas sin privilegios en los 3 sistemas, y
      en equipos de laboratorio reales.
- [ ] Revisión completa de `ml-correctness-reviewer` y `rust-reviewer`.
- [ ] **Firma de Windows con SignPath Foundation** (gratis para OSS, aprobado por el usuario
      el 2026-10-13), para que SmartScreen no bloquee el `.exe`. Requisitos: repositorio
      público (✅), **licencia OSI** (❌: hoy el repo no tiene LICENSE, y la debe elegir el
      autor), solicitud en signpath.org, y builds reproducibles desde GitHub Actions. En CI:
      firmar `mltutor.exe` antes de empaquetarlo y comprobar la firma con `signtool verify`.
      Pedirlo con antelación, porque la aprobación lleva tiempo.

### Fase 8 — Cambio de versión  *(riesgo: medio · tamaño: S)*

- [ ] README y BUILD_EXECUTABLES.md reescritos para Rust.
- [ ] Retirada del código Python del árbol principal. Se conserva en un tag
      `python-final` y en `tools/` lo necesario para generar los fixtures.
- [ ] Se sustituye `build-executables.yml` por `rust.yml` y se publica la `v0.4.0` estable
      (desde `main`, tras el merge, con confirmación del usuario).

## Paralelización con subagentes

```
F0 ──► F1 ──► F2 (piloto: fija los patrones) ──► F3 ─┐
                                               ├► F4 ─┼► F6 ─► F7 ─► F8
                                               └► F5 ─┘
```

Tras el piloto, Árboles, KNN y Redes pueden avanzar en paralelo, en worktrees
separados, porque solo comparten componentes ya estables. Patrón por lección:

1. `migration-porter` escribe el plan del módulo y los fixtures.
2. `rust-ml-engineer` hace el core y la paridad, en paralelo con `tauri-ui-engineer`
   y `ml-lesson-author`, contra un contrato de tipos IPC acordado en el plan del módulo.
3. Pasan `test-runner`, `rust-reviewer` y `ml-correctness-reviewer`.
4. Se marca en INVENTARIO.md, se hace commit y el CI queda en verde (sin release).

## Riesgos principales

| Riesgo | Mitigación |
|---|---|
| Huecos o inmadurez de linfa | Spike en F0; implementación propia y legible en core |
| Diferencias numéricas con sklearn o Keras | Fixtures con tolerancias; diferencias explicadas en el texto si el estudiante las ve |
| WebView en Linux (WebKitGTK) con rendimiento o render distinto | AppImage validada en Ubuntu 22.04 desde F0; Plotly con WebGL solo donde haga falta |
| WebView2 ausente en algún Windows de laboratorio | Aviso con enlace (D3); revisarlo en F7 |
| Gatekeeper o SmartScreen sin firma | Igual que ahora; la firma queda pendiente de credenciales |
| Alcance de la lección de redes (XL) | Va al final, cuando los componentes ya son estables; se puede dividir en subfases |
| Pérdida de formatos de exportación | D1: ONNX más paquete Python, con validación en CI |

## Preparación del entorno local (antes de la Fase 0)

- ✅ Resuelto en el repo: el toolchain por defecto de rustup era
  `stable-x86_64-apple-darwin` y fallaba en este Mac arm64 ("Bad CPU type").
  `rust-toolchain.toml` fija `stable`, que se resuelve a aarch64 (cargo 1.99). Fuera del
  repo sigue fallando hasta ejecutar `rustup default stable-aarch64-apple-darwin`.
- `gh` (GitHub CLI) no está instalado; hace falta para `release.sh`: `brew install gh`.
- Node 24 disponible. Linux necesita `libwebkit2gtk-4.1-dev` y dependencias (en CI).
