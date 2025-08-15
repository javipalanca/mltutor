import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset.dataset_manager import load_data
from utils import create_info_box, get_image_download_link, show_code_with_download
from algorithms.code_examples import SCATTERPLOT_MATRIX


def run_dataset_tab(active_tab, tips=None):
    # SELECTOR UNIFICADO DE DATASET (solo mostrar en pestañas que lo necesiten)
    if active_tab in [0, 1]:  # Exploración y Entrenamiento
        st.header("📊 Selección y Preparación de Datos")
        if tips:
            # Tips educativos sobre datos
            st.info(tips)
        run_select_dataset()
        if active_tab == 0:
            run_explore_dataset_tab()
    else:
        # Para otras pestañas, mostrar qué dataset está seleccionado actualmente
        if hasattr(st.session_state, 'selected_dataset'):
            st.info(
                f"📊 **Dataset actual:** {st.session_state.selected_dataset}")
            st.markdown("---")


def run_select_dataset():
    st.markdown("### 🎯 Selección de Dataset")

    # Inicializar dataset seleccionado si no existe
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = "🌸 Iris - Clasificación de flores"

    # Lista base de datasets predefinidos
    builtin_datasets = [
        "🌸 Iris - Clasificación de flores",
        "🍷 Vino - Clasificación de vinos",
        "🔬 Cáncer - Diagnóstico binario",
        "🚢 Titanic - Supervivencia",
        "💰 Propinas - Predicción de propinas",
        "🏠 Viviendas California - Precios",
        "🐧 Pingüinos - Clasificación de especies"
    ]

    # Añadir datasets CSV cargados si existen
    available_datasets = builtin_datasets.copy()
    if 'csv_datasets' in st.session_state:
        available_datasets.extend(st.session_state.csv_datasets.keys())

    # Asegurar que el dataset seleccionado esté en la lista disponible
    if st.session_state.selected_dataset not in available_datasets:
        st.session_state.selected_dataset = builtin_datasets[0]

    # Selector unificado
    dataset_option = st.selectbox(
        "Dataset:",
        available_datasets,
        index=available_datasets.index(st.session_state.selected_dataset),
        key="unified_dataset_selector",
        help="El dataset seleccionado se mantendrá entre las pestañas de Exploración y Entrenamiento"
    )

    # Explicación sobre el dataset seleccionado
    if "Iris" in dataset_option:
        st.markdown(
            "🌸 **Iris**: Perfecto para empezar. 3 clases de flores, 4 características simples.")
    elif "Vino" in dataset_option:
        st.markdown(
            "🍷 **Vino**: Clasificación multiclase con 13 características químicas.")
    elif "Cáncer" in dataset_option:
        st.markdown(
            "🔬 **Cáncer**: Problema binario médico con 30 características.")
    elif "Titanic" in dataset_option:
        st.markdown(
            "🚢 **Titanic**: Predicción de supervivencia con datos categóricos y numéricos.")
    elif "Propinas" in dataset_option:
        st.markdown(
            "💰 **Propinas**: Regresión para predecir cantidad de propina.")
    elif "Viviendas" in dataset_option:
        st.markdown(
            "🏠 **Viviendas**: Regresión para predecir precios de casas.")
    elif "Pingüinos" in dataset_option:
        st.markdown(
            "🐧 **Pingüinos**: Clasificación de especies con datos biológicos.")

    # Actualizar la variable de sesión
    st.session_state.selected_dataset = dataset_option

    # Separador después del selector
    st.markdown("---")


def show_dataset_info():
    # Mostrar información del dataset seleccionado
    if 'selected_dataset' in st.session_state:
        dataset_name = st.session_state.selected_dataset
        st.success(f"✅ Dataset seleccionado: **{dataset_name}**")

        # Cargar y mostrar datos
        try:
            # Cargar datos usando la función load_data común
            X, y, feature_names, class_names, dataset_info, task_type = load_data(
                dataset_name)

            # Crear DataFrame
            df = pd.DataFrame(X, columns=feature_names)

            # Determinar el nombre de la columna objetivo
            if class_names is not None and len(class_names) > 0:
                # Para clasificación, usar el nombre de la variable objetivo del dataset_info
                target_col = 'target'  # Nombre por defecto
                if hasattr(dataset_info, 'target_names'):
                    target_col = 'target'
                df[target_col] = y
            else:
                # Para regresión
                target_col = 'target'
                df[target_col] = y

            # Almacenar información del dataset
            st.session_state.df = df
            st.session_state.target_col = target_col
            st.session_state.feature_names = feature_names
            st.session_state.class_names = class_names
            st.session_state.task_type = task_type
            st.session_state.dataset_info = dataset_info

            # Mostrar información básica con explicaciones
            st.markdown("### 📊 Información del Dataset")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "📏 Filas", df.shape[0], help="Número total de ejemplos para entrenar")
            with col2:
                st.metric(
                    "📊 Columnas", df.shape[1], help="Características + variable objetivo")
            with col3:
                st.metric("🎯 Variable Objetivo", target_col,
                          help="Lo que la red va a predecir")
            with col4:
                task_icon = "🏷️" if task_type == "Clasificación" else "📈"
                st.metric(f"{task_icon} Tipo de Tarea", task_type,
                          help="Clasificación o Regresión")
        except Exception as e:
            st.error(f"Error cargando dataset: {str(e)}")
            st.info("Por favor, selecciona un dataset válido.")
    else:
        st.warning("⚠️ Por favor, selecciona un dataset para continuar.")


def run_explore_dataset_tab():

    show_dataset_info()
    # Pestaña de Datos
    st.header("Exploración de Datos")

    try:
        # Cargar datos para exploración
        X, y, feature_names, class_names, dataset_info, task_type = load_data(
            st.session_state.selected_dataset)

        # Convertir a DataFrames para facilitar el manejo
        # FIXED: No sobrescribir las columnas si X ya es DataFrame para evitar NaN
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()  # Usar las columnas originales
            # Crear mapeo de nombres para mostrar
            column_mapping = {}
            if len(feature_names) == len(X_df.columns):
                column_mapping = dict(zip(X_df.columns, feature_names))
        else:
            X_df = pd.DataFrame(X, columns=feature_names)
            column_mapping = {}

        # Determinar el nombre de la columna objetivo
        if class_names is not None and len(class_names) > 0:
            # Para clasificación, usar el nombre de la variable objetivo del dataset_info
            target_col = 'target'  # Nombre por defecto
            if hasattr(dataset_info, 'target_names'):
                target_col = dataset_info.target_names[0]
        else:
            # Para regresión
            target_col = 'target'

        y_df = pd.Series(y, name=target_col)

        # Mapear nombres de clases si están disponibles
        if task_type == "Clasificación" and class_names is not None:
            y_df = y_df.map(
                {i: name for i, name in enumerate(class_names)})

        df = pd.concat([X_df, y_df], axis=1)

        # Renombrar columnas para mostrar nombres amigables si existe el mapeo
        if column_mapping:
            df_display = df.rename(columns=column_mapping)
        else:
            df_display = df

        # Mostrar información del dataset
        # st.markdown("### Información del Dataset")
        # st.markdown(create_info_box(dataset_info), unsafe_allow_html=True)

        # Mostrar las primeras filas de los datos
        st.markdown("### 👀 Vista previa de datos")
        st.dataframe(df_display.head(10))

        # Tip sobre los datos
        with st.expander("💡 ¿Qué significan estos datos?"):
            st.markdown(f"""
            - **Filas**: Cada fila es un ejemplo que el modelo usará para aprender
            - **Columnas de características**: Las variables que el modelo analiza para hacer predicciones
            - **Variable objetivo ({target_col})**: Lo que queremos predecir
            """)

        # Estadísticas descriptivas
        st.markdown("### Estadísticas Descriptivas")
        st.dataframe(df_display.describe())

        # Distribución de clases o valores objetivo
        st.markdown("### 🎯 Distribución del Objetivo")

        fig, ax = plt.subplots(figsize=(10, 6))

        if task_type == "Clasificación":
            # Gráfico de barras para clasificación
            value_counts = y_df.value_counts().sort_index()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title("Distribución de Clases")
            ax.set_xlabel("Clase")
            ax.set_ylabel("Cantidad")

            # Rotar etiquetas si son muchas
            if len(value_counts) > 3:
                plt.xticks(rotation=45, ha='right')

            # Explicación sobre balance de clases
            class_counts = df[target_col].value_counts()
            balance_ratio = class_counts.max() / class_counts.min()
            if balance_ratio > 3:
                st.warning(
                    f"⚠️ **Dataset desbalanceado**: La clase más frecuente tiene {balance_ratio:.1f}x más ejemplos que la menos frecuente")
                st.info(
                    "💡 Los modelos de IA funcionan mejor con clases balanceadas. Considera técnicas de balanceo si es necesario.")
            else:
                st.success(
                    "✅ **Dataset bien balanceado**: Las clases tienen cantidad similar de ejemplos")
        else:
            # Histograma para regresión
            # Convertir a numérico en caso de que sean strings
            y_numeric = pd.to_numeric(y_df, errors='coerce')
            # Usar matplotlib directamente para evitar problemas de tipo
            ax.hist(y_numeric.dropna(), bins=30,
                    alpha=0.7, edgecolor='black')
            ax.set_title("Distribución de Valores Objetivo")
            ax.set_xlabel("Valor")
            ax.set_ylabel("Frecuencia")

            # Estadísticas básicas para regresión
            with st.expander("📊 Estadísticas de la Variable Objetivo"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "🎯 Media", f"{df[target_col].mean():.2f}")
                with col2:
                    st.metric("📏 Desv. Estándar",
                              f"{df[target_col].std():.2f}")
                with col3:
                    st.metric(
                        "📉 Mínimo", f"{df[target_col].min():.2f}")
                with col4:
                    st.metric(
                        "📈 Máximo", f"{df[target_col].max():.2f}")

        # Mostrar la figura con tamaño reducido pero expandible
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.pyplot(fig, use_container_width=True)

        # Análisis avanzado con selector de visualización
        st.markdown("### 📊 Análisis Avanzado de Datos")

        # Inicializar el estado del análisis si no existe
        if 'active_analysis' not in st.session_state:
            st.session_state.active_analysis = None

        # Selector de tipo de análisis con botones
        st.markdown("#### Elige el tipo de análisis que quieres realizar:")
        col1, col2 = st.columns(2)

        with col1:
            correlation_clicked = st.button(
                "🔗 Matriz de Correlación",
                use_container_width=True,
                help="Analiza las relaciones lineales entre características",
                type="primary" if st.session_state.active_analysis == "correlation" else "secondary"
            )
            if correlation_clicked:
                st.session_state.active_analysis = "correlation"
                st.rerun()

        with col2:
            pairplot_clicked = st.button(
                "📈 Matriz de Dispersión (Pairplot)",
                use_container_width=True,
                help="Visualiza relaciones entre todas las parejas de características",
                type="primary" if st.session_state.active_analysis == "pairplot" else "secondary"
            )
            if pairplot_clicked:
                st.session_state.active_analysis = "pairplot"
                st.rerun()

        # Mostrar el análisis seleccionado
        if st.session_state.active_analysis == "correlation":
            st.markdown("#### Matriz de Correlación")
            st.info("💡 **Correlación**: Mide la relación lineal entre características. Valores cercanos a 1 o -1 indican correlación fuerte.")

            # Matriz de correlación
            corr = X_df.corr()

            # Generar máscara para el triángulo superior
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Generar mapa de calor
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
            ax.set_title("Matriz de Correlación de Características")

            # Mostrar la figura con tamaño reducido pero expandible
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig_corr, use_container_width=True)

            # Enlace para descargar
            st.markdown(
                get_image_download_link(
                    fig_corr, "matriz_correlacion", "📥 Descargar matriz de correlación"),
                unsafe_allow_html=True
            )

        elif st.session_state.active_analysis == "pairplot":
            st.markdown("#### Matriz de Dispersión (Pairplot)")
            st.info("💡 **Pairplot**: Muestra las relaciones entre todas las parejas de características. Útil para detectar patrones y separabilidad entre clases.")

            # Opciones de visualización
            st.markdown("##### Opciones de visualización")
            col1, col2 = st.columns(2)

            with col1:
                # Seleccionar tipo de gráfico para la diagonal
                diag_kind = st.radio(
                    "Tipo de gráfico en la diagonal:",
                    ["Histograma", "KDE (Estimación de Densidad)"],
                    index=1,
                    horizontal=True
                )
                diag_kind = "hist" if diag_kind == "Histograma" else "kde"

            with col2:
                # Seleccionar número máximo de características
                max_features_selected = st.slider(
                    "Número máximo de características:",
                    min_value=2,
                    max_value=min(6, len(X_df.columns)),
                    value=min(4, len(X_df.columns)),
                    help="Un número mayor de características puede hacer que el gráfico sea más difícil de interpretar."
                )

            # Permitir al usuario seleccionar las características específicas
            st.markdown("##### Selecciona las características para visualizar")

            # Limitar a max_features_selected
            # Usar nombres amigables si están disponibles, sino usar originales
            if column_mapping:
                available_features = list(
                    column_mapping.values())  # Nombres amigables
                display_to_original = {
                    v: k for k, v in column_mapping.items()}  # Mapeo inverso
            else:
                available_features = X_df.columns.tolist()
                display_to_original = {}

            # Usar multiselect para seleccionar características
            selected_features = st.multiselect(
                "Características a incluir en la matriz de dispersión:",
                available_features,
                default=available_features[:max_features_selected],
                max_selections=max_features_selected,
                help=f"Selecciona hasta {max_features_selected} características para incluir en la visualización."
            )

            # Si no se seleccionó ninguna característica, usar las primeras por defecto
            if not selected_features:
                selected_features = available_features[:max_features_selected]
                st.info(
                    f"No se seleccionaron características. Usando las primeras {max_features_selected} por defecto.")

            # Convertir nombres amigables a nombres originales si es necesario
            if column_mapping:
                original_features = [display_to_original[feat]
                                     for feat in selected_features]
            else:
                original_features = selected_features

            if len(X_df) > 2000:
                enable_sampling = True
                high_quality = True
                sample_size = 2000
            else:
                enable_sampling = False
                high_quality = False
                sample_size = len(X_df)

            # Validaciones de seguridad y advertencias
            total_plots = len(selected_features) ** 2
            estimated_time = total_plots * \
                len(X_df) / 50000  # Estimación aproximada

            # Mostrar advertencias basadas en el tamaño
            if total_plots > 25:
                st.warning(
                    f"⚠️ **Cuidado**: {len(selected_features)} características = {total_plots} gráficos. Puede ser muy lento.")
            elif total_plots > 16:
                st.info(
                    f"ℹ️ **Nota**: {len(selected_features)} características = {total_plots} gráficos. Tiempo estimado: ~{estimated_time:.1f}s")

            # Botones de control
            col1, col2 = st.columns([3, 1])
            with col1:
                generate_clicked = st.button(
                    "🚀 Generar Matriz de Dispersión", type="primary")
            with col2:
                if st.button("🛑 Reiniciar", help="Usa esto si la generación se queda colgada"):
                    st.rerun()

            if generate_clicked:
                # Crear el dataframe para la visualización
                plot_df = X_df[original_features].copy()
                # Renombrar a nombres amigables para visualización
                if column_mapping:
                    plot_df = plot_df.rename(columns=column_mapping)
                # Añadir la variable objetivo para colorear
                plot_df['target'] = y_df

                # Aplicar límites de seguridad automáticos
                original_size = len(plot_df)
                max_safe_points = 2000  # Límite de seguridad absoluto

                # Aplicar muestreo si está habilitado O si es necesario por seguridad
                if enable_sampling and len(plot_df) > sample_size:
                    # Muestreo estratificado para mantener proporción de clases
                    if task_type == "Clasificación":
                        plot_df = plot_df.groupby('target', group_keys=False).apply(
                            lambda x: x.sample(min(len(x), sample_size // len(plot_df['target'].unique())),
                                               random_state=42)
                        ).reset_index(drop=True)
                    else:
                        # Para regresión, muestreo aleatorio simple
                        plot_df = plot_df.sample(
                            n=sample_size, random_state=42).reset_index(drop=True)
                elif len(plot_df) > max_safe_points:
                    # Límite de seguridad automático
                    st.warning(
                        f"⚠️ Aplicando límite de seguridad: {max_safe_points} puntos máximo para evitar timeouts")
                    if task_type == "Clasificación":
                        plot_df = plot_df.groupby('target', group_keys=False).apply(
                            lambda x: x.sample(min(len(x), max_safe_points // len(plot_df['target'].unique())),
                                               random_state=42)
                        ).reset_index(drop=True)
                    else:
                        plot_df = plot_df.sample(
                            n=max_safe_points, random_state=42).reset_index(drop=True)

                if len(plot_df) < original_size:
                    st.info(
                        f"📊 Usando {len(plot_df)} puntos de {original_size} totales para optimizar velocidad")

                # Configurar parámetros optimizados
                if high_quality:
                    plot_kws = {'alpha': 0.7, 's': 40,
                                'edgecolor': 'white', 'linewidth': 0.5}
                    diag_kws = {'alpha': 0.8}
                    height = 2.5
                else:
                    # Configuración optimizada para velocidad
                    # Sin bordes, rasterizado
                    plot_kws = {'alpha': 0.6, 's': 20, 'rasterized': True}
                    diag_kws = {'alpha': 0.6}
                    height = 2.0

                # Generar el pairplot con optimizaciones y control de tiempo
                progress_container = st.container()
                with progress_container:
                    # Crear un placeholder para el progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    cancel_button_placeholder = st.empty()

                    # Botón de cancelación (simulado)
                    status_text.info("🔄 Iniciando generación de la matriz...")
                    progress_bar.progress(10)

                    try:
                        # Verificar tamaño antes de proceder
                        estimated_time = len(
                            selected_features) ** 2 * len(plot_df) / 10000
                        if estimated_time > 30:  # Si se estima más de 30 segundos
                            st.warning(
                                f"⚠️ **Advertencia**: La generación puede tardar ~{estimated_time:.1f} segundos")
                            st.info(
                                "💡 **Sugerencia**: Reduce el número de características o el tamaño de muestra")

                        status_text.info(
                            f"🔄 Generando matriz... ({len(plot_df)} puntos, {len(selected_features)} características)")
                        progress_bar.progress(20)

                        # Configurar matplotlib para mejor rendimiento
                        plt.rcParams['figure.max_open_warning'] = 0
                        progress_bar.progress(30)

                        # Generar el pairplot con timeout simulado
                        import time
                        start_time = time.time()

                        status_text.info("🎨 Creando gráficos...")
                        progress_bar.progress(50)

                        pair_plot = sns.pairplot(
                            plot_df,
                            hue='target',
                            diag_kind=diag_kind,
                            plot_kws=plot_kws,
                            diag_kws=diag_kws,
                            height=height,
                            aspect=1.0  # Aspect ratio fijo para mejor rendimiento
                        )

                        progress_bar.progress(80)
                        status_text.info("🎨 Finalizando visualización...")

                        # Verificar si ha tardado demasiado
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 60:  # Más de 1 minuto
                            st.warning(
                                f"⏱️ La generación tardó {elapsed_time:.1f} segundos. Considera reducir el tamaño.")

                        # Configurar título con información de optimización
                        title = "Matriz de Dispersión de Características"
                        if enable_sampling and len(X_df) > sample_size:
                            title += f" (Muestra: {len(plot_df)}/{len(X_df)} puntos)"

                        pair_plot.fig.suptitle(title, y=1.02, fontsize=14)
                        progress_bar.progress(90)

                        status_text.success(
                            f"✅ Matriz generada exitosamente en {elapsed_time:.1f}s")
                        progress_bar.progress(100)

                        # Limpiar indicadores de progreso después de un momento
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()

                        st.pyplot(pair_plot.fig, use_container_width=True)

                        # Enlace para descargar
                        st.markdown(
                            get_image_download_link(
                                pair_plot.fig, "matriz_dispersion", "📥 Descargar matriz de dispersión"),
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Error generando la matriz: {str(e)}")
                        st.info(
                            "💡 Intenta reducir el número de características o el tamaño de muestra")
            else:
                st.info(
                    "👆 Haz clic en el botón para generar la matriz de dispersión con las características seleccionadas.")

                # Consejos para evitar cuelgues
                with st.expander("💡 Consejos para evitar que se cuelgue la generación"):
                    st.markdown("""
                    **Si la generación tarda mucho o se cuelga:**
                    
                    1. 🎯 **Reduce características**: Usa máximo 4-5 características
                    2. 🛑 **Usa el botón 'Reiniciar'**: Si se queda colgado más de 2 minutos
                    3. 🔄 **Recarga la página**: Como último recurso (Ctrl+F5)
                    
                    **Tiempos aproximados:**
                    - 3 características, 500 puntos: ~5 segundos
                    - 4 características, 1000 puntos: ~15 segundos  
                    - 5 características, 2000 puntos: ~45 segundos
                    - 6+ características: Puede tardar varios minutos
                    """)

        # Generar código para este análisis
        code = SCATTERPLOT_MATRIX

        show_code_with_download(
            code, "Código para análisis exploratorio", "analisis_exploratorio.py")

    except Exception as e:
        st.error(f"Error al cargar el dataset: {str(e)}")
        st.info(
            "Por favor, selecciona un dataset válido para continuar con la exploración.")
