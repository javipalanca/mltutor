"""
Metadata enriquecida para datasets de MLTutor.
Incluye mapeos de valores originales/traducidos y descripciones de campos.
"""

def get_dataset_metadata(dataset_name):
    """
    Obtiene metadata enriquecida para un dataset específico.
    
    Parameters:
    -----------
    dataset_name : str
        Nombre del dataset
        
    Returns:
    --------
    dict
        Metadata que incluye:
        - feature_descriptions: Descripciones de cada característica
        - value_mappings: Mapeos de valores categóricos originales a traducidos
        - original_to_display: Mapeo de nombres de columnas originales a nombres para mostrar
        - categorical_features: Lista de características categóricas
    """
    
    if "Titanic" in dataset_name:
        return {
            'feature_descriptions': {
                'pclass': 'Clase de pasajero (1=Primera, 2=Segunda, 3=Tercera)',
                'age': 'Edad en años del pasajero',
                'sibsp': 'Número de hermanos/esposos a bordo',
                'parch': 'Número de padres/hijos a bordo',
                'fare': 'Tarifa pagada por el boleto',
                'sex_encoded': 'Sexo del pasajero',
                'embarked_encoded': 'Puerto de embarque',
                'class_encoded': 'Clase social del pasajero',
                'who_encoded': 'Categoría de persona',
                'adult_male': 'Si es hombre adulto',
                'alone': 'Si viajaba solo'
            },
            'value_mappings': {
                'sex_encoded': {0: 'Mujer', 1: 'Hombre'},
                'embarked_encoded': {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'},
                'class_encoded': {0: 'Primera', 1: 'Segunda', 2: 'Tercera'},
                'who_encoded': {0: 'Niño', 1: 'Hombre', 2: 'Mujer'},
                'adult_male': {0: 'No', 1: 'Sí'},
                'alone': {0: 'No', 1: 'Sí'},
                'pclass': {1: 'Primera Clase', 2: 'Segunda Clase', 3: 'Tercera Clase'}
            },
            'original_to_display': {
                'pclass': 'Clase_Pasajero',
                'age': 'Edad',
                'sibsp': 'Hermanos_Conyuges',
                'parch': 'Padres_Hijos',
                'fare': 'Tarifa',
                'sex_encoded': 'Sexo',
                'embarked_encoded': 'Puerto_Embarque',
                'class_encoded': 'Clase_Social',
                'who_encoded': 'Categoria_Persona',
                'adult_male': 'Hombre_Adulto',
                'alone': 'Solo'
            },
            'categorical_features': ['pclass', 'sex_encoded', 'embarked_encoded', 'class_encoded', 'who_encoded', 'adult_male', 'alone']
        }
    
    elif "Iris" in dataset_name:
        return {
            'feature_descriptions': {
                'sepal_length': 'Longitud del sépalo en centímetros',
                'sepal_width': 'Ancho del sépalo en centímetros',
                'petal_length': 'Longitud del pétalo en centímetros',
                'petal_width': 'Ancho del pétalo en centímetros'
            },
            'value_mappings': {},
            'original_to_display': {
                'sepal_length': 'Longitud_Sépalo',
                'sepal_width': 'Ancho_Sépalo',
                'petal_length': 'Longitud_Pétalo',
                'petal_width': 'Ancho_Pétalo'
            },
            'categorical_features': []
        }
    
    elif "Propinas" in dataset_name or "Tips" in dataset_name:
        return {
            'feature_descriptions': {
                'total_bill': 'Total de la cuenta en dólares',
                'size': 'Número de personas en la mesa',
                'sex_encoded': 'Sexo del cliente que paga',
                'smoker_encoded': 'Si el cliente fuma',
                'day_encoded': 'Día de la semana',
                'time_encoded': 'Momento del día (almuerzo/cena)'
            },
            'value_mappings': {
                'sex_encoded': {0: 'Mujer', 1: 'Hombre'},
                'smoker_encoded': {0: 'No Fumador', 1: 'Fumador'},
                'day_encoded': {0: 'Jueves', 1: 'Viernes', 2: 'Sábado', 3: 'Domingo'},
                'time_encoded': {0: 'Cena', 1: 'Almuerzo'}
            },
            'original_to_display': {
                'total_bill': 'Cuenta_Total',
                'size': 'Tamaño_Grupo',
                'sex_encoded': 'Sexo',
                'smoker_encoded': 'Fumador',
                'day_encoded': 'Día_Semana',
                'time_encoded': 'Comida'
            },
            'categorical_features': ['sex_encoded', 'smoker_encoded', 'day_encoded', 'time_encoded']
        }
    
    elif "Pingüinos" in dataset_name or "Penguins" in dataset_name:
        return {
            'feature_descriptions': {
                'bill_length_mm': 'Longitud del pico en milímetros',
                'bill_depth_mm': 'Profundidad del pico en milímetros',
                'flipper_length_mm': 'Longitud de la aleta en milímetros',
                'body_mass_g': 'Masa corporal en gramos',
                'island_encoded': 'Isla donde fue observado',
                'sex_encoded': 'Sexo del pingüino'
            },
            'value_mappings': {
                'island_encoded': {0: 'Biscoe', 1: 'Dream', 2: 'Torgersen'},
                'sex_encoded': {0: 'Hembra', 1: 'Macho'}
            },
            'original_to_display': {
                'bill_length_mm': 'Longitud_Pico',
                'bill_depth_mm': 'Profundidad_Pico',
                'flipper_length_mm': 'Longitud_Aleta',
                'body_mass_g': 'Masa_Corporal',
                'island_encoded': 'Isla',
                'sex_encoded': 'Sexo'
            },
            'categorical_features': ['island_encoded', 'sex_encoded']
        }
    
    elif "Viviendas California" in dataset_name:
        return {
            'feature_descriptions': {
                'MedInc': 'Ingresos medios del distrito en decenas de miles de dólares',
                'HouseAge': 'Edad media de las casas en años',
                'AveRooms': 'Número promedio de habitaciones por hogar',
                'AveBedrms': 'Número promedio de dormitorios por hogar',
                'Population': 'Población del distrito',
                'AveOccup': 'Ocupación promedio (personas por hogar)',
                'Latitude': 'Latitud del distrito',
                'Longitude': 'Longitud del distrito'
            },
            'value_mappings': {},
            'original_to_display': {
                'MedInc': 'Ingresos_Medios',
                'HouseAge': 'Edad_Casa',
                'AveRooms': 'Habitaciones_Promedio',
                'AveBedrms': 'Dormitorios_Promedio',
                'Population': 'Población',
                'AveOccup': 'Ocupación_Promedio',
                'Latitude': 'Latitud',
                'Longitude': 'Longitud'
            },
            'categorical_features': []
        }
    
    elif "Vino" in dataset_name:
        return {
            'feature_descriptions': {
                'alcohol': 'Contenido de alcohol',
                'malic_acid': 'Ácido málico',
                'ash': 'Ceniza',
                'alcalinity_of_ash': 'Alcalinidad de la ceniza',
                'magnesium': 'Magnesio',
                'total_phenols': 'Fenoles totales',
                'flavanoids': 'Flavonoides',
                'nonflavanoid_phenols': 'Fenoles no flavonoides',
                'proanthocyanins': 'Proantocianinas',
                'color_intensity': 'Intensidad del color',
                'hue': 'Tono',
                'od280/od315_of_diluted_wines': 'Ratio OD280/OD315 de vinos diluidos',
                'proline': 'Prolina'
            },
            'value_mappings': {},
            'original_to_display': {},  # Los nombres originales son técnicos pero apropiados
            'categorical_features': []
        }
    
    elif "Cáncer" in dataset_name:
        return {
            'feature_descriptions': {
                # Simplificado - solo algunas características principales
                'mean radius': 'Radio promedio del núcleo celular',
                'mean texture': 'Textura promedio del núcleo',
                'mean perimeter': 'Perímetro promedio del núcleo',
                'mean area': 'Área promedio del núcleo',
                'mean smoothness': 'Suavidad promedio del núcleo'
            },
            'value_mappings': {},
            'original_to_display': {},  # Demasiadas características para traducir todas
            'categorical_features': []
        }
    
    else:
        # Metadata por defecto para datasets no reconocidos
        return {
            'feature_descriptions': {},
            'value_mappings': {},
            'original_to_display': {},
            'categorical_features': []
        }
