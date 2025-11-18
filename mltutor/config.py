"""
Configuración global de MLTutor.
"""
import os

# Control de GPU/CPU para TensorFlow
# Set USE_GPU=1 como variable de entorno para habilitar GPU
USE_GPU = os.environ.get('USE_GPU', '0') == '1'

# Configuración de TensorFlow
TF_LOG_LEVEL = os.environ.get('TF_LOG_LEVEL', '2')  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
