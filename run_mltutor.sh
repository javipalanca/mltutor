#!/bin/bash
# Script para ejecutar MLTutor

echo "====================================="
echo "     Iniciando MLTutor              "
echo "====================================="
echo "MLTutor es una plataforma educativa para aprender Machine Learning de forma interactiva."
echo ""
echo "Abriendo aplicación en tu navegador..."
echo ""

# Ejecutar la aplicación
python -m streamlit run app_refactored.py

# Este script nunca llegará aquí mientras la aplicación esté en ejecución
echo "Aplicación terminada."
