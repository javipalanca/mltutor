<!-- mltutor/apps/neural_network_app.py — st.info(tips) en 📊 Datos, debajo de st.header("📊 Selección y Preparación de Datos"). CORREGIDO en Rust (revisión de contenido de F5, 2026-10-16): (1) Python: "- Las redes neuronales funcionan mejor con **datos normalizados** (valores entre 0 y 1 o -1 y 1)". La app no normaliza a un rango: estandariza (media 0, desviación 1) con StandardScaler, y lo hace sola; (2) Python: "- Necesitan **suficientes datos** para entrenar bien (mínimo 100 ejemplos por clase)". No hay tal mínimo y contradecía los datasets de la app (Iris tiene 50 por clase). -->
🎓 **Tips para Redes Neuronales:**
- Las redes neuronales funcionan mejor con **datos escalados**: MLTutor estandariza cada característica (media 0 y desviación 1) antes de entrenar
- Necesitan **suficientes datos** para entrenar bien: con pocos (🌸 Iris tiene 50 por clase) el resultado depende mucho de la semilla y sobreajustan con facilidad
- Son excelentes para **patrones complejos** y **relaciones no lineales**
- Pueden funcionar tanto para **clasificación** como para **regresión**
