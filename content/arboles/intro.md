<!-- mltutor/apps/decission_tree_app.py — expander "ℹ️ ¿Qué son los Árboles de Decisión?" (cerrado por defecto), encima de las pestañas. CORREGIDO en Rust (revisión ML de F3, 2026-10-15): la tercera viñeta decía "- Pueden manejar tanto datos numéricos como categóricos"; scikit-learn solo acepta números, así que las categorías se codifican antes (MLTutor, en orden alfabético) y el árbol las trata como valores ordenados (ver parametros.md, «Creer que el árbol entiende las categorías»). -->
Los árboles de decisión son algoritmos de aprendizaje supervisado que se pueden usar tanto para tareas de clasificación como de regresión.

**Características principales:**
- Funcionan dividiendo el conjunto de datos en subconjuntos más pequeños basándose en condiciones sobre las características
- Son fáciles de interpretar y visualizar
- Admiten características categóricas si se codifican como números (aquí, en orden alfabético), pero las tratan como valores ordenados
- No requieren escalado de datos

**Limitaciones:**
- Pueden sobreajustarse fácilmente (esto se puede controlar con parámetros como max_depth)
- Pueden ser inestables (pequeños cambios en los datos pueden generar árboles muy diferentes)
- No son tan precisos como algoritmos más complejos para algunas tareas

Experimenta con diferentes configuraciones para ver cómo afectan al rendimiento del modelo.
