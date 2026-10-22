# analisis

Notebooks y scripts de análisis que **usan** el paquete `schelling_matricial`.
No forman parte del paquete: pueden usar pandas, matplotlib, seaborn, sympy,
scikit-learn, SALib, etc., que no son dependencias de `pyproject.toml`.

## Cómo usarlo

1. Activa el entorno y instala el paquete en modo editable (una sola vez):

   ```bash
   source .venv_schelling/bin/activate
   pip install -e ".[dev]"
   ```

2. En VS Code abre un notebook y elige el kernel `.venv_schelling`.

## Convenciones

- Numera los notebooks: `01_primeros_pasos.ipynb`, `02_...`.
- Fija siempre la semilla (`seed`) para que los resultados sean reproducibles.
- Guarda los datos generados en `resultados/` (en la raíz del repositorio),
  que está en `.gitignore`, no dentro de esta carpeta.
- Antes de hacer commit de un notebook, considera limpiar sus salidas
  (en VS Code: "Clear All Outputs") para que los cambios sean fáciles de revisar.

## Contenido

- `01_primeros_pasos.ipynb`: una simulación del modelo clásico, retícula
  inicial y final, evolución de la segregación y barrido de `tau` con réplicas.
- `02_verificacion_articulo.ipynb`: comprueba que el código implementa la
  dinámica del artículo (`docs/Articulo_Matriz_Promedio`): modelo clásico,
  aislados con `iota="max"`, regla de movimiento, invariancia afín,
  incomodidad total, orientación de cada tipo y reducción al modelo clásico
  (Teorema 1). Cada sección imprime `OK` o falla.
