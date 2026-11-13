# C.2 Comandos de reproducción

Desde la raíz del repositorio y con el entorno instalado:

```powershell
# Recuperación léxica: no representa evaluación de MiniLM.
.venv/Scripts/python.exe -m automl_agents.cli kb-eval --backend hash --k 3

# Recuperación semántica prevista: requiere dependencia y modelo disponibles.
.venv/Scripts/python.exe -m automl_agents.cli kb-eval --backend sentence-transformers --k 3

# Crea una ejecución MNIST nueva con manifiestos de referencia.
.venv/Scripts/python.exe scripts/verify_mnist.py --dataset runs/20261102-133428-608f/dataset --reference-run runs/20261102-133428-608f

# Recomprueba la entrega MNIST; requiere motor Docker disponible.
.venv/Scripts/python.exe -m automl_agents.cli verify-artifact 20261108-235255-e1aa --runs-dir runs/mnist-validation --timeout 1800

# Suite funcional: la prueba de proveedor real es optativa.
.venv/Scripts/python.exe -m pytest -q --disable-warnings
```

Los comandos permiten repetir procedimientos; no garantizan cifras idénticas en otro entorno. La disponibilidad de la muestra y de los modelos descargables debe comprobarse. No se incluyen credenciales ni se requiere enviar imágenes al proveedor para repetir la comprobación léxica.
