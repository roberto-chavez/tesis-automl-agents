# Web y jobs persistentes (etapa 10)

La interfaz se sirve con `automl-agents serve` en `http://127.0.0.1:8000`.
FastAPI entrega un único `web/index.html`, con `app.css` y `app.js` locales,
incluidos en el paquete Python. No necesita Node, fuentes externas ni un build.
Las variantes `index2.html`, `index3.html` e `index4.html` se retiraron.

## Recorrido

1. **Inicio y carga:** ZIP, objetivo, presets de 10/30/60 minutos de búsqueda,
   requisitos y errores junto al archivo. Opciones adicionales: equipo, reglas o
   asistente y pesos preentrenados. Reglas y embeddings hash permiten trabajar sin
   credenciales; desactivar pesos preentrenados permite la prueba sin descargas.
   Los presets limitan también candidatas, intentos y épocas; los tiempos no son
   una promesa de entrega. La preparación y el cierre quedan fuera de la búsqueda.
2. **Seguimiento:** estado del job, etapa, plan, decisiones, eventos, intentos
   terminados frente al límite, presupuesto de tiempo y mejor validación.
   Las curvas y épocas proceden de los archivos publicados por el callback de
   entrenamiento. No hay porcentaje total ni animaciones que simulen progreso.
3. **Resultados:** modelo seleccionado, validación y prueba separadas, F1 macro,
   motivo de parada, advertencias, calidad, entrega y verificación. Las descargas
   dependen de archivos reales y del estado. Una entrega parcial conserva lo útil.

Las tres vistas tienen etiquetas, foco visible, mensajes accesibles y disposición
adaptable. El ID actual se conserva en la URL y en `localStorage`; la lista del
servidor permite recuperar proyectos incluso desde otro navegador. Las consultas
fallidas reintentan con espera creciente, sin borrar la referencia al proyecto.
Los textos del servidor y el reporte se insertan como texto, nunca como HTML.

## Persistencia y concurrencia

`AUTOML_RUNS_DIR` (por defecto `runs/`) contiene `jobs.sqlite`. Cada job guarda
configuración validada, estado, fechas, mensaje y solicitud de cancelación. Antes
de encolarlo se publican configuración, dataset validado y manifiestos persistentes.
La carga está limitada a 512 MB, 2 GB descomprimidos y 50 000 entradas. Se rechazan
rutas fuera del destino, enlaces simbólicos y archivos protegidos por contraseña.

Un worker iniciado por el lifespan de FastAPI consulta SQLite. El bloqueo del SO
en `.web-worker/.run.lock` abarca reconciliación, toma del job y ejecución completa:
varios procesos HTTP pueden servir consultas, pero sólo uno procesa jobs. El
bloqueo `.training/.run.lock` también se adquiere en el pipeline por CLI. Si una
CLI está entrenando, el job web vuelve a espera sin perder su configuración.
El bloqueo por ejecución existente sigue protegiendo su journal. Estos bloqueos
se liberan al morir el proceso; no dependen de un PID o un timeout de lease.

Al adquirir el bloqueo, el worker reconcilia los jobs que quedaron activos:
importa resultados publicados, marca intentos sin resultado como interrumpidos y
conserva la entrega terminal si el pipeline ya la había confirmado. Los jobs
pendientes continúan automáticamente. Los interrumpidos requieren **Reanudar**;
no se repite trabajo confirmado ni se reinicia el presupuesto. Una época o un
intento interrumpido no se retoma desde sus pesos parciales: se crea un intento
nuevo según las políticas existentes.

La cancelación se guarda tanto en SQLite como en `cancel.requested` dentro del
directorio de ejecución. Si estaba en espera, pasa directamente a cancelada.
Si estaba activa, muestra **Cancelación solicitada** hasta que el callback termine
la época o el pipeline alcance el siguiente límite cooperativo. Una vez congelada
la búsqueda, la evaluación y el empaquetado terminan sin ofrecer cancelación.
Reanudar limpia el marcador y encola la configuración original mediante una
transacción; no permite reanudar simultáneamente el mismo job. El reloj de búsqueda
existente es de pared e incluye las pausas: no se recupera tiempo al reanudar.

## API

| Operación | Comportamiento |
|---|---|
| `POST /api/runs` | Multipart: `file`, `use_case`, `budget`, `tier`, `mode`, `pretrained`. Valida datos y splits antes de encolar. |
| `GET /api/runs` | Jobs guardados, con estado y fechas. |
| `GET /api/runs/{id}` | Estado de cola y ejecución, plan/tarea, progreso, presupuesto, validación, decisiones, eventos, entrega, capacidades y URLs de descarga. |
| `POST /api/runs/{id}/cancel` | Sólo pendientes/activos antes del cierre. Persiste intención; estados incompatibles devuelven 409. |
| `POST /api/runs/{id}/resume` | Sólo cancelados/interrumpidos/fallidos. Conserva configuración, intentos y presupuesto. |
| `GET /api/runs/{id}/report` | Reporte como texto después de detenerse, si existe. |
| `GET /api/runs/{id}/download?kind=…` | `model`, `reports`, `artifact`, `all` (por defecto). URLs autorizadas en `downloads`. |

Los errores legibles tienen `detail.message`, `detail.field` y `detail.code`.
No se admiten rutas arbitrarias para descargar. Los ZIP de descarga son temporales
y se eliminan al terminar la respuesta; no se reutiliza un ZIP de una entrega anterior.

**Reportes:** se ofrecen si existen y el job es terminal, incluso con fallo o pausa.
**Modelo:** sólo el checkpoint de la selección congelada, dentro de la ejecución.
**API:** entrega completada o con objetivo no alcanzado, con modelo y comprobación
local aprobada. Si Docker no se ejecutó, se muestra explícitamente verificación
completa pendiente; nunca se presenta como `verified=true`. Una API fallida se
excluye del ZIP conjunto. No se entrega un modelo provisional de un trial activo.

Los jobs nuevos usan SQLite. Las ejecuciones antiguas que sólo tenían `status.json`
no se importan como jobs recuperables; sus archivos siguen en disco y la CLI permite
consultar las ejecuciones con journal de etapas anteriores.

## Verificación reproducible

```powershell
.venv/Scripts/python.exe -m pytest tests/test_api.py tests/test_persistence.py tests/test_search_blocks.py -q

# Terminal 1: servidor aislado con dataset sintético y preset pequeño
.venv/Scripts/python.exe scripts/serve_web_smoke.py

# Terminal 2: Playwright para desarrollo; no es dependencia de la aplicación
node scripts/verify_web.cjs runs/web-smoke/screenshots
```

El script del navegador usa Edge headless (`WEB_SMOKE_BROWSER=chrome` permite
Chrome), genera capturas a 1440 y 390 px y rechaza desbordamientos horizontales y
errores JavaScript. Recorre validación de formulario, ZIP inválido, carga real,
entrenamiento, cancelación, reanudación, resultados, descargas, lectura del reporte,
pérdida de conexión y recuperación por URL. Guarda hechos en `verification.json`.
El servidor de prueba usa `runs/web-smoke/`, sin alterar presets productivos.
Para repetir sólo las comprobaciones de presentación, descargas y reconexión sobre
un proyecto ya terminado, configure `WEB_SMOKE_RUN` con su ID antes del script.
Las pruebas Python cubren reinicio/reconciliación, workers competidores, espera
por CLI, solicitudes incompatibles, progreso por época y entregas parciales, además
de un recorrido real que verifica los endpoints de la API exportada.

La aplicación mantiene el alcance local existente y no incorpora autenticación.
LLM real, pesos externos y Docker se comprueban por separado cuando estén disponibles.

### Resultado de la verificación del 21 de octubre de 2026

- Suite completa: **286 aprobadas y 1 omitida**, con entrenamiento y artefactos reales.
  La omitida es la integración optativa con proveedor LLM real (`AUTOML_TEST_LIVE_LLM`).
- Edge headless: carga, cancelación por época, reanudación, evaluación, descargas,
  reporte, pérdida de conexión y restauración de la vista por URL. Sin errores JavaScript.
- Presentación: capturas de las tres vistas y entrega cancelada en 1440 y 390 px;
  comprobación adicional de desbordamiento horizontal a 320 y 768 px y navegación
  de teclado, incluidos foco y errores del formulario.
  La entrega con fallo de empaquetado también se visualiza mediante una respuesta
  simulada, identificada como tal en las capturas y el registro de verificación.
- Wheel construido sin descargar dependencias; extraído en un directorio aislado
  para comprobar las rutas HTML/CSS/JS. Sólo contiene una variante HTML activa.
- API exportada: carga del modelo, preprocesamiento, `/health`, `/labels`,
  `/predict` e imagen inválida aprobados. Docker no instalado: `not_run` y
  `verified=false`, como muestra la web. Sin llamadas LLM ni pesos externos en este recorrido.
