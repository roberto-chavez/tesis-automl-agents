**PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ**

**FACULTAD DE CIENCIAS E INGENIERÍA**

**Tesis para obtener el título profesional de Ingeniero Informático**

**AUTOR:**

**CHAVEZ JAVIER, ROBERTO ABEL**

**ASESOR:**

**FONSECA ARROYO, PABLO ALEJANDRO**

**Lima, septiembre, 2026**

# Resumen

(NO NECESARIO EN LOS CURSOS DE TESIS)

- La extensión debe ser de 200 a 300 palabras, sin exceder a una página.

- Escriba en tiempo verbal presente

- El resumen debe contener información sobre:

- La justificación de la investigación

- Los objetivos o hipótesis

- La teoría o supuestos teóricos o metodológicos en la que se sustenta

- El método o procedimiento realizado (de ser necesario)

- Los resultados (de ser necesario)

- La conclusión principal

# Tema FCI

Aquí se debe colocar una copia del tema aprobado por el Decano de la FCI. No es necesario colocarlo durante los cursos de tesis.

# Tabla de Contenido

- Resumen (p. ii)
- Tema FCI (p. iii)
- Capítulo 1. Generalidades (p. 6)
  - 1.1 Problemática (p. 6)
    - 1.1.1 Árbol de Problemas (p. 7)
    - 1.1.2 Descripción (p. 8)
    - 1.1.3 Problema seleccionado (p. 9)
  - 1.2 Objetivos (p. 10)
    - 1.2.1 Objetivo general (p. 10)
    - 1.2.2 Objetivos específicos (p. 11)
    - 1.2.3 Resultados esperados (p. 12)
    - 1.2.4 Mapeo de objetivos, resultados y verificación (p. 15)
  - 1.3 Métodos y Procedimientos (p. 20)
- Capítulo 2. Marco Legal/Regulatorio/Conceptual/otros (p. 28)
  - 2.1 Introducción (p. 28)
  - 2.2 Desarrollo del marco (p. 28)
- Capítulo 3. Estado del Arte (p. 37)
  - 3.1 Introducción (p. 37)
  - 3.2 Objetivos de revisión (p. 37)
  - 3.3 Preguntas de revisión (p. 38)
  - 3.4 Estrategia de búsqueda (p. 39)
    - 3.4.1 Motores de búsqueda a usar (p. 42)
    - 3.4.2 Cadenas de búsqueda a usar (p. 43)

    - 3.4.3 Documentos encontrados (p. 46)
    - 3.4.4 Criterios de inclusión/exclusión (p. 50)
  - 3.5 Formulario de extracción de datos (p. 53)
  - 3.6 Resultados de la revisión (p. 55)
    - 3.6.1 Respuesta a pregunta P1 (p. 55)
    - 3.6.2 Respuesta a pregunta P2 (p. 60)
    - 3.6.3 Respuesta a pregunta P3 (p. 63)
  - 3.7 Conclusiones (p. 67)
- Capítulo 4. Presentación de los resultados esperados (p. 70)
- Capítulo 5. Conclusiones y trabajos futuros (p. 71)
  - 5.1 Conclusiones (p. 71)
  - 5.2 Trabajos futuros (p. 71)
- Referencias (p. 72)
- Anexos (p. 78)
- Anexo A: Plan de Proyecto (p. 79)

# Capítulo 1. Generalidades

## 1.1 Problemática

Durante la última década, el aprendizaje profundo (*deep learning*) se ha convertido en el enfoque dominante para resolver tareas de clasificación de imágenes, pero construir un modelo con buen desempeño para un dataset etiquetado específico continúa siendo un proceso costoso: exige diseñar o elegir una arquitectura, ajustar sus hiperparámetros, entrenarla y validarla, habitualmente mediante ciclos iterativos de prueba y error (Elsken et al., 2019; Xue et al., 2025). Este proceso involucra un espacio de configuraciones (arquitecturas, hiperparámetros de entrenamiento, estrategias de preprocesamiento) tan amplio que explorarlo manualmente resulta poco escalable, incluso para practicantes con experiencia (Xue et al., 2025).

Precisamente para reducir esta carga manual surgió el aprendizaje automático automatizado (*AutoML*), un campo orientado a automatizar el pipeline de construcción de modelos, desde el preprocesamiento hasta la selección de arquitectura y el ajuste de hiperparámetros, con el fin de disminuir la dependencia de conocimiento técnico especializado (He et al., 2021; Karmaker Santu et al., 2022). Sin embargo, el propio desarrollo del *AutoML* confirma que la selección de un modelo adecuado no es un problema trivial: sigue requiriendo comprender los datos del dominio, definir el problema de predicción y elegir una técnica de aprendizaje prometedora, pasos que suelen implicar un proceso iterativo entre el usuario y la herramienta (Karmaker Santu et al., 2022).

Esta dificultad técnica tiene un correlato organizacional. La literatura sobre adopción de inteligencia artificial (IA) en empresas señala de forma consistente que la falta de conocimiento y de personal especializado en IA/ML es una de las barreras más citadas para adoptar estas tecnologías, particularmente en pequeñas y medianas empresas (pymes) cuyo core de negocio no es la investigación en IA (Ayinaddis, 2025). A diferencia de las grandes organizaciones, que cuentan con equipos propios de investigación y desarrollo (I+D) capaces de asumir el costo de experimentar con modelos de aprendizaje profundo, las pymes suelen carecer de esa capacidad técnica y de los recursos computacionales necesarios para sostener ciclos de experimentación prolongados (Ayinaddis, 2025; Karmaker Santu et al., 2022). Esta asimetría se traduce en una brecha de adopción de IA entre empresas grandes y pequeñas, y en una presión competitiva sobre estas últimas, ya que las organizaciones que sí logran incorporar IA de forma más ágil ganan ventajas frente a las que no pueden hacerlo (Ayinaddis, 2025).

En consecuencia, el problema no debe entenderse como la ausencia de una aplicación o de un sistema informático puntual, sino como una dificultad estructural: las pymes cuyo core de negocio no es la IA/ML enfrentan una dificultad real para construir, seleccionar, entrenar y dejar listos para su despliegue modelos de clasificación de imágenes adecuados a partir de un dataset etiquetado, debido a la combinación de un espacio de configuración amplio, un proceso manual e iterativo, y la falta de conocimiento especializado y de recursos computacionales dentro de la organización.

### 1.1.1 Árbol de Problemas

Para organizar la problemática, se emplea la técnica del árbol de problemas, la cual permite identificar las causas directas del problema central y sus efectos (Ortegón et al., 2005). En la

Figura 1 se plantea el árbol de problemas de la problemática encontrada.

**Figura 1**

*Árbol de problemas*

![Figura 1. Árbol de problemas](images/figura1.png)

### 1.1.2 Descripción

Las cuatro causas se agrupan en dos dimensiones. Por un lado, hay causas de naturaleza técnica: el espacio de configuraciones posibles (arquitecturas, hiperparámetros) para un modelo de clasificación de imágenes es amplio, y explorarlo hoy en día depende de un proceso manual e iterativo de prueba y error, lo cual es intrínsecamente lento y demandante incluso para especialistas (Elsken et al., 2019; Xue et al., 2025). Por otro lado, hay causas de naturaleza organizacional, propias de empresas cuyo core de negocio no es la IA: la falta de personal especializado en ML/IA y la falta de recursos computacionales suficientes para sostener múltiples ciclos de entrenamiento y evaluación, ambas reportadas como barreras recurrentes para la adopción de IA en pymes (Ayinaddis, 2025; He et al., 2021). La combinación de estas cuatro causas configura el problema central: estas empresas enfrentan una dificultad real para construir, seleccionar y entrenar modelos de clasificación de imágenes adecuados a partir de sus propios datos etiquetados, y para dejarlos en un estado listo para desplegarse, sin tener que igualar la capacidad técnica y de infraestructura de una gran empresa con equipo propio de I+D (Karmaker Santu et al., 2022). De no resolverse, este problema central deriva en cuatro efectos. En el corto plazo, se traduce en mayor tiempo y costo de investigación y desarrollo antes de disponer de un modelo utilizable y en un mayor riesgo de terminar con una configuración de modelo subóptima, es decir, con menor exactitud o eficiencia de la que sería alcanzable (Xue et al., 2025). En el mediano plazo, la empresa queda en desventaja frente a otras organizaciones, grandes o pequeñas, que sí logran adoptar e iterar sobre soluciones de IA con mayor agilidad, lo que se traduce en pérdida de competitividad, o termina dependiendo de consultores y proveedores externos para poder construir sus propios modelos, profundizando la brecha de capacidades internas (Ayinaddis, 2025).

### 1.1.3 Problema seleccionado

El problema seleccionado para la presente investigación es la **dificultad de las pequeñas y medianas empresas (pymes) cuyo core de negocio no es la investigación en inteligencia artificial (IA) o *machine learning* (ML) para construir, seleccionar y entrenar de manera eficiente modelos adecuados de clasificación de imágenes a partir de un dataset etiquetado, y dejarlos listos para su despliegue.**

Este problema se selecciona porque concentra, en un único enunciado, las dos causas técnicas (amplio espacio de configuración y proceso manual/iterativo de construcción de modelos) y las dos causas organizacionales (falta de personal especializado y de recursos computacionales) identificadas en el árbol, y porque acota el alcance de la solución a un tipo de organización concreto, empresas cuyo negocio principal no es la IA, en lugar de dejarlo abierto a cualquier actor (grande, pequeño, con o sin equipo de I+D propio). Esta acotación es la que permite, en los siguientes capítulos, plantear una solución igualmente delimitada: una arquitectura multiagente que, apoyada en generación aumentada por recuperación (RAG), automatice la selección, entrenamiento y evaluación de modelos de clasificación de imágenes para este tipo de empresas.

## 1.2 Objetivos

En esta sección se definen el objetivo general y los cuatro objetivos específicos que se desprenden del análisis de la problemática presentada en la sección 1.1, cada uno orientado a atender una de las cuatro causas identificadas en el árbol de problemas, así como los resultados esperados de cada objetivo y el mapeo de verificación que permitirá evaluar objetivamente el cumplimiento de dichos resultados.

### 1.2.1 Objetivo general

Conforme al análisis de la problemática presentada, se define como objetivo general del proyecto: **“Desarrollar una arquitectura multiagente, apoyada en generación aumentada por recuperación (RAG), que automatice la construcción, selección, entrenamiento y preparación para el despliegue de modelos de clasificación de imágenes a partir de un dataset etiquetado, orientada a pequeñas y medianas empresas (pymes) cuyo core de negocio no es la investigación en inteligencia artificial (IA) o *machine learning* (ML).”**

### 1.2.2 Objetivos específicos

Durante el análisis detallado del objetivo general se determinaron cuatro objetivos específicos, uno por cada causa identificada en el árbol de problemas, los cuales se describen en la Tabla 1.

**Tabla 1**

*Tabla de objetivos específicos*

| Código | Objetivo específico | Causa que atiende |
|---|---|---|
| O1 | Diseñar un mecanismo, apoyado en RAG, que acote de manera automática el espacio de arquitecturas e hiperparámetros de modelos de clasificación de imágenes a un conjunto reducido de configuraciones candidatas relevantes para el dataset del usuario. | Amplio espacio de configuraciones de arquitecturas e hiperparámetros para modelos de clasificación de imágenes |
| O2 | Automatizar, mediante una arquitectura multiagente, el ciclo de entrenamiento, evaluación y ajuste de las configuraciones candidatas, sin requerir intervención manual continua del usuario. | Proceso de construcción y ajuste de modelos basado en decisiones manuales e iterativas de prueba y error |
| O3 | Diseñar un mecanismo de interacción que permita a usuarios sin conocimientos especializados en ML/IA proporcionar su dataset y requerimientos, y recibir el modelo resultante junto con una explicación comprensible de las decisiones tomadas por el sistema. | Ausencia de personal especializado en ML/IA dentro de la organización |
| O4 | Optimizar el consumo de recursos computacionales durante la búsqueda, entrenamiento y evaluación de las configuraciones candidatas, de modo que el proceso completo sea viable con los recursos limitados típicos de una pyme. | Alto consumo de recursos computacionales para entrenar y evaluar de forma repetida distintas configuraciones de modelos |

### 1.2.3 Resultados esperados

A continuación, se describe cada objetivo específico junto con los resultados concretos que se espera obtener al alcanzarlo. Estos resultados constituyen los entregables verificables del proyecto y serán la base del mapeo de verificación presentado en la sección 1.2.4.

**O1. Acotar el espacio de configuraciones de modelos mediante RAG**

Este objetivo busca reducir, con apoyo de una base de conocimiento recuperable, el amplio espacio de arquitecturas e hiperparámetros posibles a un subconjunto acotado y justificado de configuraciones candidatas para el dataset de imágenes proporcionado por el usuario.

- **Resultado 1 del objetivo 1 (R1.1):** Base de conocimiento (RAG) que indexa documentación técnica, buenas prácticas y arquitecturas de referencia para clasificación de imágenes, consultable por los agentes del sistema.

- **Resultado 2 del objetivo 1 (R1.2):** Componente que, a partir del dataset etiquetado y las clases del usuario, genera un conjunto acotado de configuraciones candidatas (arquitecturas e hiperparámetros).

- **Resultado 3 del objetivo 1 (R1.3):** Reporte automático de los criterios usados para acotar el espacio de búsqueda y de las configuraciones descartadas.

**O2. Automatizar el ciclo de entrenamiento, evaluación y ajuste**

Este objetivo busca eliminar la dependencia de ciclos manuales de prueba y error, delegando en agentes especializados la ejecución, evaluación y ajuste iterativo de las configuraciones candidatas generadas en O1.

- **Resultado 1 del objetivo 2 (R2.1):** Agente de entrenamiento que ejecuta de forma automática el entrenamiento de cada configuración candidata sobre el dataset del usuario.

- **Resultado 2 del objetivo 2 (R2.2):** Agente de evaluación que mide y compara el desempeño de los modelos entrenados bajo métricas estandarizadas (p. ej., exactitud, *F1-score*).

- **Resultado 3 del objetivo 2 (R2.3):** Mecanismo de iteración automática que, con base en los resultados de evaluaciones previas, decide continuar, ajustar o descartar configuraciones sin intervención humana.

**O3. Facilitar el uso del sistema a usuarios sin conocimientos especializados**

Este objetivo busca que una persona sin formación en ML/IA pueda utilizar el sistema de principio a fin, entregando como resultado un modelo entrenado y una explicación de las decisiones tomadas en un lenguaje comprensible para un usuario no técnico.

- **Resultado 1 del objetivo 3 (R3.1):** Interfaz de entrada mediante la cual el usuario carga su dataset (o un enlace a este) y define sus clases, sin requerir conocimientos técnicos de ML.

- **Resultado 2 del objetivo 3 (R3.2):** Reporte final, en lenguaje no técnico, que explica el modelo seleccionado, su desempeño esperado y las razones detrás de la selección.

- **Resultado 3 del objetivo 3 (R3.3):** Artefacto final del modelo entrenado (p. ej., una *API* dockerizada) entregado listo para su despliegue, sin requerir configuración manual adicional por parte del usuario.

**O4. Optimizar el consumo de recursos computacionales del proceso**

Este objetivo busca que la automatización propuesta en O1 y O2 sea viable en la práctica para una pyme, evitando que el ahorro de tiempo de las personas se traduzca en un consumo de cómputo desproporcionado.

- **Resultado 1 del objetivo 4 (R4.1):** Estrategia de priorización o descarte temprano de configuraciones candidatas que reduce el número de entrenamientos completos necesarios.

- **Resultado 2 del objetivo 4 (R4.2):** Registro del consumo de recursos (tiempo de cómputo, uso de CPU/GPU) asociado a cada configuración evaluada durante el proceso.

- **Resultado 3 del objetivo 4 (R4.3):** Reporte comparativo del consumo de recursos del proceso automatizado frente a un proceso de búsqueda manual o exhaustivo tomado como referencia (*baseline*).

### 1.2.4 Mapeo de objetivos, resultados y verificación

Con el fin de asegurar que cada resultado esperado pueda evaluarse de forma objetiva al final del proyecto, a continuación, se detalla el mapeo correspondiente a cada uno de los cuatro objetivos específicos, indicando para cada resultado su medio de verificación (dónde y cómo se comprobará su existencia) y su indicador objetivamente verificable (con qué criterio medible se dará por cumplido). A continuación, la Tabla 2 detalla el mapeo correspondiente al primer objetivo específico del proyecto, orientado al acotamiento del espacio de configuraciones.

**Tabla 2**

*Tabla de mapeo de resultados y verificación del objetivo 1*

| Objetivo O1: Acotar el espacio de configuraciones de modelos mediante RAG |  |  |
|---|---|---|
| Resultado | Medio de verificación | Indicador objetivamente verificable |
| R1.1. Base de conocimiento RAG de arquitecturas y configuraciones | Repositorio del proyecto con la base de conocimiento indexada y el código del módulo de recuperación | Al menos 10 consultas de prueba, con precisión ≥ 0.70 frente a un conjunto de referencia etiquetado manualmente |
| R1.2. Conjunto acotado de configuraciones candidatas | Registros (logs) de ejecución con las configuraciones generadas por el sistema para los datasets de prueba | Reducción ≥ 90 % del espacio de búsqueda considerado (por ejemplo, de N configuraciones posibles a un máximo de 5 candidatas) en cada dataset de prueba |

| R1.3. Reporte de criterios de acotación | Reporte generado automáticamente por el sistema al final de cada ejecución | Reporte generado en el 100 % de las ejecuciones (mínimo 3 datasets de prueba), incluyendo criterios de acotación y configuraciones descartadas con su motivo |
|---|---|---|

La Tabla 3 detalla el mapeo del segundo objetivo específico del proyecto, orientado a la automatización que propone la solución.

**Tabla 3**

*Tabla de mapeo de resultados y verificación del objetivo 2*

| Objetivo O2: Automatizar el ciclo de entrenamiento, evaluación y ajuste |  |  |
|---|---|---|
| Resultado | Medio de verificación | Indicador objetivamente verificable |
| R2.1. Agente de entrenamiento automático | Registros (logs) de ejecución del agente de entrenamiento | Porcentaje de ≥ 90 % de las configuraciones candidatas entrenadas sin intervención manual, por dataset de prueba. |
| R2.2. Agente de evaluación de modelos | Tablas comparativas de métricas generadas automáticamente por el agente evaluador | Exactitud, F1-score y matriz de confusión registradas para el 100 % de los modelos entrenados. |

| R2.3. Mecanismo de iteración automática | Historial de iteraciones del sistema (configuración evaluada, resultado obtenido, ajuste aplicado) | Al menos 3 ciclos completos de evaluación-ajuste por ejecución, sin intervención humana. |
|---|---|---|

La Tabla 4 detalla el mapeo del tercer objetivo específico del proyecto, orientado a la usabilidad del sistema para los usuarios.

**Tabla 4**

*Tabla de mapeo de resultados y verificación del objetivo 3*

| Objetivo O3: Facilitar el uso del sistema a usuarios sin conocimientos especializados |  |  |
|---|---|---|
| Resultado | Medio de verificación | Indicador objetivamente verificable |
| R3.1. Interfaz de carga de dataset | Prototipo funcional de la interfaz, probado con usuarios sin experiencia en ML | Al menos el 80 % de una muestra de n = 5 usuarios sin experiencia en ML completan la carga del dataset y la definición de clases sin asistencia técnica, y el puntaje SUS promedio es ≥ 68. |

| R3.2. Reporte final comprensible | Reportes generados por el sistema durante pruebas con usuarios | Al menos el 80 % de la misma muestra califica la comprensión del reporte, en base a cuestionario proporcionado, con 4 o más en una escala Likert de 1 a 5, sin apoyo técnico |
|---|---|---|
| R3.3. Artefacto final listo para despliegue | Artefacto (API dockerizada) entregado al finalizar una ejecución de prueba | Contenedor desplegado en entorno de prueba y 100 % de los smoke tests aprobados en los endpoints definidos, sin configuración manual adicional |

La Tabla 5 detalla el mapeo del cuarto objetivo específico del proyecto, orientado a la optimización de recursos.

**Tabla 5**

*Tabla de mapeo de resultados y verificación del objetivo 4*

| Objetivo O4: Optimizar el consumo de recursos computacionales del proceso |  |  |
|---|---|---|
| Resultado | Medio de verificación | Indicador objetivamente verificable |
| R4.1. Estrategia de priorización/descarte temprano | Código y documentación de la estrategia implementada | Reducción ≥ 50 % del número de entrenamientos completos frente a la búsqueda exhaustiva de referencia, con una pérdida |

|  |  | de exactitud ≤ 3 puntos porcentuales |
|---|---|---|
| R4.2. Registro de consumo de recursos | Registro (log) de tiempo de cómputo y uso de CPU/GPU por configuración evaluada | Registro de tiempo de cómputo y uso de CPU/GPU para el 100 % de las configuraciones evaluadas |
| R4.3. Reporte comparativo de consumo de recursos | Reporte comparativo entre el proceso automatizado y un proceso manual/exhaustivo de referencia | Reducción ≥ 40 % del tiempo total de cómputo frente al baseline manual o exhaustivo, medida en los mismos datasets y hardware |

Las metas numéricas de los IOV se definieron con los siguientes criterios. Primero, se establece una única muestra de usuarios de prueba para R3.1 y R3.2, conformada por entre 5 y 8 personas sin experiencia en ML/IA, idealmente vinculadas a pymes; el porcentaje de usuarios de los IOV de ambos resultados se calcula siempre sobre esa muestra. Segundo, los datasets de prueba son al menos tres conjuntos de datos etiquetados de clasificación de imágenes, y los mismos se emplean en la evaluación de los resultados de los objetivos O1, O2 y O4, de modo que las comparaciones sean consistentes. Tercero, la precisión (R1.1) se entiende como la proporción de los cinco documentos recuperados por consulta que son relevantes según el conjunto de referencia. Cuarto, el umbral de 68 puntos en la Escala de Usabilidad del Sistema (SUS) se toma como el valor de referencia de usabilidad promedio reportado en la literatura (Sauro, 2011), sobre la escala creada por Brooke (1996). Los demás umbrales (por ejemplo, la reducción de 90 % del espacio de búsqueda o de 50 % de los entrenamientos completos) se plantean como metas de partida que serán ajustadas, de ser necesario, según los resultados de las pruebas preliminares. Los métodos de validación correspondientes se detallan en la siguiente sección.

## 1.3 Métodos y Procedimientos

En esta sección se detalla, para cada uno de los doce resultados esperados definidos en la sección 1.2.3, al menos una herramienta o método concreto mediante el cual se obtendrá, así como la forma en que se validará su cumplimiento. La organización sigue la misma agrupación por objetivo específico empleada en las tablas de mapeo de verificación (sección 1.2.4), de modo que cada tabla de métodos pueda leerse en conjunto con su tabla de verificación correspondiente. En la Tabla 6 se describe brevemente las herramientas o métodos seleccionados del objetivo 1.

**Tabla 6**

*Tabla de métodos y herramientas para los resultados del objetivo 1*

| Resultado | Herramienta o método | Forma de validación |
|---|---|---|
| R1.1. Base de conocimiento RAG | Generación aumentada por recuperación (RAG) sobre una base de datos vectorial (ChromaDB), con un modelo de embeddings para indexar documentación técnica y arquitecturas de referencia | Ejecución de consultas de prueba contra la base de conocimiento, midiendo precisión y exhaustividad de los documentos recuperados frente a un conjunto de referencia etiquetado manualmente |
| R1.2. Conjunto acotado de configuraciones candidatas | Agente generador de configuraciones basado en un modelo de lenguaje (LLM), orquestado como grafo de estados con LangGraph, que combina la salida del componente RAG con el catálogo de arquitecturas preentrenadas de Keras Applications | Comparación cuantitativa entre el tamaño del espacio de búsqueda total considerado y el del subconjunto de configuraciones generado, más revisión cualitativa de su pertinencia con el asesor |
| R1.3. Reporte de criterios de acotación | Generación automática de reportes estructurados (plantillas en Markdown/JSON) a partir de los registros (logs) de ejecución del agente generador | Verificación automatizada (test suite) de que el reporte se genera correctamente en el 100 % de las ejecuciones de prueba |

La base de conocimiento (R1.1) se apoya en el paradigma de generación aumentada por recuperación (RAG), que combina un modelo de lenguaje preentrenado con un mecanismo de recuperación sobre una memoria no paramétrica externa para condicionar sus respuestas en información explícita y actualizable (Lewis et al., 2020). Para implementar esta memoria no paramétrica se empleará ChromaDB, una base de datos vectorial de código abierto orientada a aplicaciones de IA que permite almacenar e indexar *embeddings* de documentos y recuperarlos por similitud semántica (Chroma, s.f.). Sobre esta base de conocimiento se construye el agente generador de configuraciones (R1.2), coordinado mediante LangGraph, un framework para construir aplicaciones multiagente con estado como grafos cíclicos de nodos y aristas, lo que permite modelar bucles de razonamiento y decisión propios de un agente (LangChain AI, s.f.).

El catálogo de arquitecturas candidatas se ancla en Keras Applications, el módulo de modelos preentrenados de la librería Keras utilizado como referencia de modelo zoo para clasificación de imágenes (Chollet, 2015). Finalmente, el reporte de criterios de acotación (R1.3) se obtiene generando automáticamente, al final de cada ejecución, un documento estructurado con los registros del agente, siguiendo el mismo principio de trazabilidad documental que sustenta el protocolo de revisión sistemática aplicado en el capítulo 3 (Kitchenham & Charters, 2007).

En la Tabla 7 se describe brevemente las herramientas o métodos seleccionados del objetivo 2.

**Tabla 7**

*Tabla de métodos y herramientas para los resultados del objetivo 2*

| Resultado | Herramienta o método | Forma de validación |
|---|---|---|
| R2.1. Agente de entrenamiento automático | Entrenamiento supervisado de modelos de clasificación de imágenes con TensorFlow/Keras, ejecutado por un agente orquestado con LangGraph, combinando GPU local (NVIDIA GTX 1650) y servicios en la nube. | Revisión de los registros (logs) de ejecución para determinar el porcentaje de configuraciones entrenadas sin intervención manual del usuario |
| R2.2. Agente de evaluación de modelos | Cálculo de métricas estandarizadas de desempeño (exactitud, F1-score, matriz de confusión) con la librería scikit-learn | Verificación de que las métricas de desempeño quedan registradas para el 100 % de los modelos entrenados, mediante tablas comparativas generadas automáticamente |
| R2.3. Mecanismo de iteración automática | Política de decisión agéntica apoyada en optimización de hiperparámetros mediante búsqueda bayesiana con Optuna, incluyendo descarte temprano tipo Hyperband | Revisión del historial de iteraciones del sistema, contabilizando el número de iteraciones ejecutadas sin intervención humana |

El agente de entrenamiento (R2.1) se apoyará en TensorFlow/Keras, framework de referencia para la construcción y el entrenamiento de redes neuronales profundas (Chollet, 2015), ejecutado localmente sobre una GPU NVIDIA GTX 1650 para las configuraciones más ligeras y complementado con cómputo en la nube para los modelos y tareas de razonamiento agéntico de mayor exigencia, según el diseño de arquitectura ya definido para el proyecto. El agente de evaluación (R2.2) empleará scikit-learn, librería de Python ampliamente utilizada para el cálculo estandarizado de métricas de aprendizaje automático como exactitud y F1-score (Pedregosa et al., 2011), lo que garantiza que todas las configuraciones candidatas se comparen bajo el mismo criterio objetivo. El mecanismo de iteración automática (R2.3) integrará Optuna, un framework de optimización de hiperparámetros de próxima generación que permite definir de forma programática el espacio de búsqueda y podar automáticamente las configuraciones poco prometedoras (Akiba et al., 2019), apoyado en el principio de asignación adaptativa de recursos de Hyperband, que descarta tempranamente las configuraciones de peor desempeño parcial para concentrar el cómputo en las más prometedoras (Li et al., 2018); este mismo principio de descarte temprano se reutiliza en el objetivo 4 (R4.1) para optimizar el consumo de recursos.

En la Tabla 8 se describe brevemente las herramientas o métodos seleccionados del objetivo 3.

**Tabla 8**

*Tabla de métodos y herramientas para los resultados del objetivo 3*

| Resultado | Herramienta o método | Forma de validación |
|---|---|---|
| R3.1. Interfaz de carga de dataset | Prototipo de interfaz web (React/Node.js) para carga de datasets y definición de clases, evaluado con la Escala de Usabilidad del Sistema (SUS), incluyendo una advertencia sobre datos personales | Pruebas de usabilidad con usuarios sin experiencia en ML, midiendo el porcentaje que completa la carga sin asistencia técnica y el puntaje SUS obtenido |
| R3.2. Reporte final comprensible | Generación de lenguaje natural (NLG) mediante un LLM que traduce las métricas técnicas y la configuración elegida a una explicación en lenguaje no técnico, siguiendo la estructura de reporte de un model card | Encuestas de comprensión aplicadas a usuarios de prueba tras leer el reporte, midiendo el porcentaje que declara comprenderlo sin apoyo técnico adicional |
| R3.3. Artefacto final listo para despliegue | Contenerización del modelo entrenado con Docker y exposición como servicio mediante una API construida con FastAPI | Despliegue del contenedor en un entorno de prueba (staging) y ejecución de pruebas funcionales (smoke tests) sobre los endpoints de la API |

La interfaz de carga de dataset (R3.1) se validará con la Escala de Usabilidad del Sistema (SUS), un cuestionario breve y de aplicación rápida, ampliamente utilizado en ingeniería de software para obtener una medida confiable de la usabilidad percibida de un sistema con un número reducido de usuarios de prueba (Brooke, 1996). Para el reporte final comprensible (R3.2) se aprovechará la capacidad de generación de lenguaje natural de un LLM para traducir las métricas técnicas y la configuración seleccionada a un lenguaje accesible para personas sin formación en ML/IA, adoptando como referencia estructural el formato de *model cards*, propuesto para documentar de forma estandarizada las características, el desempeño y las limitaciones de un modelo entrenado (Mitchell et al., 2019). El artefacto final (R3.3) se empaquetará con Docker, la plataforma estándar de contenerización de aplicaciones que garantiza que el modelo se ejecute de manera reproducible en cualquier entorno sin configuración manual adicional (Docker Inc., s.f.), exponiéndose como servicio mediante FastAPI, un framework de Python para construir APIs de alto rendimiento con documentación autogenerada (Ramírez, s.f.), lo que facilita las pruebas funcionales de validación.

En la Tabla 9 se describe brevemente las herramientas o métodos seleccionados del objetivo 4.

**Tabla 9**

*Tabla de métodos y herramientas para los resultados del objetivo 4*

| Resultado | Herramienta o método | Forma de validación |
|---|---|---|
| R4.1. Estrategia de priorización/descarte temprano | Descarte temprano y asignación adaptativa de recursos tipo Hyperband, implementado sobre Optuna | Cálculo del porcentaje de reducción en el número de entrenamientos completos frente a una búsqueda exhaustiva de referencia |
| R4.2. Registro de consumo de recursos | Instrumentación y monitoreo de cómputo (tiempo, uso de CPU/GPU) mediante herramientas de perfilado como psutil y utilidades de monitoreo de GPU (nvidia-smi) | Revisión de la completitud de los registros (logs) de consumo de recursos por cada configuración evaluada en cada ejecución |
| R4.3. Reporte comparativo de consumo de recursos | Metodología de benchmarking comparando el pipeline automatizado frente a un proceso manual/exhaustivo tomado como línea base (baseline) | Elaboración de un reporte comparativo con el porcentaje de reducción de tiempo y/o recursos computacionales, revisado por el asesor |

La estrategia de priorización y descarte temprano (R4.1) reutiliza el mecanismo de Hyperband implementado sobre Optuna descrito para R2.3 (Li et al., 2018; Akiba et al., 2019), aplicándolo esta vez con el objetivo explícito de reducir el número de entrenamientos completos necesarios y no solo de mejorar la calidad de la configuración final. El registro de consumo de recursos (R4.2) se obtendrá instrumentando cada ejecución con herramientas de perfilado de uso común en Python, como psutil para el monitoreo de CPU y memoria, y las utilidades de línea de comandos del fabricante para el monitoreo de uso de GPU, de forma que cada configuración evaluada quede asociada a su tiempo de cómputo y su consumo de recursos. Finalmente, el reporte comparativo (R4.3) se construirá aplicando una metodología de *benchmarking* que contrasta el desempeño y el consumo de recursos del pipeline automatizado frente a un proceso de búsqueda manual o exhaustivo tomado como línea base, en línea con la práctica reportada en la literatura de comparar explícitamente automatización agéntica contra búsqueda exhaustiva para cuantificar la ganancia de eficiencia (Pchelnikov & Pchelnikov, 2026; Yang et al., 2025).

# Capítulo 2. Marco Legal/Regulatorio/Conceptual/otros

## 2.1 Introducción

El objetivo de este capítulo es presentar los marcos legal, regulatorio y conceptual que delimitan y fundamentan la propuesta descrita en la problemática (sección 1.1) y en los objetivos (sección 1.2). Estos marcos cumplen tres funciones complementarias. Primero, el marco legal (sección 2.2) identifica las normas peruanas de cumplimiento obligatorio que condicionan el tratamiento de los datasets etiquetados y el uso de sistemas de inteligencia artificial por parte de las pymes descritas en la problemática, de modo que la arquitectura propuesta pueda diseñarse siendo consciente de dichas obligaciones desde el inicio y no como un ajuste posterior. Segundo, el marco regulatorio (sección 2.3) recoge los estándares técnicos y las prácticas de gobernanza de IA, de adopción voluntaria pero cada vez más exigidos por clientes y reguladores, que orientan cómo debe documentarse y hacerse trazable un modelo automáticamente seleccionado y entrenado, en línea con la ausencia de personal especializado identificada como una de las causas del problema central (sección 1.1.1). Tercero, el marco conceptual (sección 2.4) define de manera precisa los términos técnicos empleados a lo largo de la problemática y de los objetivos específicos (AutoML, búsqueda de arquitecturas neuronales, optimización de hiperparámetros, arquitecturas multiagente, RAG, gobernanza de modelos y pymes), de forma que exista un vocabulario común y sin ambigüedades para el resto de la tesis.

## 2.2 Desarrollo del marco legal

El marco legal reúne las normas peruanas de cumplimiento obligatorio directamente relacionadas con los dos elementos centrales de la problemática: el uso de un *dataset* etiquetado como insumo del sistema (sección 1.1, párrafo 4) y la adopción de un sistema de inteligencia artificial por parte de una pyme cuyo *core* de negocio no es la IA/ML (sección 1.1.3).

### 2.2.1 Protección de datos personales

La Ley N° 29733, Ley de Protección de Datos Personales, tiene por objeto garantizar el derecho fundamental a la protección de los datos personales reconocido en el artículo 2, numeral 6, de la Constitución Política del Perú, y se aplica a todo tratamiento de datos personales realizado en el país, sea automatizado o no (Congreso de la República del Perú, 2011). Su reglamento vigente, aprobado mediante Decreto Supremo N° 016-2024-JUS, actualizó el régimen de 2013 para reforzar la seguridad, la transparencia y la responsabilidad en el tratamiento de datos, ampliar su alcance territorial a empresas extranjeras que ofrezcan bienes o servicios a personas en el país, y reconocer a los titulares el derecho a conocer la finalidad del tratamiento, acceder a su información, rectificarla y solicitar su eliminación, entre otros (CMS, 2025). La entidad encargada de velar por su cumplimiento es la Autoridad Nacional de Protección de Datos Personales, ejercida por el Ministerio de Justicia y Derechos Humanos (Congreso de la República del Perú, 2011). Esta norma es relevante para la problemática porque el dataset etiquetado que la pyme aporta al sistema (sección 1.1, párrafo 4) no necesariamente está compuesto por imágenes neutras: en casos de uso frecuentes en pymes, como el control de calidad en planta, el monitoreo de anaqueles en retail o la clasificación de residuos, el dataset puede incluir, de forma incidental, rostros de trabajadores o de clientes captados por las cámaras del proceso. Por ejemplo, una pyme que entrena un clasificador de defectos de producción a partir de fotografías de su línea de ensamblaje puede incluir sin darse cuenta a operarios en el encuadre de las imágenes, lo que convierte a ese dataset en un banco de datos personales sujeto a la Ley N° 29733 y obliga a aplicar medidas como el anonimizado o el recorte de las imágenes antes de indexarlas en el sistema. Dado que estas pymes carecen, según la problemática (causa "ausencia de personal especializado en ML/IA dentro de la organización", sección 1.1.1), de personal técnico propio, es poco probable que cuenten con criterios claros para identificar cuándo su dataset cae bajo esta norma, lo que refuerza la necesidad de que el propio sistema (en particular, el mecanismo de interacción de O3, sección 1.2.2) incorpore advertencias o validaciones básicas sobre el contenido del *dataset* cargado.

### 2.2.2 Ley que promueve el uso de la Inteligencia Artificial

La Ley N° 31814, Ley que promueve el uso de la inteligencia artificial en favor del desarrollo económico y social del país, tiene por objeto promover el uso de la IA en el marco del proceso de transformación digital del país, privilegiando el respeto de los derechos humanos y garantizando un entorno de uso ético, sostenible, transparente, replicable y responsable (Congreso de la República del Perú, 2023). La Ley establece seis principios rectores para el desarrollo y uso de la IA: estándares de seguridad basados en riesgos, pluralidad de participantes, gobernanza de internet, sociedad digital, desarrollo ético para una IA responsable, y privacidad de la inteligencia artificial (Congreso de la República del Perú, 2023). Su reglamento, aprobado mediante Decreto Supremo N° 115-2025-PCM el 9 de setiembre de 2025, desarrolla estos principios en seis títulos y 36 artículos, estableciendo una clasificación de los sistemas de IA según su nivel de riesgo ("uso indebido" y "uso de alto riesgo"), así como obligaciones de transparencia algorítmica, supervisión humana y protección de datos personales tanto para entidades públicas como privadas (Gestión, 2025; LP Derecho, 2025). La Presidencia del Consejo de Ministros, a través de la Secretaría de Gobierno y Transformación Digital, es la autoridad técnico-normativa nacional encargada de dirigir, evaluar y supervisar el uso de la IA en el país (Forbes Perú, 2023). Este marco se relaciona directamente con la causa organizacional de "ausencia de personal especializado en ML/IA dentro de la organización" y con el objetivo específico O3 de la problemática (secciones 1.1.1 y 1.2.2): una pyme sin conocimientos especializados en IA difícilmente podrá determinar por sí misma si el sistema de clasificación de imágenes que va a adoptar corresponde a un uso de "alto riesgo" bajo el reglamento (por ejemplo, si el modelo resultante se usara para tamizar candidatos a un puesto de trabajo a partir de fotografías, o para decisiones de control de calidad con impacto en la seguridad del producto), y por tanto tampoco podrá cumplir por sí sola con las obligaciones de supervisión humana y transparencia que ese nivel de riesgo exigiría. Esto justifica que el reporte no técnico contemplado como resultado R3.2 (sección 1.2.3) no se limite a explicar el desempeño del modelo, sino que además deje constancia trazable de los criterios de selección y de las limitaciones del modelo entrenado, de forma que la pyme cuente con evidencia mínima de supervisión y transparencia frente a la autoridad competente si su caso de uso llegara a clasificarse como de alto riesgo.

## 2.3 Desarrollo del marco regulatorio

El marco regulatorio reúne los estándares técnicos y las prácticas de gobernanza de IA de adopción voluntaria que, sin tener fuerza de ley en el Perú, definen buenas prácticas internacionalmente reconocidas para la documentación y trazabilidad de modelos de aprendizaje automático, un aspecto explícitamente mencionado como resultado esperado del proyecto (sección 1.2.3, resultados R1.3 y R3.2) y como acuerdo con el asesor respecto de los entregables finales de la tesis.

### 2.3.1 Sistema de gestión de la inteligencia artificial

La norma ISO/IEC 42001:2023 es el primer estándar internacional que especifica los requisitos para establecer, implementar, mantener y mejorar continuamente un sistema de gestión de la inteligencia artificial (AIMS, por sus siglas en inglés) dentro de una organización, aplicable a cualquier entidad que provea o utilice productos o servicios basados en IA, independientemente de su tamaño (International Organization for Standardization & International Electrotechnical Commission, 2023). El estándar aborda de manera estructurada la gestión de riesgos, la transparencia y la trazabilidad de los sistemas de IA, mediante siete cláusulas operativas y un conjunto de controles organizados en un anexo (Hansen, 2026). Este estándar es relevante para la problemática porque ofrece un marco de referencia externo y ya validado internacionalmente para lo que en esta tesis se define como "gobernanza de modelos" (sección 2.4): una pyme que adopte el sistema propuesto podría, en el futuro, apoyarse en el reporte de criterios de acotación (R1.3) y en el reporte final comprensible (R3.2) generados automáticamente como insumos iniciales para acreditar frente a un cliente o auditor que su proceso de selección de modelos sigue una lógica documentada y trazable, en lugar de partir de cero.

### 2.3.2 Documentación estandarizada de modelos y datasets

Ante la ausencia de procedimientos estandarizados para comunicar las características de desempeño de un modelo entrenado, Mitchell et al. (2019) proponen el formato de *model cards*: documentos breves que acompañan a un modelo entrenado y detallan su uso previsto, los factores relevantes para su desempeño, las métricas de evaluación empleadas y sus limitaciones conocidas. De manera complementaria, Gebru et al. (2018) proponen las *datasheets for* *datasets*, un formato equivalente orientado a documentar el origen, la composición y las limitaciones de un conjunto de datos. Ambos formatos son directamente aplicables al reporte final comprensible definido como resultado R3.2 (sección 1.2.3): en lugar de diseñar desde cero la estructura de dicho reporte, el sistema propuesto adoptará como referencia estructural el esquema de secciones de una model card (detalles del modelo, uso previsto, factores relevantes, métricas y consideraciones éticas), adaptando su redacción a un lenguaje comprensible para usuarios sin conocimientos especializados en ML/IA, tal como lo exige el objetivo específico O3. **2.3.3 Enfoque basado en riesgos para la gestión de sistemas de IA** De manera consistente con el principio de "estándares de seguridad basados en riesgos" de la Ley N° 31814 (sección 2.2.2), diversos marcos internacionales de gobernanza de IA, como el NIST AIRisk Management Framework, promueven clasificar los sistemas de IA según su nivel de riesgo e impacto, y ajustar en consecuencia el nivel de supervisión humana, documentación y controles exigidos (National Institute of Standards and Technology, 2023). Este enfoque respalda la decisión de diseño de que el sistema propuesto no trate a todos los casos de uso de la pyme por igual, sino que el reporte generado (R3.2) explicite el nivel de riesgo aparente del caso de uso declarado por el usuario (por ejemplo, distinguiendo un clasificador de defectos de manufactura de un clasificador que involucre datos biométricos de personas), de modo que la pyme reciba una señal temprana sobre el nivel de supervisión humana que debería mantener sobre el modelo resultante.

## 2.4 Desarrollo del marco conceptual

El marco conceptual define los términos técnicos centrales empleados en la problemática (sección 1.1) y en los objetivos específicos (sección 1.2.2), con el fin de fijar un vocabulario común para el resto de la tesis.

### 2.4.1 Contexto organizacional

#### 2.4.1.1 Pequeña y mediana empresa (pyme)

En el Perú, la clasificación empresarial vigente, establecida por la Ley N° 30056 (que modificó la Ley N° 28015) y consolidada en el Texto Único Ordenado de la Ley de Impulso al Desarrollo Productivo y al Crecimiento Empresarial, aprobado por Decreto Supremo N° 013-2013- PRODUCE, clasifica a las empresas según su nivel de ventas anuales expresado en Unidades Impositivas Tributarias (UIT): microempresa (ventas anuales de hasta 150 UIT), pequeña empresa (más de 150 y hasta 1700 UIT) y mediana empresa (más de 1700 y hasta 2300 UIT) (Instituto Nacional de Estadística e Informática, 2013). Esta definición formaliza el término "pymes" empleado en la problemática (sección 1.1) para acotar el tipo de organización objetivo de la solución: empresas que, por su nivel de ventas y de personal, típicamente no cuentan con un área de investigación y desarrollo (I+D) propia en inteligencia artificial.

### 2.4.2 Automatización del aprendizaje automático

#### 2.4.2.1 Aprendizaje automático automatizado (AutoML)

Es el campo orientado a automatizar el pipeline completo de construcción de un modelo de aprendizaje automático, desde el preprocesamiento de los datos hasta la selección de la arquitectura y el ajuste de hiperparámetros, con el fin de reducir la dependencia de conocimiento técnico especializado (He et al., 2021; Karmaker Santu et al., 2022). Este concepto, ya introducido en la problemática (sección 1.1, párrafo 2) como la respuesta histórica al costo manual de construir modelos, es el campo en el que se inserta la arquitectura multiagente propuesta en esta tesis.

#### 2.4.2.2 Búsqueda de arquitecturas neuronales (Neural Architecture Search, NAS)

Es una subdisciplina del AutoML orientada a automatizar el diseño de la arquitectura de una red neuronal para una tarea determinada, mediante estrategias como el aprendizaje por refuerzo, la optimización bayesiana, los algoritmos evolutivos o la optimización diferenciable (Elsken et al., 2019). En esta tesis, el mecanismo de acotación del espacio de configuraciones candidatas (resultado R1.2, sección 1.2.3) puede entenderse como una forma de acotar, mediante RAG, el espacio de búsqueda que en NAS suele explorarse mediante estas estrategias de búsqueda automatizada.

#### 2.4.2.3 Optimización de hiperparámetros (Hyperparameter Optimization, HPO)

Es el proceso de encontrar la combinación de hiperparámetros de entrenamiento (por ejemplo, tasa de aprendizaje, tamaño de lote o número de épocas) que maximiza el desempeño de un modelo, mediante técnicas que van desde la búsqueda exhaustiva (*grid search*) hasta métodos más eficientes como la optimización bayesiana o la asignación adaptativa de recursos de tipo Hyperband (Li et al., 2018; Xue et al., 2025). Esta noción sustenta directamente el mecanismo de iteración automática (R2.3) y la estrategia de descarte temprano (R4.1) descritos en la sección 1.3.

### 2.4.3 Arquitecturas basadas en modelos de lenguaje

#### 2.4.3.1 Arquitectura multiagente

Un agente de IA es una entidad capaz de percibir su entorno y actuar sobre él de manera autónoma para alcanzar un objetivo (Wooldridge, 2009); un sistema multiagente coordina a varios de estos agentes, cada uno con un rol especializado, para resolver conjuntamente una tarea que sería más difícil de abordar mediante un único agente monolítico. En el dominio de esta tesis, este concepto corresponde a arquitecturas como AutoML-Agent, que emplea agentes especializados para traducir requerimientos informales del usuario en tareas ejecutables de AutoML (Trirat et al., 2025), y es la base de la solución propuesta en el objetivo general (sección 1.2.1): un sistema en el que agentes especializados (generación de configuraciones, entrenamiento, evaluación e iteración) coordinan de forma autónoma el ciclo completo descrito en los objetivos O1 y O2. **2.4.3.2 Generación aumentada por recuperación (RAG)** La generación aumentada por recuperación (RAG - *Retrieval-Augmented Generation*) es una técnica que combina un modelo de lenguaje preentrenado (memoria paramétrica) con un mecanismo de recuperación sobre una base de conocimiento externa (memoria no paramétrica), de forma que las respuestas del modelo se condicionen en información explícita, verificable y actualizable sin necesidad de reentrenarlo (Lewis et al., 2020). Este concepto sustenta el resultado R1.1 (sección 1.2.3) y es, junto con las arquitecturas multiagente, uno de los dos pilares técnicos mencionados explícitamente en el objetivo general de la tesis (sección 1.2.1). **2.4.4 Trazabilidad y gobernanza 2.4.4.1 Gobernanza de modelos** En el contexto de esta tesis, la gobernanza de modelos se entiende en un sentido técnico de trazabilidad y auditabilidad: la capacidad de determinar, para cualquier modelo producido por el sistema, qué configuración fue seleccionada, con qué criterios y bajo qué evaluación de desempeño, sin que ello implique necesariamente el cumplimiento de un marco regulatorio de gobernanza de IA en sentido amplio. Esta noción se apoya en la lógica de documentación estandarizada de las *model cards* (Mitchell et al., 2019) y en los principios de trazabilidad de ISO/IEC 42001:2023 (sección 2.3.1), y se materializa de forma concreta en los resultados R1.3 y R3.2 (sección 1.2.3), que constituyen los artefactos documentales trazables del sistema propuesto.

# Capítulo 3. Estado del Arte

## 3.1 Introducción

Este capítulo presenta el protocolo de revisión sistemática de literatura (RSL) diseñado para construir el estado del arte que sustentará el diseño de la solución propuesta. El protocolo sigue las guías de Kitchenham y Charters (2007) para revisiones sistemáticas, y emplea el enfoque PICOC (Petticrew & Roberts, 2006) para estructurar tanto las preguntas de revisión como la estrategia de búsqueda. El propósito de esta revisión no es evaluar la ausencia de una aplicación

- sistema informático puntual, sino identificar y analizar de forma sistemática los enfoques, arquitecturas y técnicas ya reportados en la literatura para automatizar la construcción y selección de modelos de clasificación de imágenes, así como el uso de arquitecturas multiagente y de generación aumentada por recuperación (RAG) en este tipo de tareas. Los resultados de este capítulo alimentarán directamente el diseño de la arquitectura propuesta en los capítulos posteriores, en particular su alineamiento con los objetivos específicos O1 y O2 planteados en la sección 1.2.2.

## 3.2 Objetivos de revisión

Con el fin de construir el estado del arte que sustente el diseño de la arquitectura propuesta, se definieron los siguientes objetivos de revisión, alineados con las causas técnicas identificadas en el árbol de problemas (sección 1.1.1) y con los objetivos específicos del proyecto (sección 1.2.2):

- **OR1.** Identificar los enfoques y arquitecturas existentes para automatizar la construcción y selección de modelos de clasificación de imágenes a partir de un dataset etiquetado.

- **OR2.** Identificar cómo se han empleado arquitecturas multiagente combinadas con generación aumentada por recuperación (RAG) para automatizar tareas de aprendizaje automático, y qué ventajas o limitaciones reportan frente a enfoques que no emplean agentes.

- **OR3.** Identificar cómo los sistemas o arquitecturas existentes para automatizar la construcción y selección de modelos de clasificación de imágenes abordan su uso por parte de usuarios o empresas sin conocimientos especializados en ML/IA.

## 3.3 Preguntas de revisión

A partir de los objetivos de revisión anteriores, se formularon las siguientes preguntas de revisión, las cuales se detallan en la Tabla 10.

**Tabla 10**

*Tabla de alineamiento entre objetivos de revisión y preguntas de revisión*

| Código | Objetivo específico de revisión | Pregunta asociada |
|---|---|---|
| OR1 | Identificar los enfoques y arquitecturas existentes para automatizar la construcción y selección de modelos de clasificación de imágenes a partir de un dataset etiquetado | P1. ¿Qué arquitecturas o enfoques han sido propuestos en la literatura para automatizar la construcción y selección de modelos de visión por computadora (con énfasis en clasificación de imágenes) a partir de un dataset etiquetado? |

| OR2 | Identificar el uso de arquitecturas multiagente combinadas con RAG para automatizar tareas de aprendizaje automático, y sus ventajas/limitaciones frente a enfoques no agénticos | P2. ¿Cómo se han empleado arquitecturas agénticas (multiagente o basadas en LLM) y/o técnicas de generación aumentada por recuperación (RAG) para automatizar tareas de aprendizaje automático y visión por computadora, y qué ventajas o limitaciones se reportan frente a enfoques no agénticos? |
|---|---|---|
| OR3 | Identificar cómo los sistemas o arquitecturas existentes para automatizar la construcción y selección de modelos de clasificación de imágenes abordan su uso por parte de usuarios o empresas sin conocimientos especializados en ML/IA | P3. ¿Qué estrategias, enfoques o mecanismos de interacción han propuesto los sistemas automatizados de construcción y selección de modelos de visión por computadora para facilitar su uso por parte de usuarios o empresas sin conocimientos especializados en ML/IA? |

## 3.4 Estrategia de búsqueda

La estrategia de búsqueda tiene como objetivo responder de manera coherente a las tres preguntas de revisión planteadas en la sección 3.3. La construcción de la cadena de búsqueda se apoya en el enfoque PICOC (Población, Intervención, Comparación, Resultado y Contexto), técnica ampliamente utilizada en revisiones sistemáticas para estructurar preguntas de investigación y construir cadenas de búsqueda de forma organizada y trazable (Petticrew & Roberts, 2006). En este trabajo, el enfoque PICOC permite organizar los términos en bloques conceptuales que representan la población de modelos de clasificación de imágenes a optimizar, la intervención basada en arquitecturas multiagente y RAG para su automatización, la comparación con métodos tradicionales de selección de modelos, los resultados orientados a mejorar el rendimiento y la facilidad de uso, y el contexto de aplicación en organizaciones sin áreas especializadas en inteligencia artificial. En la Tabla 11 se detallan los términos para la construcción de la cadena de búsqueda, empleando el enfoque descrito.

**Tabla 11**

*Tabla de criterios conceptuales usados para construir la cadena de búsqueda*

| Criterio | Significado en la revisión | Términos relacionados |
|---|---|---|
| Población | Modelos y procesos de clasificación de imágenes basados en aprendizaje profundo, sobre los cuales se aplica algún tipo de automatización | "image classification", "image classifier", "image recognition", "deep learning model" |
| Intervención | Técnicas y arquitecturas orientadas a automatizar la | "agent", "multi-agent", "multiagent", "AutoML", "automated machine learning", "neural architecture search", |

|  | construcción, selección o ajuste de modelos: arquitecturas multiagente, generación aumentada por recuperación (RAG), AutoML y búsqueda de arquitecturas neuronales | "retrieval-augmented generation", "RAG", "large language model" |
|---|---|---|
| Comparación | Enfoques tradicionales o manuales frente a los cuales se contrastan las técnicas de automatización identificadas | "manual model selection", "traditional machine learning pipeline", "hyperparameter optimization", "grid search", "random search" |
| Resultado | Efectos esperados de automatizar el proceso: desempeño del modelo resultante, grado de automatización alcanzado, y nivel de accesibilidad o facilidad de uso para personas sin | "model accuracy", "performance metrics", "degree of automation", "ease of use", "usability", "no-code", "low- code", "human-in-the-loop" |

|  | conocimientos especializados en ML/IA |  |
|---|---|---|
| Contexto | Organizaciones y usuarios que aplican clasificación de imágenes sin contar con un área especializada en IA/ML como núcleo de negocio | "computer vision applications", "small and medium enterprises", “SMEs”, "non-expert users", "domain experts", "practical application" |

### 3.4.1 Motores de búsqueda a usar

Se utilizaron ACM Digital Library, Google Académico, y arXiv como motores de búsqueda principales.

- **ACM Digital Library**, por ser la base de datos bibliográfica de referencia en computación, que reúne el texto completo de revistas, actas de conferencias y publicaciones técnicas de la Association for Computing Machinery y otros editores del área, incluyendo comunidades activas en inteligencia artificial, aprendizaje automático e interacción humano-computador (Association for Computing Machinery, s.f.).

- **Google Académico**, porque complementa la cobertura de las bases de datos indexadas al incluir preprints, tesis y publicaciones recientes que aún podrían no estar catalogadas en bases más curadas, lo que resulta útil dado el carácter emergente del tema (Martín- Martín et al., 2018).

- **arXiv**, por tratarse de un repositorio de acceso abierto de preprints en ciencia de la computación, inteligencia artificial y áreas afines, que permite acceder de forma rápida a investigaciones recientes (incluyendo trabajos aún no publicados formalmente en revistas o conferencias indexadas), lo cual resulta especialmente relevante dado el ritmo acelerado de publicación en el campo de los sistemas multiagente, los modelos de lenguaje y RAG (Cornell University, s.f.).

**Nota sobre motores descartados:** inicialmente se consideraron también Scopus, debido a su cobertura multidisciplinaria en ciencia de la computación, inteligencia artificial, ingeniería de software y sistemas de información (Elsevier, 2023), e IEEE Xplore, por su relevancia en ingeniería, computación y aplicaciones de inteligencia artificial (IEEE, s.f.). Sin embargo, al ejecutar en ellos la cadena de búsqueda adaptada, la cual se expone en la Tabla 12, no se obtuvieron resultados relevantes para el tema de esta revisión, por lo que se optó por reemplazarlos por ACM Digital Library y Google Académico. Se deja constancia de este hecho por transparencia y reproducibilidad del protocolo de búsqueda (Kitchenham & Charters, 2007).

### 3.4.2 Cadena de búsqueda a usar

La cadena de búsqueda propuesta tiene como objetivo identificar estudios relevantes sobre la automatización de la clasificación de imágenes mediante arquitecturas multiagente y técnicas de Machine Learning. Su construcción se fundamenta en los bloques conceptuales articulados a través del enfoque PICOC, incorporando términos vinculados al dominio del problema, los mecanismos de automatización (sistemas multiagente y AutoML), los métodos de selección tradicional y el contexto de aplicación en pymes o usuarios no especialistas. Aspectos como RAG y la usabilidad no formaron parte de la ecuación de búsqueda, sino que emergieron inductivamente de la lectura de los estudios. En este sentido, la cadena fue formulada para responder de manera integrada a las preguntas de revisión planteadas en el estudio.

**Tabla 12**

*Tabla de cadena de búsqueda lista para su ejecución*

| Uso | Cadena booleana propuesta |
|---|---|
| Cadena de búsqueda principal | ("image classification" OR "image classifier" OR "image recognition" OR "deep learning model") AND ("agent" OR "multi-agent" OR "large language model") AND ("AutoML" OR "automated machine learning" OR "neural architecture search") AND ("manual model selection" OR "hyperparameter optimization" OR "grid search") AND ("small and medium enterprises" OR "SMEs" OR "small business" OR "non-expert users" OR "computer vision applications" OR "domain experts" OR "practical application") |

La cadena se adapta a la sintaxis específica de cada motor de búsqueda, aplicando los mismos bloques conceptuales del enfoque PICOC. Se empleó el campo de búsqueda predeterminado ("*All*"), el cual ejecuta un rastreo exhaustivo en la base de datos al examinar simultáneamente todos los metadatos del registro (título, resumen, autores, palabras clave) y el texto completo de las publicaciones cuando se encuentra disponible. En Google Académico, la cadena de búsqueda principal se pudo ejecutar íntegramente conservando todos sus términos y bloques conceptuales originales. No obstante, para adaptarse al formato de texto libre y a las restricciones de sintaxis de la plataforma, se omitieron los operadores booleanos AND explícitos entre los paréntesis, ya que el motor aplica automáticamente la intersección AND de forma implícita al yuxtaponer las agrupaciones de términos (Gusenbauer & Haddaway, 2020). Finalmente, en arXiv se utiliza la modalidad de búsqueda por texto completo ("*Full text*"). Esta opción analiza simultáneamente el título, el resumen y la totalidad del texto del artículo, lo que resulta fundamental para recuperar literatura reciente e investigaciones en formato *preprint*.

En la Tabla 13 se muestra la cadena de búsqueda adaptada para los tres motores seleccionados.

**Tabla 13**

*Tabla de cadenas de búsqueda adaptadas para las bases de datos consultadas*

| Motor | Cadena |
|---|---|
| ACM Digital Library | [[All: "image classification"] OR [All: "image classifier"] OR [All: "image recognition"] OR [All: "deep learning model"]] AND [[All: "agent"] OR [All: "multi-agent"] OR [All: "large language model"]] AND [[All: "automl"] OR [All: "automated machine learning"] OR [All: "neural architecture search"]] AND [[All: "manual model selection"] OR [All: "hyperparameter optimization"] OR [All: "grid search"]] AND [[All: "small and medium enterprises"] OR [All: "smes"] OR [All: "small business"] OR [All: "non- expert users"] OR [All: "computer vision applications"] OR [All: "domain experts"] OR [All: "practical application"]] |
| Google Académico | ("image classification" OR "image classifier" OR "image recognition" OR "deep learning model") ("agent" OR "multi-agent" OR "large language model") ("AutoML" OR "automated machine learning" OR "neural architecture search") ("manual model selection" OR "hyperparameter optimization" OR "grid search") ("small and medium enterprises" OR "SMEs" OR "small business" OR "non-expert users" OR "computer vision applications" OR "domain experts" OR "practical application") |
| arXiv | ("image classification" OR "image classifier" OR "image recognition" OR "deep learning model") AND ("agent" OR "multi-agent" OR "large language |

|  | model") AND ("AutoML" OR "automated machine learning" OR "neural architecture search") AND ("manual model selection" OR "hyperparameter optimization" OR "grid search") AND ("small and medium enterprises" OR "SMEs" OR "small business" OR "non-expert users" OR "computer vision applications" OR "domain experts" OR "practical application") |
|---|---|

### 3.4.3 Documentos encontrados

Tras ejecutar la cadena de búsqueda adaptada en los tres motores seleccionados, se registra el número de documentos recuperados por cada uno, así como los resultantes tras la eliminación de duplicados. La Tabla 14 resume este registro de ejecución del protocolo de búsqueda.

**Tabla 14**

*Tabla de registro de ejecución del protocolo de búsqueda*

| Motor | Fecha de ejecución | Cadena utilizada | Documentos encontrados | Duplicados | Seleccionados tras filtro de duplicados |
|---|---|---|---|---|---|
| ACM Digital Library | 10/09/2026 | Cadena de búsqueda | 10 | 10 | 10 |
| Google Académico | 10/09/2026 | Cadena de búsqueda | 82 |  | 82 |
| arXiv | 10/09/2026 | Cadena de búsqueda | 124 |  | 114 |

Los criterios de inclusión y exclusión se aplicaron sobre los 206 documentos resultantes tras la eliminación de duplicados. El proceso de evaluación se ejecutó en dos fases: un filtrado preliminar por título, resumen y palabras clave para contrastar su pertinencia con las preguntas de revisión (P1, P2 y P3) y con el alcance del proyecto definido en la problemática (sección 1.1) y los objetivos (sección 1.2), seguido por la verificación detallada frente a los criterios definidos en las tablas 17 y 18, de la sección 3.4.4. Para cumplir con el alcance del proyecto, se conformó un conjunto de 14 estudios primarios que satisfacían los criterios de inclusión y exclusión planteados. La Tabla 15 resume la cantidad de artículos encontrados en cada etapa.

**Tabla 15**

*Tabla de artículos encontrados según las etapas de revisión*

| Etapa | Artículos |
|---|---|
| 1. Búsqueda inicial | 216 |
| 2. Después de la eliminación de duplicados | 206 |
| 3. Después de la aplicación de los criterios CI1–CI6 | 57 |
| 4. Después de la aplicación de los criterios CE1–CE3 | 14 |

La información detallada de cada trabajo se organiza en el formulario de extracción, por lo que la Tabla 16 presenta únicamente el registro bibliográfico de los estudios seleccionados.

**Tabla 16**

*Tabla de registro de estudios primarios*

| ID | Referencia APA | Motor |
|---|---|---|
| E1 | Zhang, Y., Zong, R., Shang, L., Kou, Z., Zeng, H., & Wang, D. (2022). CrowdOptim: A Crowd-driven Neural Network Hyperparameter Optimization Approach to AI-based Smart Urban Sensing. Proceedings of the ACM on Human-Computer Interaction, 6(CSCW2), Article 435. https://doi.org/10.1145/3555536 | ACM Digital Library |

| E2 | Santu, S. K. K., Hassan, M. M., Smith, M. J., Xu, L., Zhai, C., & Veeramachaneni, K. (2022). AutoML to Date and Beyond: Challenges and Opportunities. ACM Computing Surveys, 54(8), Article 175. https://doi.org/10.1145/3470918 | ACM Digital Library |
|---|---|---|
| E3 | Jan, N.-Y., Chen, W.-C., Lin, Y.-R., Wu, C. T., Liu, S.-H., & Li, Y.- H. (2026). SmartDS-Solver: Agentic AI for Vertical Domain Problem Solving in Data Science. Under review as a conference paper at ICLR 2026. https://openreview.net/pdf?id=r7gmePFADZ | Google Académico |
| E4 | Baratchi, M., Wang, C., Limmer, S., van Rijn, J. N., Hoos, H., Bäck, T., & Olhofer, M. (2024). Automated machine learning: Past, present and future. Artificial Intelligence Review, 57, 122. https://doi.org/10.1007/s10462-024-10726-1 | Google Académico |
| E5 | Wu, X., Huang, T.-Z., Deng, L.-J., Yu, X., Zhong, Y., Deng, S., Khan, U., Wu, J., Liu, X., Razzak, I., Chang, X., & Xie, Y. (2026). SelfAI: A self-directed framework for long-horizon scientific discovery. arXiv preprint arXiv:2512.00403. https://doi.org/10.48550/arXiv.2512.00403 | Google Académico |
| E6 | Wang, X., Horstmann, K. A., Lin, E., Chen, J., Farhang, A. R., Stiles, S., Sehgal, A., Light, J., Van Valen, D., Yue, Y., & Sun, J. J. (2026). Simple Agents Outperform Experts in Biomedical Imaging Workflow Optimization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026). https://doi.org/10.48550/arXiv.2512.06006 | arXiv |
| E7 | Pchelnikov, A., & Pchelnikov, A. (2026). CellDX AI Autopilot: Agent-Guided Training and Deployment of Pathology Classifiers. | arXiv |

|  | arXiv preprint arXiv:2605.10362. https://doi.org/10.48550/arXiv.2605.10362 |  |
|---|---|---|
| E8 | Fang, H., Han, B., Erickson, N., Zhang, X., Zhou, S., Dagar, A., Zhang, J., Turkmen, A. C., Hu, T., Rangwala, H., Wu, Y. N., Wang, Y. (Bernie), & Karypis, G. (2025). MLZero: A Multi-Agent System for End-to-end Machine Learning Automation. Advances in Neural Information Processing Systems, 38. https://doi.org/10.52202/085713-2322 | arXiv |
| E9 | Trirat, P., Jeong, W., & Hwang, S. J. (2025). AutoML-Agent: A Multi- Agent LLM Framework for Full-Pipeline AutoML. Proceedings of the 42nd International Conference on Machine Learning, 267, 60099– 60146. https://arxiv.org/abs/2410.02958 | arXiv |
| E10 | Meng, G., Gu, P., Li, X., Shi, Y., Chambers, E. W., & Chen, D. Z. (2026). TopoAgent: An Agentic Framework for Automated Topology Learning in Medical Imaging. arXiv preprint arXiv:2606.29763. https://doi.org/10.48550/arXiv.2606.29763 | arXiv |
| E11 | Vignesh, M., Santhoshkumar, S. P., Vikram, D., Gajalakshmi, S., Ramamoorthi, S., & Mohammed Uveise, S. A. (2025). Beyond Conventional Programming: Empirical Perspectives on AI Model Accuracy and Development Efficiency. 2025 International Conference on Emerging Research in Computational Science (ICERCS). https://doi.org/10.1109/ICERCS65898.2025.11580564 | Google Académico |
| E12 | Yang, Z., Zeng, W., Jin, S., Qian, C., Luo, P., & Liu, W. (2025). AutoMMLab: Automatically Generating Deployable Models from Language Instructions for Computer Vision Tasks. Proceedings of the | arXiv |

|  | AAAI Conference on Artificial Intelligence, 39(21), 22056–22064. https://doi.org/10.1609/aaai.v39i21.34358 |  |
|---|---|---|
| E13 | Chen, B., Bök, T. E., Rasti, B., Markl, V., & Demir, B. (2026). REMSA: Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent. European Conference on Computer Vision (ECCV 2026). https://doi.org/10.48550/arXiv.2511.17442 | arXiv |
| E14 | Le, D., Le, D.-C., Nguyen, A.-S., Bui, T.-D., Nguyen, T.-T., Nguyen, S., & Vo, H. D. (2026). iML: Executable, Problem-Grounded, and Broadly Exploratory Code-Driven AutoML. arXiv preprint arXiv:2602.13937. https://doi.org/10.48550/arXiv.2602.13937 | arXiv |

### 3.4.4 Criterios de inclusión/exclusión

Para seleccionar los estudios primarios a partir de los documentos preliminarmente filtrados (sección 3.4.3), se definieron los siguientes criterios de inclusión y exclusión, aplicados durante la fase de verificación detallada del protocolo de búsqueda. En la Tabla 17 se detallan los criterios de inclusión, con la pertinente justificación de su aplicación.

**Tabla 17**

*Tabla de criterios de inclusión*

| Código | Criterio | Justificación |
|---|---|---|
| CI1 | El estudio aborda la automatización de la construcción, selección o ajuste de modelos de visión por computadora o de clasificación en dominios multimodales, donde las imágenes sean una modalidad relevante. | Es el núcleo temático directamente relacionado con P1 y con el problema central de la tesis. |

| CI2 | El estudio describe una arquitectura o sistema que emplea agentes basados en LLM, RAG, o ambos, para automatizar tareas de aprendizaje automático o visión por computadora. | Permite responder P2, referida específicamente a la técnica en la que se apoya la solución propuesta. |
|---|---|---|
| CI3 | El estudio menciona o evalúa aspectos de interacción y/o usabilidad para usuarios o empresas sin conocimientos especializados en ML/IA. | Aporta evidencia empírica para responder P3, sobre estrategias de accesibilidad para usuarios no especializados. |
| CI4 | El estudio fue publicado entre 2020 y 2026. | Prioriza literatura reciente, dado el avance acelerado de AutoML, agentes basados en modelos de lenguaje y RAG en los últimos años. |
| CI5 | El estudio está disponible en texto completo, en inglés o español. | Garantiza el acceso íntegro al contenido necesario para completar el formulario de extracción de datos (sección 3.5). |
| CI6 | El estudio es una contribución científica primaria o secundaria (incluyendo surveys, revisiones, tesis, reportes técnicos, benchmarks y preprints) con metodología explícita. | La revisión abarca tanto investigación primaria como contribuciones secundarias para obtener una visión integral del estado del arte. Asimismo, se consideran preprints y literatura académica gris provenientes de |

|  |  | fuentes como arXiv (sección 3.4.1), dada la velocidad de evolución de las arquitecturas multiagente y las técnicas de IA, asegurando así la inclusión de desarrollos metodológicos recientes. |
|---|---|---|

En la Tabla 18 se detallan los criterios de exclusión, con la pertinente justificación de su aplicación.

**Tabla 18**

*Tabla de criterios de exclusión*

| Código | Criterio | Justificación |
|---|---|---|
| CE1 | El estudio se limita a proponer o evaluar una arquitectura de red neuronal específica para una tarea concreta, sin explorar, comparar ni proponer mecanismos de automatización para el ciclo de vida del modelo (construcción, selección o ajuste). | No aporta evidencia para responder P1 ni a la problemática de automatización planteada. |
| CE2 | El estudio corresponde a un resumen extendido de menos de 4 páginas, póster o tutorial | Garantiza un mínimo de rigor y profundidad metodológica en los estudios primarios seleccionados. |
| CE3 | El estudio no se encuentra disponible en texto completo dentro de los medios de acceso del proyecto. | Impide completar la extracción de datos exigida en la sección 3.5. |

## 3.5 Formulario de extracción de datos

El formulario de extracción de datos se diseñó para registrar, de manera uniforme, la información bibliográfica y de contenido necesaria para responder las preguntas de revisión planteadas en la sección 3.3, a partir de los estudios primarios seleccionados. La Tabla 19 presenta los campos generales de identificación del estudio, junto con los campos vinculados directamente a cada pregunta de revisión.

**Tabla 19**

*Tabla de campos propuestos para el formulario de extracción*

| Campo | Descripción | Pregunta a la que aporta |
|---|---|---|
| ID | Identificador del estudio (E1, E2, …) | General |
| Título | Título del artículo | General |
| Autor(es) | Autores del estudio | General |
| Año de publicación | Año de publicación | General |
| Motor de búsqueda | ACM Digital Library, Google Académico o arXiv | General |
| DOI | Enlace al documento (DOI) | General |
| Dominio de aplicación | Tarea de clasificación de imágenes o visión por computadora abordada, y contexto de uso (industrial, médico, genérico, etc.) | P1 |
| Modalidad de visión por computadora | Clasificación, detección, segmentación, multimodal, o combinación | P1 |
| Enfoque o arquitectura propuesta | Descripción del enfoque, técnica o arquitectura para automatizar la construcción, selección o | P1 |

|  | ajuste de modelos de visión por computadora o clasificación |  |
|---|---|---|
| Etapa del pipeline automatizada | Etapas automatizadas (selección de arquitectura, ajuste de hiperparámetros, entrenamiento, evaluación, selección de modelo fundacional, etc.) | P1 |
| Métricas de desempeño reportadas | Métricas utilizadas (exactitud, F1-score, tiempo de entrenamiento, etc.) | P1 |
| Uso de arquitecturas agénticas | Indica si emplea agentes basados en LLM, multiagente, o ambos. Describe el rol de cada agente. | P2 |
| Tipo de agente/LLM utilizado | Modelo base (GPT-4, Claude, Llama, etc.) y tipo de agente (reactivo, deliberativo, híbrido) | P2 |
| Uso de RAG | Indica si emplea RAG y con qué propósito. | P2 |
| Ventajas/limitaciones frente a enfoques no agénticos | Ventajas o limitaciones reportadas al comparar con alternativas sin agentes. | P2 |
| Mecanismo de interacción con usuarios no especializados | Interfaz, flujo o mecanismo (NL interface, human-centered, democratización) para usuarios sin conocimientos técnicos | P3 |
| Nivel de conocimiento técnico asumido o requerido | Grado de experiencia en ML/IA que el sistema asume o exige del usuario final | P3 |
| Explicabilidad de las decisiones del sistema | Si ofrece explicaciones comprensibles para no expertos. | P3 |

| Evidencia empírica de usabilidad y accesibilidad | Indica si el estudio aporta evidencia empírica sobre la usabilidad, accesibilidad o eficacia del sistema para usuarios no especializados en ML/IA. | P3 |
|---|---|---|
| Limitaciones declaradas por los autores | Limitaciones que los propios autores reconocen | General |

## 3.6 Resultados de la revisión

El presente apartado consolida los hallazgos obtenidos a partir del proceso de revisión sistemática de la literatura, cuyo alcance se definió en torno a tres ejes: (i) la automatización de la construcción, selección o ajuste de modelos de visión por computadora, con énfasis en clasificación de imágenes; (ii) el uso de arquitecturas agénticas (multiagente o basadas en modelos de lenguaje) y/o técnicas de generación aumentada por recuperación (RAG) aplicadas a tareas de aprendizaje automático y visión por computadora; y (iii) los mecanismos de interacción, usabilidad y accesibilidad orientados a usuarios o empresas sin conocimientos especializados en ML/IA.

A continuación, se sintetizan las evidencias y patrones identificados en el estado del arte para dar respuesta a las preguntas de investigación formuladas, estableciendo la base teórica y técnica que fundamenta la propuesta de este trabajo.

### 3.6.1 Respuesta a pregunta P1

**“¿Qué arquitecturas o enfoques han sido propuestos en la literatura para automatizar la construcción y selección de modelos de visión por computadora (con énfasis en clasificación de imágenes) a partir de un dataset etiquetado?”**

La literatura de Aprendizaje Automático Automatizado (AutoML) y de visión por computadora ha propuesto diversas arquitecturas y enfoques para automatizar la construcción y selección de modelos de clasificación de imágenes a partir de datasets etiquetados (Baratchi et al., 2024; Feurer et al., 2022). Los estudios primarios revisados permiten agruparlos en tres familias: la búsqueda de arquitecturas neuronales (NAS), que constituye la base histórica de AutoML; los marcos automatizados basados en agentes de IA y modelos de lenguaje (LLM), que representan la propuesta más reciente; y la selección de modelos fundacionales junto con la optimización asistida, que automatizan etapas puntuales del ciclo de vida del modelo.

**1. Búsqueda de Arquitecturas Neuronales ( *Neural Architecture Search* - NAS)** La búsqueda de arquitecturas neuronales (NAS) es una subdisciplina de AutoML orientada a encontrar automáticamente arquitecturas de redes neuronales adecuadas para una tarea determinada, de modo que no sea necesario diseñarlas manualmente (Baratchi et al., 2024). Esta subsección resume las estrategias de NAS que la encuesta de Baratchi et al. (2024) (E4) recoge, ordenadas según la manera en que definen y recorren el espacio de búsqueda. Un primer criterio de distinción es el espacio de búsqueda. Los enfoques de nivel macro exploran la configuración global de la red, definiendo sus componentes de forma amplia, mientras que los de nivel micro o basados en celdas buscan bloques convolucionales que luego se apilan para formar arquitecturas más complejas (Baker et al., 2017; Zoph et al., 2018). Sobre estos espacios se han aplicado distintas estrategias de exploración. Las basadas en aprendizaje por refuerzo emplean un controlador recurrente que genera descripciones de arquitecturas y utiliza el desempeño de validación como señal de recompensa (Zoph & Le, 2016). De manera similar, sistemas como AutoKeras exploran de forma automatizada distintas configuraciones de modelos y ajustan sus parámetros de manera eficiente (Jin et al., 2019). En contraste, la optimización diferenciable basada en gradientes, representada por DARTS, relaja la selección discreta de operaciones mediante representaciones continuas, lo que permite optimizar conjuntamente los parámetros de la arquitectura y los pesos de la red (Liu et al., 2019). Finalmente, los algoritmos evolutivos generan progresivamente arquitecturas con mejores características de desempeño mediante mecanismos de mutación, selección y evolución (Real et al., 2019). En conjunto, estos enfoques muestran que la automatización del diseño de arquitecturas es un problema ampliamente estudiado, pero que requiere definir de antemano el espacio de búsqueda y suele demandar numerosos ciclos de entrenamiento (Elsken et al., 2019). Esto refuerza, para el caso de las pymes, la conveniencia de acotar ese espacio antes de explorarlo. **2. Marcos Automatizados Basados en Agentes de IA y Modelos de Lenguaje (LLMs)** Recientemente han surgido marcos multiagente impulsados por LLM que automatizan distintas etapas del flujo de aprendizaje automático, desde la interpretación de los requerimientos y la preparación de los datos hasta la selección, el entrenamiento y el despliegue de modelos (Yang et al., 2025; Trirat et al., 2025). Ocho de los estudios primarios pertenecen a esta familia (E5, E6, E7, E8, E9, E10, E12 y E14) y pueden agruparse según su alcance en marcos de pipeline completo, marcos de dominio específico y marcos de exploración científica y adaptación. Los marcos de pipeline completo buscan cubrir todas las etapas del flujo con un mismo sistema. AutoML-Agent implementa un sistema multiagente que automatiza el pipeline completo de AutoML mediante una estrategia de planificación aumentada por recuperación, agentes especializados para resolver subtareas y un mecanismo de verificación en múltiples etapas (Trirat et al., 2025). Por su parte, MLZero emplea un sistema multiagente impulsado por LLM para automatizar procesos de aprendizaje automático de extremo a extremo con distintas modalidades de datos, y su arquitectura incorpora percepción cognitiva, memoria semántica y episódica, y ciclos iterativos de generación y refinamiento de código (Fang et al., 2025). De

manera similar, iML utiliza agentes y la exploración de múltiples estrategias para construir soluciones de aprendizaje automático a partir de problemas planteados por los usuarios (Le et al., 2026). A diferencia de los anteriores, que parten de instrucciones generales, AutoMMLab introduce la tarea *request-to-model*, en la cual el sistema interpreta requerimientos expresados en lenguaje natural y automatiza la selección de datos, la selección del modelo, el entrenamiento y el despliegue, incluyendo la optimización de hiperparámetros mediante HPO- LLaMA (Yang et al., 2025). Los marcos de dominio específico aplican la misma lógica a problemas de visión de un área concreta. CellDX AI Autopilot automatiza el entrenamiento, la evaluación y el despliegue de clasificadores de imágenes histopatológicas mediante la interacción en lenguaje natural con un agente, e incorpora el ajuste automatizado de hiperparámetros y la comparación entre múltiples estrategias de modelado (Pchelnikov & Pchelnikov, 2026). En una línea cercana, TopoAgent combina razonamiento basado en agentes LLM con análisis de datos topológicos para determinar los descriptores y configuraciones más adecuados para un conjunto de imágenes médicas, y luego los utiliza en tareas de clasificación (Meng et al., 2026). Por último, otros trabajos emplean agentes para la exploración científica y la adaptación de herramientas existentes. SelfAI propone un sistema multiagente autodirigido para la exploración científica de largo horizonte, en el que agentes especializados gestionan y optimizan de manera autónoma configuraciones experimentales y procesos de descubrimiento (Wu et al., 2026). Asimismo, los agentes de código se han utilizado para adaptar herramientas y modelos de visión preentrenados a conjuntos de datos científicos específicos, mediante la generación automatizada de procedimientos de procesamiento y evaluación (Wang et al., 2026). En síntesis, los marcos de pipeline completo son los más cercanos al problema de esta tesis, pues atienden la construcción, selección y entrenamiento de modelos a partir de un dataset y de requerimientos expresados por el usuario, mientras que los de dominio específico y de adaptación muestran que el enfoque agéntico es transferible a distintas tareas de visión (P1). **3. Selección de Modelos Fundacionales y Optimización Asistida** Además de los métodos de NAS y de los marcos multiagente que construyen pipelines completos, la literatura reporta propuestas que no diseñan la arquitectura desde cero, sino que asisten la elección del modelo o la optimización de sus hiperparámetros, así como estudios que contrastan el grado de automatización con el control que conserva el desarrollador. En este grupo se identificaron tres aportes (E13, E1 y E11). En el primero, REMSA (Remote-sensing Model Selection Agent) emplea un agente consciente de restricciones para seleccionar modelos fundacionales de visión o visión-lenguaje en tareas de teledetección, considerando tanto las características de los datos como las restricciones del problema (Chen et al., 2026). En una línea distinta, CrowdOptim aprovecha la inteligencia colectiva (*crowdsourcing*) para apoyar la optimización de hiperparámetros en modelos de aprendizaje profundo aplicados a tareas de detección y clasificación asociadas con sensores urbanos (Zhang et al., 2022). Por último, Vignesh et al. (2025) comparan plataformas de desarrollo de IA sin código con pipelines basados en código, y concluyen que las primeras pueden reducir la barrera de entrada para usuarios no especialistas al automatizar parte de la construcción y el entrenamiento de modelos, aunque los segundos mantienen ventajas de flexibilidad y control sobre el proceso de desarrollo. En conjunto, estos trabajos muestran que la automatización puede ubicarse en distintos puntos del ciclo de vida del modelo y que existe un compromiso entre accesibilidad y control, lo cual es relevante para pymes sin personal especializado.

### 3.6.2 Respuesta a pregunta P2

**“¿Cómo se han empleado arquitecturas agénticas (multiagente o basadas en LLM) y/o técnicas de generación aumentada por recuperación (RAG) para automatizar tareas de aprendizaje automático y visión por computadora, y qué ventajas o limitaciones se reportan frente a enfoques no agénticos?”**

Las arquitecturas agénticas y las técnicas de generación aumentada por recuperación (RAG) se han consolidado como un paradigma emergente para automatizar el ciclo de vida del aprendizaje automático (AutoML) y de la visión por computadora (Trirat et al., 2025; Fang et al., 2025). Para responder a la pregunta, esta sección describe primero cómo se aplican ambas técnicas en los estudios revisados y luego resume las ventajas y las limitaciones que sus autores reportan frente a los enfoques no agénticos.

**1. Aplicación de Arquitecturas Agénticas y Técnicas RAG** Los estudios que emplean agentes o RAG lo hacen en tres modalidades complementarias: la coordinación de varios agentes especializados, el diseño de agentes específicos para tareas de visión y la incorporación de conocimiento externo mediante RAG y memoria. Esta subsección las describe en ese orden, con base en los estudios E5 a E10, E12 y E13, principalmente. En primer lugar, se han diseñado marcos donde múltiples agentes con roles especializados (como análisis de datos, diseño de modelos, optimización de hiperparámetros y escritura de código) colaboran dinámicamente para construir pipelines completos (Trirat et al., 2025; Fang et al., 2025). AutoML-Agent, por ejemplo, coordina un gestor de agentes con agentes de datos, de modelos y de operaciones para procesar instrucciones en lenguaje natural y generar modelos desplegables (Trirat et al., 2025). De igual forma, MLZero emplea un sistema jerárquico con nueve agentes especializados para transformar datos multimodales brutos en modelos listos para producción sin intervención humana (Fang et al., 2025). En segundo lugar, en tareas de visión se han propuesto agentes especializados. AutoMMLab utiliza modelos de lenguaje afinados (RU-LLaMA y HPO-LLaMA) para interpretar requerimientos, seleccionar arquitecturas de visión (como MobileNet o Swin-Transformer) y ajustar sus hiperparámetros (Yang et al., 2025). En visión médica, TopoAgent implementa un ciclo de Percepción–Razonamiento–Acción–Reflexión (PRAR), soportado por 21 herramientas especializadas, para seleccionar descriptores topológicos adaptativos según la morfología de cada imagen (Meng et al., 2026). Por otro lado, agentes mínimos de código se han utilizado para automatizar la adaptación de herramientas de visión preentrenadas (como Cellpose o MedSAM) a nuevos datasets biológicos, mediante la generación de funciones de preprocesamiento y postprocesamiento (Wang et al., 2026). En tercer lugar, las técnicas RAG y la memoria se han integrado para enriquecer el razonamiento de los agentes con conocimiento externo actualizado o estructurado (Trirat et al., 2025; Chen et al., 2026). AutoML-Agent utiliza planificación aumentada por recuperación (RAP), consultando fuentes como arXiv, Kaggle o PapersWithCode a través de APIs para formular planes de ejecución informados (Trirat et al., 2025). A su vez, REMSA aplica RAG estructurado sobre la base de datos RS-FMD para filtrar, clasificar y recomendar modelos fundacionales de percepción remota ajustados a las restricciones del usuario (Chen et al., 2026). Asimismo, MLZero e iML integran memoria semántica y de contexto para recuperar documentación precisa sobre librerías de aprendizaje automático y evitar alucinaciones sintácticas en el código generado (Fang et al., 2025; Le et al., 2026). Estos resultados indican que RAG y agentes se han usado tanto para planificar (AutoML- Agent) como para acotar el conjunto de modelos candidatos (REMSA), una idea directamente alineada con el mecanismo de acotación del espacio de configuraciones propuesto en esta tesis.

**2. Ventajas Reportadas Frente a Enfoques No Agénticos Tradicionales** Los estudios revisados reportan cuatro ventajas principales de los enfoques agénticos frente a las alternativas tradicionales: mayor flexibilidad en el espacio de búsqueda, democratización mediante interfaces en lenguaje natural, verificación y autorreflexión dinámica, y razonamiento contextual adaptativo. La primera ventaja es la flexibilidad y expresividad en el espacio de búsqueda. Mientras que las herramientas de AutoML tradicional, como AutoGluon o Auto-sklearn, operan sobre espacios de búsqueda cerrados o de caja negra, los agentes basados en LLM sintetizan código ejecutable abierto que permite explorar arquitecturas personalizadas y estrategias heterogéneas de ingeniería de características (Fang et al., 2025; Le et al., 2026). La segunda es la democratización: los marcos agénticos eliminan las barreras técnicas de configuración manual y de programación en frameworks como TensorFlow o PyTorch, y permiten que usuarios no expertos construyan y desplieguen modelos complejos a partir de instrucciones simples (Baratchi et al., 2024; Trirat et al., 2025). La tercera ventaja es la verificación y autorreflexión dinámica. A diferencia de los métodos no agénticos de un solo paso o de las búsquedas estáticas, los sistemas agénticos cuentan con bucles de retroalimentación y verificación multietapa que les permiten ejecutar código, analizar errores en tiempo de ejecución y corregirlos de forma autónoma (Trirat et al., 2025; Meng et al., 2026). La cuarta es el razonamiento contextual adaptativo: TopoAgent demuestra que la determinación adaptativa de parámetros por imagen supera en un 9.32 % la precisión balanceada de oráculos fijos no agénticos en tareas de clasificación de imágenes médicas (Meng et al., 2026).

En conjunto, las ventajas reportadas coinciden con las necesidades de las pymes sin personal especializado, pues reducen la configuración manual y permiten corregir errores sin intervención experta. **3. Limitaciones Reportadas Frente a Enfoques No Agénticos** Frente a las ventajas descritas, los estudios revisados reportan cuatro limitaciones recurrentes. La primera es el riesgo de alucinaciones y fallos de ejecución, pues los agentes basados en LLM pueden generar scripts que parecen correctos en texto pero que fallan al ejecutarse o que asumen propiedades no verificadas de los datos de entrada (Fang et al., 2025; Trirat et al., 2025). La segunda es el mayor costo computacional y de tokens: la coordinación multiagente y los bucles de iteración profunda generan un número significativamente mayor de llamadas a los modelos, con costos de inferencia y latencia superiores a los de la optimización estática tradicional (Jan et al., 2026; Trirat et al., 2025). La tercera limitación es el riesgo de sobreajuste en la optimización de hiperparámetros (HPO): se ha reportado que integrar en exceso la búsqueda de hiperparámetros dentro de los bucles agénticos puede sobreajustar los modelos al conjunto de validación y reducir su rendimiento final en el conjunto de prueba (Wang et al., 2026). La cuarta es la dependencia de la calidad del conocimiento recuperado, pues el desempeño de las estrategias RAG depende directamente de la precisión, la estructuración y la cobertura de las bases de metadatos o repositorios externos que se consultan (Chen et al., 2026; Trirat et al., 2025). Estas limitaciones motivan, en la arquitectura propuesta, el uso de verificación multietapa y de descarte temprano para contener el consumo de recursos.

### 3.6.3 Respuesta a pregunta P3

**“¿Qué estrategias, enfoques o mecanismos de interacción han propuesto los sistemas automatizados de construcción y selección de modelos de visión por computadora para** **facilitar su uso por parte de usuarios o empresas sin conocimientos especializados en ML/IA?”**

Para facilitar la adopción de herramientas de visión por computadora en empresas y usuarios sin conocimientos especializados en aprendizaje automático (ML/IA), la literatura ha propuesto diversas estrategias, enfoques y mecanismos de interacción que abstraen la complejidad técnica subyacente (Karmaker et al., 2021; Trirat et al., 2025). Los estudios revisados las organizan en cinco grupos: las interfaces en lenguaje natural, el diálogo multietapa, las plataformas no-code y low-code, la inteligencia colectiva y la abstracción de infraestructura y despliegue.

**1. Interfaces Basadas en Lenguaje Natural y Tareas "Request-to-Model"** La interfaz de lenguaje natural se ha posicionado como el mecanismo principal para democratizar el aprendizaje automático al permitir que usuarios no técnicos expresen sus objetivos mediante instrucciones en texto libre (Trirat et al., 2025). El marco **AutoMMLab** introduce el concepto de “*request-to-model”*, permitiendo a los usuarios especificar restricciones de negocio o hardware (como latencia, FPS objetivo o plataforma de ejecución) a través de lenguaje natural para que el sistema configure y entrene automáticamente el modelo adecuado (Yang et al., 2025). De manera similar, **AutoML-Agent** utiliza un agente especializado para traducir los requerimientos informales del usuario en estructuras esquemáticas en formato JSON, interpretando la intención y los criterios deseados de precisión (Trirat et al., 2025). En el ámbito médico, la plataforma **CellDX AI Autopilot** permite a patólogos y clínicos guiar el ciclo completo de entrenamiento y evaluación de clasificadores de imágenes histopatológicas conversando directamente con un agente inteligente (Pchelnikov & Pchelnikov, 2026). Así, el lenguaje natural permite que el usuario exprese el problema sin conocer el vocabulario técnico del aprendizaje automático. **2. Diálogo Multietapa y Generación de Clarificaciones Interactivas** Para resolver consultas imprecisas o ambiguas formuladas por no expertos, los sistemas incorporan bucles de clarificación conversacional que solicitan parámetros clave de forma intuitiva (Chen et al., 2026). El sistema **Remsa** integra un generador de clarificaciones que analiza las restricciones ingresadas y formula preguntas interactivas en lenguaje natural antes de sugerir el modelo fundacional de percepción remota más conveniente (Chen et al., 2026).

Por su parte, algoritmos como **HPO-LLaMA** dentro de AutoMMLab utilizan diálogos multironda para presentar propuestas de hiperparámetros y recibir retroalimentación iterativa del rendimiento del modelo sin exigir ajustes manuales de código (Yang et al., 2025). Estos mecanismos muestran que el sistema puede completar la información faltante preguntando al usuario, en lugar de exigirle una especificación técnica completa desde el inicio.

**3. Plataformas No-Code, Low-Code e Interfaces Gráficas**

Las plataformas No-Code (como Microsoft Lobe y Google Teachable Machine) ofrecen interfaces visuales tipo arrastrar y soltar que permiten subir datos etiquetados y entrenar modelos en pocos minutos sin escribir código (Liwanag et al., 2025). Estos entornos automatizan la normalización, el cambio de tamaño de imágenes y las técnicas de aumento de datos, ofreciendo herramientas de visualización sencillas como matrices de confusión para evaluar el rendimiento (Liwanag et al., 2025).

Si bien el desarrollo basado en código logra mayor flexibilidad y rendimiento final, las herramientas No-Code ofrecen una velocidad de prototipado significativamente mayor y reducen drásticamente la barrera de entrada para pequeñas empresas y educadores (Karmaker et al., 2021). De este modo, el compromiso entre simplicidad y control ya identificado en la sección 3.6.1 reaparece como criterio para decidir qué interfaz es adecuada para una pyme.

**4. Inteligencia Colectiva y Simplificación para Anotadores no Expertos (Crowdsourcing)**

El marco **CrowdOptim** aborda la dificultad de configurar hiperparámetros complejos transformando el problema en tareas sencillas de verificación visual accesible para trabajadores de crowdsourcing sin formación en IA (Zhang et al., 2022). A través de un flujo estructurado de cuatro pasos sencillos (cargar imágenes, solicitar inteligencia de la multitud, optimizar y visualizar resultados), las organizaciones pueden obtener configuraciones óptimas guiadas por retroalimentación humana colectiva (Zhang et al., 2022). Este enfoque ilustra que la complejidad técnica también puede reducirse descomponiendo la decisión en tareas simples que no requieren formación especializada.

**5. Abstracción de Infraestructura y Despliegue Automático**

Para evitar que los usuarios tengan que gestionar entornos computacionales o bases de datos complejas, plataformas como **CellDX AI Autopilot** incluyen almacenes de características preintegrados (*feature stores*) que reducen drásticamente los costos de infraestructura (Pchelnikov & Pchelnikov, 2026). Asimismo, **Remsa** genera explicaciones estructuradas y justificadas sobre las fortalezas y debilidades de los modelos recomendados, garantizando la transparencia del proceso de selección (Chen et al., 2026).

Finalmente, sistemas como **AutoML-Agent** crean de forma transparente interfaces de usuario web interactivo (mediante librerías como Gradio) para que el usuario final pueda probar su modelo desplegado inmediatamente desde un navegador (Trirat et al., 2025). En conjunto, estos mecanismos de abstracción, explicación y despliegue son los que permiten que una pyme reciba un modelo utilizable sin administrar la infraestructura ni interpretar el proceso interno.

## 3.7 Conclusiones

A continuación se presentan las conclusiones del Estado del Arte estructuradas en relación con el problema seleccionado (la dificultad de las pymes no especializadas en IA para construir, seleccionar, entrenar y desplegar modelos de clasificación de imágenes a partir de *datasets* etiquetados): **1. Evolución del Aprendizaje Automático Automatizado (AutoML) y Democratización para Pymes**

El desarrollo tradicional de modelos de clasificación de imágenes ha dependido históricamente de la programación manual intensiva, la preparación de datos y el ajuste experto de hiperparámetros (Liwanag et al., 2025; Le et al., 2025). Esta alta dependencia de científicos de datos especializados representa una barrera económica y técnica crítica para las pequeñas y medianas empresas (pymes) que carecen de departamentos de investigación en IA o presupuestos computacionales elevados (Liwanag et al., 2025; Guthardt et al., 2024; Baratchi et al., 2024). En respuesta a esta problemática, el aprendizaje automático automatizado (AutoML) ha evolucionado progresivamente desde sistemas rígidos de "caja negra" hacia plataformas orientadas a democratizar el acceso al desarrollo de IA para usuarios no técnicos (Liwanag et al., 2025; Yang et al., 2025; Le et al., 2025). El objetivo principal de esta evolución es permitir que las pymes puedan transformar sus datasets etiquetados en modelos predictivos funcionales sin requerir habilidades avanzadas de codificación (Liwanag et al., 2025; Yang et al., 2025; Baratchi et al., 2024).

**2. Paradigma Agéntico y Modelos de Lenguaje (LLMs) como Frontera Tecnológica**

Las plataformas No-Code/Low-Code tradicionales como Microsoft Lobe o Google Teachable Machine permiten a las pymes construir clasificadores de imágenes en pocos minutos mediante interfaces visuales intuitivas (Liwanag et al., 2025). No obstante, la evidencia empírica muestra que las herramientas No-Code ofrecen menor flexibilidad arquitectónica y registran entre un 3% y un 4% menos de precisión en comparación con la programación en código nativo (Liwanag et al., 2025). Para superar la rigidez del No-Code tradicional, han emergido arquitecturas multiagente impulsadas por LLMs como la frontera tecnológica más avanzada en AutoML y visión por computadora (Yang et al., 2025; Fang et al., 2025; Trirat et al., 2025). Sistemas como AutoMMLab, AutoML-Agent y MLZero emplean interfaces de lenguaje natural bajo el esquema "request-to-model" para coordinar de manera autónoma la selección de la red convolucional, el entrenamiento y la optimización de hiperparámetros (HPO) (Yang et al., 2025; Fang et al., 2025; Trirat et al., 2025). Asimismo, marcos agénticos como iML desacoplan la planificación estratégica del código de implementación, garantizando la generación de pipelines modulares, ejecutables y autocorregibles mediante verificación dinámica (Le et al., 2025).

**3. Eficiencia Computacional, Almacenamiento de Características y Despliegue en Pymes**

Para hacer viables estas soluciones en pymes con recursos limitados, la literatura destaca mecanismos para optimizar la eficiencia computacional y reducir los tiempos de búsqueda (Yang et al., 2025; Baratchi et al., 2024; Pchelnikov & Pchelnikov, 2026). Por ejemplo, la utilización de LLMs especializados como HPO-LLaMA permite identificar configuraciones óptimas de hiperparámetros en sustancialmente menos rondas de entrenamiento en comparación con la búsqueda aleatoria o bayesiana (Yang et al., 2025). Por otro lado, plataformas como CellDX AI Autopilot reducen en más de un 30x los costos de ajuste mediante el uso de almacenes de características pre-extraídas que evitan el re-procesamiento masivo de imágenes (Pchelnikov & Pchelnikov, 2026). Para solucionar el cuello de botella del despliegue en producción, marcos modernos automatizan la conversión de modelos a formatos estándar como ONNX y generan automáticamente interfaces web funcionales como Gradio (Yang et al., 2025; Trirat et al., 2025).

**4. Desafíos Persistentes y Brechas Abiertas en la Literatura**

A pesar de sus marcadas fortalezas, los sistemas agénticos basados en LLMs enfrentan el riesgo de generar código con fallos de sintaxis o alucinaciones sobre las propiedades de los datos de entrada (Fang et al., 2025; Trirat et al., 2025). Para mitigar estos errores, es indispensable implementar módulos de memoria semántica y episódica, así como bucles de verificación multietapa que incrementan la latencia y los costos de inferencia por tokens (Fang et al., 2025; Trirat et al., 2025). Adicionalmente, la falta de explicabilidad en algunas decisiones automatizadas dificulta que las pymes adapten o auditen sus modelos ante cambios en sus requerimientos operativos (Liwanag et al., 2025; Le et al., 2025).

**5. Síntesis para la Investigación**

En conclusión, la literatura demuestra que la integración de sistemas multiagente impulsados por LLMs, módulos de verificación de ejecutabilidad y herramientas de despliegue automatizado ofrece una vía prometedora para resolver la brecha técnica en pymes no especializadas (Yang et al., 2025; Le et al., 2025; Fang et al., 2025). La brecha de investigación actual consiste en diseñar arquitecturas que equilibren la interacción intuitiva en lenguaje natural, la ejecución de código sin errores y la máxima eficiencia en el consumo de tokens e infraestructura computacional (SmartDS-Solver team, 2025; Trirat et al., 2025).

# Capítulo 4. Presentación de los resultados esperados

Del capítulo 4 hasta el N deben ir los resultados. Introducción, enunciar el resultado, relacionarlo con el objetivo, presentar modo de validar su construcción**.**

# Capítulo 5. Conclusiones y trabajos futuros

## 5.1 Conclusiones

## 5.2 Trabajos futuros

# Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD* *International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Association for Computing Machinery. (s.f.). *About ACM DL*. ACM Digital Library. https://dl.acm.org/about

Ayinaddis, S. G. (2025). Artificial intelligence adoption dynamics and knowledge in SMEs and large firms: A systematic review and bibliometric analysis. *Journal of Innovation &* *Knowledge, 10*(3), Article 100682. https://doi.org/10.1016/j.jik.2025.100682

Baker, B., Gupta, O., Naik, N., & Raskar, R. (2017). Designing neural network architectures using reinforcement learning. *International Conference on Learning Representations (ICLR* *2017)*. https://arxiv.org/abs/1611.02167

Baratchi, M., Wang, C., Limmer, S., van Rijn, J. N., Hoos, H., Bäck, T., & Olhofer, M. (2024). Automated machine learning: Past, present and future. *Artificial Intelligence Review,* *57*, Article 122. https://doi.org/10.1007/s10462-024-10726-1

Brooke, J. (1996). SUS: A quick and dirty usability scale. En P. W. Jordan, B. Thomas, B. A. Weerdmeester, & A. L. McClelland (Eds.), *Usability evaluation in industry* (pp. 189–194). Taylor & Francis.

Chen, B., Bök, T. E., Rasti, B., Markl, V., & Demir, B. (2026). REMSA: Foundation model selection for remote sensing via a constraint-aware agent. *European Conference on Computer* *Vision (ECCV 2026)*. https://doi.org/10.48550/arXiv.2511.17442

Chollet, F. (2015). *Keras*. https://keras.io

Chroma. (s.f.). *Chroma documentation*. https://docs.trychroma.com CMS. (2025, 16 de septiembre). *Nuevo reglamento de protección de datos personales*. CMS Law. https://cms.law/en/per/publication/nuevo-reglamento-de-proteccion-de-datos-personales

Congreso de la República del Perú. (2011). *Ley N.° 29733, Ley de Protección de Datos* *Personales*. Diario Oficial El Peruano. https://vlex.com.pe/vid/ley-n-29733-ley-578578822

Congreso de la República del Perú. (2023). *Ley N.° 31814, Ley que promueve el uso de la* *inteligencia artificial en favor del desarrollo económico y social del país*. Diario Oficial El Peruano. https://leyes.congreso.gob.pe/Documentos/2021_2026/ADLP/Texto_Consolidado/31814- TXM.pdf

Cornell University. (s.f.). *About arXiv*. arXiv.org. https://arxiv.org/about

Docker Inc. (s.f.). *Docker documentation*. https://docs.docker.com

Elsevier. (2023). *Scopus: Access and use support center*. https://www.elsevier.com/en- in/solutions/scopus

Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. *Journal* *of Machine Learning Research, 20*(55), 1–21.

Fang, H., Han, B., Erickson, N., Zhang, X., Zhou, S., Dagar, A., Zhang, J., Turkmen, A. C., Hu, T., Rangwala, H., Wu, Y. N., Wang, Y., & Karypis, G. (2025). MLZero: A multi-agent system for end-to-end machine learning automation. *Advances in Neural Information* *Processing Systems, 38*. https://doi.org/10.52202/085713-2322

Feurer, M., Eggensperger, K., Falkner, S., Lindauer, M., & Hutter, F. (2022). Auto-Sklearn 2.0: Hands-free AutoML via meta-learning. *Journal of Machine Learning Research, 23*(261), 1–61.

Forbes Perú. (2023, 5 de julio). *Gobierno publicó la ley que promueve el uso de la* *inteligencia artificial en Perú*. https://forbes.pe/actualidad/2023-07-05/gobierno-publico-la- ley-que-promueve-el-uso-de-la-inteligencia-artificial-en-peru/

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

Gestión. (2025, 9 de septiembre). *Perú ya tiene reglamento de IA: ¿tu empresa opera bajo* *riesgos altos o indebidos?* https://gestion.pe/economia/peru-ya-tiene-reglamento-de-ia-tu- empresa-opera-bajo-riesgos-altos-o-indebidos-noticia/

Gusenbauer, M., & Haddaway, N. R. (2020). Which academic search systems are suitable for systematic reviews or meta-analyses? Evaluating retrieval qualities of Google Scholar, PubMed, and 26 other resources. *Research Synthesis Methods, 11*(2), 181–217. https://doi.org/10.1002/jrsm.1378

He, X., Zhao, K., & Chu, X. (2021). AutoML: A survey of the state-of-the-art. *Knowledge-* *Based Systems, 212*, Article 106622. https://doi.org/10.1016/j.knosys.2020.106622

IEEE. (s.f.). *About IEEE Xplore*. IEEE Xplore Digital Library. https://ieeexplore.ieee.org/Xplorehelp/overview-of-ieee-xplore/about-ieee-xplore

Instituto Nacional de Estadística e Informática. (2013). *En el Perú existen más de un millón* *700 mil empresas: Micro, pequeñas y medianas empresas concentran más del 20% de las* *ventas* (Nota de prensa N.° 133). https://m.inei.gob.pe/media/MenuRecursivo/noticias/np133- 2013-inei.pdf

International Organization for Standardization & International Electrotechnical Commission. (2023). *ISO/IEC 42001:2023 — Information technology — Artificial intelligence —* *Management system*. https://www.iso.org/standard/81230.html

Jan, N.-Y., Chen, W.-C., Lin, Y.-R., Wu, C. T., Liu, S.-H., & Li, Y.-H. (2026). *SmartDS-* *Solver: Agentic AI for vertical domain problem solving in data science* [Manuscrito sometido a conferencia]. International Conference on Learning Representations (ICLR 2026), en revisión. https://openreview.net/pdf?id=r7gmePFADZ

Jin, H., Song, Q., & Hu, X. (2019). Auto-Keras: An efficient neural architecture search system. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge* *Discovery & Data Mining*, 1946–1956. https://doi.org/10.1145/3292500.3330648

Karmaker Santu, S. K., Hassan, M. M., Smith, M. J., Xu, L., Zhai, C., & Veeramachaneni, K. (2022). AutoML to date and beyond: Challenges and opportunities. *ACM Computing Surveys,* *54*(8), Article 175. https://doi.org/10.1145/3470918

Kitchenham, B., & Charters, S. (2007). *Guidelines for performing systematic literature* *reviews in software engineering* (EBSE Technical Report EBSE-2007-01). Keele University and University of Durham.

LangChain AI. (s.f.). *LangGraph documentation*. https://langchain-ai.github.io/langgraph/

Le, D., Le, D.-C., Nguyen, A.-S., Bui, T.-D., Nguyen, T.-T., Nguyen, S., & Vo, H. D. (2026). *iML: Executable, problem-grounded, and broadly exploratory code-driven AutoML* (arXiv:2602.13937). arXiv. https://doi.org/10.48550/arXiv.2602.13937

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning* *Research, 18*(185), 1–52.

Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable architecture search. *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1806.09055 LP Derecho. (2025). *Reglamento de la Ley que promueve el uso de la inteligencia artificial* *para el desarrollo económico y social del país (Decreto Supremo N.° 115-2025-PCM)*. https://lpderecho.pe/reglamento-ley-promueve-uso-inteligencia-artificial-desarrollo- economico-pais-decreto-supremo-115-2025-pcm/

Martín-Martín, A., Orduna-Malea, E., Thelwall, M., & Delgado López-Cózar, E. (2018). Google Scholar, Web of Science, and Scopus: A systematic comparison of citations in 252 subject categories. *Journal of Informetrics, 12*(4), 1160–1177. https://doi.org/10.1016/j.joi.2018.09.002

Meng, G., Gu, P., Li, X., Shi, Y., Chambers, E. W., & Chen, D. Z. (2026). *TopoAgent: An* *agentic framework for automated topology learning in medical imaging* (arXiv:2606.29763). arXiv. https://doi.org/10.48550/arXiv.2606.29763

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the* *Conference on Fairness, Accountability, and Transparency (FAT* '19)*, 220–229. https://doi.org/10.1145/3287560.3287596

National Institute of Standards and Technology. (2023). *Artificial Intelligence Risk* *Management Framework (AI RMF 1.0)* (NIST AI 100-1). U.S. Department of Commerce. https://doi.org/10.6028/NIST.AI.100-1

Ortegón, E., Pacheco, J. F., & Prieto, A. (2005). *Metodología del marco lógico para la* *planificación, el seguimiento y la evaluación de proyectos y programas* (Serie Manuales N.° 42). CEPAL, Instituto Latinoamericano y del Caribe de Planificación Económica y Social (ILPES).

Pchelnikov, A., & Pchelnikov, A. (2026). *CellDX AI Autopilot: Agent-guided training and* *deployment of pathology classifiers* (arXiv:2605.10362). arXiv. https://doi.org/10.48550/arXiv.2605.10362

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Petticrew, M., & Roberts, H. (2006). *Systematic reviews in the social sciences: A practical* *guide*. Blackwell Publishing.

Ramírez, S. (s.f.). *FastAPI documentation*. https://fastapi.tiangolo.com

Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized evolution for image classifier architecture search. *Proceedings of the AAAI Conference on Artificial Intelligence,* *33*(1), 4780–4789. https://doi.org/10.1609/aaai.v33i01.33014780

Trirat, P., Jeong, W., & Hwang, S. J. (2025). AutoML-Agent: A multi-agent LLM framework for full-pipeline AutoML. *Proceedings of the 42nd International Conference on Machine* *Learning, 267*, 60099–60146. https://arxiv.org/abs/2410.02958

Vignesh, M., Santhoshkumar, S. P., Vikram, D., Gajalakshmi, S., Ramamoorthi, S., & Mohammed Uveise, S. A. (2025). Beyond Conventional Programming: Empirical Perspectives on AI Model Accuracy and Development Efficiency. 2025 International Conference on Emerging Research in Computational Science (ICERCS). https://doi.org/10.1109/ICERCS65898.2025.11580564

Wang, X., Horstmann, K. A., Lin, E., Chen, J., Farhang, A. R., Stiles, S., Sehgal, A., Light, J., Van Valen, D., Yue, Y., & Sun, J. J. (2026). Simple agents outperform experts in biomedical imaging workflow optimization. *Proceedings of the IEEE/CVF Conference on* *Computer Vision and Pattern Recognition (CVPR 2026)*. https://doi.org/10.48550/arXiv.2512.06006

Wooldridge, M. (2009). *An introduction to multiagent systems* (2.ª ed.). John Wiley & Sons.

Wu, X., Huang, T.-Z., Deng, L.-J., Yu, X., Zhong, Y., Deng, S., Khan, U., Wu, J., Liu, X., Razzak, I., Chang, X., & Xie, Y. (2026). *SelfAI: A self-directed framework for long-horizon* *scientific discovery* (arXiv:2512.00403). arXiv. https://doi.org/10.48550/arXiv.2512.00403

Xue, C., Li, J., Wang, X., Zhan, Y., Yan, J., & Li, C.-G. (2025). On neural architecture search and hyperparameter optimization: A max-flow based approach. *Neural Networks, 188*, Article 107507. https://doi.org/10.1016/j.neunet.2025.107507 Yang, Z., Zeng, W., Jin, S., Qian, C., Luo, P., & Liu, W. (2025). AutoMMLab: Automatically generating deployable models from language instructions for computer vision tasks. *Proceedings of the AAAI Conference on Artificial Intelligence, 39*(21), 22056–22064. https://doi.org/10.1609/aaai.v39i21.34358

Zhang, Y., Zong, R., Shang, L., Kou, Z., Zeng, H., & Wang, D. (2022). CrowdOptim: A crowd-driven neural network hyperparameter optimization approach to AI-based smart urban sensing. *Proceedings of the ACM on Human-Computer Interaction, 6*(CSCW2), Article 435. https://doi.org/10.1145/3555536 Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. *International Conference on Learning Representations (ICLR 2017)*. https://arxiv.org/abs/1611.01578

Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. *Proceedings of the IEEE Conference on Computer Vision and* *Pattern Recognition (CVPR)*, 8697–8710. https://doi.org/10.1109/CVPR.2018.00907

# Anexos

Los anexos deben ser referenciados desde el documento. Por ejemplo, debe existir un párrafo donde se diga que determinada información puede ser vista en el Anexo X

Los anexos pueden numerarse con letras o número de acuerdo con su preferencia.

# Anexo A: Plan de Proyecto

- **Justificación** <<Se debe justificar el proyecto mediante la exposición de las razones por las cuales se debe realizar el estudio. Además de plantear claramente el propósito del proyecto dependiendo del caso se explica por qué es conveniente y cuáles son los beneficios de éste. (Hernández Sampieri, Fernández Collado, & Baptista, 2014) Algunos criterios, incluidos en (Hernández Sampieri et al., 2014), para evaluar la importancia de una investigación son:

- Conveniencia: ¿para qué sirve?

- Relevancia social: ¿quiénes se benefician con la investigación? ¿cómo?

- Implicaciones prácticas: ¿ayudará a resolver un problema real? ¿tiene implicaciones trascendentales para una amplia gama de problemas prácticos?

- Valor teórico: ¿se llenará algún vacío de conocimiento?

- Utilidad metodológica. Es importante resaltar que Hernández también incide en que es muy difícil que una investigación pueda responder positivamente a todos estos criterios.>>

- **Viabilidad** <<Se debe considerar si se cuenta con la disponibilidad de recursos para el proyecto, así como los conocimientos requeridos para desarrollarlo. También debe ser posible acceder al contexto o fuentes de información necesarios para modelar el problema o la solución. No sólo debe ser viable el proceso sino los productos del proyecto>>

- **Alcance** <<Se debe indicar las actividades principales o equivalentes que describen cuáles sí se incluyen y cuáles no se incluyen en el **proyecto**. Tener presente que éstos se refieren a los de alto nivel

- primer nivel de descomposición de su proyecto.>>

- **Limitaciones** << Las limitaciones del proyecto establecen condicionantes que afectan su planificación y operación. Algunos ejemplos son: de tiempo, de presupuesto, de funcionalidades, de herramientas, etc. >>

- **Identificación de los riesgos del proyecto** << Haga una lista de todos riesgos del proyecto indicando para cada uno de ellos:

Descripción: descripción del riesgo, identifica claramente lo que puede ocurrir. Síntomas: conjunto de hechos, situaciones que hacen posible que el riesgo se concrete. Probabilidad: percepción del nivel de certeza de que el riesgo vaya a ocurrir. Impacto: percepción del nivel de daño que puede afectar al proyecto Severidad: valor = probabilidad x impacto, que permite calificar a los riesgos Mitigación: acciones que realizará en breve para evitar que los síntomas se den. Contingencia: acciones que realizará en caso el riesgo se concrete. >>

- **Estructura de descomposición del trabajo (EDT)** << Desarrolle la estructura de descomposición del trabajo de su proyecto. La representación del EDT es de libre elección: gráfico, esquemático, etc.>>

- **Lista de tareas** << Elabore una lista de tarea del proyecto − incluya la duración estimada de cada tarea (calculada) cuya unidad es el tiempo: horas, días, semanas, etc. − incluya el esfuerzo asociado a cada tarea (calculada) cuya unidad es tiempo/recurso (o tiempo-recurso): horas-persona, semanas-persona, etc. − incluya el costo estimado de cada tarea (calculada) − la lista de tareas debe incluir actividades de verificación y validación. − la lista de tareas debe incluir las reuniones con su orientador (asesor, profesor de curso, revisor, posibles usuarios o clientes) y las actividades dentro del curso. >>

- **Cronograma del proyecto** << Incluya en un cronograma en donde las tareas incluyan lo siguiente: − fecha de inicio y fin de cada tarea, obtenidas por la planificación, asegurándose que la duración planificada/programada sea mayor a la duración estimada − dependencias entre las tareas Se puede mostrar un diagrama que recoja toda esa información>>

- **Lista de recursos** << Incluya una lista de los recursos indicando descripción del recurso, cantidad y oportunidad en su uso dentro de su proyecto:>>

- **Personas involucradas y necesidades de capacitación** << Haga una lista de todos los involucrados en el proyecto y necesidades de capacitación >>

- **Materiales requeridos para el proyecto** << Haga una lista de todos los materiales requerido. En caso no sea necesario indique que “No aplica” >>

- **Estándares utilizados en el proyecto** << Haga una lista de todos los estándares utilizados sean estos internacionales, nacionales, regionales, distritales, industriales, sectoriales. >>

- **Equipamiento requerido** << Haga una lista de todos los equipos necesarios para su proyecto >>

- **Herramientas requeridas** << Haga una lista de todas las herramientas requeridas para su proyecto >>

- **Costeo del Proyecto** << Debe desarrollar un análisis de costos que incluya siempre:

- costos por el equipo humano

- costos por utilización de equipamiento (usar el concepto de depreciación)

- costos por utilización de software, usar el concepto de TCO (Total cost of ownership)

- amortización

- Debe tener claro que costo es un concepto distinto al de gasto A continuación, se presenta una plantilla mínima genérica que puede ser usada de base para establecer un presupuesto. Para los proyectos de curso es suficiente un solo nivel de partida.

| Ítem | Descripción | Unidad | Cantidad | Valor Unitario (S/.) | Monto Total (S/.) | Monto Acumulado (S/.) |
|---|---|---|---|---|---|---|

− Ejemplo:

| Íte m | Descripción | Unidad |  |  | Cantidad | Valor Unidad (S/.) | Monto Parcial (S/.) | Monto Total (S/.) |
|---|---|---|---|---|---|---|---|---|
| 0 | Costo total del proyecto | --- |  |  | --- | --- | --- | 6,520 |
| 1. | Estudiantes o tesistas | --- |  |  | --- | --- | --- | 890 |
| 1.1 | Estudiante 1 | Horas |  |  | 40 | 10 | 400 |  |
| 1.2 | Estudiante 2 | Horas |  |  | 39 | 10 | 390 |  |
|  | : | Horas |  |  |  |  |  |  |
| 1.n | Estudiante n | Horas |  |  | 10 | 10 | 100 |  |
| 2. | Otros participantes (en caso aplique) | --- |  |  | --- | --- | --- | 440 |
| 2.1 | Otro participante 1 | Horas |  |  | 2 | 20 | 40 |  |
|  |  | Horas |  |  |  |  |  |  |
| 2.m | Otro participante m | Horas |  |  | 10 | 10 | 100 |  |
| 3. | Servicios y consultoría (en caso aplique) | --- |  |  | --- | --- | --- | 500 |
| 3.1 | Servicio o consultoría 1 | Informe |  |  | 1 | 500 | 500 |  |
| 3.2 | Servicio o consultoría 2 | Informe |  |  |  |  |  |  |
|  |  | Informe |  |  |  |  |  |  |
| 3.p | Servicio o consultoría p | Informe |  |  |  |  |  |  |
| 4. | Materiales e insumos (en caso aplique) | --- |  |  | --- | --- | --- | 310 |
| 4.1 | Materiales e insumos 1 | Unidad |  |  | 3 | 20 | 60 |  |
|  |  | Unidad |  |  |  |  |  |  |
| 4.q | Materiales e insumos q | Unidad |  |  | 2 | 120 | 240 |  |
| 5. | Bienes y equipos | Unid1 | C a n t | Unid2 | Cant2- | - | - | 1,900 |

|  |  |  | 1 - |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| 5.1 | Computadoras | Equipo | 3 | Horas | 40 | 5 | 600 |  |
| 5.2 | Servidores | Equipo | 1 | Horas | 30 | 10 | 300 |  |
| 5.3 | Licencia de software ZZZZ | Unidad | 1 | Horas | 10 | 50 | 500 |  |
| 5.5 | Bien o equipo 5 |  |  |  |  |  |  |  |
| 5.r | Bien o equipo r |  |  |  |  |  |  |  |
| 6. | Pasajes y viáticos | Unid1 | C a n t 1 - | Unid2 | Cant2- | - | - | 2,480 |
| 6.1 | Movilidad local (trabajo de campo) | Viajes/ Día | 4 | Día | 10 | 10 | 400 |  |
| 6.2 | Pasaje para evento nacional | Unidad | 1 | Person as | 2 | 500 | 1000 |  |
| 6.3 | Viáticos | Días | 4 | Person as | 2 | 60 | 480 |  |
| 6.4 | Alojamiento | Noches | 3 | Person as | 2 | 100 | 600 |  |
| 6.5 | Pasajes o viáticos 5 |  |  |  |  |  |  |  |
| 6.s | Pasajes o viáticos s |  |  |  |  |  |  |  |
