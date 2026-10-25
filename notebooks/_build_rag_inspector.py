"""Regenera el notebook autónomo desde la KB actual, sin incluir datos ni secretos."""
from pathlib import Path
import ast
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    item = {"cell_type": kind, "id": f"rag-{len(cells):02d}", "metadata": {},
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


def md(source):
    cell("markdown", source)


def code(source):
    cell("code", source)


files = {"kb.py": (ROOT / "src/automl_agents/kb.py").read_text("utf-8")}
for p in sorted((ROOT / "src/automl_agents/knowledge").iterdir()):
    if p.suffix in {".md", ".json"}:
        files[f"knowledge/{p.name}"] = p.read_text("utf-8")
for name in ("pipeline.py", "candidates.py"):
    files[f"contexto/{name}"] = (ROOT / "src/automl_agents" / name).read_text("utf-8")
manifest = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
snapshot_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]

md("""
    # Mira por dentro el RAG de tu proyecto

    **Abre este `.ipynb` en Google Colab → Archivo → Subir notebook. Ejecuta las celdas en orden.**
    Basta un entorno de CPU; no necesitas GPU, claves de API, dataset de imágenes ni subir el repositorio.
    La primera instalación necesita internet. El modo semántico opcional también descarga un modelo.

    Este notebook incluye una **instantánea de los 23 Markdown y del `kb.py` real del proyecto**.
    Reconstruye una copia del índice en Colab: **no abre ni sincroniza la base de tu PC**.
    No contiene `.env`, imágenes ni credenciales. Si cambias los documentos locales, regenera el notebook
    con `python notebooks/_build_rag_inspector.py` antes de volver a subirlo.

    ## 1. ¿Dónde está la «base RAG»?

    | Pieza | Archivo o carpeta del proyecto | Función |
    |---|---|---|
    | Conocimiento legible | `src/automl_agents/knowledge/*.md` | 11 fichas de arquitecturas y 12 guías |
    | Indexación y búsqueda | `src/automl_agents/kb.py` | Calcula vectores, guarda documentos y recupera por similitud |
    | Índice persistido de la web | `runs/.chroma/` (o `$AUTOML_RUNS_DIR/.chroma/`) | `chroma.sqlite3` y archivos del índice vectorial |
    | Consultas del sistema | `Pipeline.n_retrieve()` en `pipeline.py` | Busca según el perfil y caso de uso |
    | Uso de evidencia en generación | `propose_llm()` en `candidates.py` | Envía textos recuperados al LLM para proponer candidatas |
    | Preguntas de evaluación | `knowledge/reference_queries.json` | 10 consultas y sus IDs relevantes; no se indexa como documento |

    **RAG es el proceso completo**, no un único archivo: documentos → vectores → recuperación →
    contexto para el LLM → propuesta. Chroma es el almacén; los Markdown son la fuente legible.
    Los PDF de `papers/` **no se incorporan automáticamente**: `_load()` solo lee `knowledge/*.md`.
    En esta versión cada Markdown se indexa completo como un registro; no hay partición en fragmentos.

    La web configura persistencia. En cambio, `RunConfig.kb_persist_dir` vale `None` por defecto:
    la CLI normal y `kb-eval` usan Chroma en memoria y reconstruyen la base al iniciar.
    En la inspección local del 19/10/2026 existía `kb_hash`, con 23 registros de 512 dimensiones.
""")
md("""
    ## 2. Instalar solo lo necesario para inspeccionar

    Usamos la misma versión de Chroma comprobada localmente. No instalamos todo el proyecto:
    así evitamos cargar entrenamiento, TensorFlow, Optuna y el servidor web.
    Si Colab pide reiniciar la sesión tras instalar, hazlo y continúa desde esta celda.
""")
code("""
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "chromadb==1.5.9"])
""")
md("""
    ## 3. Restaurar la copia incluida

    La siguiente celda contiene los textos y el código original como un diccionario de Python,
    junto con sus huellas SHA-256. Puedes desplegarla para leerlos. Solo escribe en una carpeta temporal.
    `contexto/` sirve para leer cómo se integra el RAG; esos módulos no se ejecutan.
""")
code("SNAPSHOT = " + repr(files) + "\n\nMANIFEST = " + repr(manifest) + "\nSNAPSHOT_ID = " + repr(snapshot_id))
code("""
    import hashlib
    import importlib.util
    import json
    import tempfile
    from pathlib import Path
    from html import escape
    import numpy as np
    from IPython.display import display, Markdown, HTML, SVG

    BASE = (Path('/content') if Path('/content').is_dir() else Path(tempfile.gettempdir())) / ('automl_rag_' + SNAPSHOT_ID)
    SOURCE = BASE / 'fuentes'
    for name, text in SNAPSHOT.items():
        assert hashlib.sha256(text.encode()).hexdigest() == MANIFEST[name], name
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')

    spec = importlib.util.spec_from_file_location('rag_proyecto_snapshot', SOURCE / 'kb.py')
    kb_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kb_module  # requerido por dataclasses
    spec.loader.exec_module(kb_module)

    def tabla(rows):
        if not rows:
            print('Sin filas.')
            return
        headers = list(rows[0])
        html = '<table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>'
        for row in rows:
            html += '<tr>' + ''.join('<td>' + escape(str(row[h])) + '</td>' for h in headers) + '</tr>'
        display(HTML(html + '</tbody></table>'))

    print('Instantánea:', SNAPSHOT_ID)
    print('Fuentes restauradas en:', SOURCE)
    print('Markdown incluidos:', len(list((SOURCE / 'knowledge').glob('*.md'))))
""")
md("""
    ## 4. Leer la base antes de convertirla en vectores

    Esta es la parte que más conviene revisar para tu tesis. Son fichas iniciales breves;
    su existencia no prueba que el conocimiento esté validado o respaldado con citas por afirmación.
    El inventario y los textos permiten revisar su cobertura, precisión y fuentes.
""")
code("""
    documentos = sorted((SOURCE / 'knowledge').glob('*.md'))
    tabla([{'ID': p.stem, 'Tipo': p.stem.split('_')[0],
            'Título': p.read_text('utf-8').splitlines()[0].lstrip('# '),
            'Caracteres': len(p.read_text('utf-8'))} for p in documentos])
    print('Arquitecturas:', sum(p.stem.startswith('arch_') for p in documentos))
    print('Guías:', sum(p.stem.startswith('guide_') for p in documentos))
""")
code("""
    DOCUMENTO = 'guide_small_dataset'  # cambia por cualquier ID del inventario
    path = SOURCE / 'knowledge' / (DOCUMENTO + '.md')
    if path not in documentos:
        raise ValueError('Elige un ID del inventario.')
    display(Markdown(path.read_text('utf-8')))

    VER_TODOS = False  # True muestra los 23 textos completos
    if VER_TODOS:
        for p in documentos:
            display(Markdown('---\\n**' + p.name + '**\\n\\n' + p.read_text('utf-8')))
""")
md("""
    ## 5. Crear el índice con el mismo `KnowledgeBase` del proyecto

    `hash` reproduce el modo encontrado en tu base local: vectores léxicos de 512 dimensiones,
    sin descarga de modelo. Se basa en coincidencias de tokens normalizados y pares de tokens;
    **no es un modelo semántico entrenado** y puede perder sinónimos o paráfrasis.

    Para probar embeddings semánticos cambia `BACKEND` a `sentence-transformers` y vuelve a ejecutar
    desde esta celda. Descargará `paraphrase-multilingual-MiniLM-L12-v2`; la colección se llamará
    `kb_st`. Cada ejecución crea un directorio nuevo para evitar residuos de ensayos anteriores.
    El proyecto usa `auto` por defecto: intenta el modelo semántico y cae a hash si falla.
""")
code("""
    BACKEND = 'hash'  # 'hash' o 'sentence-transformers'
    if BACKEND not in {'hash', 'sentence-transformers'}:
        raise ValueError('Usa hash o sentence-transformers.')
    if BACKEND == 'sentence-transformers':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sentence-transformers>=3,<6'])

    import uuid
    import chromadb
    INDEX_DIR = BASE / ('indice_' + BACKEND + '_' + uuid.uuid4().hex[:8])
    kb = kb_module.KnowledgeBase(persist_dir=INDEX_DIR, backend=BACKEND,
                                 knowledge_dir=SOURCE / 'knowledge')
    print('Versión de Chroma:', chromadb.__version__)
    print('Backend real:', kb.embedder.name)
    print('Colección:', kb.collection.name)
    print('Registros:', kb.collection.count())
    print('Índice creado en:', INDEX_DIR)
    print('Archivos:', [p.name for p in INDEX_DIR.iterdir()])
    assert kb.collection.count() == len(documentos)
""")
md("""
    ## 6. Ver lo que almacena Chroma

    Cada registro incluye ID, texto, metadatos y un vector numérico. No necesitas abrir el archivo
    SQLite para inspeccionarlo: `collection.get()` permite ver estos datos desde Python.
    No edites manualmente `chroma.sqlite3`; el índice también tiene archivos asociados.
""")
code("""
    almacen = kb.collection.get(include=['documents', 'metadatas', 'embeddings'])
    orden = sorted(range(len(almacen['ids'])), key=lambda i: almacen['ids'][i])
    ids = [almacen['ids'][i] for i in orden]
    vectores = np.asarray(almacen['embeddings'], dtype=float)[orden]
    textos = [almacen['documents'][i] for i in orden]
    metadatos = [almacen['metadatas'][i] for i in orden]
    tabla([{'ID': doc_id, 'Título': meta['title'], 'Tipo': meta['kind'],
            'Dimensiones': vectores.shape[1], 'Inicio del texto': text[:140]}
           for doc_id, meta, text in zip(ids, metadatos, textos)])
    print('Matriz (documentos × dimensiones):', vectores.shape)
    print('Primeras 20 coordenadas de', ids[0], ':', vectores[0, :20].round(4).tolist())
    print('Normas de los vectores:', np.linalg.norm(vectores, axis=1).round(4).tolist())
""")
md("""
    ## 7. Hacer una consulta y leer la evidencia recuperada

    Cambia `CONSULTA` y `K`. El proyecto calcula `score = 1 − distancia coseno`.
    Un valor mayor indica más similitud en este espacio; **no es una probabilidad de acierto**.
    En modo hash los valores son especialmente sensibles al vocabulario compartido.
""")
code("""
    CONSULTA = 'dataset con muy pocas imágenes por clase'
    K = 3
    if not 1 <= K <= len(ids):
        raise ValueError('K debe estar entre 1 y el número de documentos.')
    recuperados = kb.query(CONSULTA, k=K)
    tabla([{'Puesto': i, 'ID': d.id, 'Título': d.title, 'Similitud coseno': round(d.score, 4)}
           for i, d in enumerate(recuperados, 1)])
    for d in recuperados:
        display(Markdown('### ' + d.id + '\\n**' + d.title + '**\\n\\n' + d.text))
""")
md("""
    ## 8. Ver los vectores como mapa

    El gráfico proyecta documentos y consulta a dos dimensiones con PCA (SVD con NumPy).
    Es una ayuda visual: la búsqueda real se hace en **todas** las dimensiones. La proyección puede
    distorsionar las distancias, y los grupos de hash no deben interpretarse como comprensión semántica.
    Los puntos tienen su ID al pasar el cursor; la consulta aparece en rojo y los recuperados con borde oscuro.
""")
code("""
    vector_consulta = np.asarray(kb.embedder.embed([CONSULTA]), dtype=float)
    matriz = np.vstack([vectores, vector_consulta])
    centrada = matriz - matriz.mean(axis=0)
    u, s, vt = np.linalg.svd(centrada, full_matrices=False)
    xy = u[:, :2] * s[:2]
    varianza = (s[:2] ** 2).sum() / (s ** 2).sum() if np.any(s) else 0.0
    minimo, rango = xy.min(axis=0), np.ptp(xy, axis=0)
    pantalla = (xy - minimo) / np.where(rango == 0, 1, rango)
    seleccion = {d.id for d in recuperados}
    partes = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 590" role="img" aria-label="PCA de documentos y consulta">',
              '<rect width="940" height="590" fill="#f8fafc"/>',
              '<text x="30" y="32" font-size="20" fill="#0f172a">Documentos y consulta · proyección PCA</text>',
              '<text x="30" y="58" font-size="13">Azul: arquitectura · verde: guía · rojo: consulta</text>']
    for i, nombre in enumerate(ids + ['CONSULTA']):
        x, y = 55 + pantalla[i, 0] * 820, 505 - pantalla[i, 1] * 400
        color = '#dc2626' if nombre == 'CONSULTA' else '#2563eb' if nombre.startswith('arch_') else '#059669'
        radio = 9 if nombre == 'CONSULTA' else 6
        borde = '#0f172a' if nombre in seleccion else color
        partes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radio}" fill="{color}" stroke="{borde}" stroke-width="3"><title>{escape(nombre)}</title></circle>')
        etiqueta = 'CONSULTA' if nombre == 'CONSULTA' else str(i + 1)
        partes.append(f'<text x="{x + 10:.2f}" y="{y - 5:.2f}" font-size="11">{etiqueta}</text>')
    partes.append(f'<text x="30" y="562" font-size="13">Varianza representada: {varianza:.1%}. Las coordenadas son una proyección, no el ranking.</text></svg>')
    grafico_svg = ''.join(partes)
    display(SVG(grafico_svg))
    tabla([{'Número': i + 1, 'ID': nombre, 'Recuperado': nombre in seleccion} for i, nombre in enumerate(ids)])
""")
md("""
    ## 9. Evaluar las 10 preguntas de referencia

    Precisión@k = relevantes recuperados / k. Recall@k (exhaustividad) = relevantes recuperados /
    relevantes etiquetados. La tabla permite ver errores concretos. Las etiquetas también necesitan revisión:
    un documento sin etiqueta relevante no necesariamente es irrelevante.

    Con la referencia actual hay 23 etiquetas en 10 preguntas (2–3 por pregunta). A k=5, el techo
    matemático de la precisión media es **23 / 50 = 0.46**, incluso si recuperas todos los relevantes.
    Esta referencia no permite demostrar una meta de precisión@5 ≥ 0.70 sin revisar el protocolo y las etiquetas.
""")
code("""
    referencias = json.loads((SOURCE / 'knowledge/reference_queries.json').read_text('utf-8'))
    K_EVAL = 3
    if not 1 <= K_EVAL <= len(ids):
        raise ValueError('K_EVAL fuera de rango.')
    filas_eval = []
    for ref in referencias:
        esperados = set(ref['relevant'])
        assert esperados <= set(ids), 'Referencia a un documento inexistente'
        obtenidos = [d.id for d in kb.query(ref['query'], k=K_EVAL)]
        aciertos = len(set(obtenidos) & esperados)
        filas_eval.append({'Consulta': ref['query'], 'Esperados': ', '.join(ref['relevant']),
                           'Obtenidos': ', '.join(obtenidos),
                           'Precisión': round(aciertos / K_EVAL, 3),
                           'Recall': round(aciertos / len(esperados), 3)})
    tabla(filas_eval)
    metricas = kb.evaluate(k=K_EVAL)
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print('Techo de precisión con estas etiquetas:',
          round(sum(min(K_EVAL, len(set(r['relevant']))) for r in referencias) / (len(referencias) * K_EVAL), 3))
""")
md("""
    ## 10. ¿Dónde entra el LLM? ¿Esto ya genera una respuesta?

    Hasta aquí inspeccionamos **indexación y recuperación**. El sistema completo añade los textos al
    contexto del LLM en `candidates.propose_llm()` y valida sus propuestas contra catálogo y rejilla.
    La función `Pipeline.n_retrieve()` hace varias consultas y `retrieve_many()` une los resultados,
    deduplica IDs y conserva el mayor score de cada documento. Aquí puedes ver ambos métodos originales.

    **Sin LLM disponible**, `propose_rules()` selecciona por reglas, perfil y recursos. Los documentos
    recuperados se usan para asociar IDs de fuentes; su texto no determina ese ranking. Por tanto,
    ejecutar recuperación por sí sola no demuestra que las decisiones hayan sido generadas mediante RAG.
    Este notebook no llama APIs ni genera propuestas con LLM.
""")
code("""
    import ast
    def ver_funcion(archivo, nombre):
        source = (SOURCE / 'contexto' / archivo).read_text('utf-8')
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre)
        display(Markdown('```python\\n' + ast.get_source_segment(source, fn) + '\\n```'))

    ver_funcion('pipeline.py', 'n_retrieve')
    ver_funcion('candidates.py', 'propose_llm')
""")
code("""
    consultas_ejemplo = [
        'dataset pequeño con pocas imágenes de clasificación de imágenes',
        'tamaño de lote y memoria de GPU',
        'arquitecturas ligeras con transferencia de aprendizaje',
    ]  # subconjunto ilustrativo: el pipeline usa más consultas según el perfil
    evidencia = kb.retrieve_many(consultas_ejemplo, k=3)
    contexto_para_llm = [{'id': d.id, 'titulo': d.title, 'texto': d.text} for d in evidencia]
    print('Consultas:', len(consultas_ejemplo), '→ documentos únicos:', len(evidencia))
    print('Así luce el campo evidencia_recuperada del contexto para el LLM:')
    print(json.dumps(contexto_para_llm, ensure_ascii=False, indent=2))
""")
md("""
    ## 11. Guardar una copia legible de la inspección

    Se exportan JSON con documentos, metadatos, vectores, consulta y métricas, y el gráfico SVG.
    El JSON sirve para inspección; no es un respaldo nativo completo de Chroma.
    Cambia `DESCARGAR` a `True` si quieres descargar el ZIP desde Colab.
    Las carpetas temporales de Colab pueden desaparecer cuando termine la sesión.
""")
code("""
    import zipfile
    EXPORT = BASE / 'exportacion'
    EXPORT.mkdir(exist_ok=True)
    registros = [{'id': doc_id, 'document': text, 'metadata': meta, 'embedding': v.tolist()}
                 for doc_id, text, meta, v in zip(ids, textos, metadatos, vectores)]
    contenido = {'snapshot_id': SNAPSHOT_ID, 'chroma_version': chromadb.__version__,
                 'backend': kb.embedder.name, 'collection': kb.collection.name,
                 'records': registros, 'query': CONSULTA,
                 'retrieved': [d.to_dict() for d in recuperados], 'evaluation': metricas}
    (EXPORT / 'inspeccion_rag.json').write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding='utf-8')
    (EXPORT / 'mapa_rag.svg').write_text(grafico_svg, encoding='utf-8')
    (EXPORT / 'manifest_fuentes.json').write_text(json.dumps(MANIFEST, indent=2), encoding='utf-8')
    zip_path = BASE / 'inspeccion_rag.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in EXPORT.iterdir():
            zf.write(p, p.name)
        for p in SOURCE.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                zf.write(p, 'fuentes/' + p.relative_to(SOURCE).as_posix())
    print('Exportación:', zip_path)
    DESCARGAR = False
    if DESCARGAR:
        from google.colab import files
        files.download(str(zip_path))
""")
md("""
    ## Para seguir explorando tu proyecto

    - En tu PC, abre cualquier `src/automl_agents/knowledge/*.md`: ya puedes leer la base sin Colab.
    - Revisa `runs/<id>/events.jsonl`, evento `rag`, para ver IDs y scores de una corrida real.
    - En `runs/<id>/reports/01_criterios_acotacion.json`, revisa `mode`, `criteria.retrieved_docs`
      y las fuentes de cada candidata. `mode` distingue si se usó `llm` o `reglas`.
    - Para ampliar el conocimiento, añade Markdown al directorio `knowledge/` y vuelve a construir
      el índice. El código actual hace `upsert`, pero **no elimina registros antiguos** si borras o
      renombras documentos: usa un índice nuevo al experimentar con una KB modificada.
    - Compara `hash` y `sentence-transformers` con las mismas consultas y etiquetas. Mide la mejora;
      no la des por supuesta.

    Referencias de uso: [documentación oficial de Chroma](https://docs.trychroma.com/docs/collections/manage-collections)
    y [preguntas frecuentes de Google Colab](https://research.google.com/colaboratory/faq.html).
""")

for c in cells:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "colab": {"name": "Explorar_RAG_Colab.ipynb", "provenance": []},
    "rag_snapshot": {"sha256_manifest": manifest, "id": snapshot_id, "date": "2026-10-19"}},
    "nbformat": 4, "nbformat_minor": 5}
output = ROOT / 'notebooks/Explorar_RAG_Colab.ipynb'
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'{output}: {len(cells)} celdas, instantánea {snapshot_id}')
