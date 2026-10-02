from automl_agents.catalog import CATALOG, GRID_SIZE, full_space_size


def test_grid_and_full_space():
    assert GRID_SIZE == 90
    assert full_space_size() == 90 * len(CATALOG)


def test_kb_indexes_all_docs(kb):
    assert kb.collection.count() >= 20


def test_kb_query_finds_relevant_guide(kb):
    ids = [d.id for d in kb.query("dataset con muy pocas imágenes por clase", k=3)]
    assert "guide_small_dataset" in ids


def test_kb_reference_eval_meets_baseline(kb):
    """R1.1: consultas de prueba frente a un conjunto de referencia etiquetado a mano."""
    m = kb.evaluate(k=3)
    assert m["recall_at_k"] >= 0.5, m
