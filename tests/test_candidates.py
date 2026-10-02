from automl_agents.candidates import propose
from automl_agents.catalog import CATALOG
from automl_agents.config import RunConfig
from automl_agents.dataset import DatasetProfile


def _profile(n=500):
    return DatasetProfile("x", ["a", "b"], {"a": n // 2, "b": n // 2}, n, 1.0, (200, 200), 100)


class FakeLLM:
    available = True

    def __init__(self, payload=None, boom=False):
        self.payload, self.boom = payload, boom

    def complete_json(self, system, user, max_tokens=0):
        if self.boom:
            raise RuntimeError("sin red")
        return self.payload


def test_rules_reduce_space_and_respect_tier():
    r = propose(_profile(), [], RunConfig(dataset_dir="x", resource_tier="low"), None)
    assert r["mode"] == "reglas"
    assert r["space"]["reduction_pct"] > 90
    assert all(CATALOG[c.architecture].tier == "light" for c in r["candidates"])
    assert any(d["architecture"] == "VGG16" for d in r["discarded"])


def test_llm_output_is_validated_against_catalog():
    payload = {"candidatas": [
        {"arquitectura": "ModeloInventado", "justificacion": "x"},
        {"arquitectura": "VGG16", "justificacion": "pesado, no permitido en tier low"},
        {"arquitectura": "EfficientNetB0", "justificacion": "ok", "lr": [3e-4, 0.5], "dropout": [0.2], "batch": [32], "augment": [True]},
    ], "descartadas": []}
    r = propose(_profile(), [], RunConfig(dataset_dir="x"), FakeLLM(payload))
    assert r["mode"] == "llm"
    assert [c.architecture for c in r["candidates"]] == ["EfficientNetB0"]
    assert r["candidates"][0].lr_values == [3e-4]  # 0.5 no está en la rejilla


def test_llm_failure_falls_back_to_rules():
    r = propose(_profile(), [], RunConfig(dataset_dir="x"), FakeLLM(boom=True))
    assert r["mode"] == "reglas" and "Fallo del LLM" in r["note"]
