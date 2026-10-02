from automl_agents.candidates import Candidate
from automl_agents.config import RunConfig
from automl_agents.controller import decide


def _c(cid):
    return Candidate(cid, "MobileNetV2", "", [], [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.0, 0.2, 0.4], [16, 32], [True], 96)


def _rec(cid, acc, lr=1e-3, pruned=False):
    return {"candidate_id": cid, "val_acc": acc, "pruned": pruned, "epochs": 3,
            "params": {"lr": lr, "dropout": 0.2, "batch": 16, "augment": True}}


def test_discards_laggard_and_adjusts_leader():
    cfg = RunConfig(dataset_dir="x", rounds=2)
    nxt, decs, stop = decide(1, [_c("c1"), _c("c2")], [_rec("c1", 0.9), _rec("c2", 0.5)], cfg)
    acts = {d["candidate_id"]: d["action"] for d in decs}
    assert acts == {"c1": "ajustar", "c2": "descartar"}
    assert [c.id for c in nxt] == ["c1"] and not stop
    assert nxt[0].lr_values == [3e-4, 1e-3, 3e-3]  # vecinos de 1e-3


def test_stops_on_last_round_and_on_target():
    cfg = RunConfig(dataset_dir="x", rounds=2, target_accuracy=0.95)
    assert decide(2, [_c("c1")], [_rec("c1", 0.6)], cfg)[2] is True
    assert decide(1, [_c("c1")], [_rec("c1", 0.97)], cfg)[2] is True
