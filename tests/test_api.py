import io
import zipfile

from fastapi.testclient import TestClient

from automl_agents import api
from automl_agents.demo import make_demo_dataset


def _zip_bytes(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in folder.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent))
    return buf.getvalue()


def test_upload_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(api, "_execute", lambda cfg: api._set_status(cfg.run_dir, "done"))
    c = TestClient(api.app)
    assert "Construya su clasificador" in c.get("/").text

    bad = c.post("/api/runs", files={"file": ("x.zip", b"no es zip")})
    assert bad.status_code == 400 and "ZIP" in bad.json()["detail"]

    make_demo_dataset(tmp_path / "mi_dataset", per_class=10)
    ok = c.post("/api/runs", files={"file": ("d.zip", _zip_bytes(tmp_path / "mi_dataset"))},
                data={"use_case": "formas", "budget": "rapido"})
    assert ok.status_code == 200, ok.text
    rid = ok.json()["run_id"]
    assert ok.json()["dataset"]["n_classes"] == 3
    assert c.get(f"/api/runs/{rid}").json()["status"] == "done"
    assert c.get("/api/runs/../../etc").status_code in (404, 422)


def test_zip_slip_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_RUNS_DIR", str(tmp_path / "runs"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    r = TestClient(api.app).post("/api/runs", files={"file": ("d.zip", buf.getvalue())})
    assert r.status_code == 400
