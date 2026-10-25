import io
import json
import threading
import zipfile

from fastapi.testclient import TestClient
import pytest

from automl_agents import api
from automl_agents.config import RunConfig
from automl_agents.demo import make_demo_dataset
from automl_agents.events import CancellationToken, RunLogger, write_json
from automl_agents.jobs import JobQueue, JobWorker
from automl_agents.state_store import StateStore, run_lock, training_lock


def _zip_bytes(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in folder.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent))
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_RUNS_DIR", str(tmp_path / "runs"))
    # Sin lifespan: cada prueba avanza el worker de forma explícita.
    return TestClient(api.app)


@pytest.fixture
def upload(tmp_path):
    return _zip_bytes(make_demo_dataset(tmp_path / "imagenes", per_class=10))


def create(client, upload, **data):
    response = client.post("/api/runs", files={"file": ("imagenes.zip", upload)}, data=data)
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


def test_upload_and_assets(client, upload):
    assert "Construya su clasificador" in client.get("/").text
    assert client.get("/assets/app.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/index4.html").status_code == 404
    rid = create(client, upload, budget="rapido", mode="rules", pretrained="false")
    d = client.get(f"/api/runs/{rid}").json()
    assert d["status"] == "queued" and d["capabilities"] == {"cancel": True, "resume": False}
    assert d["budget"]["limits"]["max_training_attempts"] == 12
    cfg = json.loads(api.queue().get(rid)["config"])
    assert not cfg["pretrained"] and not cfg["use_llm"]
    assert (api.runs_dir() / rid / "data/split_manifest.json").is_file()
    assert client.get(f"/api/runs/{rid}/download").status_code == 409
    assert client.get(f"/api/runs/{rid}/report").status_code == 409
    assert client.post(f"/api/runs/{rid}/resume").status_code == 409
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == rid
    assert client.get("/api/runs/../../etc").status_code in (404, 422)
    assert client.get("/api/runs/unknown123").status_code == 404


@pytest.mark.parametrize("data,field", [({"budget":"x"},"budget"),({"mode":"x"},"mode"),({"tier":"x"},"tier")])
def test_bad_options(client, upload, data, field):
    response=client.post("/api/runs", files={"file":("d.zip",upload)}, data=data)
    assert response.status_code == 400 and response.json()["detail"]["field"] == field


def test_bad_datasets(client, tmp_path, monkeypatch):
    for payload in (b"not zip",):
        response=client.post("/api/runs",files={"file":("x.zip",payload)})
        assert response.status_code == 400 and "ZIP" in response.json()["detail"]["message"]
    for name in ("../evil.txt", "C:/evil.txt"):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w") as z:
            z.writestr(name,"x")
        assert client.post("/api/runs",files={"file":("d.zip",buffer.getvalue())}).status_code == 400
    ds=make_demo_dataset(tmp_path / "small",per_class=3)
    response=client.post("/api/runs",files={"file":("d.zip",_zip_bytes(ds))})
    assert response.status_code == 400
    ds=make_demo_dataset(tmp_path / "corrupt",per_class=10)
    next(ds.rglob("*.png")).write_bytes(b"broken")
    response=client.post("/api/runs",files={"file":("d.zip",_zip_bytes(ds))})
    assert response.status_code == 400 and response.json()["detail"]["code"] == "corrupt_image"
    monkeypatch.setattr(api,"MAX_UPLOAD",10)
    assert client.post("/api/runs",files={"file":("d.zip",b"x"*11)}).status_code == 400
    assert api.queue().list() == []


def test_queued_cancel_resume_is_durable(client, upload):
    rid=create(client,upload)
    assert client.post(f"/api/runs/{rid}/cancel").json()["status"] == "cancelled"
    queue=JobQueue(api.runs_dir())  # nuevo objeto tras un reinicio
    assert queue.get(rid)["cancel_requested"] == 1
    assert CancellationToken(api.runs_dir()/rid).is_cancelled()
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 409
    assert client.post(f"/api/runs/{rid}/resume").status_code == 200
    assert not CancellationToken(api.runs_dir()/rid).is_cancelled()
    assert queue.get(rid)["status"] == "queued"
    assert client.post(f"/api/runs/{rid}/resume").status_code == 409


def test_running_cancel_and_global_worker_lock(client, upload):
    rid=create(client,upload)
    second=create(client,upload)
    entered,release=threading.Event(),threading.Event()
    executed=[]
    def runner(cfg):
        executed.append(cfg.run_id)
        entered.set()
        assert release.wait(10)
        assert CancellationToken(cfg.run_dir).is_cancelled()
        return {"status":"cancelled"}
    worker=JobWorker(api.queue(),runner)
    thread=threading.Thread(target=worker.tick)
    thread.start()
    try:
        assert entered.wait(10)
        JobWorker(api.queue(),lambda cfg: pytest.fail("Dos workers entrenando")).tick()
        assert api.queue().get(second)["status"] == "queued"
        assert client.post(f"/api/runs/{rid}/cancel").json()["status"] == "cancelling"
        assert client.post(f"/api/runs/{rid}/resume").status_code == 409
        assert client.get(f"/api/runs/{rid}").json()["capabilities"]["cancel"] is False
    finally:
        release.set();thread.join(10)
    assert executed == [rid]
    assert api.queue().get(rid)["status"] == "cancelled"


def test_restart_reconciles_and_preserves_queue(client, upload):
    rid=create(client,upload)
    second=create(client,upload)
    queue=api.queue()
    cfg=queue.claim()
    assert cfg.run_id == rid
    store=StateStore(cfg.run_dir)
    meta=dict(attempt_id="attempt",trial_id="trial",candidate_id="candidate",candidate_version=1,
              study_id="study",params={},seed=42,round=1,trial=0,architecture="MobileNetV2")
    store.begin_attempt("slot",meta);store.put("consumed",{"training_attempts":1});store.close()
    with run_lock(queue.root / ".web-worker"):
        JobQueue(queue.root).reconcile()
    assert queue.get(rid)["status"] == "interrupted"
    store=StateStore(cfg.run_dir)
    assert store.attempts()[0]["status"] == "interrupted"
    assert store.get("consumed")["training_attempts"] == 1
    store.close()
    assert queue.get(second)["status"] == "queued"
    assert client.get(f"/api/runs/{rid}").json()["capabilities"]["resume"]


def test_telemetry_and_partial_downloads(client, upload):
    rid=create(client,upload)
    queue=api.queue();cfg=queue.claim();rd=cfg.run_dir
    store=StateStore(rd)
    store.put("node","entrenar");store.put("plan",{"tasks":[{"task_id":"task","status":"running","description":"Comparar modelos"}]})
    meta=dict(attempt_id="attempt",trial_id="trial",candidate_id="candidate",candidate_version=1,
              study_id="study",params={},seed=42,round=1,trial=0,architecture="MobileNetV2")
    store.begin_attempt("slot",meta)
    write_json(rd/"attempts/attempt/curves.json",{"accuracy":[.5,.7],"val_accuracy":[.4,.6]})
    RunLogger(rd).emit("train","Entrenando")
    with (rd/"events.jsonl").open("a") as f:f.write('{"ts":')
    d=client.get(f"/api/runs/{rid}").json()
    assert d["progress"]["current_trial"]["epochs_completed"] == 2
    assert d["active_task"]["description"] == "Comparar modelos"
    assert d["events"][0]["message"] == "Entrenando"
    model=rd/"models/winner.keras";model.parent.mkdir();model.write_bytes(b"model")
    store.put("selection",{"checkpoint":str(model)})
    store.put("verification",{"status":"failed"})
    store.close()
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 409
    (rd/"reports").mkdir();(rd/"reports/06_reporte_final.md").write_text("Reporte parcial")
    (rd/"artifact").mkdir();(rd/"artifact/model.keras").write_bytes(b"broken")
    queue.finish(rid,"artifact_failed")
    d=client.get(f"/api/runs/{rid}").json()
    assert set(d["downloads"]) == {"model","reports","all"}
    assert client.get(f"/api/runs/{rid}/download?kind=artifact").status_code == 409
    assert client.get(f"/api/runs/{rid}/download?kind=model").content == b"model"
    response=client.get(f"/api/runs/{rid}/download")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        assert "model.keras" in z.namelist()
        assert all(not n.startswith("artifact/") for n in z.namelist())
    assert not list(rd.glob("download-*.zip"))


def test_worker_errors_are_persistent(client, upload):
    rid=create(client,upload)
    def fail(cfg):raise ValueError("Los datos cambiaron")
    JobWorker(api.queue(),fail).tick()
    d=client.get(f"/api/runs/{rid}").json()
    assert d["status"] == "failed" and "datos" in d["message"]
    assert d["capabilities"]["resume"]


def test_cli_training_defers_web_job(client, upload):
    rid=create(client,upload)
    from automl_agents.pipeline import run_pipeline
    with training_lock(api.runs_dir()):
        JobWorker(api.queue(),run_pipeline).tick()
    assert api.queue().get(rid)["status"] == "queued"


def test_cancel_before_first_node_can_resume(client, upload, monkeypatch):
    rid=create(client,upload)
    from automl_agents.pipeline import run_pipeline, Pipeline
    # La cancelación puede llegar mientras el worker construye el pipeline.
    def runner(cfg):
        (cfg.run_dir/"cancel.requested").touch()
        return run_pipeline(cfg)
    JobWorker(api.queue(),runner).tick()
    d=client.get(f"/api/runs/{rid}").json()
    assert d["status"] == "cancelled" and "reports" in d["downloads"]
    assert d["progress"]["attempts_started"] == 0
    assert client.post(f"/api/runs/{rid}/resume").status_code == 200
    # El marcador se limpia antes de poner de nuevo el job en cola.
    assert not (api.runs_dir()/rid/"cancel.requested").exists()


def test_lifespan_starts_pending_jobs(client, upload, monkeypatch):
    rid=create(client,upload)
    done=threading.Event()
    def runner(cfg):done.set();return {"status":"no_model"}
    monkeypatch.setattr("automl_agents.jobs.execute_job",runner)
    with TestClient(api.app):
        assert done.wait(10)
    assert api.queue().get(rid)["status"] == "no_model"


def test_web_pipeline_real(client, upload, monkeypatch):
    """Carga → worker real → modelo/reportes/API exportada, sin red ni LLM."""
    pytest.importorskip("tensorflow")
    monkeypatch.setattr("automl_agents.packaging.shutil.which", lambda _: None)
    monkeypatch.setitem(api.PRESETS,"rapido",dict(max_candidates=1,rounds=1,trials_per_round=1,max_epochs=1,
        max_search_seconds=180,allowed_architectures=["MobileNetV3Small"],img_size_override=32,
        docker_timeout_seconds=2))
    rid=create(client,upload,budget="rapido",pretrained="false",mode="rules")
    JobWorker(api.queue()).tick()
    d=client.get(f"/api/runs/{rid}").json()
    assert d["status"] in ("completed","target_not_reached"),d
    assert d["progress"]["attempts_finished"] == 1
    assert d["best_validation"] is not None
    assert d["summary"]["test"]["n_test"] > 0
    assert d["summary"]["verification"]["checks"]["local"]["status"] == "passed"
    assert set(d["downloads"]) == {"reports","model","artifact","all"}
    for url in d["downloads"].values():assert client.get(url).status_code == 200
    assert client.get(f"/api/runs/{rid}/report").status_code == 200
    assert client.post(f"/api/runs/{rid}/resume").status_code == 409
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 409
