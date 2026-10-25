"""Servidor aislado para verificar la web sin credenciales ni pesos descargados.

Ejecutar: python scripts/serve_web_smoke.py
No cambia los presets de la aplicación instalada. Datos y jobs: runs/web-smoke/.
"""
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = Path(os.getenv("WEB_SMOKE_ROOT", str(ROOT / "runs/web-smoke"))).resolve()
os.environ["AUTOML_RUNS_DIR"] = str(SMOKE_ROOT / "jobs")
os.environ["AUTOML_EMBEDDING"] = "hash"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from automl_agents import api
from automl_agents.demo import make_demo_dataset

api.PRESETS["rapido"] = dict(max_candidates=1, rounds=1, trials_per_round=3,
                            max_epochs=5, max_search_seconds=300,
                            allowed_architectures=["MobileNetV3Small"],
                            img_size_override=32, docker_timeout_seconds=300)

if __name__ == "__main__":
    dataset = make_demo_dataset(SMOKE_ROOT / "images", per_class=10, size=40)
    with zipfile.ZipFile(SMOKE_ROOT / "images.zip", "w") as archive:
        for path in dataset.rglob("*.png"):
            archive.write(path, path.relative_to(dataset.parent))
    import uvicorn
    uvicorn.run(api.app, host="127.0.0.1", port=int(os.getenv("WEB_SMOKE_PORT", "8765")), access_log=False)
