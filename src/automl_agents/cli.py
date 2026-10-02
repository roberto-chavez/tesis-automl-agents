from __future__ import annotations

import argparse
import json
import logging

from .config import RunConfig


def main(argv=None):
    ap = argparse.ArgumentParser("automl-agents", description="AutoML multiagente + RAG para clasificación de imágenes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Ejecuta el pipeline completo sobre un dataset (carpeta por clase)")
    r.add_argument("--dataset", required=True)
    r.add_argument("--use-case", default="")
    r.add_argument("--tier", choices=["low", "medium", "high"], default="low")
    r.add_argument("--max-candidates", type=int, default=3)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--trials", type=int, default=4, help="trials por candidato y ronda")
    r.add_argument("--epochs", type=int, default=8)
    r.add_argument("--output", default=None)
    r.add_argument("--no-pretrained", action="store_true", help="sin pesos ImageNet (pruebas offline)")
    r.add_argument("--img-size", type=int, default=None)
    r.add_argument("--archs", nargs="*", default=None, help="restringe arquitecturas permitidas")
    r.add_argument("--no-llm", action="store_true")

    k = sub.add_parser("kb-eval", help="Evalúa la recuperación del RAG (precisión@k / exhaustividad@k)")
    k.add_argument("--k", type=int, default=3)
    k.add_argument("--backend", default="auto")

    d = sub.add_parser("demo-dataset", help="Genera un dataset sintético de prueba")
    d.add_argument("path")
    d.add_argument("--per-class", type=int, default=30)

    s = sub.add_parser("serve", help="Levanta la plataforma web/API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd == "run":
        from .pipeline import run_pipeline
        cfg = RunConfig(dataset_dir=a.dataset, use_case=a.use_case, resource_tier=a.tier,
                        max_candidates=a.max_candidates, rounds=a.rounds, trials_per_round=a.trials,
                        max_epochs=a.epochs, pretrained=not a.no_pretrained, img_size_override=a.img_size,
                        allowed_architectures=a.archs, use_llm=not a.no_llm)
        if a.output:
            cfg.output_dir = a.output
        summary = run_pipeline(cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nResultados en: {cfg.run_dir}")
    elif a.cmd == "kb-eval":
        from .kb import KnowledgeBase
        print(json.dumps(KnowledgeBase(backend=a.backend).evaluate(k=a.k), indent=2))
    elif a.cmd == "demo-dataset":
        from .demo import make_demo_dataset
        print(make_demo_dataset(a.path, a.per_class))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("automl_agents.api:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
