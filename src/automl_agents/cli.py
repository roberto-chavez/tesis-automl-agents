from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import RunConfig
from .llm import DEFAULT_LLM_TIMEOUT_SECONDS
from .context import ModelContextBudget
from .schemas import MetricPolicy
from .state_store import RunBusyError, TrainingBusyError


def main(argv=None):
    # Los registros Docker contienen caracteres fuera de las páginas de códigos
    # de Windows. Mantener también la salida redirigida como JSON UTF-8 válido.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser("automl-agents", description="AutoML multiagente + RAG para clasificación de imágenes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Ejecuta el pipeline completo sobre un dataset (carpeta por clase)",
                       epilog="Límites globales persistentes. El plazo se comprueba entre trials y al terminar cada época.")
    r.add_argument("--dataset", required=True)
    r.add_argument("--use-case", default="")
    r.add_argument("--tier", choices=["low", "medium", "high"], default="low")
    r.add_argument("--max-candidates", type=int, default=3)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--trials", type=int, default=4, help="trials por candidato y ronda")
    r.add_argument("--epochs", type=int, default=8)
    r.add_argument("--output", "--runs-dir", dest="output", default=None)
    r.add_argument("--no-pretrained", action="store_true", help="sin pesos ImageNet (pruebas offline)")
    r.add_argument("--img-size", type=int, default=None)
    r.add_argument("--archs", nargs="*", default=None, help="restringe arquitecturas permitidas")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--mode", choices=["rules", "agentic"], default="agentic")
    r.add_argument("--primary-metric", choices=["accuracy"], default="accuracy")
    r.add_argument("--metric-direction", choices=["maximize"], default="maximize")
    r.add_argument("--target-accuracy", type=float, default=0.99)
    r.add_argument("--tie-breaker", choices=["smaller_model"], default="smaller_model")
    r.add_argument("--val-fraction", type=float, default=0.15)
    r.add_argument("--test-fraction", type=float, default=0.15)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--discard-margin", type=float, default=0.08)
    r.add_argument("--max-training-attempts", type=int, default=None)
    r.add_argument("--max-search-seconds", type=float, default=3600.0)
    r.add_argument("--closing-reserve-seconds", type=float, default=60.0,
                   help="Reserva separada de tiempo para cierre/evaluación; no se consume buscando")
    r.add_argument("--max-decisions", type=int, default=100)
    r.add_argument("--artifact-smoke-timeout-seconds", type=float, default=120)
    r.add_argument("--docker-timeout-seconds", type=float, default=300)
    r.add_argument("--max-llm-calls", type=int, default=20)
    r.add_argument("--max-configuration-llm-calls", type=int, default=1)
    r.add_argument("--max-diagnosis-llm-calls", type=int, default=3)
    r.add_argument("--max-retries", type=int, default=1)
    r.add_argument("--llm-timeout-seconds", type=float, default=DEFAULT_LLM_TIMEOUT_SECONDS)
    r.add_argument("--max-stagnant-blocks", type=int, default=3)
    r.add_argument("--max-no-progress-actions", type=int, default=6)
    r.add_argument("--rag-max-documents", type=int, default=32)
    r.add_argument("--rag-max-tokens", type=int, default=32000)
    r.add_argument("--context-read-max-chars", type=int, default=4000)
    r.add_argument("--llm-input-tokens", type=int, default=16000)
    r.add_argument("--llm-output-reserve", type=int, default=2000)
    r.add_argument("--llm-context-window", type=int, default=24000)

    for command in ("resume", "status"):
        p = sub.add_parser(command, help="Reanuda una ejecución" if command == "resume" else "Consulta estado y consumo persistidos")
        p.add_argument("run_id")
        p.add_argument("--runs-dir", "--output", dest="runs_dir", default=None,
                       help="Directorio de ejecuciones (por defecto AUTOML_RUNS_DIR o runs)")

    for command in ("report", "verify-artifact"):
        p = sub.add_parser(command, help="Regenera reportes desde SQLite" if command == "report" else
                           "Repite las pruebas del artefacto y Docker con timeout")
        p.add_argument("run_id")
        p.add_argument("--runs-dir", "--output", dest="runs_dir", default=None)
        if command == "verify-artifact":
            p.add_argument("--skip-docker", action="store_true")
            p.add_argument("--timeout", type=float, default=None, help="Timeout de Docker en segundos")

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
        try:
            cfg = RunConfig(dataset_dir=a.dataset, use_case=a.use_case, resource_tier=a.tier,
                        max_candidates=a.max_candidates, rounds=a.rounds, trials_per_round=a.trials,
                        max_epochs=a.epochs, pretrained=not a.no_pretrained, img_size_override=a.img_size,
                        allowed_architectures=a.archs, use_llm=not a.no_llm,
                        mode=a.mode, val_fraction=a.val_fraction, test_fraction=a.test_fraction,
                        seed=a.seed, discard_margin=a.discard_margin,
                        metric_policy=MetricPolicy(name=a.primary_metric, direction=a.metric_direction,
                                                   threshold=a.target_accuracy, tie_breaker=a.tie_breaker),
                        max_training_attempts=a.max_training_attempts, max_search_seconds=a.max_search_seconds,
                        closing_reserve_seconds=a.closing_reserve_seconds,
                        artifact_smoke_timeout_seconds=a.artifact_smoke_timeout_seconds,
                        docker_timeout_seconds=a.docker_timeout_seconds,
                        max_decisions=a.max_decisions, max_llm_calls=a.max_llm_calls, max_retries=a.max_retries,
                        max_configuration_llm_calls=a.max_configuration_llm_calls,
                        max_diagnosis_llm_calls=a.max_diagnosis_llm_calls,
                        llm_timeout_seconds=a.llm_timeout_seconds,
                        max_stagnant_blocks=a.max_stagnant_blocks,
                        max_no_progress_actions=a.max_no_progress_actions,
                        rag_max_documents=a.rag_max_documents, rag_max_tokens=a.rag_max_tokens,
                        context_read_max_chars=a.context_read_max_chars,
                        llm_context_budget=ModelContextBudget(input_tokens=a.llm_input_tokens,
                            output_tokens=a.llm_output_reserve, window_tokens=a.llm_context_window),
                        **({"output_dir": a.output} if a.output is not None else {}))
        except ValueError as exc:
            ap.error(str(exc))
        from .pipeline import run_pipeline
        try:
            summary = run_pipeline(cfg)
        except (ValueError, OSError, RunBusyError, TrainingBusyError) as exc:
            ap.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nResultados en: {cfg.run_dir}")
    elif a.cmd in ("resume", "status"):
        from .state_store import StateStore, load_run
        try:
            if a.cmd == "resume":
                from .pipeline import resume_pipeline
                result = resume_pipeline(a.run_id, a.runs_dir)
            else:
                cfg = load_run(a.run_id, a.runs_dir)
                store = StateStore(cfg.run_dir)
                try:
                    result = {"run_id": cfg.run_id, **store.status()}
                finally:
                    store.close()
        except (ValueError, OSError, RunBusyError, TrainingBusyError) as exc:
            ap.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif a.cmd in ("report", "verify-artifact"):
        from .state_store import StateStore, load_run, run_lock
        from .reporting import write_persistent_reports
        from .events import write_json
        try:
            cfg = load_run(a.run_id, a.runs_dir)
            if not (cfg.run_dir / "state.sqlite").exists():
                raise ValueError("La ejecución no tiene estado persistente")
            with run_lock(cfg.run_dir):
                store = StateStore(cfg.run_dir)
                try:
                    if a.cmd == "verify-artifact":
                        from .packaging import verify_artifact
                        delivery = store.get("delivery")
                        final = store.get("final_evaluation")
                        if not delivery or not final or not delivery.get("artifact"):
                            raise ValueError("La ejecución no tiene un artefacto publicado para verificar")
                        result = verify_artifact(cfg.run_dir / "artifact", docker=not a.skip_docker,
                            smoke_timeout=cfg.artifact_smoke_timeout_seconds,
                            docker_timeout=cfg.docker_timeout_seconds if a.timeout is None else a.timeout)
                        store.put("verification", result)
                        target = final["winner"]["val_acc"] >= final["winner"]["metric_policy"]["threshold"]
                        status = "artifact_failed" if result["status"] == "failed" else "completed" if target else "target_not_reached"
                        delivery = {**delivery, "status": status, "verification": result}
                        store.put("delivery", delivery)
                        store.put("status", status)
                        write_json(cfg.run_dir / "run_summary.json", delivery)
                        write_persistent_reports(store, cfg)
                        # La misma entrega puede haberse creado desde la web.
                        # Mantener su estado visible alineado con la verificación CLI.
                        if (cfg.run_dir.parent / "jobs.sqlite").is_file():
                            from .jobs import JobQueue
                            JobQueue(cfg.run_dir.parent).finish(cfg.run_id, status)
                    else:
                        result = write_persistent_reports(store, cfg)
                finally:
                    store.close()
        except (ValueError, OSError, RunBusyError) as exc:
            ap.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
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
