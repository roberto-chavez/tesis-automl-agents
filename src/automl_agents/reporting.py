"""Generación de reportes: R1.3, R2.2/R2.3 (historial), R3.2 (model card), R4.2, R4.3."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from .catalog import CATALOG
from .llm import LLM
from .events import write_text


def _w(path: Path, text: str):
    write_text(path, text)


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def wilson(acc: float, n: int, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = acc + z * z / (2 * n)
    m = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def write_criteria_report(rd: Path, profile, prop: dict, docs) -> None:
    data = {
        "mode": prop["mode"], "note": prop["note"], "error": prop.get("error"),
        "dataset": profile.to_dict(),
        "criteria": {**prop["criteria"], "retrieved_docs": [d.to_dict() for d in docs]},
        "space": prop["space"],
        "candidates": [c.to_dict() for c in prop["candidates"]],
        "discarded": prop["discarded"],
    }
    _w(rd / "reports/01_criterios_acotacion.json", json.dumps(data, ensure_ascii=False, indent=2))
    sp = prop["space"]
    md = [f"# Criterios de acotación del espacio de búsqueda (R1.3)\n",
          f"Generador: **{prop['mode']}**. {prop['note']}\n",
          f"- Dataset: {profile.n_images} imágenes, {profile.n_classes} clases, razón de desbalance {profile.imbalance_ratio:.1f}:1.",
          f"- Espacio total considerado: **{sp['total_configs']}** configuraciones; candidato: **{sp['candidate_configs']}** "
          f"(reducción **{sp['reduction_pct']} %**).\n",
          "## Evidencia recuperada (RAG)",
          _md_table(["Documento", "Tipo", "Similitud"], [(d.title, d.kind, f"{d.score:.2f}") for d in docs]),
          "\n## Candidatas",
          _md_table(["ID", "Versión", "Arquitectura", "Justificación", "Fuentes", "lr", "dropout", "batch", "aug"],
                    [(c.id, c.version, c.architecture, c.rationale, ", ".join(c.sources), c.lr_values, c.dropout_values,
                      c.batch_sizes, c.augment_values) for c in prop["candidates"]]),
          "\n## Descartadas",
          _md_table(["Arquitectura", "Motivo"], [(d["architecture"], d["reason"]) for d in prop["discarded"]])]
    _w(rd / "reports/01_criterios_acotacion.md", "\n".join(md))


def write_iterations(rd: Path, decisions: list[dict], records: list[dict]) -> None:
    _w(rd / "reports/02_historial_iteraciones.json",
       json.dumps({"decisions": decisions, "trials": records}, ensure_ascii=False, indent=2))


def write_resources_csv(rd: Path, records: list[dict]) -> None:
    p = rd / "reports/04_consumo_recursos.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["round", "candidate_id", "architecture", "trial", "lr", "dropout", "batch", "augment", "val_acc",
            "epochs", "pruned", "seconds", "cpu_percent_avg", "rss_mb_peak", "gpu_util_avg", "gpu_mem_mb_peak"]
    cols += ["candidate_version", "study_id", "trial_id", "attempt_id", "status", "stop_reason", "error_code"]
    with io.StringIO(newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([r["round"], r["candidate_id"], r["architecture"], r["trial"], r["params"].get("lr"),
                        r["params"].get("dropout"), r["params"].get("batch"), r["params"].get("augment"),
                        round(r["val_acc"], 4), r["epochs"], r["pruned"],
                        *[r["resources"].get(k) for k in ("seconds", "cpu_percent_avg", "rss_mb_peak",
                                                      "gpu_util_avg", "gpu_mem_mb_peak")],
                        *[r.get(k) for k in ("candidate_version", "study_id", "trial_id", "attempt_id", "status")],
                        r.get("result", {}).get("stop_reason"),
                        (r.get("result", {}).get("error") or {}).get("code")])
        _w(p, f.getvalue())


def write_evaluation(rd: Path, evaluation: dict, winner_id: str) -> None:
    _w(rd / "reports/03_evaluacion.json", json.dumps(evaluation, ensure_ascii=False, indent=2))
    rows = [(cid + (" ★" if cid == winner_id else ""), e["architecture"], e["val_acc"], e["test"]["accuracy"],
             e["test"]["f1_macro"], e["test"]["latency_ms_per_image"], e["test"]["model_size_mb"])
            for cid, e in evaluation.items()]
    md = ["# Evaluación comparativa de modelos (R2.2)\n",
          "La selección se hace con la exactitud de **validación**; el conjunto de **prueba** solo se reporta "
          "después del cierre de búsqueda (evita utilizar test para seleccionar).\n",
          _md_table(["Candidato", "Arquitectura", "Val acc", "Test acc", "Test F1 macro", "ms/imagen", "MB"], rows)]
    _w(rd / "reports/03_evaluacion.md", "\n".join(md))


def write_benchmark(rd: Path, b: dict) -> None:
    _w(rd / "reports/05_benchmark.json", json.dumps(b, ensure_ascii=False, indent=2))
    f, c = b["baseline_exhaustive_full_space"], b["baseline_exhaustive_candidate_space"]
    md = ["# Consumo de recursos frente a línea base (R4.1, R4.3)\n",
          f"- Entrenamientos ejecutados: **{b['trials_total']}**, de los cuales **{b['trials_pruned']}** "
          f"({b['pruned_pct']} %) se podaron temprano.",
          f"- Épocas usadas en todos los intentos: **{b['epochs_used']}**. En intentos completados/podados: "
          f"{b['epochs_eligible_for_early_stopping']} de {b['epochs_without_pruning_same_trials']} máximas; "
          f"reducción **{b['saving_from_early_stopping_pct']} %** por poda y parada anticipada. "
          "Los fallos y cancelaciones no cuentan como ahorro por parada anticipada.",
          f"- Tiempo medido: {b['seconds_measured']} s ({b['seconds_per_epoch_avg']} s/época en promedio).\n",
          _md_table(["Línea base (estimada)", "Configs", "Épocas", "Horas est.", "Reducción de épocas"],
                    [("Búsqueda exhaustiva, espacio completo", f["configs"], f["epochs"], f["est_hours"], f"{f['reduction_epochs_pct']} %"),
                     ("Búsqueda exhaustiva, espacio acotado por RAG", c["configs"], c["epochs"], c["est_hours"], f"{c['reduction_epochs_pct']} %")]),
          f"\n> {b['note']}"]
    md += ["\n## Tiempos reales por etapa (segundos)",
           _md_table(["Etapa", "Segundos medidos"], b.get("stage_seconds_measured", {}).items())]
    _w(rd / "reports/05_benchmark.md", "\n".join(md))


def _plain_summary(facts: dict, llm: LLM | None) -> str:
    template = (
        f"El sistema probó {facts['trials']} configuraciones de {facts['n_candidates']} tipos de modelo con sus "
        f"{facts['n_images']} imágenes de {facts['n_classes']} clases, y eligió **{facts['arch']}**. "
        f"Con imágenes que no vio durante el entrenamiento acertó cerca del **{facts['acc']*100:.0f} %** de las veces "
        f"(rango probable {facts['lo']*100:.0f} %–{facts['hi']*100:.0f} %, sobre {facts['n_test']} imágenes de prueba).")
    # Las cifras siempre se insertan por plantilla. Ninguna salida libre del LLM
    # reemplaza hechos ni se publica como historial/contexto de búsqueda.
    return template


def write_final_report(rd: Path, cfg, profile, prop, winner: dict, ev: dict, records, risk: dict,
                       benchmark: dict, llm: LLM | None) -> None:
    t = ev["test"]
    lo, hi = wilson(t["accuracy"], t["n_test"])
    spec = CATALOG[winner["architecture"]]
    facts = {"trials": len(records), "n_candidates": len({c.id for c in prop["candidates"]}), "n_images": profile.n_images,
             "n_classes": profile.n_classes, "arch": winner["architecture"], "acc": t["accuracy"],
             "lo": lo, "hi": hi, "n_test": t["n_test"]}
    limits = [f"Se evaluó con {t['n_test']} imágenes de prueba; con pocas imágenes el rango de incertidumbre es amplio."
              if t["n_test"] < 200 else f"Evaluado con {t['n_test']} imágenes de prueba."]
    limits += profile.warnings
    limits.append("El modelo solo reconoce las clases con las que se entrenó; ante imágenes de otro tipo "
                  "dará igualmente una respuesta, por lo que conviene revisar los casos de baja confianza.")
    if not cfg.pretrained:
        limits.append("Se entrenó SIN pesos preentrenados (modo de prueba); no use este modelo en producción.")
    md = [f"# Reporte final del modelo — ejecución {cfg.run_id}\n",
          "## Resumen en lenguaje sencillo\n", _plain_summary(facts, llm), "\n## Ficha del modelo (model card)\n",
          "### Detalles del modelo",
          f"- Arquitectura: {winner['architecture']} (~{spec.params_m} M de parámetros, entrada {winner['img_size']}x{winner['img_size']} px).",
          f"- Hiperparámetros elegidos: {winner['params']}.",
          f"- Se entrenó con transferencia de aprendizaje (pesos ImageNet): {'sí' if cfg.pretrained else 'no'}.\n",
          "### Uso previsto",
          f"- {cfg.use_case or 'No se describió un caso de uso.'}\n",
          "### Datos",
          f"- Perfil de entrenamiento: {profile.n_images} imágenes, clases: {', '.join(profile.classes)}. Partición solicitada "
          f"{int(100*(1-cfg.val_fraction-cfg.test_fraction))}/{int(100*cfg.val_fraction)}/{int(100*cfg.test_fraction)} (entrenamiento/validación/prueba).\n",
          "### Desempeño (conjunto de prueba)",
          f"- Exactitud: **{t['accuracy']*100:.1f} %** (IC 95 % aprox. {lo*100:.1f}–{hi*100:.1f} %). F1 macro: {t['f1_macro']:.3f}.",
          f"- Velocidad: {t['latency_ms_per_image']} ms por imagen; tamaño del modelo: {t['model_size_mb']} MB.\n",
          _md_table(["Clase", "Precisión", "Exhaustividad (recall)", "F1", "Imágenes"],
                    [(c, m["precision"], m["recall"], m["f1"], m["support"]) for c, m in t["per_class"].items()]),
          "\n### Limitaciones", *[f"- {x}" for x in limits],
          "\n### Consideraciones éticas y legales",
          f"- Nivel de riesgo aparente: **{risk['risk_level']}**.",
          *[f"- {n}" for n in risk["notes"]], f"- {risk['disclaimer']}\n",
          "## Por qué el sistema tomó estas decisiones\n",
          f"- Se acotó el espacio de {prop['space']['total_configs']} a {prop['space']['candidate_configs']} configuraciones "
          f"({prop['space']['reduction_pct']} % menos). Detalle en `01_criterios_acotacion.md`.",
          f"- {winner['architecture']}: {next(c.rationale for c in prop['candidates'] if c.id == winner['candidate_id'] and c.version == winner['candidate_version'])}",
          f"- Ahorro de cómputo por poda y parada anticipada: {benchmark['saving_from_early_stopping_pct']} % de épocas. "
          "Detalle en `05_benchmark.md`.\n",
          "## Cómo usar el modelo\n",
          "Consulte el estado de entrega y las comprobaciones en `07_entrega.json` antes de usar `artifact/`."]
    _w(rd / "reports/06_reporte_final.md", "\n".join(md))


def write_persistent_reports(store, cfg):
    """Reconstruye reportes sin motor de búsqueda, dataset original ni cliente LLM."""
    from types import SimpleNamespace
    from .benchmark import compute_benchmark
    from .schemas import Candidate

    rd = store.run_dir
    context = store.get("context", {})
    proposal = store.get("proposal", context.get("prop", {}))
    records = store.records()
    decisions = store.get("orchestrator", {}).get("history", [])
    write_iterations(rd, decisions, records)
    write_resources_csv(rd, records)
    selection = store.get("selection")
    final = store.get("final_evaluation")
    versions = dict(store.get("candidate_history", {}))
    for candidate in proposal.get("candidates", []):
        versions.setdefault(f"{candidate['id']}:v{candidate['version']}", candidate)
    if selection and selection.get("candidate"):
        c = selection["candidate"]
        versions[f"{c['id']}:v{c['version']}"] = c
    space = proposal.get("space", {"candidate_configs": 0})
    bench = compute_benchmark(records, cfg, space, store.get("stage_timings", {}))
    write_benchmark(rd, bench)
    delivery = store.get("delivery", {"status": "pending"})
    facts = {"delivery": delivery, "selection": selection, "metric_policy": (
                 selection["metric_policy"] if selection else cfg.metric_policy.to_dict()),
             "stop_reason": store.get("stop_reason"), "candidate_versions": versions,
             "decisions": decisions, "trials": records, "stage_seconds_measured": store.get("stage_timings", {}),
             "verification": store.get("verification"), "benchmark": bench,
             "final_evaluation": final}
    _w(rd / "reports/07_entrega.json", json.dumps(facts, ensure_ascii=False, indent=2))
    profile = store.get("training_profile")
    if profile and {"mode", "note", "criteria", "space", "candidates", "discarded"} <= proposal.keys():
        from .kb import Doc
        initial_docs = [Doc(**d) for d in store.get("proposal_evidence", context.get("docs", []))]
        write_criteria_report(rd, SimpleNamespace(**profile, to_dict=lambda: profile),
                              {**proposal, "candidates": [Candidate.from_dict(c) for c in proposal["candidates"]]},
                              initial_docs)
    if final and profile and proposal:
        winner = final["winner"]
        ev = {winner["candidate_id"]: {"architecture": winner["architecture"],
               "val_acc": winner["val_acc"], "test": final["metrics"], "selection": winner}}
        write_evaluation(rd, ev, winner["candidate_id"])
        prop = {**proposal, "candidates": [Candidate.from_dict(c) for c in versions.values()]}
        write_final_report(rd, cfg, SimpleNamespace(**profile), prop, winner,
                           ev[winner["candidate_id"]], records,
                           store.get("risk", context.get("risk", {})), bench, None)
        base = (rd / "reports/06_reporte_final.md").read_text("utf-8")
    else:
        base = f"# Reporte de ejecución {cfg.run_id}\n\nNo hay evaluación final disponible.\n"
    details = ["\n## Estado de entrega", f"- Estado: {delivery['status']}.",
               f"- Motivo de parada: {facts['stop_reason']}.",
               f"- Verificación: {(facts['verification'] or {}).get('status', 'not_run')}.",
               "\n## Versiones de candidatas",
               _md_table(["Candidata", "Versión", "Arquitectura"],
                         [(c["id"], c["version"], c["architecture"]) for c in versions.values()]),
               "\n## Intentos y errores",
               _md_table(["Intento", "Candidata", "Versión", "Estado", "Parada", "Error"],
                         [(r.get("attempt_id"), r["candidate_id"], r.get("candidate_version"), r["status"],
                           r.get("result", {}).get("stop_reason"),
                           (r.get("result", {}).get("error") or {}).get("code", "")) for r in records]),
               "\n## Tiempos medidos (segundos)",
               _md_table(["Etapa", "Segundos"], facts["stage_seconds_measured"].items()),
               "\nDetalles y decisiones en `02_historial_iteraciones.json` y `07_entrega.json`."]
    _w(rd / "reports/06_reporte_final.md", base + "\n".join(details))
    return facts
