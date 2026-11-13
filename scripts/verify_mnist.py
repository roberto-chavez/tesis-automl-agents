"""Reproducible MNIST acceptance run, optionally reusing an earlier run's splits."""
import argparse
import json
import shutil
from pathlib import Path

from automl_agents.config import RunConfig
from automl_agents.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/mnist-validation"))
    args = parser.parse_args()
    cfg = RunConfig(dataset_dir=str(args.dataset.resolve()),
                    output_dir=str(args.output.resolve()),
                    allowed_architectures=["SmallCNN"], max_candidates=1,
                    rounds=2, trials_per_round=2, max_epochs=20,
                    target_accuracy=0.95, pretrained=False, use_llm=False,
                    mode="rules", embedding_backend="hash", max_search_seconds=600)
    if args.reference_run:
        old = json.loads((args.reference_run / "config.json").read_text("utf-8"))
        cfg.seed = old["seed"]
        cfg.val_fraction = old["val_fraction"]
        cfg.test_fraction = old["test_fraction"]
        (cfg.run_dir / "data").mkdir(parents=True)
        for name in ("dataset_manifest.json", "split_manifest.json"):
            shutil.copy2(args.reference_run / "data" / name, cfg.run_dir / "data" / name)
    run_pipeline(cfg)
    # Acceptance uses validation from persisted trials, never test accuracy.
    from automl_agents.state_store import StateStore
    store = StateStore(cfg.run_dir)
    try:
        scores = [r["val_acc"] for r in store.records() if r["status"] == "completed"]
    finally:
        store.close()
    best = max(scores, default=0.0)
    result = {"run_dir": str(cfg.run_dir), "validation_accuracy": best,
              "threshold": 0.95, "passed": best >= 0.95}
    (cfg.run_dir / "acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("MNIST validation accuracy is below 95%")


if __name__ == "__main__":
    main()
