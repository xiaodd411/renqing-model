"""Choose canonical aggregation on development data, then freeze config."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from renqing.causal import CausalPredictor
from renqing.metrics import evaluate
from renqing.schema import read_examples
from train_causal_candidate import select_development


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/causal-round1-lora")
    parser.add_argument("--output", default="runs/causal-round2")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    predictor = CausalPredictor(args.source)
    examples = select_development(read_examples("data/expanded-v2/validation.jsonl"), 360)
    results = {}
    for rotations in (1, 3):
        scores = predictor.score_examples(examples, order_policy="canonical", rotations=rotations)
        metrics = {}
        for kind in ("choice", "noul", "score"):
            idx = [i for i, x in enumerate(examples) if x.question.kind == kind]
            metrics[kind] = evaluate([scores[i] for i in idx], [examples[i].target for i in idx])
        results[rotations] = {"macro_nll": float(np.mean([x["nll"] for x in metrics.values()])), "by_type": metrics}
    chosen = min(results, key=lambda r: results[r]["macro_nll"])
    root.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(Path(args.source) / name, root / name)
    metadata = {**predictor.meta, "model_id": "renqing-causal-round2", "order_policy": "canonical", "rotations": chosen,
                "parent_candidate": str(args.source), "selection_rule": "canonical order required; rotations chosen by development macro NLL",
                "round2_development_results": results}
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chosen_rotations": chosen, "development_macro_nll": {r: m["macro_nll"] for r, m in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
