"""Separate validation/development evaluation from final regression evaluation."""
import argparse
import json
from pathlib import Path
import random
import time

from renqing.causal import CausalPredictor
from renqing.schema import read_examples
from renqing.train import sliced_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/expanded-v2/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-model")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    examples = read_examples(args.data)
    if args.limit:
        rng = random.Random(128)
        # Keep source/task strata represented in bounded development checks.
        strata = {}
        for ex in examples:
            strata.setdefault((ex.metadata.get("source", "demo"), ex.question.kind), []).append(ex)
        selected = []
        for group in strata.values():
            selected.extend(rng.sample(group, min(len(group), max(1, args.limit // len(strata)))))
        examples = selected
    predictor = CausalPredictor(args.model, base_model=args.base_model)
    start = time.perf_counter()
    logits = predictor.score_examples(examples, args.batch_size)
    temperature = predictor.calibration["temperature"] if predictor.calibration else 1.0
    report = {"data": args.data, "examples": len(examples), "seconds": time.perf_counter() - start,
              "model": predictor.meta, "metrics": sliced_metrics(examples, logits, temperature, 0.8),
              "by_source": {}, "predictions": []}
    for source in sorted({ex.metadata.get("source", "demo") for ex in examples}):
        indices = [i for i, ex in enumerate(examples) if ex.metadata.get("source", "demo") == source]
        report["by_source"][source] = sliced_metrics([examples[i] for i in indices], [logits[i] for i in indices], temperature, 0.8)
    for ex, score in zip(examples, logits):
        report["predictions"].append({"id": ex.id, "group_id": ex.group_id, "kind": ex.question.kind,
                                      "source": ex.metadata.get("source", "demo"), "target": list(ex.target), "logits": score.tolist()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"examples": len(examples), "seconds": report["seconds"], "overall": report["metrics"]["overall"],
                      "by_source": {s: {k: v["accuracy_against_target_mode"] for k, v in m["by_type"].items()}
                                    for s, m in report["by_source"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
