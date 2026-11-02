"""Compare checkpoints on identical frozen sets; group bootstrap, no model tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from renqing.metrics import evaluate, softmax
from renqing.model import Predictor, collect_logits
from renqing.schema import read_examples
from renqing.train import sliced_metrics


def cluster_interval(examples, logits, temperature, iterations=2000):
    """95% percentile bootstrap over scenario/question groups, not derived rows."""
    groups = {}
    for ex, logit in zip(examples, logits):
        target = np.asarray(ex.target)
        if np.count_nonzero(np.isclose(target, target.max(), atol=1e-10, rtol=0)) != 1:
            continue
        correct = int(int(np.argmax(logit)) == int(target.argmax()))
        group = groups.setdefault(ex.group_id, [0, 0])
        group[0] += correct
        group[1] += 1
    values = np.asarray(list(groups.values()), dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(984)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    sampled = values[indices].sum(axis=1)
    ratios = sampled[:, 0] / sampled[:, 1]
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "groups": len(values), "iterations": iterations,
            "method": "95_percent_group_percentile_bootstrap", "meaning": "sampling variation within these datasets, not proof of real-world social reliability"}


def assess(predictor, examples):
    logits = collect_logits(predictor.model, predictor.tokenizer, examples, predictor.meta["max_length"], 24)
    temperature = predictor.calibration["temperature"] if predictor.calibration else 1.0
    result = sliced_metrics(examples, logits, temperature, 0.8)
    result["accuracy_interval"] = cluster_interval(examples, logits, temperature)
    result["by_source"] = {}
    for source in sorted({x.metadata.get("source", "original_demo") for x in examples}):
        indices = [i for i, x in enumerate(examples) if x.metadata.get("source", "original_demo") == source]
        subset = [examples[i] for i in indices]
        outputs = [logits[i] for i in indices]
        scoped = sliced_metrics(subset, outputs, temperature, 0.8)
        scoped["groups"] = len({x.group_id for x in subset})
        scoped["accuracy_interval"] = cluster_interval(subset, outputs, temperature)
        for kind, metrics in scoped["by_type"].items():
            if kind == "noul":
                # Three polarity predicates produce two negatives per public row.
                target = [x.target for x in subset if x.question.kind == "noul"]
                metrics["always_no_accuracy"] = evaluate([[0, 1] for _ in target], target)["accuracy_against_target_mode"]
        result["by_source"][source] = scoped
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name=directory")
    parser.add_argument("--test", default="data/expanded-v2/test.jsonl")
    parser.add_argument("--legacy-test", default="data/demo/test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    test, legacy = read_examples(args.test), read_examples(args.legacy_test)
    report = {"selection_rule": "epochs and checkpoints selected by validation only; test outcomes never select a checkpoint",
              "comparison": "identical frozen examples for each model", "expanded_test_groups": len({x.group_id for x in test}),
              "caveat": "public stance recognition and synthetic social judgments are different capabilities; report separately",
              "models": {}}
    torch.set_num_threads(4)
    for item in args.models:
        name, path = item.split("=", 1)
        predictor = Predictor(path, args.device)
        report["models"][name] = {"path": path, "best_epoch": predictor.meta.get("best_epoch"),
                                  "expanded_test": assess(predictor, test), "legacy_test_diagnostic": assess(predictor, legacy)}
        print(json.dumps({"model": name, "overall": report["models"][name]["expanded_test"]["overall"]}, ensure_ascii=False), flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
