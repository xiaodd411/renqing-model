"""Frozen regression and one reserved external evaluation; never a training input."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from renqing.artifacts import create_json
from renqing.model import Predictor
from renqing.causal import CausalPredictor
from renqing.reliability import example_profile
from renqing.schema import read_examples
from renqing.service import predictor_scores
from renqing.train import sliced_metrics


def main():
    output = Path("audits/final-regression-v3.json")
    if output.exists():
        raise FileExistsError(output)
    datasets = {"expanded_regression": read_examples("data/expanded-v2/test.jsonl"),
                "legacy_regression": read_examples("data/demo/test.jsonl"),
                "xcopa_reserved_test": read_examples("data/xcopa-zh-eval/test.jsonl")}
    report = {"scope": "v1/v2 sets are known regression tests, XCOPA test held out from this iteration's tuning",
              "external_caveat": "public causal commonsense benchmark; foundation pretraining exposure cannot be excluded; not a social intelligence benchmark",
              "models": {}, "bundle": {}}
    torch.set_num_threads(4)
    cache = {}
    for name, path, cls in (("v1", "runs/renqing-v1", Predictor), ("v2", "runs/renqing-v2", Predictor),
                            ("round1", "runs/causal-round1-lora", CausalPredictor), ("round2", "runs/causal-round2", CausalPredictor)):
        predictor = cls(path)
        report["models"][name], cache[name] = {}, {}
        for dataset, examples in datasets.items():
            start = time.perf_counter()
            scores = predictor_scores(predictor, examples)
            cache[name][dataset] = scores
            metrics = sliced_metrics(examples, scores, 1, 0.8)
            metrics["by_source"] = {}
            for source in sorted({x.metadata.get("source", "original_demo") for x in examples}):
                idx = [i for i, x in enumerate(examples) if x.metadata.get("source", "original_demo") == source]
                metrics["by_source"][source] = sliced_metrics([examples[i] for i in idx], [scores[i] for i in idx], 1, 0.8)
            report["models"][name][dataset] = {"seconds": time.perf_counter() - start, **metrics}
            print(json.dumps({"model": name, "dataset": dataset, "accuracy": metrics["overall"]["accuracy_against_target_mode"],
                              "nll": metrics["overall"]["nll"]}), flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # These are the exact predeclared bundle routes, not test-dependent choices.
    for dataset, examples in datasets.items():
        scores = []
        for i, ex in enumerate(examples):
            profile = example_profile(ex)
            name = "v2" if profile == "stance" else "v1" if profile == "social" and ex.question.kind == "choice" else "round2"
            scores.append(cache[name][dataset][i])
        metrics = sliced_metrics(examples, scores, 1, 0.8)
        metrics["by_source"] = {}
        for source in sorted({x.metadata.get("source", "original_demo") for x in examples}):
            idx = [i for i, x in enumerate(examples) if x.metadata.get("source", "original_demo") == source]
            metrics["by_source"][source] = sliced_metrics([examples[i] for i in idx], [scores[i] for i in idx], 1, 0.8)
        report["bundle"][dataset] = metrics
    create_json(output, report)
    # Retain logits for independent recomputation without rerunning GPU inference.
    import gzip
    payload = {model: {dataset: [{"id": x.id, "group_id": x.group_id, "kind": x.question.kind,
                                 "source": x.metadata.get("source", "original_demo"), "target": list(x.target), "logits": score.tolist()}
                                for x, score in zip(datasets[dataset], scores)] for dataset, scores in datasets_map.items()}
               for model, datasets_map in cache.items()}
    with gzip.open("audits/final-regression-v3-logits.json.gz", "xt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(json.dumps({dataset: m["overall"] for dataset, m in report["bundle"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
