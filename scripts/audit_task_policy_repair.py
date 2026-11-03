"""Evaluate a known-regression repair and enforce explicit preservation gates."""
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from renqing.artifacts import create_json
from renqing.causal import CausalPredictor
from renqing.metrics import softmax
from renqing.reliability import example_profile
from renqing.schema import read_examples
from renqing.service import predictor_scores
from renqing.train import sliced_metrics


def main():
    output = Path("audits/task-policy-repair.json")
    if output.exists():
        raise FileExistsError(output)
    with gzip.open("audits/final-regression-v3-logits.json.gz", "rt", encoding="utf-8") as f:
        cache = json.load(f)
    old = json.loads(Path("audits/final-regression-v3.json").read_text(encoding="utf-8"))
    bundle = json.loads(Path("runs/renqing-v3.1/metadata.json").read_text(encoding="utf-8"))
    expert_names = {"social_choice_v1": "v1", "stance_v2": "v2", "causal_round3": "round3"}
    datasets = {"expanded_regression": "data/expanded-v2/test.jsonl", "legacy_regression": "data/demo/test.jsonl",
                "xcopa_reserved_test": "data/xcopa-zh-eval/test.jsonl"}
    predictor = CausalPredictor("runs/causal-round3")
    report = {"scope": "known regression repair, all listed tests have already been inspected; not new held-out evidence",
              "default_changed": False, "round3": {}, "bundle": {}, "gates": [], "per_example_changes": {}}
    torch.set_num_threads(4)

    def gate(name, passed, detail):
        report["gates"].append({"name": name, "passed": bool(passed), "detail": detail})

    for dataset, path in datasets.items():
        examples = read_examples(path)
        for model in cache:
            if [r["id"] for r in cache[model][dataset]] != [ex.id for ex in examples]:
                raise ValueError("缓存样本顺序与数据不匹配")
        started = time.perf_counter()
        scores = predictor_scores(predictor, examples)
        report["round3"][dataset] = {"seconds": time.perf_counter() - started, **sliced_metrics(examples, scores, 1, 0.8)}
        selected = []
        for i, ex in enumerate(examples):
            route = bundle["routes"][f"{example_profile(ex)}:{ex.question.kind}"]
            model = expert_names[route]
            selected.append(scores[i] if model == "round3" else np.array(cache[model][dataset][i]["logits"]))
        metrics = sliced_metrics(examples, selected, 1, 0.8)
        metrics["by_source"] = {}
        for source in sorted({x.metadata.get("source", "original_demo") for x in examples}):
            indices = [i for i, x in enumerate(examples) if x.metadata.get("source", "original_demo") == source]
            metrics["by_source"][source] = sliced_metrics([examples[i] for i in indices], [selected[i] for i in indices], 1, 0.8)
        report["bundle"][dataset] = metrics
        # Exact frozen expert routing is checked separately from aggregate accuracy.
        for profile, kind, reference in (("social", "choice", "v1"), ("stance", None, "v2")):
            indices = [i for i, ex in enumerate(examples) if example_profile(ex) == profile and (kind is None or ex.question.kind == kind)]
            if indices:
                delta = max(float(np.max(np.abs(softmax(selected[i]) - softmax(cache[reference][dataset][i]["logits"])))) for i in indices)
                gate(f"{dataset}:{profile}:frozen_probabilities", delta == 0.0, {"examples": len(indices), "max_delta": delta})
        if dataset != "xcopa_reserved_test":
            source = "synthetic_challenge_v2" if dataset == "expanded_regression" else "original_demo"
            for kind in ("noul", "score"):
                actual = metrics["by_source"][source]["by_type"][kind]["accuracy_against_target_mode"]
                floor = max(old["models"][ref][dataset]["by_source"][source]["by_type"][kind]["accuracy_against_target_mode"]
                            for ref in ("v1", "v2", "round1"))
                gate(f"{dataset}:social:{kind}:accuracy", actual >= floor, {"actual": actual, "protected_floor": floor})
        else:
            actual = metrics["overall"]["accuracy_against_target_mode"]
            floor = old["models"]["round2"][dataset]["overall"]["accuracy_against_target_mode"]
            gate("causal_choice_accuracy", actual >= floor, {"actual": actual, "protected_floor": floor})
        # Show lost and gained individual predictions; aggregate pass is not universal nonregression.
        changes = {"regressed": [], "improved": []}
        for i, ex in enumerate(examples):
            target = np.asarray(ex.target)
            if np.count_nonzero(np.isclose(target, target.max(), atol=1e-10, rtol=0)) != 1:
                continue
            baseline = "v2" if example_profile(ex) == "stance" else "v1"
            before = int(np.argmax(cache[baseline][dataset][i]["logits"])) == int(target.argmax())
            after = int(np.argmax(selected[i])) == int(target.argmax())
            if before and not after:
                changes["regressed"].append(ex.id)
            if after and not before:
                changes["improved"].append(ex.id)
        report["per_example_changes"][dataset] = changes
        print(json.dumps({"dataset": dataset, "bundle_accuracy": metrics["overall"]["accuracy_against_target_mode"],
                          "regressed": len(changes["regressed"]), "improved": len(changes["improved"])}), flush=True)
    baseline = json.loads(Path("audits/baseline-v0.2.json").read_text(encoding="utf-8"))
    mismatches = [name for name, expected in baseline["files"].items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected]
    gate("old_artifact_bytes", not mismatches, {"checked_files": len(baseline["files"]), "mismatches": mismatches})
    report["experimental_gate_passed"] = all(g["passed"] for g in report["gates"])
    report["eligible_for_default"] = False
    report["default_blockers"] = ["social labels not independently human-reviewed", "known regressions are not new generalization evidence",
                                  "aggregate protected metrics do not preserve every individual prediction", "no profile certified for automatic acceptance"]
    create_json(output, report)
    print(json.dumps({"experimental_gate_passed": report["experimental_gate_passed"], "eligible_for_default": False,
                      "gates": report["gates"]}, ensure_ascii=False, indent=2))
    if not report["experimental_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
