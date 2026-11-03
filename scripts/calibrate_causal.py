"""Disjoint calibration groups for temperature fitting and abstention evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from renqing.causal import CausalPredictor
from renqing.metrics import evaluate, fit_temperature
from renqing.reliability import example_profile, fit_review_rule
from renqing.schema import read_examples
from renqing.artifacts import create_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/causal-round2")
    args = parser.parse_args()
    root = Path(args.model)
    if (root / "calibration.json").exists() or (root / "reliability_policy.json").exists():
        raise FileExistsError("不覆盖已冻结的校准与复核策略")
    examples = read_examples("data/expanded-v2/calibration.jsonl") + read_examples("data/xcopa-zh-eval/val.jsonl")
    predictor = CausalPredictor(root)
    logits = predictor.score_examples(examples)
    fit_indices = [i for i, x in enumerate(examples) if int(hashlib.sha256(x.group_id.encode()).hexdigest()[:8], 16) % 2 == 0]
    evidence_indices = [i for i in range(len(examples)) if i not in set(fit_indices)]
    if {examples[i].group_id for i in fit_indices} & {examples[i].group_id for i in evidence_indices}:
        raise AssertionError("温度拟合组与阈值证据组发生重叠")
    global_t = fit_temperature([logits[i] for i in fit_indices], [examples[i].target for i in fit_indices])
    calibration = {"scope": "mixed_profile_calibration_only_social_labels_unreviewed", "temperature": global_t,
                   "decision_fingerprint": predictor.fingerprint, "profile_temperatures": {}, "profiles": {}}
    policy = {"decision_fingerprint": predictor.fingerprint, "profiles": {}, "temperature_and_threshold_groups_disjoint": True,
              "automatic_acceptance_note": "only declared profiles with sufficient in-distribution calibration evidence; never an OOD guarantee"}
    keys = sorted({f"{example_profile(x)}:{x.question.kind}" for x in examples})
    for key in keys:
        temp = [i for i in fit_indices if f"{example_profile(examples[i])}:{examples[i].question.kind}" == key]
        cert = [i for i in evidence_indices if f"{example_profile(examples[i])}:{examples[i].question.kind}" == key]
        if not temp or not cert:
            continue
        t = fit_temperature([logits[i] for i in temp], [examples[i].target for i in temp])
        calibration["profile_temperatures"][key] = t
        calibration["profiles"][key] = {"temperature_fit_examples": len(temp), "threshold_evidence_examples": len(cert),
                                         "threshold_before": evaluate([logits[i] for i in cert], [examples[i].target for i in cert]),
                                         "threshold_after": evaluate([logits[i] for i in cert], [examples[i].target for i in cert], t)}
        policy["profiles"][key] = fit_review_rule([examples[i] for i in cert], [logits[i] for i in cert], t, trusted_labels=not key.startswith("social:"))
    calibration["temperature_fit_groups"] = sorted({examples[i].group_id for i in fit_indices})
    calibration["threshold_evidence_groups"] = sorted({examples[i].group_id for i in evidence_indices})
    # Validate both payloads before publishing either, including numpy scalars.
    json.dumps(calibration, allow_nan=False)
    json.dumps(policy, allow_nan=False)
    for name, value in (("calibration.json", calibration), ("reliability_policy.json", policy)):
        create_json(root / name, value)
    print(json.dumps({"temperatures": calibration["profile_temperatures"],
                      "certified_profiles": [key for key, value in policy["profiles"].items() if value["certified"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
