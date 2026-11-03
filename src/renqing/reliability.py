"""Explicit scope and group-aware evidence for abstention; never implies OOD trust."""
from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist

import numpy as np
from .metrics import softmax


def example_profile(example):
    declared = getattr(example.question, "profile", None)
    if declared is not None:
        return declared
    source = example.metadata.get("source", "original_demo")
    if source == "dureader_yesno":
        return "stance"
    if source == "xcopa_zh":
        return "causal"
    return "social"


def decision_fingerprint(metadata, adapter_sha256, system_prompt):
    fields = {k: metadata.get(k) for k in ("base_model", "base_revision", "base_model_sha256", "max_length", "order_policy", "rotations")}
    if "task_policies" in metadata:
        fields["task_policies"] = metadata["task_policies"]
    fields.update(adapter_sha256=adapter_sha256, system_prompt_sha256=hashlib.sha256(system_prompt.encode()).hexdigest())
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def wilson_lower(successes, count, z):
    if count <= 0:
        return 0.0
    p = successes / count
    return (p + z * z / (2 * count) - z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))) / (1 + z * z / count)


def fit_review_rule(examples, logits, temperature, trusted_labels, target=0.9, min_groups=30):
    if not 0 < target < 1 or min_groups < 1:
        raise ValueError("非法可靠性目标")
    grid = np.linspace(0.5, 0.99, 25)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(grid)))
    checks = []
    for threshold in grid:
        groups = {}
        for ex, score in zip(examples, logits):
            p, t = softmax(score, temperature), np.asarray(ex.target)
            if p.max() < threshold:
                continue
            unique_target = np.count_nonzero(np.isclose(t, t.max(), atol=1e-10, rtol=0)) == 1
            correct = unique_target and int(p.argmax()) == int(t.argmax())
            groups[ex.group_id] = groups.get(ex.group_id, True) and correct
        count, successes = len(groups), int(sum(groups.values()))
        lower = float(wilson_lower(successes, count, z))
        checks.append({"threshold": float(threshold), "groups": count, "all_accepted_correct_groups": successes,
                       "group_success_lower_bound": lower})
    eligible = [c for c in checks if c["groups"] >= min_groups and c["group_success_lower_bound"] >= target]
    chosen = max(eligible, key=lambda c: c["groups"]) if eligible and trusted_labels else None
    return {"certified": chosen is not None, "threshold": chosen["threshold"] if chosen else None,
            "trusted_label_scope": trusted_labels, "target_group_success": target, "min_groups": min_groups,
            "selected": chosen, "checks": checks,
            "reason": "fitted_on_known_calibration_scope" if chosen else "unreviewed_social_labels" if not trusted_labels else "insufficient_calibration_evidence",
            "method": "group-all-accepted-correct Wilson lower bound; Bonferroni adjustment over threshold grid",
            "limitations": "depends on group independence and matching calibration distribution; no real-world or OOD guarantee"}


def review_reasons(policy, question, probabilities, fingerprint):
    if policy is None:
        return ["no_reliability_policy"]
    if policy.get("decision_fingerprint") != fingerprint:
        return ["stale_or_mismatched_reliability_policy"]
    if question.profile is None:
        return ["task_profile_not_declared"]
    key = f"{question.profile}:{question.kind}"
    rule = policy.get("profiles", {}).get(key)
    if rule is None or not rule.get("certified"):
        return [rule.get("reason", "profile_not_validated") if rule else "profile_not_validated"]
    threshold = rule.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return ["invalid_reliability_policy"]
    if not rule.get("trusted_label_scope"):
        return ["unreviewed_social_labels"]
    p = np.asarray(probabilities)
    if np.count_nonzero(np.isclose(p, p.max(), atol=1e-12, rtol=0)) > 1:
        return ["tied_candidates"]
    if p.max() < threshold:
        return ["below_validated_threshold"]
    return []
