"""Proper scoring rules and conservative, explicit review routing."""
from __future__ import annotations

import math
import numpy as np


def softmax(logits, temperature=1.0):
    if isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, (int, float, np.integer, np.floating)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为正数")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("logits 必须是至少包含两个有限数值的一维向量")
    x = values / temperature
    x -= x.max()
    p = np.exp(x)
    return p / p.sum()


def evaluate(logits, targets, temperature=1.0, threshold=0.8):
    if not isinstance(threshold, (int, float, np.integer, np.floating)) or isinstance(threshold, (bool, np.bool_)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 [0,1] 内")
    probabilities = [softmax(x, temperature) for x in logits]
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("预测和目标必须非空且数量一致")
    targets = [np.asarray(t) for t in targets]
    for p, target in zip(probabilities, targets):
        if target.ndim != 1 or len(target) != len(p) or target.dtype.kind not in "iuf" or not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1, atol=1e-5, rtol=0):
            raise ValueError("target 必须与 logits 对齐且为总和为 1 的非负有限概率")
    predictions = [int(p.argmax()) for p in probabilities]
    correct = np.array([i == int(t.argmax()) for i, t in zip(predictions, targets)])
    unambiguous = np.array([np.count_nonzero(np.isclose(t, t.max(), atol=1e-10, rtol=0)) == 1 for t in targets])
    # For soft targets, reliability = target mass on the selected candidate.
    support = np.array([float(t[i]) for i, t in zip(predictions, targets)])
    conf = np.array([float(p.max()) for p in probabilities])
    ece = 0.0
    bins = []
    for i in range(10):
        selected = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1))
        n = int(selected.sum())
        if n:
            mean_conf, mean_support = float(conf[selected].mean()), float(support[selected].mean())
            ece += n / len(conf) * abs(mean_conf - mean_support)
            bins.append({"lower": i / 10, "count": n, "confidence": mean_conf, "target_support": mean_support})
    accept = conf >= threshold
    accepted_unambiguous = accept & unambiguous
    return {"count": len(targets), "unambiguous_count": int(unambiguous.sum()), "ambiguous_count": int((~unambiguous).sum()),
            "accuracy_against_target_mode": float(correct[unambiguous].mean()) if unambiguous.any() else None,
            "nll": float(np.mean([-np.sum(t * np.log(np.maximum(p, 1e-12))) for p, t in zip(probabilities, targets)])),
            "brier": float(np.mean([np.sum((p - t) ** 2) for p, t in zip(probabilities, targets)])),
            "ece_soft": float(ece), "coverage": float(accept.mean()),
            "accepted_accuracy_against_target_mode": float(correct[accepted_unambiguous].mean()) if accepted_unambiguous.any() else None,
            "ambiguous_accepted_count": int((accept & ~unambiguous).sum()),
            "reliability_bins": bins}


def fit_temperature(logits, targets):
    # Grid search keeps the scalar bounded and works with variable candidate counts.
    candidates = np.unique(np.concatenate([np.geomspace(0.1, 10.0, 161), [1.0]]))
    best = min(candidates, key=lambda t: evaluate(logits, targets, float(t))["nll"])
    return float(best)


def decision_result(question, logits, temperature, threshold, calibrated, calibration_scope="heldout_dataset_only", tie_breaker="input"):
    if not isinstance(threshold, (int, float, np.integer, np.floating)) or isinstance(threshold, (bool, np.bool_)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 [0,1] 内")
    p = softmax(logits, temperature)
    if len(p) != len(question.options):
        raise ValueError("logits 数量必须与候选项一致")
    selected = int(p.argmax())
    if tie_breaker not in ("input", "lexical"):
        raise ValueError("tie_breaker 必须为 input 或 lexical")
    if tie_breaker == "lexical":
        tied = np.flatnonzero(np.isclose(p, p.max(), atol=1e-12, rtol=0))
        selected = min(map(int, tied), key=lambda i: question.options[i])
    # This is maximum class probability, NOT a separate epistemic-confidence head.
    confidence = float(p.max())
    result = {"type": question.kind, "probabilities": [float(x) for x in p],
              "confidence": confidence, "confidence_definition": "max_class_probability",
              "calibration_status": calibration_scope if calibrated else "uncalibrated",
              "needs_review": not calibrated or confidence < threshold}
    if question.kind == "choice":
        result.update(choice=question.options[selected], index=selected)
    elif question.kind == "score":
        result.update(score=float(np.dot(p, question.values)), level=question.options[selected],
                      values=list(question.values))
    else:
        result.update(noul=float(p[0]))
    return result
