"""Exercise real checkpoints, profiles, frozen defaults and policy freshness."""
import json
from pathlib import Path

import numpy as np
import torch

from renqing.artifacts import create_json
from renqing.service import load_predictor


def main():
    output = Path("audits/runtime-contracts.json")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    bundle = load_predictor("runs/renqing-v3.1")
    request = json.loads(Path("examples/request.json").read_text(encoding="utf-8"))
    legacy = load_predictor("runs/renqing-v1")
    expected, actual = legacy.predict(request), bundle.predict(request)
    if expected != actual:
        raise AssertionError("无 profile 请求不再逐字段复现旧入口")
    profiled = json.loads(Path("examples/request-profile.json").read_text(encoding="utf-8"))
    result = bundle.predict(profiled)
    if not all(answer["needs_review"] for answer in result["answers"].values()):
        raise AssertionError("未经人工验证的人际输出必须复核")
    candidate = bundle.expert("causal_round3")
    if not candidate.calibration_valid or candidate.policy["decision_fingerprint"] != candidate.fingerprint:
        raise AssertionError("当前校准/策略与决策配置不匹配")
    calibration = candidate.calibration
    overlap = set(calibration["temperature_fit_groups"]) & set(calibration["threshold_evidence_groups"])
    if overlap:
        raise AssertionError("拟合温度与认证阈值的场景组重叠")
    for answer in result["answers"].values():
        p = np.array(answer["probabilities"])
        if not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
            raise AssertionError("预测概率无效")
    evidence = {"legacy_default_exact_match": True, "profiled_outputs_need_review": True,
                "calibration_matches_current_decision": True, "temperature_and_evidence_group_overlap": len(overlap),
                "certified_profiles": [k for k, v in candidate.policy["profiles"].items() if v["certified"]],
                "request_outputs": result}
    create_json(output, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "request_outputs"}, indent=2))


if __name__ == "__main__":
    main()
