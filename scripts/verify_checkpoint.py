"""Offline integration checks against the real saved checkpoint."""
import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from renqing.model import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/renqing-v1")
    parser.add_argument("--request", default="examples/request.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    predictor = Predictor(args.model, "cpu")
    result = predictor.predict(request)
    for answer in result["answers"].values():
        assert all(np.isfinite(p) and 0 <= p <= 1 for p in answer["probabilities"])
        assert abs(sum(answer["probabilities"]) - 1) < 1e-8
    isolated = predictor.predict({"state": request["state"], "questions": {"response": request["questions"]["response"]}})
    np.testing.assert_allclose(result["answers"]["response"]["probabilities"],
                               isolated["answers"]["response"]["probabilities"], atol=2e-5, rtol=0)
    reversed_request = json.loads(json.dumps(request))
    reversed_request["questions"]["response"]["options"].reverse()
    reversed_result = predictor.predict(reversed_request)
    np.testing.assert_allclose(result["answers"]["response"]["probabilities"],
                               reversed_result["answers"]["response"]["probabilities"][::-1], atol=2e-5, rtol=0)
    gpu_error = None
    if torch.cuda.is_available():
        gpu = Predictor(args.model, "cuda").predict(request)
        gpu_error = max(float(np.max(np.abs(np.array(a["probabilities"]) - gpu["answers"][key]["probabilities"])))
                        for key, a in result["answers"].items())
        assert gpu_error < 2e-5, gpu_error
    report = {"offline_cpu_load_and_inference": "passed", "probability_normalization": "passed",
              "question_isolation": "passed", "candidate_permutation": "passed",
              "cpu_gpu_max_probability_difference": gpu_error, "cpu_gpu_parity_tolerance": 2e-5,
              "example_prediction": result}
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "example_prediction"}, indent=2))


if __name__ == "__main__":
    main()
