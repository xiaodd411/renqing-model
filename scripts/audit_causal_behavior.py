import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from renqing.causal import CausalPredictor
from renqing.metrics import decision_result, evaluate, softmax
from renqing.schema import Question, read_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/expanded-v2/validation.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    examples = [x for x in read_examples(args.data) if x.question.kind == "choice"]
    examples = random.Random(91).sample(examples, 36)
    predictor = CausalPredictor(args.model)
    report = {"data_scope": "development diagnostics; not a fresh test benchmark", "policies": {}}
    reversed_questions = [Question(x.question.kind, x.question.prompt, tuple(reversed(x.question.options))) for x in examples]
    for policy, rotations in (("as_given", 1), ("canonical", 1), ("canonical", 3)):
        start = time.perf_counter()
        scores = predictor.score_examples(examples, order_policy=policy, rotations=rotations)
        with torch.inference_mode():
            reversed_scores = predictor.raw_scores([(x.state, q) for x, q in zip(examples, reversed_questions)], order_policy=policy, rotations=rotations)
        differences = [float(np.max(np.abs(softmax(a) - softmax(b)[::-1]))) for a, b in zip(scores, reversed_scores)]
        raw_flips = sum(int(np.argmax(a)) != len(b) - 1 - int(np.argmax(b)) for a, b in zip(scores, reversed_scores))
        flips = sum(decision_result(ex.question, a, 1, 0.8, False, tie_breaker="lexical")["choice"] !=
                    decision_result(q, b, 1, 0.8, False, tie_breaker="lexical")["choice"]
                    for ex, q, a, b in zip(examples, reversed_questions, scores, reversed_scores))
        report["policies"][f"{policy}_rotations_{rotations}"] = {
            "examples": len(examples), "choice_flip_count": flips, "raw_argmax_flip_count": raw_flips, "max_probability_difference": max(differences),
            "metrics": evaluate(scores, [x.target for x in examples]), "seconds": time.perf_counter() - start}
    report["remaining_limitations"] = ["canonical ordering removes input-order variance but does not prove absence of option wording bias",
                                        "rotations increase inference cost", "diagnostic labels are not independent human social annotations"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
