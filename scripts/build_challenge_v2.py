import argparse
import hashlib
import json
from pathlib import Path
import random
from renqing.schema import Example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/challenge_cases_v2.json")
    parser.add_argument("--output", default="data/challenge_v2.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("挑战集已冻结，不覆盖")
    cases = json.loads(Path(args.source).read_text(encoding="utf-8"))["cases"]
    rng, examples = random.Random(527), []
    for domain, state, options, positive, negative, yes, risk in cases:
        group = "challenge-" + hashlib.sha256(state.encode()).hexdigest()[:20]
        order = list(range(len(options)))
        rng.shuffle(order)
        meta = {"source": "synthetic_challenge_v2", "label_kind": "author_assigned_not_observed_frequency",
                "human_reviewed": False, "frozen_before_training": True}
        tasks = [("choice", {"type": "choice", "prompt": "结合具体背景和当事人目标，哪种处理方式更合适？", "options": [options[i] for i in order]},
                  [0.9 if i == 0 else 0.05 for i in order]),
                 ("score", {"type": "score", "prompt": "当前互动中的分歧、信任损害或边界冲突有多明显？",
                            "options": ["低：没有明显冲突或信任损害", "中：存在分歧或压力，需要沟通", "高：出现明确越界或严重信任损害"], "values": [0, 0.5, 1]},
                  [[0.9, 0.09, 0.01], [0.08, 0.84, 0.08], [0.01, 0.09, 0.9]][risk]),
                 ("positive", {"type": "noul", "prompt": positive}, [yes, 1 - yes]),
                 ("negative", {"type": "noul", "prompt": negative}, [1 - yes, yes])]
        for task, question, target in tasks:
            examples.append(Example.parse({"id": group + "-" + task, "group_id": group, "domain": domain, "state": state,
                                           "question": question, "target": target, "metadata": meta}))
    output.write_text("".join(json.dumps(x.as_dict(), ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")
    print(json.dumps({"scenarios": len(cases), "examples": len(examples), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
