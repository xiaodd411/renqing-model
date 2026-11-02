"""Export unreviewed social examples; never fabricate independent annotations."""
import argparse
import collections
import json
from pathlib import Path
import random

from renqing.schema import read_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expanded-v2")
    parser.add_argument("--output", default="data/expanded-v2/review_queue.jsonl")
    args = parser.parse_args()
    root, groups = Path(args.data), collections.defaultdict(list)
    for ex in read_examples(root / "test.jsonl"):
        if ex.metadata.get("source") == "synthetic_challenge_v2":
            groups[ex.state].append(ex)
    training = collections.defaultdict(list)
    for ex in read_examples(root / "train.jsonl"):
        if ex.metadata.get("source") in ("original_demo", "synthetic_contrast_v2"):
            training[ex.state].append(ex)
    rng = random.Random(53)
    for state in rng.sample(sorted(training), min(32, len(training))):
        groups[state].extend(training[state])
    with Path(args.output).open("x", encoding="utf-8") as f:
        for state, examples in groups.items():
            row = {"state": state, "group_id": examples[0].group_id, "domain": examples[0].domain,
                   "review_status": "pending", "tasks": [{"id": x.id, "question": x.question.as_dict(),
                   "proposed_target": list(x.target), "annotations": []} for x in examples]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"review_scenarios": len(groups), "review_tasks": sum(map(len, groups.values())), "status": "pending_not_human_reviewed"}))


if __name__ == "__main__":
    main()
