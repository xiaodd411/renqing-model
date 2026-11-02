"""Build deterministic, domain-stratified splits from authored scenario groups."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from renqing.schema import Example, validate_splits


def build(source, output, seed=42):
    cases = json.loads(Path(source).read_text(encoding="utf-8"))["cases"]
    by_domain = collections.defaultdict(list)
    for case in cases:
        by_domain[case[0]].append(case)
    if any(len(rows) != 10 for rows in by_domain.values()):
        raise ValueError("演示划分要求每个领域恰好 10 个场景")
    rng = random.Random(seed)
    splits = {k: [] for k in ("train", "validation", "calibration", "test")}
    risk_options = ["低：没有明显冲突或关系损害迹象", "中：已有分歧或压力，需要沟通协调", "高：已有明显冒犯、反复越界或严重信任损害"]
    risk_targets = [[0.88, 0.11, 0.01], [0.08, 0.84, 0.08], [0.01, 0.14, 0.85]]
    for domain, rows in by_domain.items():
        rows = rows.copy()
        rng.shuffle(rows)
        for i, (_, state, options, risk, criterion, yes) in enumerate(rows):
            split = "train" if i < 7 else ("validation", "calibration", "test")[i - 7]
            group = hashlib.sha256(state.encode()).hexdigest()[:16]
            order = list(range(3))
            rng.shuffle(order)
            preference = [0.9, 0.08, 0.02]
            definitions = [
                ("choice", {"type": "choice", "prompt": "兼顾当前目标、尊重和必要边界，哪种回应更得体？",
                            "options": [options[j] for j in order]}, [preference[j] for j in order]),
                ("score", {"type": "score", "prompt": "仅根据已发生的互动，当前冲突或关系损害的风险有多大？",
                           "options": risk_options, "values": [0, 0.5, 1]}, risk_targets[risk]),
                ("noul", {"type": "noul", "prompt": criterion}, [yes, round(1 - yes, 8)])]
            for kind, question, target in definitions:
                splits[split].append(Example.parse({"id": f"{group}-{kind}", "group_id": group, "domain": domain,
                                                   "state": state, "question": question, "target": target}))
    stats = validate_splits(splits)
    output = Path(output)
    if any((output / f"{name}.jsonl").exists() for name in splits):
        raise FileExistsError("输出数据已存在；请使用新的输出目录，避免覆盖人工标注")
    output.mkdir(parents=True, exist_ok=True)
    for split, examples in splits.items():
        rng.shuffle(examples)
        (output / f"{split}.jsonl").write_text("".join(json.dumps(x.as_dict(), ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")
    manifest = {"seed": seed, "source": "original_synthetic_authored_by_assistant", "human_reviewed": False,
                "label_probabilities": "author_assigned_not_observed_frequencies", "splits": stats,
                "seed_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/seed_cases.json")
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.source, args.output, args.seed)
