"""Inspectable, deterministic quality checks; no external service calls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from renqing.schema import read_examples, validate_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expanded-v2")
    parser.add_argument("--output", default="data/expanded-v2/quality_report.json")
    args = parser.parse_args()
    root = Path(args.data)
    splits = {s: read_examples(root / f"{s}.jsonl") for s in ("train", "validation", "calibration", "test")}
    stats = validate_splits(splits)
    profiles = {}
    for split, examples in splits.items():
        profiles[split] = {
            **stats[split], "distinct_states": len({x.state for x in examples}),
            "sources": dict(collections.Counter(x.metadata.get("source", "unknown") for x in examples)),
            "source_groups": {source: len({x.group_id for x in examples if x.metadata.get("source") == source})
                              for source in sorted({x.metadata.get("source", "unknown") for x in examples})},
            "domains": dict(collections.Counter(x.domain for x in examples)),
            "noul_target_classes": dict(collections.Counter("yes" if x.target[0] > 0.5 else "no" if x.target[0] < 0.5 else "ambiguous"
                                                            for x in examples if x.question.kind == "noul")),
            "score_target_levels": dict(collections.Counter(str(max(range(len(x.target)), key=x.target.__getitem__))
                                                             for x in examples if x.question.kind == "score")),
            "file_sha256": hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()}
    all_examples = [x for rows in splits.values() for x in rows]
    challenge = [x for x in all_examples if x.metadata.get("source") == "synthetic_challenge_v2"]
    assert challenge and all(x.metadata.get("frozen_before_training") for x in challenge)
    report = {"schema_and_split_integrity": "passed", "cross_split_group_leakage": 0, "cross_split_exact_state_leakage": 0,
              "unique_ids": len({x.id for x in all_examples}), "examples": len(all_examples),
              "independent_groups": len({x.group_id for x in all_examples}),
              "distinct_states": len({x.state for x in all_examples}), "profiles": profiles,
              "findings": [
                  {"severity": "high", "issue": "部分人际标签仍来自合成规则，尚未经独立人工复核", "impact": "不能据内部准确率认定真实社交可靠性"},
                  {"severity": "high", "issue": "公开人工标签是回答立场，不是客观事实或情商", "impact": "仅能证明辅助观点理解能力，应与人际挑战表现分开报告"},
                  {"severity": "medium", "issue": "原始问题派生多个相关任务；模板有多种改写", "impact": "行数高于独立证据数量，按问题簇或模板族隔离和统计"},
                  {"severity": "medium", "issue": "词面 MinHash 近重复检查并非完整语义去重", "impact": "仍需人工审查语义近重复"}]}
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("profiles", "findings")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
