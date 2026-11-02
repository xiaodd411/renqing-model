import importlib.util
import math
from pathlib import Path
import unittest

from renqing.metrics import evaluate
from renqing.schema import Example, read_examples, validate_splits

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_expanded", ROOT / "scripts/prepare_expanded.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ExpandedDataTests(unittest.TestCase):
    def test_ambiguous_targets_not_counted_as_yes(self):
        metrics = evaluate([[5, 0], [0, 5]], [[0.5, 0.5], [0, 1]], threshold=0.8)
        self.assertEqual(metrics["unambiguous_count"], 1)
        self.assertEqual(metrics["ambiguous_count"], 1)
        self.assertEqual(metrics["accuracy_against_target_mode"], 1)
        self.assertEqual(metrics["ambiguous_accepted_count"], 1)
        self.assertIsNone(evaluate([[1, 1]], [[0.5, 0.5]])["accuracy_against_target_mode"])

    def test_provenance_roundtrip(self):
        row = next(builder.synthetic_examples([["职场", "同事", "帮忙检查一页文档"]], 1))[1]
        self.assertEqual(Example.parse(row.as_dict()).metadata, row.metadata)

    def test_public_polarity_is_not_objective_truth(self):
        for label in ("Yes", "No", "Depends"):
            row = {"question": "是否同意这项安排？", "answer": "这是示例回答", "group_id": "public-q",
                   "label": label, "source_split": "dev", "source_id": "1", "domain": "沟通"}
            examples = list(builder.public_examples([row]))
            self.assertEqual(len(examples), 4)
            self.assertTrue(all(s == "test" for s, _ in examples))
            positives = [ex for _, ex in examples if ex.question.kind == "noul" and ex.target[0] == 1]
            self.assertEqual(len(positives), 1)
            self.assertEqual(positives[0].metadata["scope"], "answer_stance_not_world_truth")

    def test_context_changes_answer_with_same_candidate_set(self):
        examples = [ex for _, ex in builder.synthetic_examples([["职场", "同事", "帮忙检查一页文档"]], 1)
                    if ex.question.kind == "choice"]
        self.assertEqual(len(examples), 18)
        self.assertEqual(len({frozenset(ex.question.options) for ex in examples}), 1)
        selected = {ex.question.options[max(range(len(ex.target)), key=ex.target.__getitem__)] for ex in examples}
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({ex.group_id for ex in examples}), 1)

    def test_pii_filter(self):
        self.assertIsNotNone(builder.PII.search("联系QQ:123456789"))
        self.assertIsNotNone(builder.PII.search("请联系 test@example.com"))
        self.assertIsNone(builder.PII.search("明天14点见面讨论一次安排"))

    def test_minhash_groups_lexical_near_duplicates(self):
        left = "我打算明天下午和这位朋友一起到附近的公园散步并且讨论周末的活动安排"
        right = "我打算明天下午和那位朋友一起到附近的公园散步并且讨论周末的活动安排"
        suffix = "如果下雨我们会一起讨论其他可行的室内活动并在出发之前确认彼此的时间"
        left, right = left + suffix, right + suffix
        lg, rg = {left[i:i + 3] for i in range(len(left) - 2)}, {right[i:i + 3] for i in range(len(right) - 2)}
        self.assertGreaterEqual(len(lg & rg) / len(lg | rg), 0.85)
        result, audit = builder.near_question_clusters([left, right, "完全不同的一个问题"])
        self.assertEqual(result[left], result[right])
        self.assertNotEqual(result[left], result["完全不同的一个问题"])
        self.assertGreaterEqual(audit["near_question_merges"], 1)

    def test_actual_splits_and_legacy_holdout(self):
        data = ROOT / "data/expanded-v2"
        if not data.exists():
            self.skipTest("扩展数据尚未生成")
        splits = {s: read_examples(data / f"{s}.jsonl") for s in ("train", "validation", "calibration", "test")}
        validate_splits(splits)
        legacy = {x.group_id for s in ("validation", "calibration", "test") for x in read_examples(ROOT / "data/demo" / f"{s}.jsonl")}
        self.assertFalse(legacy & {x.group_id for x in splits["train"]})
        self.assertTrue(any(x.metadata.get("source") == "synthetic_challenge_v2" for x in splits["test"]))
        self.assertFalse(any(x.metadata.get("source") == "synthetic_challenge_v2" for x in splits["train"]))


if __name__ == "__main__":
    unittest.main()
