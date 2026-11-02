import json
import math
from pathlib import Path
import tempfile
import unittest

from renqing.schema import Example, Question, parse_request, validate_splits


def example(group="g1", identity="e1", state="同事今天请我帮忙。"):
    return Example.parse({"id": identity, "group_id": group, "domain": "职场", "state": state,
                          "question": {"type": "choice", "prompt": "如何回应？", "options": ["先确认时间", "立刻辱骂"]},
                          "target": [0.95, 0.05]})


class SchemaTests(unittest.TestCase):
    def test_group_and_state_leakage(self):
        with self.assertRaisesRegex(ValueError, "场景组跨集合"):
            validate_splits({"train": [example()], "test": [example(identity="e2", state="另一种描述")]})
        with self.assertRaisesRegex(ValueError, "相同场景跨集合"):
            validate_splits({"train": [example()], "test": [example("g2", "e2")]})

    def test_invalid_probabilities_and_score(self):
        data = example().as_dict()
        for invalid in ([0.2, 0.2], [math.nan, 0.1], [-0.1, 1.1], [1.0], [True, 0.0]):
            data["target"] = invalid
            with self.assertRaises(ValueError):
                Example.parse(data)
        for values in ([1, 0], [0, float("inf")], [0, 0], [0]):
            with self.assertRaises(ValueError):
                Question.parse({"type": "score", "prompt": "风险？", "options": ["低", "高"], "values": values})

    def test_request_and_noul_order(self):
        _, qs = parse_request({"state": "对方没回复。", "questions": {"q": {"type": "noul", "prompt": "能断定生气吗？"}}})
        self.assertEqual(qs["q"].options, ("是", "否"))
        with self.assertRaises(ValueError):
            parse_request({"state": "有效", "questions": {}})

    def test_demo_no_leakage(self):
        from renqing.schema import read_examples
        root = Path(__file__).resolve().parents[1] / "data" / "demo"
        if not root.exists():
            self.skipTest("请先运行 scripts/build_data.py")
        result = validate_splits({s: read_examples(root / f"{s}.jsonl") for s in ("train", "validation", "calibration", "test")})
        self.assertEqual(sum(x["examples"] for x in result.values()), 240)
        self.assertEqual(result["train"]["groups"], 56)


class ProbabilityTests(unittest.TestCase):
    def test_calibration_improves_nll(self):
        from renqing.metrics import evaluate, fit_temperature, softmax
        logits, targets = [[8.0, 0.0], [0.0, 8.0]], [[0.8, 0.2], [0.2, 0.8]]
        t = fit_temperature(logits, targets)
        self.assertGreater(t, 1.0)
        self.assertLess(evaluate(logits, targets, t)["nll"], evaluate(logits, targets)["nll"])
        self.assertAlmostEqual(sum(softmax([10000, 9999, -20000])), 1.0)

    def test_score_expectation_and_review(self):
        from renqing.metrics import decision_result
        q = Question.parse({"type": "score", "prompt": "风险？", "options": ["低", "高"], "values": [0, 10]})
        result = decision_result(q, [0, 0], 1.0, 0.8, True)
        self.assertEqual(result["score"], 5.0)
        self.assertTrue(result["needs_review"])
        self.assertIsNone(__import__("renqing.metrics", fromlist=["evaluate"]).evaluate([[0, 0]], [[1, 0]])["accepted_accuracy_against_target_mode"])
        self.assertTrue(decision_result(q, [100, 0], 1.0, 0.8, False)["needs_review"])
        self.assertEqual(decision_result(q, [0, 0], 1.0, 0.8, True, "custom_heldout")["calibration_status"], "custom_heldout")

    def test_variable_option_counts(self):
        from renqing.metrics import evaluate
        result = evaluate([[3, 0], [0, 4, 0]], [[1, 0], [0, 1, 0]])
        self.assertEqual(result["accuracy_against_target_mode"], 1.0)
        self.assertTrue(math.isfinite(result["nll"]))


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from renqing.model import DecisionModel
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "是", "否", "好", "坏", "同", "事", "。", "问", "题"]), encoding="utf-8")
        cls.tokenizer = BertTokenizerFast(vocab_file=str(cls.root / "vocab.txt"))
        torch.manual_seed(1)
        cls.model = DecisionModel(BertModel(BertConfig(vocab_size=14, hidden_size=24, num_hidden_layers=1,
                                                     num_attention_heads=2, intermediate_size=32))).eval()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_candidate_permutation_and_save_reload(self):
        import torch
        from renqing.model import DecisionModel, encode_pairs
        first = encode_pairs(self.tokenizer, ["同事。"] * 2, ["是好", "否坏"], 32)
        second = encode_pairs(self.tokenizer, ["同事。"] * 2, ["否坏", "是好"], 32)
        with torch.no_grad():
            expected = self.model(**first)
            reversed_values = self.model(**second)
        self.assertTrue(torch.allclose(expected, reversed_values.flip(0), atol=1e-6))
        path = self.root / "saved"
        self.model.save(path, self.tokenizer, {"max_length": 64})
        restored, _, meta = DecisionModel.load(path, "cpu")
        with torch.no_grad():
            self.assertTrue(torch.allclose(expected, restored(**first), atol=1e-6))
        self.assertEqual(meta["max_length"], 64)

    def test_training_reaches_encoder_and_head(self):
        import torch
        from renqing.model import encode_pairs
        self.model.train()
        self.model.zero_grad()
        inputs = encode_pairs(self.tokenizer, ["同事。"] * 2, ["是好", "否坏"], 32)
        logits = self.model(**inputs)
        loss = -logits.log_softmax(0)[0]
        loss.backward()
        self.assertGreater(self.model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.head[1].weight.grad.abs().sum().item(), 0)
        self.model.eval()

    def test_long_query_is_rejected(self):
        from renqing.model import encode_pairs
        with self.assertRaisesRegex(ValueError, "过长"):
            encode_pairs(self.tokenizer, ["同事。"], ["好" * 40], 32)
        with self.assertRaisesRegex(ValueError, "不会静默截断"):
            encode_pairs(self.tokenizer, ["同事" * 40], ["好"], 32)


if __name__ == "__main__":
    unittest.main()
