import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from renqing.metrics import decision_result
from renqing.reliability import decision_fingerprint, fit_review_rule, review_reasons
from renqing.schema import Question
from renqing.artifacts import create_json


class ScopePolicyTests(unittest.TestCase):
    def test_many_derived_rows_do_not_become_independent_evidence(self):
        examples = [SimpleNamespace(group_id="one-question", target=(1.0, 0.0)) for _ in range(500)]
        result = fit_review_rule(examples, [[8, 0]] * 500, 1.0, True)
        self.assertFalse(result["certified"])
        self.assertEqual(max(x["groups"] for x in result["checks"]), 1)

    def test_sufficient_independent_correct_groups_can_pass(self):
        examples = [SimpleNamespace(group_id=f"q{i}", target=(1.0, 0.0)) for i in range(150)]
        rule = fit_review_rule(examples, [[8, 0]] * 150, 1.0, True)
        self.assertTrue(rule["certified"])
        self.assertGreaterEqual(rule["selected"]["group_success_lower_bound"], 0.9)
        json.dumps(rule, allow_nan=False)

    def test_synthetic_social_labels_cannot_certify_even_when_correct(self):
        examples = [SimpleNamespace(group_id=f"q{i}", target=(1.0, 0.0)) for i in range(150)]
        rule = fit_review_rule(examples, [[8, 0]] * 150, 1.0, False)
        self.assertFalse(rule["certified"])
        self.assertEqual(rule["reason"], "unreviewed_social_labels")

    def test_scope_and_fingerprint_fail_closed(self):
        q = Question.parse({"type": "noul", "prompt": "这段话表示同意吗？", "profile": "stance"})
        policy = {"decision_fingerprint": "same", "profiles": {"stance:noul": {"certified": True, "threshold": 0.9, "trusted_label_scope": True}}}
        self.assertEqual(review_reasons(policy, q, [0.95, 0.05], "same"), [])
        self.assertEqual(review_reasons(policy, q, [0.95, 0.05], "changed"), ["stale_or_mismatched_reliability_policy"])
        self.assertEqual(review_reasons(policy, q, [0.8, 0.2], "same"), ["below_validated_threshold"])
        unspecified = Question.parse({"type": "noul", "prompt": q.prompt})
        self.assertEqual(review_reasons(policy, unspecified, [0.99, 0.01], "same"), ["task_profile_not_declared"])
        policy["profiles"]["stance:noul"]["threshold"] = float("nan")
        self.assertEqual(review_reasons(policy, q, [0.99, 0.01], "same"), ["invalid_reliability_policy"])

    def test_fingerprint_tracks_behavior_not_display_name(self):
        meta = {"base_model": "a", "base_revision": "123", "base_model_sha256": "hash", "max_length": 1024, "order_policy": "canonical", "rotations": 1}
        old = decision_fingerprint(meta, "weights", "prompt")
        self.assertNotEqual(old, decision_fingerprint({**meta, "rotations": 3}, "weights", "prompt"))
        self.assertNotEqual(old, decision_fingerprint(meta, "changed", "prompt"))
        self.assertNotEqual(old, decision_fingerprint(meta, "weights", "changed prompt"))
        self.assertEqual(old, decision_fingerprint({**meta, "model_id": "display only"}, "weights", "prompt"))

    def test_lexical_tie_break_keeps_answer_stable(self):
        q = Question.parse({"type": "choice", "prompt": "选哪一个？", "options": ["乙", "甲"]})
        reversed_q = Question.parse({"type": "choice", "prompt": q.prompt, "options": list(reversed(q.options))})
        a = decision_result(q, [0, 0], 1, 0.8, False, tie_breaker="lexical")
        b = decision_result(reversed_q, [0, 0], 1, 0.8, False, tie_breaker="lexical")
        self.assertEqual(a["choice"], b["choice"])
        self.assertNotEqual(decision_result(q, [0, 0], 1, 0.8, False)["choice"],
                            decision_result(reversed_q, [0, 0], 1, 0.8, False)["choice"])

    def test_profile_roundtrip_and_legacy_shape(self):
        old = {"type": "noul", "prompt": "同意吗？"}
        self.assertEqual(Question.parse(old).as_dict(), old)
        new = {**old, "profile": "social"}
        self.assertEqual(Question.parse(new).as_dict(), new)
        with self.assertRaises(ValueError):
            Question.parse({**old, "profile": "arbitrary"})

    def test_artifact_write_does_not_destroy_existing_or_leave_partial_json(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "policy.json"
            with self.assertRaises(ValueError):
                create_json(target, {"invalid": float("nan")})
            self.assertFalse(target.exists())
            create_json(target, {"approved": False})
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                create_json(target, {"approved": True})
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
