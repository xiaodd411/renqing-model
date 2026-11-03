import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from renqing.causal import task_policy
from renqing.reliability import decision_fingerprint, example_profile
from renqing.schema import Example, Question
from renqing.service import BundlePredictor, load_predictor


class ServiceContracts(unittest.TestCase):
    def make_bundle(self, root):
        (root / "metadata.json").write_text(json.dumps({
            "backend": "expert_bundle", "model_id": "bundle", "legacy_default": "legacy",
            "routes": {"stance:noul": "stance"}, "experts": {"legacy": "../v1", "stance": "../v2"}}), encoding="utf-8")
        return BundlePredictor(root)

    def test_no_profile_delegates_exact_original_request_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.make_bundle(Path(folder))
            expected = {"model": "old", "answers": {"q": {"needs_review": False}}}
            expert = Mock()
            expert.predict.return_value = expected
            bundle.experts["legacy"] = expert
            request = {"state": "场景", "questions": {"q": {"type": "noul", "prompt": "同意吗？"}}}
            self.assertIs(bundle.predict(request, 0.7), expected)
            expert.predict.assert_called_once_with(request, 0.7)

    def test_profile_routes_preserve_probabilities_and_require_review(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.make_bundle(Path(folder))
            class Frozen:
                def predict(self, data, threshold):
                    return {"model": "v2", "answers": {"q": {"probabilities": [0.91, 0.09], "needs_review": False}}}
            bundle.experts["stance"] = Frozen()
            request = {"state": "场景", "questions": {"q": {"type": "noul", "prompt": "同意吗？", "profile": "stance"}}}
            answer = bundle.predict(request)["answers"]["q"]
            self.assertEqual(answer["probabilities"], [0.91, 0.09])
            self.assertEqual(answer["selected_model"], "v2")
            self.assertTrue(answer["needs_review"])

    def test_unknown_backend_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "metadata.json").write_text('{"backend": "unknown"}')
            with self.assertRaises(ValueError):
                load_predictor(root)

    def test_mixed_request_keeps_unprofiled_question_review_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.make_bundle(Path(folder))
            class Frozen:
                def predict(self, data, threshold):
                    return {"model": "old", "answers": {k: {"probabilities": [0.91, 0.09], "needs_review": False}
                                                        for k in data["questions"]}}
            bundle.experts.update(legacy=Frozen(), stance=Frozen())
            result = bundle.predict({"state": "场景", "questions": {
                "old": {"type": "noul", "prompt": "同意吗？"},
                "new": {"type": "noul", "prompt": "同意吗？", "profile": "stance"}}})
            self.assertFalse(result["answers"]["old"]["needs_review"])
            self.assertTrue(result["answers"]["new"]["needs_review"])

    def test_explicit_profile_precedes_provenance_in_evaluation(self):
        ex = Example("a", "a", "常识", "场景", Question("noul", "同意吗？", ("是", "否"), profile="causal"), (1, 0))
        self.assertEqual(example_profile(ex), "causal")

    def test_task_policies_keep_fixed_answer_spaces(self):
        meta = {"order_policy": "canonical", "rotations": 3, "task_policies": {
            "score": {"order_policy": "as_given", "rotations": 1},
            "noul": {"order_policy": "as_given", "rotations": 1}}}
        self.assertEqual(task_policy(meta, "choice"), ("canonical", 3))
        for kind in ("noul", "score"):
            self.assertEqual(task_policy(meta, kind), ("as_given", 1))
            self.assertEqual(task_policy(meta, kind, "canonical", 3), ("canonical", 3))
        for rotations in (0, True, 1.5):
            with self.assertRaises(ValueError):
                task_policy(meta, "choice", rotations=rotations)
        original = decision_fingerprint(meta, "adapter", "prompt")
        self.assertNotEqual(original, decision_fingerprint({**meta, "task_policies": {}}, "adapter", "prompt"))


if __name__ == "__main__":
    unittest.main()
