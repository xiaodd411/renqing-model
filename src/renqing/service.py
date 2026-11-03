"""Backward-compatible loading plus explicit-profile composition of frozen experts."""
import json
from pathlib import Path

from .model import Predictor as LegacyPredictor, collect_logits
from .schema import parse_request


def load_predictor(model_dir, device="auto", **kwargs):
    metadata = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    backend = metadata.get("backend", "encoder")
    if backend == "encoder":
        return LegacyPredictor(model_dir, device)
    if backend == "causal_letter":
        from .causal import CausalPredictor
        return CausalPredictor(model_dir, device, **kwargs)
    if backend == "expert_bundle":
        return BundlePredictor(model_dir, device)
    raise ValueError(f"未知模型后端：{backend}")


class BundlePredictor:
    def __init__(self, model_dir, device="auto"):
        self.root = Path(model_dir).resolve()
        self.meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.device, self.experts = device, {}

    def expert(self, name):
        if name not in self.experts:
            path = (self.root / self.meta["experts"][name]).resolve()
            self.experts[name] = load_predictor(path, self.device)
        return self.experts[name]

    def predict(self, data, threshold=0.8):
        state, questions = parse_request(data)
        # Existing callers without profiles follow precisely the old code path.
        if all(q.profile is None for q in questions.values()):
            return self.expert(self.meta["legacy_default"]).predict(data, threshold)
        grouped, answers = {}, {}
        for key, q in questions.items():
            route = self.meta["routes"].get(f"{q.profile}:{q.kind}", self.meta["legacy_default"])
            grouped.setdefault(route, {})[key] = q.as_dict()
        for name, subset in grouped.items():
            result = self.expert(name).predict({"state": state, "questions": subset}, threshold)
            for key, answer in result["answers"].items():
                answer = {**answer, "selected_model": result["model"], "task_profile": questions[key].profile}
                if questions[key].profile is not None and not hasattr(self.expert(name), "policy"):
                    # Frozen encoders keep their probabilities; no new reliability claim.
                    answer.update(needs_review=True, review_reasons=["legacy_profile_reliability_not_certified"])
                answers[key] = answer
        return {"model": self.meta["model_id"], "answers": {key: answers[key] for key in questions}}


def predictor_scores(predictor, examples, batch_size=4):
    if isinstance(predictor, BundlePredictor):
        from .reliability import example_profile
        import numpy as np
        results = []
        for ex in examples:
            q = {**ex.question.as_dict(), "profile": example_profile(ex)}
            result = predictor.predict({"state": ex.state, "questions": {"q": q}})
            results.append(np.log(np.maximum(result["answers"]["q"]["probabilities"], 1e-12)))
        return results
    if hasattr(predictor, "score_examples"):
        from .reliability import example_profile
        scores = predictor.score_examples(examples, batch_size)
        calibration = predictor.calibration if predictor.calibration_valid else None
        if calibration:
            return [score / calibration.get("profile_temperatures", {}).get(f"{example_profile(ex)}:{ex.question.kind}", calibration["temperature"])
                    for ex, score in zip(examples, scores)]
        return scores
    scores = collect_logits(predictor.model, predictor.tokenizer, examples, predictor.meta["max_length"], batch_size)
    temperature = predictor.calibration["temperature"] if predictor.calibration else 1
    return [score / temperature for score in scores]
