"""Freeze old artifacts and replay saved predictions before an improvement round."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from renqing.metrics import softmax
from renqing.model import DecisionModel, collect_logits
from renqing.schema import read_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="audits/baseline-v0.2.json")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    paths = [p for folder in (Path("runs/renqing-v1"), Path("runs/renqing-v2"), Path("data/demo"), Path("data/expanded-v2"))
             for p in folder.rglob("*") if p.is_file()]
    report = {"purpose": "immutable old artifacts and replay evidence", "files": {}, "replays": {}}
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        report["files"][path.as_posix()] = h.hexdigest()
    torch.set_num_threads(4)
    for name, data in (("renqing-v1", "data/demo/test.jsonl"), ("renqing-v2", "data/expanded-v2/test.jsonl")):
        root = Path("runs") / name
        model, tokenizer, metadata = DecisionModel.load(root)
        examples = read_examples(data)
        saved = {x["id"]: x["probabilities"] for x in json.loads((root / "test_predictions.json").read_text(encoding="utf-8"))}
        t = json.loads((root / "calibration.json").read_text(encoding="utf-8"))["temperature"]
        logits = collect_logits(model, tokenizer, examples, metadata["max_length"], metadata["hyperparameters"]["batch_size"])
        delta = max(float(np.max(np.abs(softmax(logit, t) - saved[ex.id]))) for ex, logit in zip(examples, logits))
        if delta > 1e-5:
            raise AssertionError(f"{name}: historical prediction mismatch {delta}")
        report["replays"][name] = {"examples": len(examples), "max_probability_difference": delta, "tolerance": 1e-5, "status": "passed"}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["replays"], indent=2))


if __name__ == "__main__":
    main()
