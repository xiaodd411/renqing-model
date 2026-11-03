import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", default="runs/causal-round1-base")
    args = parser.parse_args()
    root, base = Path(args.output).resolve(), Path(args.base).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    source = json.loads((base / "source.json").read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"backend": "causal_letter", "model_id": "renqing-causal-round1-base",
                "base_model": source["model"], "base_revision": source["revision"],
                "base_model_sha256": source["model_sha256"], "base_model_local_hint": os.path.relpath(base, root),
                "max_length": 1024, "order_policy": "as_given", "rotations": 1,
                "status": "candidate_not_promoted", "adapter": False}
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
