"""Repair ordinal/binary regressions without modifying any earlier checkpoint.

This policy follows inspection of the known regression suite. Those tests are
regressions from this point on, not new unbiased generalization evidence.
"""
import json
from pathlib import Path
import shutil

from renqing.artifacts import create_json


def main():
    source, output = Path("runs/causal-round2"), Path("runs/causal-round3")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(source / name, output / name)
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    meta.update(model_id="renqing-causal-round3", parent_candidate="runs/causal-round2",
                task_policies={"noul": {"order_policy": "as_given", "rotations": 1},
                               "score": {"order_policy": "as_given", "rotations": 1}},
                selection_rule="choice retains canonical rotations; fixed binary/ordinal spaces preserve round1 policy",
                selection_scope="repair after known regression inspection; no new held-out accuracy claim")
    create_json(output / "metadata.json", meta)
    bundle = json.loads(Path("runs/renqing-v3/metadata.json").read_text(encoding="utf-8"))
    bundle.update(model_id="renqing-v0.3.1-profile-bundle", status="experimental_opt_in",
                  selection_note="v3.1 repairs v3 known regressions; no fresh test claim")
    bundle["experts"]["causal_round3"] = "../causal-round3"
    bundle["experts"].pop("causal_round2")
    bundle["routes"] = {k: "causal_round3" if v == "causal_round2" else v for k, v in bundle["routes"].items()}
    create_json("runs/renqing-v3.1/metadata.json", bundle)
    print("Created separate task-policy candidate and opt-in bundle; defaults unchanged")


if __name__ == "__main__":
    main()
