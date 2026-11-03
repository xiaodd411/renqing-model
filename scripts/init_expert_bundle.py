"""A frozen composition; promotion still requires a separate regression gate."""
from pathlib import Path
from renqing.artifacts import create_json


root = Path("runs/renqing-v3")
metadata = {
    "backend": "expert_bundle", "model_id": "renqing-v0.3-profile-bundle", "status": "candidate_not_promoted",
    "legacy_default": "social_choice_v1", "experts": {
        "social_choice_v1": "../renqing-v1", "stance_v2": "../renqing-v2", "causal_round2": "../causal-round2"},
    "routes": {
        "social:choice": "social_choice_v1", "social:noul": "causal_round2", "social:score": "causal_round2",
        "stance:choice": "stance_v2", "stance:noul": "stance_v2", "stance:score": "stance_v2",
        "causal:choice": "causal_round2", "causal:noul": "causal_round2", "causal:score": "causal_round2"},
    "compatibility": "no profile delegates exactly to v0.1; explicit old model paths remain unchanged",
    "selection_note": "routes declared before final regression and XCOPA test evaluation; no individual test-ID routing"
}
create_json(root / "metadata.json", metadata)
print("Created frozen profile bundle candidate")
