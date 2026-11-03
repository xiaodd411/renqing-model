"""Pinned public Chinese causal benchmark; evaluation only, no training use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from renqing.schema import Example

REVISION = "e2e9d7f105a758ee869cd13e0fe251ef93a29840"
ROOT_URL = f"https://raw.githubusercontent.com/cambridgeltl/xcopa/{REVISION}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="只补全内容一致的不完整下载，不覆盖文件")
    args = parser.parse_args()
    root = Path("data/xcopa-zh-eval")
    if root.exists() and any(root.iterdir()):
        if not args.resume or (root / "manifest.json").exists():
            raise FileExistsError(root)
    payloads = {}
    manifest = {"source": "XCOPA Chinese human translation of causal commonsense questions", "revision": REVISION,
                "project": "https://github.com/cambridgeltl/xcopa", "license": "CC-BY-4.0", "training_use": False,
                "scope": "causal commonsense; not an interpersonal or emotional-intelligence benchmark",
                "pretraining_contamination": "cannot exclude exposure in foundation-model pretraining", "files": {}}
    for split in ("val", "test"):
        url = f"{ROOT_URL}/data/zh/{split}.zh.jsonl"
        payload = urllib.request.urlopen(url, timeout=60).read()
        examples = []
        for index, line in enumerate(payload.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label"])
            question = "哪个候选更可能是上述事件的原因？" if row["question"] == "cause" else "哪个候选更可能是上述事件导致的结果？"
            examples.append(Example.parse({"id": f"xcopa-zh-{split}-{index}", "group_id": f"xcopa-original-{split}-{index}",
                "domain": "因果常识", "state": row["premise"], "question": {"type": "choice", "prompt": question,
                    "options": [row["choice1"], row["choice2"]]}, "target": [float(label == 0), float(label == 1)],
                "metadata": {"source": "xcopa_zh", "source_split": split, "source_id": index, "label_kind": "public_human_causal_choice", "license": "CC-BY-4.0"}}))
        converted = "".join(json.dumps(x.as_dict(), ensure_ascii=False) + "\n" for x in examples).encode("utf-8")
        payloads[f"{split}.jsonl"] = converted
        manifest["files"][split] = {"url": url, "upstream_sha256": hashlib.sha256(payload).hexdigest(), "examples": len(examples),
                                   "converted_sha256": hashlib.sha256(converted).hexdigest()}
    payloads["LICENSE.txt"] = urllib.request.urlopen(f"{ROOT_URL}/LICENSE.md", timeout=60).read()
    payloads["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    if root.exists():
        for existing in root.iterdir():
            if not existing.is_file() or existing.name not in payloads or existing.read_bytes() != payloads[existing.name]:
                raise FileExistsError(f"不会覆盖不一致的现有内容：{existing}")
    root.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        target = root / name
        if not target.exists():
            with target.open("xb") as f:
                f.write(value)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
