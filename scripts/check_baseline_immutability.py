import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="audits/baseline-v0.2.json")
    args = parser.parse_args()
    saved = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in saved["files"].items():
        path = Path(name)
        h = hashlib.sha256()
        if not path.is_file():
            mismatches.append(name)
            continue
        with path.open("rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        if h.hexdigest() != expected:
            mismatches.append(name)
    print(json.dumps({"checked_files": len(saved["files"]), "mismatches": mismatches}, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
