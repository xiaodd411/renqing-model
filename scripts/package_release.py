"""Package source, authored data, trained checkpoint and reports, excluding caches."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    excluded = {"__pycache__", ".venv", "cache", ".pytest_cache", ".git"}
    files = [p for p in root.rglob("*") if p.is_file() and not any(x in excluded or x.endswith(".egg-info") for x in p.relative_to(root).parts)]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in files:
            z.write(p, "renqing-model/" + str(p.relative_to(root)).replace("\\", "/"))
        z.writestr("renqing-model/SHA256SUMS.json", json.dumps(hashes, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert "renqing-model/runs/renqing-v1/encoder/model.safetensors" in z.namelist()
    print(json.dumps({"archive": str(output), "files": len(files) + 1, "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
