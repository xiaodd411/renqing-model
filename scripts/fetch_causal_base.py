"""Explicit download of the pinned public base, or verify an existing local copy."""
import argparse
import hashlib
from pathlib import Path


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


def main():
    parser = argparse.ArgumentParser(description="下载约 3.1GB 公开底座，不上传训练或场景数据")
    parser.add_argument("--verify-path", help="仅核验已有本地底座，不联网")
    args = parser.parse_args()
    if args.verify_path:
        source = Path(args.verify_path)
    else:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(MODEL, revision=REVISION,
                      cache_dir=Path(__file__).resolve().parents[1] / "cache" / "huggingface",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"]))
    digest = hashlib.sha256()
    with (source / "model.safetensors").open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("底座 SHA256 不匹配")
    print(f"Verified {MODEL}@{REVISION}: {source.resolve()}")


if __name__ == "__main__":
    main()
