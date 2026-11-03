"""Create complete JSON artifacts without truncating or replacing existing work."""
import json
import os
from pathlib import Path
import tempfile


def create_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".artifact-", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # Atomic no-clobber publication on NTFS/POSIX, unlike replace().
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
