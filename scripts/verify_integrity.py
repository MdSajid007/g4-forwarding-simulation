"""Verify the release allowlist against its SHA-256 file manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for record in manifest["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append({"path": record["path"], "error": "unsafe path"})
            continue
        path = ROOT.joinpath(*relative.parts)
        if not path.is_file():
            errors.append({"path": record["path"], "error": "missing"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append({"path": record["path"], "error": "hash mismatch"})
    print(json.dumps({"verified": not errors, "files": len(manifest["files"]),
                      "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
