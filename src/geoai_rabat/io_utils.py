from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else value.as_posix()
    return str(value)


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Écrit un JSON de façon atomique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)
        stream.write("\n")
        temp_name = stream.name
    os.replace(temp_name, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, source_id: str, generated: bool = True) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "source_id": source_id,
        "generated": generated,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
