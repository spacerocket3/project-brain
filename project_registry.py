"""Safe, local registry for repositories authorized in Project Brain."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path


PROJECT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class ProjectRegistry(Mapping[str, Path]):
    """Reload the registry on access so long-lived MCP servers see CLI changes."""

    def __init__(self, path: Path):
        self.path = path

    def _data(self) -> dict:
        try:
            payload = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {"version": 1, "projects": {}}
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(payload.get("projects"), dict)
        ):
            raise RuntimeError(f"Invalid project registry: {self.path}")
        return payload

    def entries(self) -> dict[str, dict]:
        projects = self._data()["projects"]
        result: dict[str, dict] = {}
        for name, entry in projects.items():
            if not PROJECT_NAME_RE.fullmatch(name) or not isinstance(entry, dict):
                raise RuntimeError(f"Invalid project entry {name!r} in {self.path}")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                raise RuntimeError(f"Project {name!r} must use an absolute path")
            result[name] = {
                "path": str(Path(raw_path).resolve()),
                "description": str(entry.get("description") or ""),
            }
        return result

    def __getitem__(self, name: str) -> Path:
        try:
            return Path(self.entries()[name]["path"])
        except KeyError as exc:
            raise KeyError(name) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries())

    def __len__(self) -> int:
        return len(self.entries())

    def note(self, name: str) -> str:
        entry = self.entries().get(name)
        if not entry:
            raise ValueError(f"Unknown project: {name}")
        return entry["description"] or f"Registered checkout at {entry['path']}"

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    def add(self, name: str, root: Path, description: str = "") -> dict:
        if not PROJECT_NAME_RE.fullmatch(name):
            raise ValueError(
                "Project names must start with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens (max 64 characters)."
            )
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Repository directory does not exist: {resolved}")
        result = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"Not a Git repository: {resolved}")
        top_level = Path(result.stdout.strip()).resolve()
        if top_level != resolved:
            raise ValueError(
                f"Register the repository root {top_level}, not its subdirectory {resolved}"
            )
        payload = self._data()
        if name in payload["projects"]:
            raise ValueError(f"Project already registered: {name}")
        if any(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and Path(item["path"]).resolve() == resolved
            for item in payload["projects"].values()
        ):
            raise ValueError(f"Repository path already registered: {resolved}")
        payload["projects"][name] = {
            "path": str(resolved),
            "description": description.strip(),
        }
        payload["projects"] = dict(sorted(payload["projects"].items()))
        self._write(payload)
        return {"project": name, **payload["projects"][name]}

    def remove(self, name: str) -> dict:
        payload = self._data()
        try:
            entry = payload["projects"].pop(name)
        except KeyError as exc:
            raise ValueError(f"Unknown project: {name}") from exc
        self._write(payload)
        return {"project": name, **entry}
