"""Exhaustive graph-guided repository analysis for Project Brain.

The graph is a navigation atlas, not behavioral proof.  Every query freezes a
single repository/index snapshot, presents the whole atlas when it fits, and
otherwise maps every owned edge through deterministic token-bounded batches.
The model then requests complete source evidence and may iterate until it can
answer, reports insufficient evidence, or a bounded stop condition is reached.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from structural_index import extractor_fingerprint


PROMPT_VERSION = "graph-loop-v2"
ATLAS_VERSION = 2
EXECUTABLE_KINDS = {"function", "method", "hook", "component"}
REQUEST_OPS = {"symbol", "callers", "callees", "tests", "sql", "range"}
CLAIM_KINDS = {
    "direct", "indexed_relationship", "runtime_behavior", "inference", "absence",
}
SUPPORT_STATES = {"support", "refute", "unknown"}
RESULT_STATUSES = {"need_more", "answer", "insufficient"}


class SnapshotChanged(RuntimeError):
    """The repository or index changed while a query was running."""


class CoverageError(RuntimeError):
    """The atlas did not account for every edge exactly once."""


class ModelProtocolError(RuntimeError):
    """The local model did not produce a usable structured response."""


def digest(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()
    return hashlib.sha256(payload).hexdigest()


def base36(number: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if number == 0:
        return "0"
    result = ""
    while number:
        number, remainder = divmod(number, 36)
        result = alphabet[remainder] + result
    return result


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL,
    ).strip()


@dataclass(frozen=True)
class SymbolRecord:
    symbol_id: str
    path: str
    kind: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    active: bool
    authority: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    source_path: str
    source_name: str
    source_qualified_name: str
    relation: str
    target_name: str
    target_path: str | None
    target_qualified_name: str | None
    line: int
    confidence: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SnapshotDescriptor:
    project: str
    commit: str
    content_digest: str
    graph_digest: str
    snapshot_id: str
    extractor_version: str = ""
    generation_id: str = ""
    prompt_version: str = PROMPT_VERSION


@dataclass
class FrozenSnapshot:
    descriptor: SnapshotDescriptor
    files: dict[str, str]
    symbols: tuple[SymbolRecord, ...]
    edges: tuple[EdgeRecord, ...]
    authorities: dict[str, str]
    symbol_by_id: dict[str, SymbolRecord] = field(init=False)
    edge_by_id: dict[str, EdgeRecord] = field(init=False)

    def __post_init__(self) -> None:
        self.symbol_by_id = {item.symbol_id: item for item in self.symbols}
        self.edge_by_id = {item.edge_id: item for item in self.edges}
        if len(self.symbol_by_id) != len(self.symbols):
            raise CoverageError("duplicate symbol IDs in frozen snapshot")
        if len(self.edge_by_id) != len(self.edges):
            raise CoverageError("duplicate edge IDs in frozen snapshot")

    @property
    def project(self) -> str:
        return self.descriptor.project

    @property
    def snapshot_id(self) -> str:
        return self.descriptor.snapshot_id


@dataclass(frozen=True)
class AtlasBatch:
    batch_id: str
    atlas: dict[str, Any]
    owned_edge_ids: tuple[str, ...]
    context_edge_ids: tuple[str, ...] = ()

    @property
    def batch_digest(self) -> str:
        return digest([
            self.batch_id, self.atlas, self.owned_edge_ids, self.context_edge_ids,
        ])


@dataclass
class Evidence:
    evidence_id: str
    project: str
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol_id: str | None
    symbol: str
    complete: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def for_model(self) -> dict[str, Any]:
        numbered = "\n".join(
            f"{number}: {line}"
            for number, line in enumerate(
                self.content.splitlines(), start=self.start_line,
            )
        )
        return {
            "evidence_id": self.evidence_id,
            "path": self.path,
            "lines": [self.start_line, self.end_line],
            "kind": self.kind,
            "symbol": self.symbol,
            "complete": self.complete,
            "metadata": self.metadata,
            "content": numbered,
        }


def _json_metadata(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"raw": raw or ""}
    return value if isinstance(value, dict) else {"value": value}


def _canonical_graph_payload(
    db: sqlite3.Connection, project: str,
) -> dict[str, list[dict[str, Any]]]:
    edges = [dict(row) for row in db.execute(
        """SELECT source_path,source_name,source_qualified_name,relation,
                  target_name,target_path,target_qualified_name,line,
                  resolution_confidence,metadata
           FROM edges WHERE project=?
           ORDER BY source_path,source_qualified_name,line,relation,target_name,
                    coalesce(target_path,''),coalesce(target_qualified_name,''),metadata,id""",
        (project,),
    )]
    symbols = [dict(row) for row in db.execute(
        """SELECT path,kind,qualified_name,start_line,end_line,active,metadata
           FROM symbols WHERE project=?
           ORDER BY path,start_line,end_line,kind,qualified_name,id""",
        (project,),
    )]
    return {"edges": edges, "symbols": symbols}


def repository_content_digest(repo: Path) -> str:
    """Hash paths and exact bytes for all tracked files."""
    state = hashlib.sha256()
    for relative in sorted(git(repo, "ls-files").splitlines()):
        if not relative:
            continue
        absolute = repo / relative
        if not absolute.is_file():
            raise SnapshotChanged(f"tracked file disappeared: {relative}")
        raw_path = relative.encode("utf-8", "surrogateescape")
        raw = absolute.read_bytes()
        state.update(len(raw_path).to_bytes(8, "big"))
        state.update(raw_path)
        state.update(len(raw).to_bytes(8, "big"))
        state.update(raw)
    return state.hexdigest()


def require_clean_repository(repo: Path) -> None:
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        preview = " | ".join(dirty.splitlines()[:5])
        raise SnapshotChanged(f"repository is dirty: {preview}")


def freeze_snapshot(
    db: sqlite3.Connection, project: str, repo: Path,
) -> FrozenSnapshot:
    """Copy the current clean indexed generation into immutable query memory."""
    repo_row = db.execute(
        "SELECT commit_hash,content_digest,extractor_version,generation_id "
        "FROM repos WHERE project=?", (project,),
    ).fetchone()
    structure_row = db.execute(
        "SELECT commit_hash,graph_digest,content_digest,extractor_version,generation_id "
        "FROM structure_snapshots WHERE project=?", (project,),
    ).fetchone()
    if not repo_row or not structure_row:
        raise SnapshotChanged(f"{project} has no complete indexed generation")
    commit = git(repo, "rev-parse", "HEAD")
    require_clean_repository(repo)
    if repo_row["commit_hash"] != commit or structure_row["commit_hash"] != commit:
        raise SnapshotChanged("text and graph indexes are not on repository HEAD")
    content_digest = repository_content_digest(repo)
    if content_digest != repo_row["content_digest"]:
        raise SnapshotChanged("repository byte digest does not match text index")
    if content_digest != structure_row["content_digest"]:
        raise SnapshotChanged("text and graph indexes have different byte digests")
    if (
        repo_row["extractor_version"] != extractor_fingerprint()
        or structure_row["extractor_version"] != extractor_fingerprint()
    ):
        raise SnapshotChanged("indexed extractor version is stale")
    generation_id = repo_row["generation_id"]
    if not generation_id or generation_id != structure_row["generation_id"]:
        raise SnapshotChanged("text and graph indexes are from different generations")

    commit_sets = {
        "chunks": {row[0] for row in db.execute(
            "SELECT DISTINCT commit_hash FROM chunks WHERE project=?", (project,),
        )},
        "authority": {row[0] for row in db.execute(
            "SELECT DISTINCT commit_hash FROM source_authority WHERE project=?", (project,),
        )},
        "symbols": {row[0] for row in db.execute(
            "SELECT DISTINCT commit_hash FROM symbols WHERE project=?", (project,),
        )},
        "edges": {row[0] for row in db.execute(
            "SELECT DISTINCT commit_hash FROM edges WHERE project=?", (project,),
        )},
    }
    if any(values != {commit} for values in commit_sets.values()):
        raise SnapshotChanged(f"mixed index commits: {commit_sets}")
    generation_sets = {
        table: {row[0] for row in db.execute(
            f"SELECT DISTINCT generation_id FROM {table} WHERE project=?", (project,),
        )}
        for table in ("chunks", "source_authority", "symbols", "edges")
    }
    if any(values != {generation_id} for values in generation_sets.values()):
        raise SnapshotChanged(f"mixed index generations: {generation_sets}")

    authorities = {
        row["path"]: row["authority"]
        for row in db.execute(
            "SELECT path,authority FROM source_authority WHERE project=? ORDER BY path",
            (project,),
        )
    }
    indexed_paths = {
        row[0] for row in db.execute(
            "SELECT DISTINCT path FROM chunks WHERE project=?", (project,),
        )
    } | {
        row[0] for row in db.execute(
            "SELECT DISTINCT path FROM symbols WHERE project=?", (project,),
        )
    }
    files: dict[str, str] = {}
    file_hashes = []
    for relative in sorted(indexed_paths):
        if relative not in authorities:
            raise SnapshotChanged(f"indexed path lacks authority classification: {relative}")
        absolute = repo / relative
        if not absolute.is_file():
            raise SnapshotChanged(f"indexed file disappeared: {relative}")
        try:
            content = absolute.read_text("utf-8")
        except UnicodeDecodeError:
            continue
        files[relative] = content
        file_hashes.append([relative, hashlib.sha256(content.encode()).hexdigest()])
    frozen_indexed_digest = digest(file_hashes)

    raw_symbols = [dict(row) for row in db.execute(
        """SELECT path,kind,name,qualified_name,start_line,end_line,active,metadata
           FROM symbols WHERE project=?
           ORDER BY path,start_line,end_line,kind,qualified_name""",
        (project,),
    )]
    symbols = tuple(
        SymbolRecord(
            symbol_id=f"S{base36(index)}",
            path=row["path"], kind=row["kind"], name=row["name"],
            qualified_name=row["qualified_name"], start_line=row["start_line"],
            end_line=row["end_line"], active=bool(row["active"]),
            authority=authorities.get(row["path"], "unknown"),
            metadata=_json_metadata(row["metadata"]),
        )
        for index, row in enumerate(raw_symbols)
    )

    graph_payload = _canonical_graph_payload(db, project)
    raw_edges = graph_payload["edges"]
    graph_digest = digest(graph_payload)
    if graph_digest != structure_row["graph_digest"]:
        raise SnapshotChanged("stored graph digest does not match current graph rows")
    edges = tuple(
        EdgeRecord(
            edge_id=f"E{base36(index)}",
            source_path=row["source_path"], source_name=row["source_name"],
            source_qualified_name=row["source_qualified_name"] or row["source_name"],
            relation=row["relation"], target_name=row["target_name"],
            target_path=row["target_path"],
            target_qualified_name=row["target_qualified_name"], line=row["line"],
            confidence=row["resolution_confidence"],
            metadata=_json_metadata(row["metadata"]),
        )
        for index, row in enumerate(raw_edges)
    )
    snapshot_id = digest([
        project, commit, content_digest, frozen_indexed_digest, graph_digest,
        extractor_fingerprint(), generation_id, PROMPT_VERSION, ATLAS_VERSION,
    ])
    descriptor = SnapshotDescriptor(
        project=project, commit=commit, content_digest=content_digest,
        graph_digest=graph_digest, snapshot_id=snapshot_id,
        extractor_version=extractor_fingerprint(), generation_id=generation_id,
    )
    require_clean_repository(repo)
    if repository_content_digest(repo) != content_digest:
        raise SnapshotChanged("repository bytes changed while freezing snapshot")
    return FrozenSnapshot(descriptor, files, symbols, edges, authorities)


def assert_snapshot_current(
    db: sqlite3.Connection, snapshot: FrozenSnapshot, repo: Path,
) -> None:
    if git(repo, "rev-parse", "HEAD") != snapshot.descriptor.commit:
        raise SnapshotChanged("repository commit changed during query")
    if extractor_fingerprint() != snapshot.descriptor.extractor_version:
        raise SnapshotChanged("extractor changed during query")
    require_clean_repository(repo)
    if repository_content_digest(repo) != snapshot.descriptor.content_digest:
        raise SnapshotChanged("repository bytes changed during query")
    row = db.execute(
        "SELECT commit_hash,graph_digest,content_digest,extractor_version,generation_id "
        "FROM structure_snapshots WHERE project=?",
        (snapshot.project,),
    ).fetchone()
    if not row or (
        row["commit_hash"] != snapshot.descriptor.commit
        or row["graph_digest"] != snapshot.descriptor.graph_digest
        or row["content_digest"] != snapshot.descriptor.content_digest
        or row["extractor_version"] != snapshot.descriptor.extractor_version
        or row["generation_id"] != snapshot.descriptor.generation_id
    ):
        raise SnapshotChanged("structural index changed during query")
    if digest(_canonical_graph_payload(db, snapshot.project)) != snapshot.descriptor.graph_digest:
        raise SnapshotChanged("graph rows changed during query")


def compact_atlas(edges: Iterable[EdgeRecord]) -> dict[str, Any]:
    ordered = sorted(
        edges,
        key=lambda edge: (
            edge.source_path, edge.source_qualified_name, edge.line, edge.relation,
            edge.target_name, edge.target_path or "", edge.edge_id,
        ),
    )
    paths = sorted(
        {edge.source_path for edge in ordered}
        | {edge.target_path for edge in ordered if edge.target_path},
    )
    names = sorted(
        {edge.source_qualified_name for edge in ordered}
        | {edge.target_name for edge in ordered}
        | {edge.target_qualified_name for edge in ordered if edge.target_qualified_name},
    )
    relations = sorted({edge.relation for edge in ordered})
    confidences = sorted({edge.confidence for edge in ordered})
    metadata_keys = sorted({key for edge in ordered for key in edge.metadata})
    path_id = {value: index for index, value in enumerate(paths)}
    name_id = {value: index for index, value in enumerate(names)}
    relation_id = {value: index for index, value in enumerate(relations)}
    confidence_id = {value: index for index, value in enumerate(confidences)}
    metadata_id = {value: index for index, value in enumerate(metadata_keys)}
    groups: dict[tuple[int, int], list[list[Any]]] = {}
    for edge in ordered:
        key = (path_id[edge.source_path], name_id[edge.source_qualified_name])
        groups.setdefault(key, []).append([
            edge.edge_id,
            relation_id[edge.relation],
            name_id[edge.target_name],
            path_id[edge.target_path] if edge.target_path else -1,
            name_id[edge.target_qualified_name] if edge.target_qualified_name else -1,
            edge.line,
            confidence_id[edge.confidence],
            [[metadata_id[name], edge.metadata[name]] for name in sorted(edge.metadata)],
        ])
    return {
        "v": ATLAS_VERSION,
        "legend": "g=[source_path_id,source_name_id,edges]; edge=[id,rel,target_name,target_path,target_qname,line,confidence,metadata_pairs]; metadata_pair=[metadata_key_id,value]",
        "p": paths,
        "n": names,
        "r": relations,
        "c": confidences,
        "m": metadata_keys,
        "g": [[path_index, name_index, groups[(path_index, name_index)]]
              for path_index, name_index in sorted(groups)],
    }


def atlas_skeleton(edges: Iterable[EdgeRecord]) -> dict[str, Any]:
    by_path: dict[str, Counter[str]] = {}
    for edge in edges:
        by_path.setdefault(edge.source_path, Counter())[edge.relation] += 1
    return {
        "total_edges": sum(sum(counts.values()) for counts in by_path.values()),
        "files": [[path, sum(counts.values()), dict(sorted(counts.items()))]
                  for path, counts in sorted(by_path.items())],
    }


def validate_batch_ownership(
    all_edge_ids: Iterable[str], batches: Iterable[AtlasBatch],
) -> None:
    expected = Counter({edge_id: 1 for edge_id in all_edge_ids})
    owned = Counter(
        edge_id for batch in batches for edge_id in batch.owned_edge_ids
    )
    if owned != expected:
        missing = sorted((expected - owned).elements())[:10]
        duplicate = sorted(
            edge_id for edge_id, count in owned.items() if count != 1
        )[:10]
        raise CoverageError(
            f"atlas ownership mismatch: missing={missing} duplicate={duplicate}",
        )


def strict_json_loads(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {item}"),
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("model response must be one JSON object")
    return value


RESPONSE_KEYS = {
    "snapshot_id", "call_id", "batch_id", "batch_digest", "status",
    "inspect_edge_ids", "claims", "requests", "contradictions", "answer",
    "answer_claim_ids",
}
CLAIM_KEYS = {
    "claim_id", "text", "kind", "support", "evidence_ids", "edge_ids",
    "missing_evidence",
}
REQUEST_KEYS = {
    "op", "path", "name", "qualified_name", "start_line", "end_line", "reason",
}


def validate_response_schema(
    value: dict[str, Any], *, snapshot_id: str, call_id: str,
    batch_id: str, batch_digest: str, allowed_edge_ids: set[str],
    allowed_evidence_ids: set[str], allowed_claim_ids: set[str],
) -> dict[str, Any]:
    """Fail closed on protocol/schema drift before a response can be cached."""
    missing = RESPONSE_KEYS - set(value)
    extra = set(value) - RESPONSE_KEYS
    if missing or extra:
        raise ModelProtocolError(
            f"response keys mismatch missing={sorted(missing)} extra={sorted(extra)}",
        )
    expected = {
        "snapshot_id": snapshot_id, "call_id": call_id,
        "batch_id": batch_id, "batch_digest": batch_digest,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ModelProtocolError(f"response {key} does not match request")
    if value["status"] not in RESULT_STATUSES:
        raise ModelProtocolError("invalid response status")
    for key in ("inspect_edge_ids", "claims", "requests", "contradictions", "answer_claim_ids"):
        if not isinstance(value[key], list):
            raise ModelProtocolError(f"{key} must be an array")
    if not isinstance(value["answer"], str):
        raise ModelProtocolError("answer must be a string")
    if not all(isinstance(item, str) for item in value["inspect_edge_ids"]):
        raise ModelProtocolError("inspect_edge_ids must contain strings")
    foreign = set(value["inspect_edge_ids"]) - allowed_edge_ids
    if foreign:
        raise ModelProtocolError(f"map cited edges outside its batch: {sorted(foreign)[:5]}")
    claim_ids: set[str] = set()
    for claim in value["claims"]:
        if not isinstance(claim, dict) or set(claim) != CLAIM_KEYS:
            raise ModelProtocolError("claim does not match strict schema")
        if not all(isinstance(claim[key], str) for key in (
            "claim_id", "text", "kind", "support", "missing_evidence",
        )):
            raise ModelProtocolError("claim scalar fields must be strings")
        if claim["kind"] not in CLAIM_KINDS or claim["support"] not in SUPPORT_STATES:
            raise ModelProtocolError("invalid claim kind/support")
        if not claim["claim_id"] or claim["claim_id"] in claim_ids:
            raise ModelProtocolError("claim IDs must be non-empty and unique per response")
        claim_ids.add(claim["claim_id"])
        for key in ("evidence_ids", "edge_ids"):
            if not isinstance(claim[key], list) or not all(
                isinstance(item, str) for item in claim[key]
            ):
                raise ModelProtocolError(f"claim {key} must be a string array")
        foreign = set(claim["edge_ids"]) - allowed_edge_ids
        if foreign:
            raise ModelProtocolError(
                f"claim cited edges outside its batch: {sorted(foreign)[:5]}",
            )
        foreign_evidence = set(claim["evidence_ids"]) - allowed_evidence_ids
        if foreign_evidence:
            raise ModelProtocolError(
                f"claim cited unopened evidence: {sorted(foreign_evidence)[:5]}",
            )
    for request in value["requests"]:
        if (
            not isinstance(request, dict) or not set(request) <= REQUEST_KEYS
            or "reason" not in request or not isinstance(request["reason"], str)
        ):
            raise ModelProtocolError("request does not match strict schema")
        if request.get("op") not in REQUEST_OPS:
            raise ModelProtocolError("invalid request op")
        if request.get("op") == "range":
            if (
                not isinstance(request.get("path"), str)
                or not request["path"]
                or not isinstance(request.get("start_line"), int)
                or not isinstance(request.get("end_line"), int)
                or request["start_line"] < 1
                or request["end_line"] < request["start_line"]
            ):
                raise ModelProtocolError(
                    "range request requires path and a valid inclusive line range",
                )
        if set(request) != REQUEST_KEYS:
            raise ModelProtocolError("request does not contain every strict schema field")
        if not all(isinstance(request[key], str) for key in (
            "path", "name", "qualified_name", "reason",
        )) or not all(isinstance(request[key], int) and request[key] >= 1 for key in (
            "start_line", "end_line",
        )):
            raise ModelProtocolError("request fields have invalid types or bounds")
    for contradiction in value["contradictions"]:
        if (
            not isinstance(contradiction, dict)
            or set(contradiction) != {"text", "claim_ids"}
            or not isinstance(contradiction["text"], str)
            or not isinstance(contradiction["claim_ids"], list)
            or not all(isinstance(item, str) for item in contradiction["claim_ids"])
        ):
            raise ModelProtocolError("contradiction does not match strict schema")
    if not all(isinstance(item, str) for item in value["answer_claim_ids"]):
        raise ModelProtocolError("answer_claim_ids must contain strings")
    return value


class NemotronGateway:
    """Small vLLM client with strict JSON and exact tokenizer calls."""

    def __init__(
        self, base_url: str, model: str, timeout: int = 300,
        max_output_tokens: int = 2400,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.provider = "nemotron"
        self.cache_namespace = f"nemotron:{self.base_url}:{self.model}"

    @staticmethod
    def rendered_messages(messages: list[dict[str, str]]) -> str:
        return "\n".join(
            f"<{item['role']}>\n{item['content']}\n</{item['role']}>"
            for item in messages
        )

    def token_count(self, messages: list[dict[str, str]]) -> int:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        api_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        request = Request(
            f"{api_root}/tokenize", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Nemotron tokenizer failed closed: {exc}") from exc
        if "count" in result:
            return int(result["count"])
        if isinstance(result.get("tokens"), list):
            return len(result["tokens"])
        raise RuntimeError("Nemotron tokenizer returned no token count")

    def complete(self, messages: list[dict[str, str]]) -> str:
        response_format: dict[str, Any] = {"type": "json_object"}
        try:
            request_payload = json.loads(messages[-1]["content"])
            envelope = request_payload["response_schema"]
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "project_brain_response",
                    "strict": True,
                    "schema": model_response_json_schema(envelope, request_payload),
                },
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.max_output_tokens,
            "response_format": response_format,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Nemotron completion failed: {exc}") from exc
        message = result["choices"][0]["message"]
        return (
            message.get("content") or message.get("reasoning_content")
            or message.get("reasoning") or ""
        )


class GeminiGateway:
    """Native Gemini client with matching countTokens and generateContent inputs."""

    def __init__(
        self, api_key: str, model: str = "gemini-3.6-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 300, max_output_tokens: int = 8_192,
    ) -> None:
        if not api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is empty")
        self.api_key = api_key.strip()
        self.model = model.removeprefix("models/")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.provider = "gemini"
        self.cache_namespace = (
            f"gemini-json-local-schema-v1:{self.base_url}:{self.model}"
        )

    @staticmethod
    def _messages_payload(
        messages: list[dict[str, str]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        system_parts = [
            {"text": item["content"]}
            for item in messages if item["role"] == "system"
        ]
        contents = [
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
            for item in messages if item["role"] != "system"
        ]
        return ({"parts": system_parts} if system_parts else {}), contents

    @staticmethod
    def _repair_wire_schema() -> dict[str, Any]:
        string_array = {"type": "array", "items": {"type": "string"}}
        claim = {
            "type": "object", "additionalProperties": False,
            "required": sorted(CLAIM_KEYS),
            "properties": {
                "claim_id": {"type": "string"},
                "text": {"type": "string"},
                "kind": {"type": "string", "enum": sorted(CLAIM_KINDS)},
                "support": {"type": "string", "enum": sorted(SUPPORT_STATES)},
                "evidence_ids": string_array,
                "edge_ids": string_array,
                "missing_evidence": {"type": "string"},
            },
        }
        evidence_request = {
            "type": "object", "additionalProperties": False,
            "required": sorted(REQUEST_KEYS),
            "properties": {
                "op": {"type": "string", "enum": sorted(REQUEST_OPS)},
                "path": {"type": "string"},
                "name": {"type": "string"},
                "qualified_name": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "reason": {"type": "string"},
            },
        }
        contradiction = {
            "type": "object", "additionalProperties": False,
            "required": ["text", "claim_ids"],
            "properties": {
                "text": {"type": "string"},
                "claim_ids": string_array,
            },
        }
        return {
            "type": "object", "additionalProperties": False,
            "required": sorted(RESPONSE_KEYS),
            "properties": {
                "snapshot_id": {"type": "string"},
                "call_id": {"type": "string"},
                "batch_id": {"type": "string"},
                "batch_digest": {"type": "string"},
                "status": {"type": "string", "enum": sorted(RESULT_STATUSES)},
                "inspect_edge_ids": string_array,
                "claims": {"type": "array", "items": claim},
                "requests": {"type": "array", "items": evidence_request},
                "contradictions": {"type": "array", "items": contradiction},
                "answer": {"type": "string"},
                "answer_claim_ids": string_array,
            },
        }

    def request_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        system, contents = self._messages_payload(messages)
        generation_config: dict[str, Any] = {
            "maxOutputTokens": self.max_output_tokens,
            "responseMimeType": "application/json",
        }
        # Gemini rejects some data-dependent schemas even when they are small.
        # Normal calls use JSON mode plus strict local validation. The short repair
        # call gets a stable wire schema to prevent malformed second responses.
        try:
            is_repair = "invalid_response" in json.loads(messages[-1]["content"])
        except (IndexError, TypeError, json.JSONDecodeError):
            is_repair = False
        if is_repair:
            generation_config["responseJsonSchema"] = self._repair_wire_schema()
        payload: dict[str, Any] = {
            "model": f"models/{self.model}",
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = system
        return payload

    def _post(
        self, operation: str, payload: dict[str, Any], timeout: int,
    ) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        transient_codes = {429, 500, 502, 503, 504}
        for attempt in range(3):
            request = Request(
                f"{self.base_url}/models/{self.model}:{operation}",
                data=encoded,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:1_000]
                if exc.code in transient_codes and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Gemini {operation} failed with HTTP {exc.code}: {detail}",
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini {operation} failed: {exc}") from exc
        raise RuntimeError(f"Gemini {operation} exhausted transport retries")

    def token_count(self, messages: list[dict[str, str]]) -> int:
        request_payload = self.request_payload(messages)
        result = self._post(
            "countTokens", {"generateContentRequest": request_payload}, timeout=120,
        )
        if not isinstance(result.get("totalTokens"), int):
            raise RuntimeError("Gemini countTokens returned no totalTokens")
        return result["totalTokens"]

    def complete(self, messages: list[dict[str, str]]) -> str:
        result = self._post(
            "generateContent", self.request_payload(messages), self.timeout,
        )
        try:
            parts = result["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned no textual candidate") from exc
        if not text:
            raise RuntimeError("Gemini returned an empty textual candidate")
        return text


SYSTEM_PROMPTS = {
    "plan": (
        "You are a repository structural navigator. You receive an atlas covering every "
        "relationship assigned to this map. Use it to decide which code to open; an edge "
        "proves only that the index recorded a relationship, not runtime behavior. Return "
        "compact JSON only. Choose at most 12 edge IDs, "
        "6 requests, and 8 claims; every text or reason must be one short sentence. "
        "Preserve hypotheses, contradictions, and uncertainty. Do not answer the question "
        "before requesting executable evidence. Repeat the received protocol fields exactly "
        "and satisfy response_schema without adding fields."
    ),
    "evidence": (
        "You are a code investigator. Evaluate all complete evidence provided, compare "
        "hypotheses, and return JSON only. You may make multi-source inferences from "
        "sources and coverage-based absence claims; label those claims explicitly. "
        "Every claim must cite at least one evidence_id actually supplied in this call. "
        "Do not request already supplied evidence again without evaluating it in a claim. "
        "Never invent IDs. Request missing evidence explicitly. The critic guides and "
        "challenges but does not discard useful hypotheses merely because no single line "
        "proves them yet. Repeat protocol fields exactly and satisfy response_schema."
    ),
    "reduce": (
        "You reduce a distributed investigation. Reconcile all outputs while preserving "
        "support, refutations, contradictions, and uncertainty. Return JSON only. Never "
        "turn an atlas relationship into runtime proof. If evidence is missing, return "
        "need_more with concrete requests; otherwise return answer with a clear response. "
        "Never omit prior claims: the ledger is append-only. Repeat protocol fields exactly "
        "and satisfy response_schema."
    ),
    "repair": (
        "Repair the response into exactly one valid JSON object. Do not add facts or alter "
        "the reasoning. Return JSON only."
    ),
}


def _visible_ids(value: Any, prefix: str) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str) and re.fullmatch(prefix + r"[0-9a-z]+", item):
            found.add(item)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def model_response_json_schema(
    envelope: dict[str, Any], request_payload: dict[str, Any],
) -> dict[str, Any]:
    request_phase = request_payload.get("phase")
    evidence_phase = request_phase == "evidence"
    claims_required = request_phase in {"evidence", "final_synthesis"}
    visible_edges = _visible_ids(request_payload, "E")
    visible_evidence = _visible_ids(request_payload, "V")
    visible_claims = _visible_ids(request_payload, "C")
    eligible_answer_ids = request_payload.get("eligible_answer_claim_ids", [])
    if not (
        isinstance(eligible_answer_ids, list)
        and all(isinstance(item, str) for item in eligible_answer_ids)
    ):
        eligible_answer_ids = []

    def string_array(
        maximum: int, enum: list[str] | None = None, minimum: int = 0,
    ) -> dict[str, Any]:
        items: dict[str, Any] = {"type": "string"}
        if enum:
            items["enum"] = enum
        if enum == []:
            maximum = 0
        return {
            "type": "array", "items": items,
            "minItems": minimum, "maxItems": maximum,
        }
    claim = {
        "type": "object", "additionalProperties": False,
        "required": sorted(CLAIM_KEYS),
        "properties": {
            "claim_id": {"type": "string", "minLength": 1, "maxLength": 80},
            "text": {"type": "string", "maxLength": 500},
            "kind": {"type": "string", "enum": sorted(CLAIM_KINDS)},
            "support": {"type": "string", "enum": sorted(SUPPORT_STATES)},
            "evidence_ids": string_array(
                12, visible_evidence, minimum=1 if evidence_phase else 0,
            ),
            "edge_ids": string_array(12, visible_edges),
            "missing_evidence": {"type": "string", "maxLength": 300},
        },
    }
    request = {
        "type": "object", "additionalProperties": False,
        "required": sorted(REQUEST_KEYS),
        "properties": {
            "op": {"type": "string", "enum": sorted(REQUEST_OPS)},
            "path": {"type": "string", "maxLength": 500},
            "name": {"type": "string", "maxLength": 300},
            "qualified_name": {"type": "string", "maxLength": 500},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "reason": {"type": "string", "maxLength": 240},
        },
    }
    contradiction = {
        "type": "object", "additionalProperties": False,
        "required": ["text", "claim_ids"],
        "properties": {
            "text": {"type": "string", "maxLength": 500},
            "claim_ids": string_array(8, visible_claims),
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(RESPONSE_KEYS),
        "properties": {
            "snapshot_id": {"const": envelope["snapshot_id"]},
            "call_id": {"const": envelope["call_id"]},
            "batch_id": {"const": envelope["batch_id"]},
            "batch_digest": {"const": envelope["batch_digest"]},
            "status": (
                {"const": "answer"} if request_phase == "final_synthesis"
                else {"type": "string", "enum": sorted(RESULT_STATUSES)}
            ),
            "inspect_edge_ids": string_array(12, visible_edges),
            "claims": {
                "type": "array", "items": claim,
                "minItems": 1 if claims_required else 0, "maxItems": 8,
            },
            "requests": {"type": "array", "items": request, "maxItems": 6},
            "contradictions": {
                "type": "array", "items": contradiction, "maxItems": 8,
            },
            "answer": {
                "type": "string",
                "minLength": 1 if request_phase == "final_synthesis" else 0,
                "maxLength": 6000,
            },
            "answer_claim_ids": string_array(
                8,
                eligible_answer_ids if request_phase == "final_synthesis" else visible_claims,
                minimum=1 if request_phase == "final_synthesis" else 0,
            ),
        },
    }


def messages_for(phase: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[phase]},
        {"role": "user", "content": json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"),
        )},
    ]


def response_schema(
    snapshot_id: str, call_id: str, batch_id: str, batch_digest: str,
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "call_id": call_id,
        "batch_id": batch_id,
        "batch_digest": batch_digest,
        "status": "need_more",
        "inspect_edge_ids": ["E..."],
        "claims": [{
            "claim_id": "C...", "text": "...",
            "kind": "direct|indexed_relationship|runtime_behavior|inference|absence",
            "support": "support|refute|unknown", "evidence_ids": ["V..."],
            "edge_ids": ["E..."], "missing_evidence": "string; empty if none",
        }],
        "requests": [{
            "op": "symbol|callers|callees|tests|sql|range",
            "path": "string; empty if not applicable",
            "name": "string; empty if not applicable",
            "qualified_name": "string; empty if not applicable",
            "start_line": 1, "end_line": 1,
            "reason": "why this evidence is needed",
        }],
        "contradictions": [{"text": "...", "claim_ids": ["C..."]}],
        "answer": "",
        "answer_claim_ids": [],
    }


def protocol_payload(
    phase: str, payload: dict[str, Any], snapshot_id: str,
    batch_id: str, batch_digest: str,
) -> tuple[dict[str, Any], str]:
    base = dict(payload)
    base.update({
        "snapshot_id": snapshot_id,
        "batch_id": batch_id,
        "batch_digest": batch_digest,
    })
    call_id = "Q" + digest([
        snapshot_id, phase, batch_id, batch_digest, base, PROMPT_VERSION,
    ])[:20]
    base["call_id"] = call_id
    base["response_schema"] = response_schema(
        snapshot_id, call_id, batch_id, batch_digest,
    )
    return base, call_id


def build_plan_payload(
    question: str, snapshot: FrozenSnapshot, batch: AtlasBatch,
    mode: str, skeleton: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "phase": "global_atlas" if mode == "global" else "atlas_batch",
        "question": question,
        "snapshot": asdict(snapshot.descriptor),
        "coverage": {
            "owned": len(batch.owned_edge_ids),
            "total": len(snapshot.edges),
        },
        "atlas": batch.atlas,
    }
    if skeleton is not None:
        payload["global_skeleton"] = skeleton
    return payload


def partition_atlas(
    snapshot: FrozenSnapshot,
    question: str,
    gateway: Any,
    context_window: int = 65_536,
    completion_reserve: int = 3_000,
    safety_margin: int = 1_500,
) -> tuple[str, list[AtlasBatch], dict[str, Any]]:
    """Use one complete atlas when possible; otherwise cover all source groups."""
    edge_ids = [edge.edge_id for edge in snapshot.edges]
    full_atlas = compact_atlas(snapshot.edges)
    full_batch = AtlasBatch("B0", full_atlas, tuple(edge_ids))
    full_payload, _ = protocol_payload(
        "plan", build_plan_payload(question, snapshot, full_batch, "global", None),
        snapshot.snapshot_id, full_batch.batch_id, full_batch.batch_digest,
    )
    full_messages = messages_for(
        "plan", full_payload,
    )
    full_tokens = gateway.token_count(full_messages)
    available = context_window - completion_reserve - safety_margin
    if full_tokens <= available:
        validate_batch_ownership(edge_ids, [full_batch])
        return "global", [full_batch], {
            "global_input_tokens": full_tokens, "available_input_tokens": available,
        }

    groups: dict[tuple[str, str], list[EdgeRecord]] = {}
    for edge in snapshot.edges:
        groups.setdefault(
            (edge.source_path, edge.source_qualified_name), [],
        ).append(edge)
    skeleton = atlas_skeleton(snapshot.edges)
    batches: list[AtlasBatch] = []

    def payload_for(edges: list[EdgeRecord], batch_id: str) -> dict[str, Any]:
        candidate = AtlasBatch(
            batch_id, compact_atlas(edges),
            tuple(edge.edge_id for edge in edges),
        )
        payload = build_plan_payload(
            question, snapshot, candidate, "partitioned", skeleton,
        )
        return protocol_payload(
            "plan", payload, snapshot.snapshot_id, batch_id, candidate.batch_digest,
        )[0]

    finalized: list[tuple[list[EdgeRecord], int]] = []
    remaining = [edge for key in sorted(groups) for edge in groups[key]]
    while remaining:
        batch_id = f"B{len(finalized)}"
        low, high = 1, len(remaining)
        best_size = 0
        best_tokens = 0
        while low <= high:
            midpoint = (low + high) // 2
            tokens = gateway.token_count(messages_for(
                "plan", payload_for(remaining[:midpoint], batch_id),
            ))
            if tokens <= available:
                best_size, best_tokens = midpoint, tokens
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best_size == 0:
            raise CoverageError(
                f"single edge {remaining[0].edge_id} exceeds input budget",
            )
        finalized.append((remaining[:best_size], best_tokens))
        remaining = remaining[best_size:]
    for index, (candidate, _) in enumerate(finalized):
        batches.append(AtlasBatch(
            f"B{index}", compact_atlas(candidate),
            tuple(edge.edge_id for edge in candidate),
        ))
    validate_batch_ownership(edge_ids, batches)
    finalized_tokens = [tokens for _, tokens in finalized]
    return "partitioned", batches, {
        "global_input_tokens": full_tokens,
        "available_input_tokens": available,
        "batch_input_tokens": finalized_tokens,
        "skeleton": skeleton,
    }


class EvidenceLedger:
    def __init__(
        self, snapshot: FrozenSnapshot, max_symbol_bytes: int = 100_000,
    ) -> None:
        self.snapshot = snapshot
        self.max_symbol_bytes = max_symbol_bytes
        self.items: dict[str, Evidence] = {}
        self.seen_requests: set[str] = set()
        self.unresolved: list[dict[str, Any]] = []

    def _evidence(
        self, path: str, start: int, end: int, kind: str,
        symbol: SymbolRecord | None = None, complete: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence | None:
        content = self.snapshot.files.get(path)
        if content is None:
            self.unresolved.append({"path": path, "reason": "file not frozen as text"})
            return None
        lines = content.splitlines()
        if start < 1 or end < start or end > len(lines):
            self.unresolved.append({
                "path": path, "range": [start, end], "reason": "range outside snapshot",
            })
            return None
        selected = "\n".join(lines[start - 1:end])
        evidence_id = "V" + digest([
            self.snapshot.snapshot_id, path, start, end, kind,
            hashlib.sha256(selected.encode()).hexdigest(),
        ])[:16]
        evidence = Evidence(
            evidence_id=evidence_id, project=self.snapshot.project, path=path,
            start_line=start, end_line=end, kind=kind,
            symbol_id=symbol.symbol_id if symbol else None,
            symbol=symbol.qualified_name if symbol else "", complete=complete,
            content=selected, metadata=metadata or {},
        )
        self.items.setdefault(evidence_id, evidence)
        return self.items[evidence_id]

    def containing_symbol(self, path: str, line: int) -> SymbolRecord | None:
        candidates = [
            symbol for symbol in self.snapshot.symbols
            if symbol.path == path and symbol.kind in EXECUTABLE_KINDS
            and symbol.start_line <= line <= symbol.end_line and symbol.active
        ]
        return min(
            candidates,
            key=lambda item: (item.end_line - item.start_line, item.start_line),
            default=None,
        )

    @staticmethod
    def _name_words(value: str) -> tuple[str, ...]:
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        return tuple(
            word for word in re.findall(r"[a-z0-9]+", separated.lower())
            if len(word) > 2
        )

    def seed_question(self, question: str, limit: int = 24) -> list[Evidence]:
        """Open explicitly named paths/symbols; ranking never affects atlas coverage."""
        question_words = set(self._name_words(question))
        question_flat = "".join(self._name_words(question))
        candidates: list[tuple[int, SymbolRecord]] = []
        for symbol in self.snapshot.symbols:
            if not symbol.active or symbol.kind not in EXECUTABLE_KINDS:
                continue
            name_words = self._name_words(symbol.name)
            path_words = self._name_words(Path(symbol.path).stem)
            name_flat = "".join(name_words)
            path_flat = "".join(path_words)
            score = 0
            if name_flat and name_flat in question_flat:
                score += 200 + len(name_flat)
            if path_flat and path_flat in question_flat:
                score += 160 + len(path_flat)
            overlap = len(set(name_words) & question_words)
            if overlap >= 2 or (overlap == 1 and len(name_words) == 1):
                score += overlap * 20
            if score:
                candidates.append((score, symbol))
        opened = []
        for _, symbol in sorted(
            candidates,
            key=lambda item: (
                -item[0], item[1].path, item[1].qualified_name, item[1].start_line,
            ),
        )[:limit]:
            item = self.add_symbol(symbol)
            if item:
                item.metadata["seed_reason"] = "explicit_query_name_or_path"
                opened.append(item)
        return opened

    def add_symbol(self, symbol: SymbolRecord) -> Evidence | None:
        content = self.snapshot.files.get(symbol.path, "")
        lines = content.splitlines()
        body = "\n".join(lines[symbol.start_line - 1:symbol.end_line])
        if len(body.encode()) <= self.max_symbol_bytes:
            return self._evidence(
                symbol.path, symbol.start_line, symbol.end_line, "symbol", symbol,
            )
        preview_end = min(symbol.end_line, symbol.start_line + 160)
        return self._evidence(
            symbol.path, symbol.start_line, preview_end, "symbol", symbol,
            complete=False,
            metadata={"full_range": [symbol.start_line, symbol.end_line], "reason": "oversized"},
        )

    def add_range(self, path: str, start: int, end: int) -> list[Evidence]:
        containing = self.containing_symbol(path, start)
        if containing and containing.end_line >= end:
            content = self.snapshot.files.get(containing.path, "")
            body = "\n".join(
                content.splitlines()[containing.start_line - 1:containing.end_line],
            )
            if len(body.encode()) > self.max_symbol_bytes:
                item = self._evidence(
                    path, start, end, "range", containing, complete=True,
                    metadata={
                        "parent_symbol_range": [containing.start_line, containing.end_line],
                        "parent_symbol_complete": False,
                    },
                )
            else:
                item = self.add_symbol(containing)
        else:
            item = self._evidence(path, start, end, "range")
        return [item] if item else []

    def _symbols_named(
        self, name: str, path: str | None = None, active_only: bool = True,
    ) -> list[SymbolRecord]:
        lowered = name.lower()
        return [
            symbol for symbol in self.snapshot.symbols
            if (not path or symbol.path == path)
            and (not active_only or symbol.active)
            and (symbol.name.lower() == lowered or symbol.qualified_name.lower() == lowered)
        ]

    def expand_edge(self, edge_id: str) -> list[Evidence]:
        edge = self.snapshot.edge_by_id.get(edge_id)
        if not edge:
            self.unresolved.append({"edge_id": edge_id, "reason": "unknown edge"})
            return []
        result: list[Evidence] = []
        source = self.containing_symbol(edge.source_path, edge.line)
        if source:
            item = self.add_symbol(source)
        else:
            item = self._evidence(
                edge.source_path, edge.line, edge.line, "callsite", metadata={"edge_id": edge_id},
            )
        if item:
            if item.start_line <= edge.line <= item.end_line:
                item.metadata["edge_ids"] = sorted(
                    set(item.metadata.get("edge_ids", [])) | {edge_id},
                )
            result.append(item)
            if not item.complete and source is not None:
                callsite = self._evidence(
                    edge.source_path, max(source.start_line, edge.line - 80),
                    min(source.end_line, edge.line + 80), "callsite", source,
                    complete=True,
                    metadata={
                        "edge_ids": [edge_id],
                        "parent_symbol_range": [source.start_line, source.end_line],
                        "parent_symbol_complete": False,
                    },
                )
                if callsite:
                    result.append(callsite)
        targets = self._symbols_named(
            edge.target_qualified_name or edge.target_name,
            edge.target_path,
        )
        if not targets and edge.target_qualified_name:
            targets = self._symbols_named(edge.target_name, edge.target_path)
        if not targets and edge.relation == "invokes_edge_function":
            prefix = f"supabase/functions/{edge.target_name}/"
            targets = [
                symbol for symbol in self.snapshot.symbols
                if symbol.path.startswith(prefix) and symbol.active
            ]
        for target in sorted(
            targets, key=lambda item: (item.path, item.qualified_name, item.start_line),
        ):
            target_item = self.add_symbol(target)
            if target_item:
                target_item.metadata["target_of_edge_ids"] = sorted(
                    set(target_item.metadata.get("target_of_edge_ids", [])) | {edge_id},
                )
                result.append(target_item)
        return list({item.evidence_id: item for item in result}.values())

    @staticmethod
    def request_key(request: dict[str, Any]) -> str:
        normalized = {
            key: request[key] for key in sorted(request)
            if key != "reason" and request[key] not in (None, "", [], {})
        }
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def resolve_request(self, request: dict[str, Any]) -> list[Evidence]:
        key = self.request_key(request)
        if key in self.seen_requests:
            return []
        self.seen_requests.add(key)
        op = request.get("op")
        path = request.get("path") or None
        name = str(request.get("qualified_name") or request.get("name") or "")
        if op not in REQUEST_OPS:
            self.unresolved.append({"request": request, "reason": "invalid op"})
            return []
        if path and (path.startswith("/") or ".." in Path(path).parts or path not in self.snapshot.files):
            self.unresolved.append({"request": request, "reason": "invalid path"})
            return []
        if op == "range":
            try:
                return self.add_range(path or "", int(request["start_line"]), int(request["end_line"]))
            except (KeyError, TypeError, ValueError):
                self.unresolved.append({"request": request, "reason": "invalid range"})
                return []
        result: list[Evidence] = []
        if op in {"symbol", "sql"}:
            candidates = self._symbols_named(name, path, active_only=True)
            if op == "sql":
                candidates = [item for item in candidates if item.authority == "schema_history"]
            for symbol in candidates[:8]:
                item = self.add_symbol(symbol)
                if item:
                    result.append(item)
        elif op == "callers":
            for edge in self.snapshot.edges:
                if (
                    (edge.target_name == name or edge.target_qualified_name == name)
                    and (not path or edge.target_path == path)
                ):
                    result.extend(self.expand_edge(edge.edge_id))
        elif op == "callees":
            for edge in self.snapshot.edges:
                if (
                    (edge.source_name == name or edge.source_qualified_name == name)
                    and (not path or edge.source_path == path)
                ):
                    result.extend(self.expand_edge(edge.edge_id))
        elif op == "tests":
            target_edges = {
                edge.edge_id for edge in self.snapshot.edges
                if edge.target_name == name or edge.target_qualified_name == name
            }
            for edge_id in target_edges:
                edge = self.snapshot.edge_by_id[edge_id]
                if self.snapshot.authorities.get(edge.source_path) == "test":
                    result.extend(self.expand_edge(edge_id))
            for symbol in self.snapshot.symbols:
                if symbol.authority == "test" and name.lower() in (
                    symbol.name + " " + symbol.qualified_name
                ).lower():
                    item = self.add_symbol(symbol)
                    if item:
                        result.append(item)
        unique = {item.evidence_id: item for item in result}
        if not unique:
            self.unresolved.append({"request": request, "reason": "no indexed match"})
        return list(unique.values())


def validate_model_response(
    value: dict[str, Any], snapshot: FrozenSnapshot, ledger: EvidenceLedger,
) -> dict[str, Any]:
    """Validate references without suppressing hypotheses or inferences."""
    result = dict(value)
    invalid: list[dict[str, Any]] = []
    if value.get("snapshot_id") not in (None, snapshot.snapshot_id):
        raise ModelProtocolError("model response belongs to another snapshot")
    status = value.get("status", "need_more")
    result["status"] = status if status in RESULT_STATUSES else "need_more"
    claims = value.get("claims", value.get("hypotheses", []))
    if not isinstance(claims, list):
        claims = []
    cleaned_claims = []
    raw_to_canonical: dict[str, str] = {}
    for position, raw_claim in enumerate(claims):
        if not isinstance(raw_claim, dict):
            continue
        claim = dict(raw_claim)
        if claim.get("kind") == "relationship":
            claim["kind"] = "indexed_relationship"
        claim["kind"] = claim.get("kind") if claim.get("kind") in CLAIM_KINDS else "inference"
        claim["support"] = claim.get("support") if claim.get("support") in SUPPORT_STATES else "unknown"
        raw_id = str(claim.get("claim_id") or "")
        claim_id = "C" + digest([claim.get("text", ""), claim["kind"]])[:16]
        claim["claim_id"] = claim_id
        if raw_id:
            raw_to_canonical[raw_id] = claim_id
        evidence_ids = claim.get("evidence_ids", [])
        edge_ids = claim.get("edge_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        if not isinstance(edge_ids, list):
            edge_ids = []
        bad_evidence = [item for item in evidence_ids if item not in ledger.items]
        bad_edges = [item for item in edge_ids if item not in snapshot.edge_by_id]
        claim["evidence_ids"] = [item for item in evidence_ids if item in ledger.items]
        claim["edge_ids"] = [item for item in edge_ids if item in snapshot.edge_by_id]
        if bad_evidence or bad_edges:
            citation_error = {
                "claim": position, "evidence_ids": bad_evidence, "edge_ids": bad_edges,
            }
            invalid.append(citation_error)
            claim["invalid_references"] = citation_error
            claim["support"] = "unknown"
        if claim["kind"] in {"direct", "runtime_behavior"} and not claim["evidence_ids"]:
            claim["support"] = "unknown"
            claim.setdefault("missing_evidence", "runtime/direct claim has no opened source evidence")
        if claim["kind"] == "indexed_relationship" and not claim["edge_ids"]:
            claim["support"] = "unknown"
            claim.setdefault("missing_evidence", "indexed relationship has no valid edge")
        cleaned_claims.append(claim)
    requests = value.get("requests", [])
    result["requests"] = [
        item for item in requests
        if isinstance(item, dict) and item.get("op") in REQUEST_OPS
    ][:12]
    result["inspect_edge_ids"] = [
        item for item in value.get("inspect_edge_ids", [])
        if item in snapshot.edge_by_id
    ][:24]
    result["claims"] = cleaned_claims
    known_claim_ids = {claim["claim_id"] for claim in cleaned_claims}
    known_claim_ids.update(raw_to_canonical)
    cleaned_contradictions = []
    for position, raw_contradiction in enumerate(value.get("contradictions", [])):
        if not isinstance(raw_contradiction, dict):
            continue
        contradiction = dict(raw_contradiction)
        cited = contradiction.get("claim_ids", [])
        if not isinstance(cited, list):
            cited = []
        normalized = [raw_to_canonical.get(item, item) for item in cited]
        bad_claim_ids = [item for item in normalized if item not in known_claim_ids]
        contradiction["claim_ids"] = [
            item for item in normalized if item in known_claim_ids
        ]
        if bad_claim_ids:
            contradiction["invalid_claim_ids"] = bad_claim_ids
            invalid.append({
                "contradiction": position, "claim_ids": bad_claim_ids,
            })
        cleaned_contradictions.append(contradiction)
    result["invalid_references"] = invalid
    result["contradictions"] = cleaned_contradictions
    valid_claim_ids = {
        claim["claim_id"] for claim in cleaned_claims
        if claim["support"] != "unknown" and not claim.get("invalid_references")
    }
    result["answer_claim_ids"] = list(dict.fromkeys(
        raw_to_canonical.get(item, item)
        for item in value.get("answer_claim_ids", [])
        if raw_to_canonical.get(item, item) in valid_claim_ids
    ))
    if result["status"] == "answer" and not result["answer_claim_ids"]:
        result["status"] = "need_more"
        result["answer"] = ""
        result.setdefault("answer_rejected_reason", "no valid answer_claim_ids")
    return result


def response_compact(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": value.get("status"),
        "answer": value.get("answer", ""),
        "claims": value.get("claims", []),
        "requests": value.get("requests", []),
        "inspect_edge_ids": value.get("inspect_edge_ids", []),
        "contradictions": value.get("contradictions", []),
        "invalid_references": value.get("invalid_references", []),
        "answer_claim_ids": value.get("answer_claim_ids", []),
    }


class InvestigationLedger:
    """Deterministic append-only host state; model omission cannot erase history."""

    def __init__(self) -> None:
        self.claims: dict[str, dict[str, Any]] = {}
        self.contradictions: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, dict[str, Any]] = {}
        self.inspect_edge_ids: set[str] = set()
        self.invalid_references: dict[str, dict[str, Any]] = {}

    def merge(self, value: dict[str, Any]) -> dict[str, Any]:
        for claim in value.get("claims", []):
            claim_id = claim["claim_id"]
            previous = self.claims.get(claim_id)
            if previous is None:
                self.claims[claim_id] = dict(claim)
            else:
                merged = dict(previous)
                for key in ("evidence_ids", "edge_ids"):
                    merged[key] = sorted(set(previous.get(key, [])) | set(claim.get(key, [])))
                if previous.get("support") == "unknown" and claim.get("support") != "unknown":
                    merged["support"] = claim["support"]
                elif (
                    previous.get("support") in {"support", "refute"}
                    and claim.get("support") in {"support", "refute"}
                    and previous["support"] != claim["support"]
                ):
                    contradiction = {
                        "text": f"Conflicting support states for {claim_id}",
                        "claim_ids": [claim_id],
                    }
                    self.contradictions[digest(contradiction)] = contradiction
                self.claims[claim_id] = merged
        for contradiction in value.get("contradictions", []):
            self.contradictions[digest(contradiction)] = contradiction
        for request in value.get("requests", []):
            self.requests[EvidenceLedger.request_key(request)] = request
        self.inspect_edge_ids.update(value.get("inspect_edge_ids", []))
        for invalid in value.get("invalid_references", []):
            self.invalid_references[digest(invalid)] = invalid
        enriched = dict(value)
        enriched["claims"] = [self.claims[key] for key in sorted(self.claims)]
        enriched["contradictions"] = [
            self.contradictions[key] for key in sorted(self.contradictions)
        ]
        enriched["invalid_references"] = [
            self.invalid_references[key] for key in sorted(self.invalid_references)
        ]
        return enriched

    def for_model(self) -> dict[str, Any]:
        return {
            "claims": [self.claims[key] for key in sorted(self.claims)],
            "contradictions": [
                self.contradictions[key] for key in sorted(self.contradictions)
            ],
            "requests": [self.requests[key] for key in sorted(self.requests)],
            "inspect_edge_ids": sorted(self.inspect_edge_ids),
            "invalid_references": [
                self.invalid_references[key] for key in sorted(self.invalid_references)
            ],
        }


class ExhaustiveLoop:
    def __init__(
        self, db: sqlite3.Connection, repo: Path, gateway: Any,
        cache_dir: Path | None = None, context_window: int = 65_536,
        max_rounds: int = 4, max_requests: int = 12,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.db = db
        self.repo = repo
        self.gateway = gateway
        self.cache_dir = cache_dir
        self.context_window = context_window
        self.max_rounds = max_rounds
        self.max_requests = max_requests
        self.progress = progress
        self.call_metrics: list[dict[str, Any]] = []

    def _call(
        self, phase: str, payload: dict[str, Any], snapshot: FrozenSnapshot,
        *, batch_id: str, batch_digest: str, allowed_edge_ids: set[str],
        allowed_evidence_ids: set[str] | None = None,
        allowed_claim_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        prepared, call_id = protocol_payload(
            phase, payload, snapshot.snapshot_id, batch_id, batch_digest,
        )
        messages = messages_for(phase, prepared)
        cache_namespace = getattr(
            self.gateway, "cache_namespace", type(self.gateway).__name__,
        )
        cache_key = digest([
            snapshot.snapshot_id, phase, messages, PROMPT_VERSION, cache_namespace,
        ])
        cache_file = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        legacy_key = digest([
            snapshot.snapshot_id, phase, prepared, PROMPT_VERSION,
        ])
        legacy_file = (
            self.cache_dir / f"{legacy_key}.json"
            if (
                self.cache_dir and phase == "plan"
                and getattr(self.gateway, "provider", "") == "nemotron"
            ) else None
        )
        input_tokens = self.gateway.token_count(messages)
        if input_tokens + self.gateway.max_output_tokens + 1_500 > self.context_window:
            raise CoverageError(f"{phase} request exceeds model context window")
        cached_file = next(
            (path for path in (cache_file, legacy_file) if path and path.is_file()),
            None,
        )
        if cached_file:
            assert_snapshot_current(self.db, snapshot, self.repo)
            try:
                cached = strict_json_loads(cached_file.read_text("utf-8"))
                validated = validate_response_schema(
                    cached, snapshot_id=snapshot.snapshot_id, call_id=call_id,
                    batch_id=batch_id, batch_digest=batch_digest,
                    allowed_edge_ids=allowed_edge_ids,
                    allowed_evidence_ids=allowed_evidence_ids or set(),
                    allowed_claim_ids=allowed_claim_ids or set(),
                )
                self.call_metrics.append({
                    "phase": phase, "call_id": call_id, "batch_id": batch_id,
                    "batch_digest": batch_digest, "input_tokens": input_tokens,
                    "completion_reserve": self.gateway.max_output_tokens,
                    "safety_margin": 1_500, "cache_hit": True,
                })
                return validated
            except (json.JSONDecodeError, ValueError, ModelProtocolError):
                pass
        raw = self.gateway.complete(messages)
        try:
            value = strict_json_loads(raw)
            value = validate_response_schema(
                value, snapshot_id=snapshot.snapshot_id, call_id=call_id,
                batch_id=batch_id, batch_digest=batch_digest,
                allowed_edge_ids=allowed_edge_ids,
                allowed_evidence_ids=allowed_evidence_ids or set(),
                allowed_claim_ids=allowed_claim_ids or set(),
            )
        except (json.JSONDecodeError, ValueError, ModelProtocolError) as first_error:
            repair_payload = {
                "invalid_response": raw, "error": str(first_error),
                "allowed_edge_ids": _visible_ids(prepared, "E"),
                "allowed_evidence_ids": _visible_ids(prepared, "V"),
                "allowed_claim_ids": _visible_ids(prepared, "C"),
                "response_schema": response_schema(
                    snapshot.snapshot_id, call_id, batch_id, batch_digest,
                ),
            }
            repair_messages = messages_for("repair", repair_payload)
            repair_tokens = self.gateway.token_count(repair_messages)
            if repair_tokens + self.gateway.max_output_tokens + 1_500 > self.context_window:
                raise CoverageError("repair request exceeds context; refusing truncation")
            repaired = self.gateway.complete(repair_messages)
            try:
                value = strict_json_loads(repaired)
                value = validate_response_schema(
                    value, snapshot_id=snapshot.snapshot_id, call_id=call_id,
                    batch_id=batch_id, batch_digest=batch_digest,
                    allowed_edge_ids=allowed_edge_ids,
                    allowed_evidence_ids=allowed_evidence_ids or set(),
                    allowed_claim_ids=allowed_claim_ids or set(),
                )
            except (json.JSONDecodeError, ValueError, ModelProtocolError) as exc:
                raise ModelProtocolError(
                    f"invalid model response after one retry; first={first_error}; second={exc}",
                ) from exc
        assert_snapshot_current(self.db, snapshot, self.repo)
        self.call_metrics.append({
            "phase": phase, "call_id": call_id, "batch_id": batch_id,
            "batch_digest": batch_digest, "input_tokens": input_tokens,
            "completion_reserve": self.gateway.max_output_tokens,
            "safety_margin": 1_500, "cache_hit": False,
        })
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
            os.replace(temporary, cache_file)
        return value

    def _plan_payload(
        self, question: str, snapshot: FrozenSnapshot, batch: AtlasBatch,
        mode: str, skeleton: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return build_plan_payload(question, snapshot, batch, mode, skeleton)

    def _reduce(
        self, question: str, snapshot: FrozenSnapshot,
        outputs: list[dict[str, Any]], phase: str,
        evidence: EvidenceLedger, analysis: InvestigationLedger,
    ) -> dict[str, Any]:
        if not outputs:
            raise CoverageError("cannot reduce zero mandatory outputs")
        if len(outputs) == 1:
            return analysis.merge(outputs[0])
        current = outputs
        generation = 0
        all_edge_ids = set(snapshot.edge_by_id)
        while len(current) > 1:
            generation += 1
            next_generation = []
            offset = 0
            while offset < len(current):
                remaining = len(current) - offset
                group_size = min(6, remaining)
                if remaining - group_size == 1:
                    group_size -= 1
                if group_size < 2:
                    raise CoverageError("reduction would create a non-progressing singleton group")
                chosen: list[dict[str, Any]] | None = None
                chosen_payload: dict[str, Any] | None = None
                chosen_id = ""
                chosen_digest = ""
                for size in range(group_size, 1, -1):
                    group = current[offset:offset + size]
                    batch_id = f"R-{phase}-{generation}-{offset}"
                    batch_digest = digest([batch_id, [response_compact(item) for item in group]])
                    payload = {
                        "phase": phase,
                        "question": question,
                        "generation": generation,
                        "inputs": [response_compact(item) for item in group],
                        "ledger": analysis.for_model(),
                    }
                    prepared, _ = protocol_payload(
                        "reduce", payload, snapshot.snapshot_id, batch_id, batch_digest,
                    )
                    messages = messages_for("reduce", prepared)
                    if (
                        self.gateway.token_count(messages)
                        + self.gateway.max_output_tokens + 1_500
                        <= self.context_window
                    ):
                        chosen = group
                        chosen_payload = payload
                        chosen_id = batch_id
                        chosen_digest = batch_digest
                        break
                if chosen is None or chosen_payload is None:
                    raise CoverageError(
                        "two reduce inputs exceed context; refusing truncation",
                    )
                raw = self._call(
                    "reduce", chosen_payload, snapshot, batch_id=chosen_id,
                    batch_digest=chosen_digest, allowed_edge_ids=all_edge_ids,
                    allowed_evidence_ids=set(evidence.items),
                    allowed_claim_ids=set(analysis.claims),
                )
                validated = validate_model_response(raw, snapshot, evidence)
                analysis.merge(validated)
                next_generation.append(validated)
                offset += len(chosen)
            if len(next_generation) >= len(current):
                raise CoverageError("reduction made no progress")
            current = next_generation
        return current[0]

    def _evidence_batches(
        self, question: str, snapshot: FrozenSnapshot, ledger: EvidenceLedger,
        analysis: InvestigationLedger, prior: dict[str, Any], items: list[Evidence],
        round_number: int,
    ) -> list[tuple[str, str, list[Evidence]]]:
        available = self.context_window - self.gateway.max_output_tokens - 1_500
        batches: list[tuple[str, str, list[Evidence]]] = []
        current: list[Evidence] = []

        def prepared(evidence_items: list[Evidence], batch_index: int) -> tuple[dict[str, Any], str, str]:
            batch_id = f"EV{round_number}-{batch_index}"
            batch_digest = digest([
                batch_id, [item.evidence_id for item in evidence_items],
                snapshot.snapshot_id,
            ])
            payload = {
                "phase": "evidence",
                "question": question,
                "coverage": {
                    "graph_edges_delivered": len(snapshot.edges),
                    "opened_evidence": len(ledger.items),
                },
                "prior": response_compact(prior),
                "ledger": analysis.for_model(),
                "evidence": [item.for_model() for item in evidence_items],
            }
            wrapped, _ = protocol_payload(
                "evidence", payload, snapshot.snapshot_id, batch_id, batch_digest,
            )
            return wrapped, batch_id, batch_digest

        for item in items:
            candidate = current + [item]
            wrapped, _, _ = prepared(candidate, len(batches))
            if current and self.gateway.token_count(messages_for("evidence", wrapped)) > available:
                _, batch_id, batch_digest = prepared(current, len(batches))
                batches.append((batch_id, batch_digest, current))
                current = []
            single, _, _ = prepared([item], len(batches))
            if self.gateway.token_count(messages_for("evidence", single)) > available:
                raise CoverageError(
                    f"evidence {item.evidence_id} exceeds context and was not silently truncated",
                )
            current.append(item)
        if current:
            _, batch_id, batch_digest = prepared(current, len(batches))
            batches.append((batch_id, batch_digest, current))
        return batches

    def run(self, question: str, project: str) -> dict[str, Any]:
        started_at = time.monotonic()
        self.call_metrics.clear()
        snapshot = freeze_snapshot(self.db, project, self.repo)
        mode, batches, atlas_diagnostics = partition_atlas(
            snapshot, question, self.gateway, self.context_window,
            completion_reserve=self.gateway.max_output_tokens,
        )
        skeleton = atlas_diagnostics.get("skeleton") if mode == "partitioned" else None
        evidence = EvidenceLedger(snapshot)
        analysis = InvestigationLedger()
        all_edge_ids = set(snapshot.edge_by_id)
        map_outputs: list[dict[str, Any]] = []
        delivered: list[str] = []
        for batch in batches:
            if self.progress:
                self.progress(f"atlas {len(map_outputs) + 1}/{len(batches)} {batch.batch_id}")
            raw = self._call(
                "plan", self._plan_payload(question, snapshot, batch, mode, skeleton),
                snapshot, batch_id=batch.batch_id, batch_digest=batch.batch_digest,
                allowed_edge_ids=set(batch.owned_edge_ids) | set(batch.context_edge_ids),
            )
            validated = validate_model_response(raw, snapshot, evidence)
            analysis.merge(validated)
            map_outputs.append(validated)
            delivered.extend(batch.owned_edge_ids)
        expected_counter = Counter({edge.edge_id: 1 for edge in snapshot.edges})
        delivered_counter = Counter(delivered)
        if delivered_counter != expected_counter or set(delivered) != all_edge_ids:
            raise CoverageError("not every required atlas map completed")
        state = self._reduce(
            question, snapshot, map_outputs, "atlas_reduce", evidence, analysis,
        )
        seeded = evidence.seed_question(question)
        seen_edges: set[str] = set()
        stop_reason = ""
        rounds = 0
        for round_number in range(1, self.max_rounds + 1):
            rounds = round_number
            before_ids = set() if round_number == 1 and seeded else set(evidence.items)
            pending_edges = list(dict.fromkeys(
                list(state.get("inspect_edge_ids", []))
                + sorted(analysis.inspect_edge_ids)
            ))
            pending_edges = [edge for edge in pending_edges if edge not in seen_edges][:24]
            for edge_id in pending_edges:
                seen_edges.add(edge_id)
                evidence.expand_edge(edge_id)
            request_candidates = list(state.get("requests", [])) + [
                analysis.requests[key] for key in sorted(analysis.requests)
            ]
            pending_requests = []
            pending_keys: set[str] = set()
            for request in request_candidates:
                key = EvidenceLedger.request_key(request)
                if key in evidence.seen_requests or key in pending_keys:
                    continue
                pending_keys.add(key)
                pending_requests.append(request)
                if len(pending_requests) >= self.max_requests:
                    break
            for request in pending_requests:
                evidence.resolve_request(request)
            new_ids = sorted(set(evidence.items) - before_ids)
            if not evidence.items:
                stop_reason = "planner_requested_no_resolvable_evidence"
                state["status"] = "insufficient"
                break
            if not new_ids:
                stop_reason = "no_new_evidence"
                state["status"] = "insufficient"
                break
            evidence_outputs: list[dict[str, Any]] = []
            round_batches = self._evidence_batches(
                question, snapshot, evidence, analysis, state,
                [evidence.items[item_id] for item_id in new_ids], round_number,
            )
            if self.progress:
                self.progress(
                    f"evidence round {round_number}: {len(new_ids)} opened in {len(round_batches)} batches"
                )
            for batch_id, batch_digest, evidence_batch in round_batches:
                payload = {
                    "phase": "evidence",
                    "question": question,
                    "coverage": {
                        "graph_edges_delivered": len(snapshot.edges),
                        "opened_evidence": len(evidence.items),
                    },
                    "prior": response_compact(state),
                    "ledger": analysis.for_model(),
                    "evidence": [item.for_model() for item in evidence_batch],
                }
                raw = self._call(
                    "evidence", payload, snapshot, batch_id=batch_id,
                    batch_digest=batch_digest, allowed_edge_ids=all_edge_ids,
                    allowed_evidence_ids=set(evidence.items),
                    allowed_claim_ids=set(analysis.claims),
                )
                evidence_outputs.append(validate_model_response(raw, snapshot, evidence))
            for output in evidence_outputs:
                analysis.merge(output)
            state = self._reduce(
                question, snapshot, evidence_outputs, "evidence_reduce", evidence, analysis,
            )
            if state.get("status") in {"answer", "insufficient"}:
                stop_reason = state["status"]
                break
        else:
            state["status"] = "insufficient"
            stop_reason = "max_rounds"

        eligible_claims = [
            claim for claim in analysis.claims.values()
            if claim.get("support") != "unknown"
            and not claim.get("invalid_references")
            and (claim.get("evidence_ids") or claim.get("kind") == "absence")
        ]
        if state.get("status") != "answer" and eligible_claims:
            synthesis_id = "FINAL"
            synthesis_digest = digest([
                synthesis_id, snapshot.snapshot_id,
                [claim["claim_id"] for claim in eligible_claims],
                analysis.for_model().get("contradictions", []),
            ])
            synthesis_payload = {
                "phase": "final_synthesis",
                "question": question,
                "coverage": {
                    "graph_edges_delivered": len(snapshot.edges),
                    "opened_evidence": len(evidence.items),
                    "rounds": rounds,
                    "bounded_stop": stop_reason,
                },
                "eligible_claims": eligible_claims,
                "eligible_answer_claim_ids": [
                    claim["claim_id"] for claim in eligible_claims
                ],
                "unresolved_claims": [
                    claim for claim in analysis.claims.values()
                    if claim.get("support") == "unknown"
                ],
                "contradictions": analysis.for_model().get("contradictions", []),
                "instruction": (
                    "Synthesize an answer using only eligible_claims. Repeat in claims "
                    "every used claim while preserving claim_id, evidence_ids, kind, and support; "
                    "list the same IDs in answer_claim_ids. State what was verified and "
                    "clearly identify what remains uncertain; never convert "
                    "an unresolved_claim into fact."
                ),
            }
            if self.progress:
                self.progress(
                    f"final synthesis: {len(eligible_claims)} grounded claims"
                )
            raw = self._call(
                "reduce", synthesis_payload, snapshot,
                batch_id=synthesis_id, batch_digest=synthesis_digest,
                allowed_edge_ids=all_edge_ids,
                allowed_evidence_ids=set(evidence.items),
                allowed_claim_ids={claim["claim_id"] for claim in eligible_claims},
            )
            synthesis = validate_model_response(raw, snapshot, evidence)
            analysis.merge(synthesis)
            state = synthesis
            stop_reason = (
                "final_synthesis_answer" if synthesis.get("status") == "answer"
                else f"{stop_reason}:final_synthesis_{synthesis.get('status', 'need_more')}"
            )

        state = analysis.merge(state)
        answer_claim_ids = set(state.get("answer_claim_ids", []))
        answer_claims = [
            claim for claim in state.get("claims", [])
            if claim["claim_id"] in answer_claim_ids
        ]
        answer_grounded = bool(answer_claims) and all(
            not claim.get("invalid_references") and claim.get("support") != "unknown"
            for claim in answer_claims
        ) and any(
            claim.get("evidence_ids") or claim.get("kind") == "absence"
            for claim in answer_claims
        )
        if state.get("status") == "answer" and not answer_grounded:
            state["status"] = "insufficient"
            state["answer"] = ""
            stop_reason = "answer_claims_not_grounded"
        valid_used = {
            evidence_id
            for claim in state.get("claims", [])
            if claim.get("claim_id") in answer_claim_ids
            for evidence_id in claim.get("evidence_ids", [])
            if evidence_id in evidence.items
        }
        state["sources"] = [
            {
                "evidence_id": item.evidence_id,
                "citation": f"{project}:{item.path}:{item.start_line}-{item.end_line}",
                "commit": snapshot.descriptor.commit[:12],
                "complete": item.complete,
                "kind": item.kind,
            }
            for item in (evidence.items[item_id] for item_id in sorted(valid_used))
        ]
        state["snapshot"] = asdict(snapshot.descriptor)
        state["diagnostics"] = {
            "provider": getattr(self.gateway, "provider", "unknown"),
            "model": getattr(self.gateway, "model", "unknown"),
            "atlas_mode": mode,
            "total_edges": len(snapshot.edges),
            "covered_edges": len(delivered_counter),
            "atlas_batches": len(batches),
            "batch_digests": [batch.batch_digest for batch in batches],
            "atlas": atlas_diagnostics,
            "rounds": rounds,
            "opened_evidence": len(evidence.items),
            "opened_evidence_items": [
                {
                    "evidence_id": item.evidence_id, "path": item.path,
                    "lines": [item.start_line, item.end_line],
                    "symbol": item.symbol, "complete": item.complete,
                }
                for item in sorted(evidence.items.values(), key=lambda item: item.evidence_id)
            ],
            "evidence_bytes": sum(len(item.content.encode()) for item in evidence.items.values()),
            "complete_evidence": sum(item.complete for item in evidence.items.values()),
            "unresolved_requests": evidence.unresolved,
            "unresolved_edges": sum(
                edge.confidence in {"unresolved", "same_class_unresolved"}
                for edge in snapshot.edges
            ),
            "stop_reason": stop_reason,
            "invalid_references": state.get("invalid_references", []),
            "contradictions": state.get("contradictions", []),
            "snapshot_digest": snapshot.snapshot_id,
            "model_calls": self.call_metrics,
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }
        assert_snapshot_current(self.db, snapshot, self.repo)
        return state
