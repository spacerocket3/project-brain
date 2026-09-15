#!/usr/bin/env python3
"""Local repository memory, structural search, and repository cartography."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vector_search import (
    CODE_EMBED_MODEL,
    EMBED_MODEL,
    code_rerank_scores,
    code_reranker_backend,
    code_reranker_cached,
    ensure_schema,
    index_embeddings,
    semantic_rows,
)
from structural_index import (
    extractor_fingerprint,
    ensure_schema as ensure_structural_schema,
    index_structure,
    related,
)
from exhaustive_loop import ExhaustiveLoop, GeminiGateway, NemotronGateway
from project_registry import ProjectRegistry


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("PROJECT_BRAIN_DB", ROOT / "data" / "index.sqlite3"))
LLM_URL = os.environ.get("PROJECT_BRAIN_LLM_URL", "http://127.0.0.1:8001/v1")
LLM_MODEL = os.environ.get("PROJECT_BRAIN_LLM_MODEL", "nvidia/nemotron-3-super")
LLM_CONTAINER = os.environ.get("PROJECT_BRAIN_LLM_CONTAINER", "nemotron-super-vllm")
GEMINI_MODEL = os.environ.get("PROJECT_BRAIN_GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_API_URL = os.environ.get(
    "PROJECT_BRAIN_GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
GEMINI_SECRET_FILE = Path(os.environ.get(
    "PROJECT_BRAIN_GEMINI_SECRET_FILE",
    Path.home() / ".config" / "project-brain" / "gemini.env",
))
GEMINI_CONTEXT_WINDOW = int(os.environ.get(
    "PROJECT_BRAIN_GEMINI_CONTEXT_WINDOW", "131072",
))
PROJECTS_FILE = Path(os.environ.get(
    "PROJECT_BRAIN_PROJECTS_FILE", ROOT / "data" / "projects.json",
))
REPOS = ProjectRegistry(PROJECTS_FILE)
TEXT_EXTENSIONS = {
    ".bash", ".c", ".cc", ".cjs", ".conf", ".cpp", ".cs", ".css", ".csv",
    ".cts", ".dart", ".env.example", ".ex", ".exs", ".fs", ".fsx", ".go",
    ".gradle", ".graphql", ".groovy", ".h", ".html", ".java", ".js", ".json",
    ".jsx", ".kt", ".kts", ".lua", ".md", ".mjs", ".mts", ".php", ".pl",
    ".proto", ".py", ".rb", ".rs", ".rst", ".scala", ".scss", ".sh", ".sql",
    ".svelte", ".swift", ".toml", ".ts", ".tsx", ".txt", ".vue", ".xml",
    ".yaml", ".yml", ".zsh",
}
SKIP_NAMES = {"package-lock.json", "bun.lock", "bun.lockb", "deno.lock"}
SKIP_PARTS = {"node_modules", "dist", "build", "coverage", ".git"}
SECRET_RE = re.compile(r"(^|/)(\.env($|\.)|.*\.(pem|key|p12|pfx)$)", re.I)
WORD_RE = re.compile(r"[\w][\w.-]*", re.UNICODE)
STOPWORDS = {
    "a", "al", "and", "como", "con", "cual", "de", "del", "donde", "el", "en",
    "es", "esta", "este", "for", "how", "la", "las", "lo", "los", "of", "o",
    "para", "por", "que", "se", "the", "to", "un", "una", "y",
}
QUERY_NOISE = {
    "adjacent", "applicable", "best-located", "complete", "especially",
    "existing", "including", "relevant", "trace", "versus", "where",
    "encuentra", "identifica", "incluye", "investiga", "revisa",
}
MAX_QUERY_TERMS = 48
QUERY_EXPANSIONS = {
    "ausencia": ("absence", "absences"),
    "ausencias": ("absence", "absences"),
    "empleado": ("employee", "staff"),
    "empleados": ("employee", "employees", "staff"),
    "evaluacion": ("evaluation", "evaluations"),
    "flujo": ("flow", "workflow"),
    "huesped": ("guest",),
    "proyecto": ("project",),
    "reserva": ("reservation", "booking"),
    "saldo": ("balance",),
    "turno": ("shift",),
    "vacaciones": ("vacation", "urlaub", "absence"),
}

# Cross-language concepts that commonly describe repository invariants without
# naming their implementation vocabulary. These terms augment retrieval; they
# never become evidence by themselves.
INTENT_EXPANSIONS = {
    "reliability": {
        "triggers": ("fallo", "falla", "failed", "failure", "timeout", "conexion", "connection", "respuesta", "response"),
        "terms": ("retry", "reconcile", "idempotency", "idempotent", "command", "outbox", "dead_letter", "last_attempt", "error"),
    },
    "duplicate": {
        "triggers": ("duplic", "repet", "twice", "again", "otra vez", "reint"),
        "terms": ("unique", "idempotency", "existing", "conflict", "lock", "claim", "compare", "reservation"),
    },
    "external_write": {
        "triggers": (
            "proveedor", "provider", "extern", "third-party", "remote", "upstream",
            "webhook", "ack",
        ),
        "terms": ("external", "command", "sync", "ack", "booking", "projection", "reconcile", "worker"),
    },
    "database_guarantee": {
        "triggers": ("garantia", "garant", "constraint", "restric", "race", "carrera", "concurrent"),
        "terms": ("unique", "transaction", "advisory", "lock", "conflict", "migration", "index"),
    },
    "tests": {
        "triggers": ("test", "prueba", "spec", "coverage", "cobertura"),
        "terms": ("test", "spec", "assert", "fixture", "mock"),
    },
}


AUTHORITY_WEIGHTS = {
    "test": 1.60, "code": 1.50, "schema_history": 1.35, "canonical": 1.10,
    "config": 0.80, "active": 0.55, "generated": 0.70, "historical": 0.30,
    "unknown": 1.0,
}


@dataclass
class Hit:
    project: str
    path: str
    start_line: int
    end_line: int
    commit: str
    score: float
    content: str
    authority: str = "unknown"

    @property
    def citation(self) -> str:
        return f"{self.project}:{self.path}:{self.start_line}-{self.end_line}"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    ensure_schema(db)
    ensure_structural_schema(db)
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS repos (
            project TEXT PRIMARY KEY,
            root TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            content_digest TEXT NOT NULL DEFAULT '',
            extractor_version TEXT NOT NULL DEFAULT '',
            generation_id TEXT NOT NULL DEFAULT '',
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            project TEXT NOT NULL,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            commit_hash TEXT NOT NULL,
            generation_id TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunks_project_idx ON chunks(project);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            path, content, content='chunks', content_rowid='id',
            tokenize='unicode61 remove_diacritics 2 tokenchars ''_-'''
        );
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, path, content) VALUES (new.id, new.path, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, path, content)
            VALUES ('delete', old.id, old.path, old.content);
        END;
        """
    )
    repo_columns = {row[1] for row in db.execute("PRAGMA table_info(repos)")}
    for name in ("content_digest", "extractor_version", "generation_id"):
        if name not in repo_columns:
            db.execute(f"ALTER TABLE repos ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
    chunk_columns = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
    if "generation_id" not in chunk_columns:
        db.execute("ALTER TABLE chunks ADD COLUMN generation_id TEXT NOT NULL DEFAULT ''")
    return db


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def require_registered_project(project: str) -> Path:
    if project not in REPOS:
        available = ", ".join(REPOS) or "none"
        raise ValueError(
            f"Unknown project: {project}. Registered projects: {available}. "
            "Use `project-brain project add` first."
        )
    return REPOS[project]


def working_tree_files(repo: Path) -> Iterable[Path]:
    """Yield tracked plus untracked, non-ignored source files from the working tree."""
    listed = git(repo, "ls-files", "--cached", "--others", "--exclude-standard")
    for relative in listed.splitlines():
        path = Path(relative)
        suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".env.example") else path.suffix
        if (
            not relative
            or path.name in SKIP_NAMES
            or any(part in SKIP_PARTS for part in path.parts)
            or SECRET_RE.search(relative)
            or suffix.lower() not in TEXT_EXTENSIONS
        ):
            continue
        absolute = repo / path
        if absolute.is_file() and absolute.stat().st_size <= 1_500_000:
            yield path


def tracked_files(repo: Path) -> Iterable[Path]:
    """Compatibility name; indexing intentionally represents the working tree."""
    yield from working_tree_files(repo)


def repository_content_digest(repo: Path) -> str:
    """Hash paths and bytes for the exact indexable working-tree snapshot."""
    state = hashlib.sha256()
    for path in sorted(working_tree_files(repo), key=str):
        relative = str(path)
        absolute = repo / relative
        if not absolute.is_file():
            raise RuntimeError(f"Working-tree file disappeared while hashing: {relative}")
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
        raise RuntimeError(f"Refusing to index dirty repository: {preview}")


def chunks_for(text: str, size: int = 80, overlap: int = 15) -> Iterable[tuple[int, int, str]]:
    lines = text.splitlines()
    step = size - overlap
    for offset in range(0, len(lines), step):
        block = lines[offset : offset + size]
        if not block:
            break
        yield offset + 1, offset + len(block), "\n".join(block)
        if offset + size >= len(lines):
            break


def publish_chunks(
    db: sqlite3.Connection,
    project: str,
    commit: str,
    generation_id: str,
    desired: list[tuple[str, int, int, str]],
) -> None:
    """Publish a generation while retaining IDs (and embeddings) for unchanged chunks."""
    existing: dict[tuple[str, int, int, str], list[int]] = {}
    for row in db.execute(
        "SELECT id,path,start_line,end_line,content FROM chunks WHERE project=?",
        (project,),
    ):
        key = (row["path"], row["start_line"], row["end_line"], row["content"])
        existing.setdefault(key, []).append(row["id"])

    for path, start, end, content in desired:
        key = (path, start, end, content)
        candidates = existing.get(key)
        if candidates:
            chunk_id = candidates.pop()
            db.execute(
                "UPDATE chunks SET commit_hash=?,generation_id=? WHERE id=?",
                (commit, generation_id, chunk_id),
            )
        else:
            db.execute(
                "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
                "VALUES(?,?,?,?,?,?,?)",
                (project, path, start, end, commit, generation_id, content),
            )

    # New rows are not in `existing`; old unmatched rows are. Delete only the latter.
    stale = [chunk_id for ids in existing.values() for chunk_id in ids]
    db.executemany("DELETE FROM chunks WHERE id=?", ((chunk_id,) for chunk_id in stale))


def semantic_status(db: sqlite3.Connection, project: str) -> dict:
    chunks = db.execute(
        "SELECT count(*) FROM chunks WHERE project=?", (project,),
    ).fetchone()[0]
    embedded = db.execute(
        """SELECT count(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
           WHERE c.project=? AND e.model=?""",
        (project, EMBED_MODEL),
    ).fetchone()[0]
    pending = max(0, chunks - embedded)
    ratio = pending / chunks if chunks else 0.0
    if pending == 0:
        recommendation = "current"
    elif pending >= 100 or ratio >= 0.25:
        recommendation = "required"
    elif pending >= 50 or ratio >= 0.10:
        recommendation = "recommended"
    else:
        recommendation = "defer"
    return {
        "semantic_current": pending == 0,
        "chunks": chunks,
        "embedded_chunks": embedded,
        "pending_embeddings": pending,
        "semantic_coverage": round(embedded / chunks, 4) if chunks else 1.0,
        "recommendation": recommendation,
    }


def index_project(project: str, force: bool = False) -> dict:
    repo = require_registered_project(project)
    commit = git(repo, "rev-parse", "HEAD")
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=all"))
    content_digest = repository_content_digest(repo)
    extractor_version = extractor_fingerprint()
    generation_id = hashlib.sha256(
        f"{project}\0{commit}\0{content_digest}\0{extractor_version}".encode()
    ).hexdigest()
    db = connect()
    previous = db.execute(
        "SELECT commit_hash,content_digest,extractor_version,generation_id "
        "FROM repos WHERE project=?", (project,),
    ).fetchone()
    structural_files = db.execute(
        "SELECT count(*) FROM source_authority WHERE project=? AND commit_hash=?",
        (project, commit),
    ).fetchone()[0]
    if (
        previous and previous["commit_hash"] == commit
        and previous["content_digest"] == content_digest
        and previous["extractor_version"] == extractor_version
        and previous["generation_id"] == generation_id
        and structural_files and not force
    ):
        count = db.execute("SELECT count(*) FROM chunks WHERE project=?", (project,)).fetchone()[0]
        return {
            "project": project, "commit": commit[:12], "chunks": count,
            "status": "current", "working_tree_dirty": dirty,
            "structural_current": True,
            **semantic_status(db, project),
        }

    files = 0
    desired_chunks: list[tuple[str, int, int, str]] = []
    for relative in tracked_files(repo):
        try:
            text = (repo / relative).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files += 1
        desired_chunks.extend(
            (str(relative), start, end, content)
            for start, end, content in chunks_for(text)
        )
    chunks = len(desired_chunks)
    with db:
        publish_chunks(db, project, commit, generation_id, desired_chunks)
        db.execute(
            "INSERT INTO repos(project,root,commit_hash,content_digest,extractor_version,generation_id) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(project) DO UPDATE SET "
            "root=excluded.root, commit_hash=excluded.commit_hash, "
            "content_digest=excluded.content_digest, "
            "extractor_version=excluded.extractor_version, "
            "generation_id=excluded.generation_id, indexed_at=CURRENT_TIMESTAMP",
            (project, str(repo), commit, content_digest, extractor_version, generation_id),
        )
        structure = index_structure(
            db, project, repo, commit, content_digest=content_digest,
            generation_id=generation_id, manage_transaction=False,
        )
        if git(repo, "rev-parse", "HEAD") != commit:
            raise RuntimeError("Repository commit changed during indexing")
        if repository_content_digest(repo) != content_digest:
            raise RuntimeError("Repository bytes changed during indexing")
        if extractor_fingerprint() != extractor_version:
            raise RuntimeError("Extractor changed during indexing")
    return {
        "project": project, "commit": commit[:12], "files": files, "chunks": chunks,
        "content_digest": content_digest, "extractor_version": extractor_version,
        "generation_id": generation_id, "structure": structure, "status": "indexed",
        "working_tree_dirty": dirty,
        "structural_current": True,
        **semantic_status(db, project),
    }


def query_terms(question: str, limit: int = MAX_QUERY_TERMS) -> list[str]:
    limit = max(1, limit)
    normalized = unicodedata.normalize("NFKD", question).encode("ascii", "ignore").decode().lower()
    words: list[str] = []
    for raw in WORD_RE.findall(normalized):
        word = raw.replace("\"", "")
        if len(word) <= 1 or word in STOPWORDS or word in QUERY_NOISE:
            continue
        words.append(word)
        words.extend(QUERY_EXPANSIONS.get(word, ()))
    for intent in INTENT_EXPANSIONS.values():
        if any(trigger in normalized for trigger in intent["triggers"]):
            words.extend(intent["terms"])
    words = list(dict.fromkeys(words))
    if len(words) > limit:
        # Long agent questions commonly put exact symbols and compatibility
        # surfaces near the end. Preserve both ends instead of silently keeping
        # only the natural-language preamble.
        head = (limit + 1) // 2
        tail = limit - head
        words = list(dict.fromkeys(words[:head] + (words[-tail:] if tail else [])))
    return words


def fts_query(question: str) -> str:
    words = query_terms(question)
    if not words:
        raise ValueError("The query contains no searchable terms")
    return " OR ".join(f'"{word}"' for word in words)


def keyword_search(question: str, project: str | None = None, limit: int = 12) -> list[Hit]:
    if project:
        require_registered_project(project)
    db = connect()
    where = "AND c.project=?" if project else ""
    params: list[object] = [fts_query(question)]
    if project:
        params.append(project)
    params.append(max(1, min(limit, 30)))
    rows = db.execute(
        f"""
        SELECT c.project,c.path,c.start_line,c.end_line,c.commit_hash,c.content,
               bm25(chunks_fts, 2.5, 1.0) AS rank
        FROM chunks_fts
        JOIN chunks c ON c.id=chunks_fts.rowid
        JOIN source_authority a ON a.project=c.project AND a.path=c.path
        WHERE chunks_fts MATCH ?
          AND a.authority IN ('code','test','schema_history','generated','config') {where}
        ORDER BY rank
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        Hit(
            project=row["project"], path=row["path"], start_line=row["start_line"],
            end_line=row["end_line"], commit=row["commit_hash"][:12],
            score=round(-row["rank"], 5), content=row["content"],
        )
        for row in rows
    ]


def retrieval_multiplier(question: str, hit: Hit) -> float:
    """Boost implementation evidence that matches the query's operational intent."""
    normalized = unicodedata.normalize("NFKD", question).encode("ascii", "ignore").decode().lower()
    active = [
        (name, intent) for name, intent in INTENT_EXPANSIONS.items()
        if any(trigger in normalized for trigger in intent["triggers"])
    ]
    haystack = f"{hit.path}\n{hit.content}".lower()
    intent_matches = sum(
        1 for _, intent in active for term in intent["terms"] if term in haystack
    )
    multiplier = AUTHORITY_WEIGHTS[hit.authority] * (1.0 + min(intent_matches, 8) * 0.08)
    active_names = {name for name, _ in active}
    if active_names & {"reliability", "external_write", "duplicate"}:
        path = hit.path.lower()
        if any(term in path for term in (
            "retry", "worker", "job", "queue", "command", "reconcile", "outbox", "webhook",
        )):
            multiplier *= 1.3
        elif any(part in path for part in ("/functions/", "/migrations/", "/workers/", "/jobs/", "/server/", "/api/")):
            multiplier *= 1.12
        if any(part in path for part in ("/prompts/", "systemprompt")):
            multiplier *= 0.65
    if "tests" in active_names and hit.authority == "test":
        multiplier *= 1.5
    if "database_guarantee" in active_names and hit.authority == "schema_history":
        multiplier *= 1.4
    return multiplier


def search(question: str, project: str | None = None, limit: int = 12) -> list[Hit]:
    """Fuse local CPU vector retrieval with lexical code search."""
    candidates = max(20, min(60, limit * 4))
    keyword_hits = keyword_search(question, project, candidates)
    db = connect()
    ranked: dict[str, tuple[Hit, float]] = {}
    for rank, hit in enumerate(keyword_hits, 1):
        ranked[hit.citation] = (hit, 1.0 / (60 + rank))
    for rank, row in enumerate(semantic_rows(db, question, project, candidates), 1):
        hit = Hit(
            project=row["project"], path=row["path"], start_line=row["start_line"],
            end_line=row["end_line"], commit=row["commit_hash"][:12],
            score=round(row["semantic_score"], 5), content=row["content"],
        )
        previous = ranked.get(hit.citation)
        score = (previous[1] if previous else 0.0) + 0.9 / (60 + rank)
        ranked[hit.citation] = (previous[0] if previous else hit, score)
    weighted = []
    for hit, base_score in ranked.values():
        row = db.execute(
            "SELECT authority FROM source_authority WHERE project=? AND path=?",
            (hit.project, hit.path),
        ).fetchone()
        hit.authority = row["authority"] if row else "unknown"
        if hit.authority not in {"code", "test", "schema_history", "generated", "config"}:
            continue
        weighted.append((hit, base_score * retrieval_multiplier(question, hit)))
    results = sorted(weighted, key=lambda item: item[1], reverse=True)[:limit]
    for hit, fused_score in results:
        hit.score = round(fused_score, 6)
    return [hit for hit, _ in results]


def should_code_rerank(question: str, mode: str = "auto") -> bool:
    """Choose the slower code-aware pass only for explicit or complex searches."""
    if mode == "fast":
        return False
    if mode == "code":
        return True
    if mode != "auto":
        raise ValueError("semantic_mode must be auto, fast, or code")
    return code_reranker_cached() and len(query_terms(question)) >= 8


def rerank_code_hits(question: str, hits: list[Hit]) -> list[Hit]:
    """Fuse first-stage rank with a bounded code-aware semantic reranker."""
    if not hits:
        return []
    documents = [f"File: {hit.path}\n{hit.content}" for hit in hits]
    scores = code_rerank_scores(connect(), question, documents)
    code_order = sorted(range(len(hits)), key=lambda index: scores[index], reverse=True)
    code_ranks = {index: rank for rank, index in enumerate(code_order, 1)}
    fused = []
    for base_rank, hit in enumerate(hits, 1):
        score = 0.25 / (60 + base_rank) + 0.75 / (60 + code_ranks[base_rank - 1])
        hit.score = round(score, 6)
        fused.append((hit, score))
    return [hit for hit, _score in sorted(fused, key=lambda item: item[1], reverse=True)]


def structure_projects(project: str | None = None) -> list[dict]:
    if project:
        require_registered_project(project)
    names = [project] if project else list(REPOS)
    return [index_project(name, force=True)["structure"] for name in names]


def embed_projects(project: str | None = None, batch_size: int = 8) -> dict:
    if project:
        require_registered_project(project)
    def progress(done: int, total: int) -> None:
        print(f"Embeddings {project or 'all'}: {done}/{total}", file=sys.stderr, flush=True)
    db = connect()
    result = index_embeddings(db, project, batch_size, progress)
    if project:
        return {**result, **semantic_status(db, project)}
    return result


def purge_project_index(project: str) -> dict:
    """Delete one scope from the index without touching its Git checkout."""
    db = connect()
    tables = (
        "chunks", "source_authority", "symbols", "edges",
        "structure_snapshots", "repos",
    )
    counts = {
        table: db.execute(
            f"SELECT count(*) FROM {table} WHERE project=?", (project,),
        ).fetchone()[0]
        for table in tables
    }
    with db:
        for table in (
            "edges", "symbols", "source_authority", "structure_snapshots",
            "chunks", "repos",
        ):
            db.execute(f"DELETE FROM {table} WHERE project=?", (project,))
    return {"project": project, "deleted": counts, "checkout_deleted": False}


def register_project(name: str, root: str, description: str = "") -> dict:
    return REPOS.add(name, Path(root), description)


def unregister_project(name: str) -> dict:
    if name not in REPOS:
        raise ValueError(f"Unknown project: {name}")
    purge = purge_project_index(name)
    registration = REPOS.remove(name)
    return {**registration, **purge}


def structural_context(hits: list[Hit], limit: int = 100) -> str:
    """Render graph evidence overlapping the exact retrieved ranges."""
    if not hits:
        return ""
    db = connect()
    ranges = list(dict.fromkeys(
        (hit.project, hit.path, hit.start_line, hit.end_line) for hit in hits
    ))
    lines = ["STRUCTURAL INDEX (active=0 means superseded SQL definition):"]
    remaining = limit
    for project, path, start_line, end_line in ranges:
        if remaining <= 0:
            break
        symbols = db.execute(
            "SELECT kind,name,start_line,end_line,active FROM symbols "
            "WHERE project=? AND path=? AND start_line<=? AND end_line>=? "
            "ORDER BY active DESC,start_line LIMIT 20",
            (project, path, end_line, start_line),
        ).fetchall()
        edges = db.execute(
            "SELECT relation,target_name,target_path,line FROM edges "
            "WHERE project=? AND source_path=? AND line BETWEEN ? AND ? "
            "ORDER BY line LIMIT 20",
            (project, path, start_line, end_line),
        ).fetchall()
        for row in symbols:
            lines.append(
                f"SYMBOL {project}:{path}:{row['start_line']} {row['kind']} "
                f"{row['name']} active={row['active']}"
            )
            remaining -= 1
        for row in edges:
            target_path = f" -> {row['target_path']}" if row["target_path"] else ""
            lines.append(
                f"EDGE {project}:{path}:{row['line']} {row['relation']} "
                f"{row['target_name']}{target_path}"
            )
            remaining -= 1
    return "\n".join(lines) if len(lines) > 1 else ""


def answer_from_hits(question: str, hits: list[Hit]) -> dict:
    if not hits:
        return {"answer": "The index contains insufficient context.", "sources": []}
    source_parts = []
    used = []
    total = 0
    for hit in hits:
        block = f"SOURCE [{hit.citation}] commit={hit.commit} authority={hit.authority}\n{hit.content}"
        if total + len(block) > 24_000:
            continue
        source_parts.append(block)
        used.append(hit)
        total += len(block)
    graph = structural_context(used, limit=30)
    context_parts = source_parts + ([graph] if graph else [])
    system = (
        "You are a repository engineering assistant. Answer only from the included repository "
        "sources. State clearly when evidence is missing. Cite each technical claim as "
        "[project:path:start-end]. Never invent files, behavior, or decisions. The context "
        "contains executable code, SQL, types, and tests. Treat active=0 as a superseded SQL "
        "definition and prefer active=1. A SYMBOL records a definition or range overlap; only "
        "an EDGE records an indexed relationship. Do not confuse proximity with dependency. "
        "Answer concretely."
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\n\nCONTEXT:\n" + "\n\n".join(context_parts)},
        ],
        "temperature": 0.1,
        "max_tokens": 700,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        f"{LLM_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Nemotron did not respond: {exc}") from exc
    message = result["choices"][0]["message"]
    answer = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )
    return {
        "answer": answer,
        "sources": [
            {"citation": hit.citation, "commit": hit.commit, "score": hit.score}
            for hit in used
        ],
    }


def ask_fast(question: str, project: str | None = None, limit: int = 10) -> dict:
    """Legacy bounded retrieval for quick orientation and debugging."""
    if not container_running(LLM_CONTAINER):
        raise RuntimeError(
            "Nemotron is not running; start it with: project-brain model nemotron"
        )
    lock_path = DB_PATH.parent / "nemotron-query.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return answer_from_hits(question, search(question, project, limit))


def load_gemini_api_key() -> str:
    """Read Gemini credentials without evaluating the secret file as shell code."""
    environment_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        if GEMINI_SECRET_FILE.stat().st_mode & 0o077:
            raise RuntimeError(
                f"Gemini secret must not be group/world readable: {GEMINI_SECRET_FILE}",
            )
        lines = GEMINI_SECRET_FILE.read_text("utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Gemini secret not found: {GEMINI_SECRET_FILE}",
        ) from exc
    for line in lines:
        if line.startswith("GEMINI_API_KEY="):
            key = line.partition("=")[2].strip()
            if key:
                return key
    raise RuntimeError(f"GEMINI_API_KEY is missing from {GEMINI_SECRET_FILE}")


def gemini_configured() -> bool:
    try:
        load_gemini_api_key()
    except RuntimeError:
        return False
    return True


def _run_exhaustive(question: str, project: str | None, provider: str) -> dict:
    if not project:
        raise ValueError("An exhaustive query requires one registered project")
    require_registered_project(project)
    if provider == "nemotron":
        if not container_running(LLM_CONTAINER):
            raise RuntimeError(
                "Nemotron is not running; start it with: "
                "project-brain model nemotron",
            )
        gateway = NemotronGateway(LLM_URL, LLM_MODEL)
        context_window = 65_536
    elif provider == "gemini":
        gateway = GeminiGateway(
            load_gemini_api_key(), GEMINI_MODEL, GEMINI_API_URL,
        )
        context_window = GEMINI_CONTEXT_WINDOW
    else:
        raise ValueError(f"Unknown exhaustive-query provider: {provider}")

    lock_path = DB_PATH.parent / f"{provider}-query.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runner = ExhaustiveLoop(
            connect(), REPOS[project], gateway,
            cache_dir=DB_PATH.parent / "query-cache" / provider,
            context_window=context_window,
            progress=lambda message: print(
                f"[project-brain {provider} exhaustive] {message}",
                file=sys.stderr, flush=True,
            ),
        )
        result = runner.run(question, project)
    if not result.get("answer"):
        result["answer"] = (
            "The snapshot evidence was insufficient for a confident answer."
        )
    replacements = {
        source["evidence_id"]: f"[{source['citation']}]"
        for source in result.get("sources", [])
    }
    for evidence_id, citation in replacements.items():
        result["answer"] = result["answer"].replace(evidence_id, citation)
    return result


def ask(question: str, project: str | None = None, limit: int = 10) -> dict:
    """Run the exhaustive graph-guided evidence loop with local Nemotron."""
    return _run_exhaustive(question, project, "nemotron")


def ask_gemini(question: str, project: str | None = None, limit: int = 10) -> dict:
    """Run the same exhaustive evidence protocol with Gemini."""
    return _run_exhaustive(question, project, "gemini")


def container_running(name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            text=True, capture_output=True, check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def wait_service(url: str, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"The service did not become ready within {timeout}s: {last_error}")


def model_mode(mode: str, wait: bool = True) -> dict:
    if mode not in {"embeddings", "nemotron", "off", "status"}:
        raise ValueError(f"Unknown model mode: {mode}")
    if mode == "status":
        return {
            "encoder": "on_demand",
            "code_reranker": "cached" if code_reranker_cached() else "not_cached",
            "nemotron": container_running(LLM_CONTAINER),
        }
    if mode == "embeddings":
        return {"mode": mode, **model_mode("status")}
    elif mode == "nemotron":
        subprocess.run(["docker", "start", LLM_CONTAINER], check=True, capture_output=True)
        if wait:
            wait_service(LLM_URL.rstrip("/") + "/models")
    elif mode == "off":
        try:
            subprocess.run(
                ["docker", "stop", LLM_CONTAINER], check=False, capture_output=True,
            )
        except OSError:
            pass
    return {"mode": mode, **model_mode("status")}


def ask_max(question: str, project: str | None = None, limit: int = 10) -> dict:
    """Compatibility alias for the exhaustive graph-guided query."""
    return ask(question, project, limit)


def status() -> dict:
    db = connect()
    indexed = {row["project"]: dict(row) for row in db.execute("SELECT * FROM repos")}
    projects = []
    for name, repo in REPOS.items():
        row = indexed.get(name)
        count = db.execute("SELECT count(*) FROM chunks WHERE project=?", (name,)).fetchone()[0]
        embedded = db.execute(
            "SELECT count(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id WHERE c.project=? AND e.model=?",
            (name, EMBED_MODEL),
        ).fetchone()[0]
        symbols = db.execute("SELECT count(*) FROM symbols WHERE project=?", (name,)).fetchone()[0]
        edges = db.execute("SELECT count(*) FROM edges WHERE project=?", (name,)).fetchone()[0]
        try:
            current = git(repo, "rev-parse", "HEAD")
            digest = repository_content_digest(repo)
            dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=all"))
            repository_error = None
        except (OSError, subprocess.CalledProcessError) as exc:
            current = ""
            digest = ""
            dirty = False
            repository_error = str(exc)
        structural_current = bool(
            not repository_error and row and row["commit_hash"] == current
            and row["content_digest"] == digest
            and row["extractor_version"] == extractor_fingerprint()
        )
        projects.append({
            "project": name, "root": str(repo), "description": REPOS.note(name),
            "commit": current[:12] or None, "chunks": count, "embedded": embedded,
            "symbols": symbols, "edges": edges,
            "current": structural_current,
            "structural_current": structural_current,
            "working_tree_dirty": dirty,
            "repository_error": repository_error,
            **semantic_status(db, name),
        })
    return {
        "database": str(DB_PATH),
        "embedding_model": EMBED_MODEL,
        "code_reranker_model": CODE_EMBED_MODEL,
        "code_reranker_backend": code_reranker_backend(),
        "code_reranker_cached": code_reranker_cached(),
        "projects": projects,
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok", **status()})
        else:
            self.send_json(404, {"error": "Use GET /health or POST /search, /related, /index"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            route = re.fullmatch(
                r"\/([a-z][a-z0-9_-]{0,63})\/"
                r"(search|query|query-fast|query-max|query-gemini|related)",
                self.path,
            )
            route_project = route.group(1) if route else None
            action = f"/{route.group(2)}" if route else self.path
            requested_project = data.get("project")
            if route_project and requested_project and route_project != requested_project:
                raise ValueError("The route project does not match the request body")
            project = route_project or requested_project
            if project:
                require_registered_project(project)
            if action == "/search":
                hits = search(data["question"], project, int(data.get("limit", 12)))
                self.send_json(200, {"hits": [asdict(hit) | {"citation": hit.citation} for hit in hits]})
            elif action == "/query":
                self.send_json(200, ask(data["question"], project, int(data.get("limit", 10))))
            elif action == "/query-fast":
                self.send_json(200, ask_fast(data["question"], project, int(data.get("limit", 10))))
            elif action == "/query-gemini":
                self.send_json(200, ask_gemini(data["question"], project, int(data.get("limit", 10))))
            elif action == "/query-max":
                self.send_json(200, ask_max(data["question"], project, int(data.get("limit", 10))))
            elif action == "/related":
                if not project:
                    raise ValueError("related requires one registered project")
                self.send_json(200, related(connect(), project, data["name"], int(data.get("limit", 50))))
            elif self.path == "/structure":
                self.send_json(200, {"results": structure_projects(data.get("project"))})
            elif self.path == "/index":
                names = [data["project"]] if data.get("project") else list(REPOS)
                self.send_json(200, {"results": [index_project(name, bool(data.get("force"))) for name in names]})
            else:
                self.send_json(404, {"error": "Unknown route"})
        except (KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[project-brain] {self.address_string()} {fmt % args}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index_cmd = sub.add_parser("index", help="Index all projects or one project")
    index_cmd.add_argument("project", nargs="?")
    index_cmd.add_argument("--force", action="store_true")
    structure_cmd = sub.add_parser("structure", help="Index symbols and dependencies")
    structure_cmd.add_argument("project", nargs="?")
    related_cmd = sub.add_parser("related", help="Inspect a symbol and its relationships")
    related_cmd.add_argument("name")
    related_cmd.add_argument("--project", required=True)
    related_cmd.add_argument("--limit", type=int, default=50)
    embed_cmd = sub.add_parser("embed", help="Generate missing embeddings incrementally")
    embed_cmd.add_argument("project", nargs="?")
    embed_cmd.add_argument("--batch-size", type=int, default=8)
    search_cmd = sub.add_parser("search", help="Find relevant repository excerpts")
    search_cmd.add_argument("question")
    search_cmd.add_argument("--project")
    search_cmd.add_argument("--limit", type=int, default=12)
    ask_cmd = sub.add_parser("ask", help="Experimental grounded answer with Nemotron")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--project")
    ask_cmd.add_argument("--limit", type=int, default=10)
    ask_cmd.add_argument("--json", action="store_true", help="Print the complete result and diagnostics")
    ask_gemini_cmd = sub.add_parser(
        "ask-gemini", help="Experimental exhaustive loop with Gemini",
    )
    ask_gemini_cmd.add_argument("question")
    ask_gemini_cmd.add_argument("--project")
    ask_gemini_cmd.add_argument("--limit", type=int, default=10)
    ask_gemini_cmd.add_argument(
        "--json", action="store_true", help="Print the complete result and diagnostics",
    )
    ask_fast_cmd = sub.add_parser("ask-fast", help="Experimental bounded model answer")
    ask_fast_cmd.add_argument("question")
    ask_fast_cmd.add_argument("--project")
    ask_fast_cmd.add_argument("--limit", type=int, default=10)
    ask_fast_cmd.add_argument("--json", action="store_true", help="Print the complete result")
    ask_max_cmd = sub.add_parser("ask-max", help="Compatibility alias for experimental ask")
    ask_max_cmd.add_argument("question")
    ask_max_cmd.add_argument("--project")
    ask_max_cmd.add_argument("--limit", type=int, default=10)
    ask_max_cmd.add_argument("--json", action="store_true", help="Print the complete result and diagnostics")
    model_cmd = sub.add_parser("model", help="Manage the experimental local generation model")
    model_cmd.add_argument("mode", choices=("embeddings", "nemotron", "off", "status"))
    serve_cmd = sub.add_parser("serve", help="Start the optional local HTTP API")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)
    sub.add_parser("status", help="Show index freshness and registered projects")
    project_cmd = sub.add_parser("project", help="Register and manage repositories")
    project_sub = project_cmd.add_subparsers(dest="project_command", required=True)
    project_add = project_sub.add_parser("add", help="Authorize one local Git checkout")
    project_add.add_argument("name")
    project_add.add_argument("path")
    project_add.add_argument("--description", default="")
    project_sub.add_parser("list", help="List registered repositories")
    project_remove = project_sub.add_parser(
        "remove", help="Remove a local index and registration, never the checkout",
    )
    project_remove.add_argument("name")
    args = parser.parse_args()

    if args.command == "index":
        names = [args.project] if args.project else list(REPOS)
        print(json.dumps([index_project(name, args.force) for name in names], ensure_ascii=False, indent=2))
    elif args.command == "structure":
        print(json.dumps(structure_projects(args.project), ensure_ascii=False, indent=2))
    elif args.command == "related":
        require_registered_project(args.project)
        print(json.dumps(related(connect(), args.project, args.name, args.limit), ensure_ascii=False, indent=2))
    elif args.command == "embed":
        print(json.dumps(embed_projects(args.project, args.batch_size), ensure_ascii=False, indent=2))
    elif args.command == "search":
        print(json.dumps([asdict(hit) | {"citation": hit.citation} for hit in search(args.question, args.project, args.limit)], ensure_ascii=False, indent=2))
    elif args.command in {"ask", "ask-gemini", "ask-max", "ask-fast"}:
        runners = {
            "ask": ask,
            "ask-gemini": ask_gemini,
            "ask-max": ask_max,
            "ask-fast": ask_fast,
        }
        runner = runners[args.command]
        result = runner(args.question, args.project, args.limit)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        print(result["answer"])
        diagnostics = result.get("diagnostics")
        if diagnostics:
            print(
                "\nCoverage: "
                f"{diagnostics.get('covered_edges', 0)}/{diagnostics.get('total_edges', 0)} "
                f"relationships; {diagnostics.get('atlas_batches', 0)} batches; "
                f"{diagnostics.get('rounds', 0)} rounds; "
                f"{diagnostics.get('opened_evidence', 0)} evidence items opened; "
                f"stop={diagnostics.get('stop_reason', 'n/a')}"
            )
            print(f"Snapshot: {diagnostics.get('snapshot_digest', 'n/a')}")
            print(f"Contradictions: {len(diagnostics.get('contradictions', []))}")
        print("\nSources used:")
        for source in result["sources"]:
            print(f"- [{source['citation']}] commit={source.get('commit', 'n/a')}")
    elif args.command == "model":
        print(json.dumps(model_mode(args.mode), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "project":
        if args.project_command == "add":
            result = register_project(args.name, args.path, args.description)
        elif args.project_command == "list":
            result = [
                {"project": name, **entry}
                for name, entry in REPOS.entries().items()
            ]
        else:
            result = unregister_project(args.name)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"Project Brain listening on http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nProject Brain stopped")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
