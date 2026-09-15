"""Exact local vector storage and retrieval for Project Brain."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import warnings
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EMBED_MODEL = os.environ.get(
    "PROJECT_BRAIN_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CODE_EMBED_MODEL = os.environ.get(
    "PROJECT_BRAIN_CODE_EMBED_MODEL",
    "jinaai/jina-embeddings-v2-base-code",
)
CODE_EMBED_URL = os.environ.get("PROJECT_BRAIN_CODE_EMBED_URL", "").strip()
CODE_EMBED_API_KEY = os.environ.get("PROJECT_BRAIN_CODE_EMBED_API_KEY", "").strip()
CODE_EMBED_TIMEOUT = max(
    1, int(os.environ.get("PROJECT_BRAIN_CODE_EMBED_TIMEOUT", "300")),
)
CODE_QUERY_INSTRUCTION = os.environ.get(
    "PROJECT_BRAIN_CODE_QUERY_INSTRUCTION",
    "Instruct: Given a software engineering question, retrieve code, tests, schema, "
    "and configuration that causally explain or implement it.\nQuery: ",
)
MODEL_CACHE = Path(os.environ.get(
    "PROJECT_BRAIN_MODEL_CACHE",
    Path.home() / ".cache" / "project-brain" / "models",
))
QUERY_INSTRUCTION = ""
_MODEL = None
_CODE_MODEL = None


def ensure_schema(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector BLOB NOT NULL,
            embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS embeddings_model_idx ON embeddings(model);
        CREATE TABLE IF NOT EXISTS query_embeddings (
            question_hash TEXT NOT NULL, model TEXT NOT NULL, dimension INTEGER NOT NULL,
            vector BLOB NOT NULL, embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(question_hash,model)
        );
        CREATE TABLE IF NOT EXISTS content_embeddings (
            content_hash TEXT NOT NULL, model TEXT NOT NULL, dimension INTEGER NOT NULL,
            vector BLOB NOT NULL, embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(content_hash,model)
        );
        CREATE INDEX IF NOT EXISTS content_embeddings_model_idx
          ON content_embeddings(model);
        """
    )


def _embed(inputs: list[str]) -> np.ndarray:
    global _MODEL
    import numpy as np
    from fastembed import TextEmbedding

    if _MODEL is None:
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The model .* now uses mean pooling instead of CLS embedding.*",
                category=UserWarning,
            )
            _MODEL = TextEmbedding(
                model_name=EMBED_MODEL, cache_dir=str(MODEL_CACHE), threads=4,
            )
    vectors = np.asarray(list(_MODEL.embed(inputs, batch_size=32)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _embed_code(inputs: list[str]) -> np.ndarray:
    """Embed a small reranking set with a code-aware model on demand."""
    global _CODE_MODEL
    import numpy as np

    if CODE_EMBED_URL:
        return _embed_code_remote(inputs)
    from fastembed import TextEmbedding
    if _CODE_MODEL is None:
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        _CODE_MODEL = TextEmbedding(
            model_name=CODE_EMBED_MODEL,
            cache_dir=str(MODEL_CACHE),
            threads=max(1, int(os.environ.get("PROJECT_BRAIN_CODE_THREADS", "8"))),
        )
    batch_size = max(1, int(os.environ.get("PROJECT_BRAIN_CODE_BATCH_SIZE", "4")))
    vectors = np.asarray(list(_CODE_MODEL.embed(inputs, batch_size=batch_size)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _embed_code_remote(inputs: list[str]) -> np.ndarray:
    """Use an isolated OpenAI-compatible embedding service, normally on GPU."""
    import numpy as np

    headers = {"Content-Type": "application/json"}
    if CODE_EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {CODE_EMBED_API_KEY}"
    request = Request(
        CODE_EMBED_URL,
        data=json.dumps({
            "model": CODE_EMBED_MODEL,
            "input": inputs,
            "encoding_format": "float",
        }).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=CODE_EMBED_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"code embedding service failed at {CODE_EMBED_URL}: {exc}"
        ) from exc
    rows = sorted(payload.get("data", []), key=lambda item: int(item.get("index", 0)))
    if len(rows) != len(inputs):
        raise RuntimeError(
            f"code embedding service returned {len(rows)} vectors for {len(inputs)} inputs"
        )
    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def code_reranker_backend() -> str:
    return "remote" if CODE_EMBED_URL else "cpu"


def code_reranker_cached() -> bool:
    """Return whether automatic deep retrieval can start without a model download."""
    if os.environ.get("PROJECT_BRAIN_CODE_RERANK", "auto").casefold() == "off":
        return False
    if CODE_EMBED_URL:
        return True
    model_dir = MODEL_CACHE / f"models--{CODE_EMBED_MODEL.replace('/', '--')}"
    return model_dir.is_dir()


def _document(project: str, path: str, content: str) -> str:
    header = f"Project: {project}\nFile: {path}\n"
    if len(content) > 16_000:
        content = content[:16_000] + "\n[TRUNCATED LONG GENERATED OR MINIFIED CONTENT]"
    return header + content


def index_embeddings(
    db: sqlite3.Connection,
    project: str | None = None,
    batch_size: int = 8,
    progress: callable | None = None,
) -> dict:
    ensure_schema(db)
    where = "AND c.project=?" if project else ""
    params: list[object] = [EMBED_MODEL]
    if project:
        params.append(project)
    rows = db.execute(
        f"""
        SELECT c.id,c.project,c.path,c.content
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.model=?
        WHERE e.chunk_id IS NULL {where}
        ORDER BY c.project,c.path,c.start_line
        """,
        params,
    ).fetchall()
    completed = 0
    for offset in range(0, len(rows), max(1, batch_size)):
        batch = rows[offset : offset + max(1, batch_size)]
        vectors = _embed([_document(row["project"], row["path"], row["content"]) for row in batch])
        with db:
            for row, vector in zip(batch, vectors):
                db.execute(
                    """
                    INSERT INTO embeddings(chunk_id,model,dimension,vector)
                    VALUES(?,?,?,?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                      model=excluded.model, dimension=excluded.dimension,
                      vector=excluded.vector, embedded_at=CURRENT_TIMESTAMP
                    """,
                    (row["id"], EMBED_MODEL, int(vector.shape[0]), vector.astype("<f4").tobytes()),
                )
        completed += len(batch)
        if progress:
            progress(completed, len(rows))
    total = db.execute(
        "SELECT count(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id "
        "WHERE e.model=?" + (" AND c.project=?" if project else ""),
        ([EMBED_MODEL, project] if project else [EMBED_MODEL]),
    ).fetchone()[0]
    return {"project": project or "all", "embedded": completed, "total": total, "model": EMBED_MODEL}


def query_vector(db: sqlite3.Connection, question: str) -> np.ndarray:
    import numpy as np

    prepared = QUERY_INSTRUCTION + question
    question_hash = hashlib.sha256(prepared.encode("utf-8")).hexdigest()
    row = db.execute(
        "SELECT vector FROM query_embeddings WHERE question_hash=? AND model=?",
        (question_hash, EMBED_MODEL),
    ).fetchone()
    if row:
        return np.frombuffer(row["vector"], dtype="<f4")
    vector = _embed([prepared])[0]
    with db:
        db.execute(
            "INSERT OR REPLACE INTO query_embeddings(question_hash,model,dimension,vector) VALUES(?,?,?,?)",
            (question_hash, EMBED_MODEL, int(vector.shape[0]), vector.astype("<f4").tobytes()),
        )
    return vector


def code_rerank_scores(
    db: sqlite3.Connection, question: str, documents: list[str],
) -> list[float]:
    """Score a bounded candidate set and persist vectors by immutable content hash."""
    import numpy as np

    ensure_schema(db)
    prepared_question = question.strip()
    if CODE_EMBED_URL:
        prepared_question = CODE_QUERY_INSTRUCTION + prepared_question
    question_hash = hashlib.sha256(prepared_question.encode("utf-8")).hexdigest()
    query_row = db.execute(
        "SELECT vector FROM query_embeddings WHERE question_hash=? AND model=?",
        (question_hash, CODE_EMBED_MODEL),
    ).fetchone()

    prepared_documents = [document[:32_000] for document in documents]
    content_hashes = [
        hashlib.sha256(document.encode("utf-8")).hexdigest()
        for document in prepared_documents
    ]
    cached: dict[str, np.ndarray] = {}
    if content_hashes:
        placeholders = ",".join("?" for _ in content_hashes)
        rows = db.execute(
            f"SELECT content_hash,vector FROM content_embeddings "
            f"WHERE model=? AND content_hash IN ({placeholders})",
            [CODE_EMBED_MODEL, *content_hashes],
        ).fetchall()
        cached = {
            row["content_hash"]: np.frombuffer(row["vector"], dtype="<f4")
            for row in rows
        }

    missing_hashes = []
    missing_documents = []
    for content_hash, document in zip(content_hashes, prepared_documents):
        if content_hash not in cached and content_hash not in missing_hashes:
            missing_hashes.append(content_hash)
            missing_documents.append(document)

    inputs = ([] if query_row else [prepared_question]) + missing_documents
    embedded = _embed_code(inputs) if inputs else np.empty((0, 0), dtype=np.float32)
    cursor = 0
    if query_row:
        query = np.frombuffer(query_row["vector"], dtype="<f4")
    else:
        query = embedded[cursor]
        cursor += 1
        with db:
            db.execute(
                "INSERT OR REPLACE INTO query_embeddings"
                "(question_hash,model,dimension,vector) VALUES(?,?,?,?)",
                (
                    question_hash, CODE_EMBED_MODEL, int(query.shape[0]),
                    query.astype("<f4").tobytes(),
                ),
            )

    with db:
        for content_hash in missing_hashes:
            vector = embedded[cursor]
            cursor += 1
            cached[content_hash] = vector
            db.execute(
                "INSERT OR REPLACE INTO content_embeddings"
                "(content_hash,model,dimension,vector) VALUES(?,?,?,?)",
                (
                    content_hash, CODE_EMBED_MODEL, int(vector.shape[0]),
                    vector.astype("<f4").tobytes(),
                ),
            )
    return [float(cached[content_hash] @ query) for content_hash in content_hashes]


def semantic_rows(
    db: sqlite3.Connection,
    question: str,
    project: str | None = None,
    limit: int = 30,
) -> list[dict]:
    import numpy as np

    ensure_schema(db)
    query = query_vector(db, question)
    where = "AND c.project=?" if project else ""
    params: list[object] = [EMBED_MODEL]
    if project:
        params.append(project)
    rows = db.execute(
        f"""
        SELECT c.project,c.path,c.start_line,c.end_line,c.commit_hash,c.content,
               e.dimension,e.vector
        FROM embeddings e
        JOIN chunks c ON c.id=e.chunk_id
        JOIN source_authority a ON a.project=c.project AND a.path=c.path
        WHERE e.model=? AND a.authority IN ('code','test','schema_history','generated','config') {where}
        """,
        params,
    ).fetchall()
    if not rows:
        return []
    matrix = np.vstack([np.frombuffer(row["vector"], dtype="<f4") for row in rows])
    scores = matrix @ query
    take = min(max(1, limit), len(rows))
    indices = np.argpartition(scores, -take)[-take:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return [
        {
            "project": rows[i]["project"], "path": rows[i]["path"],
            "start_line": rows[i]["start_line"], "end_line": rows[i]["end_line"],
            "commit_hash": rows[i]["commit_hash"], "content": rows[i]["content"],
            "semantic_score": float(scores[i]),
        }
        for i in indices
    ]
