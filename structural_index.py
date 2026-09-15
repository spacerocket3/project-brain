"""Structural symbols, dependencies, and source authority for Project Brain."""

from __future__ import annotations

import json
import hashlib
import difflib
import re
import sqlite3
import subprocess
from pathlib import Path

TS_EXTENSIONS = {".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"}
SQL_EXTENSIONS = {".sql"}
EXTRACTOR_VERSION = "ts-compiler-sql-v3"
SQL_DEFINITION_PATTERNS = [
    ("function", re.compile(r"(?is)\bcreate\s+(?:or\s+replace\s+)?function\s+([\w.]+)\s*\(")),
    ("table", re.compile(r"(?is)\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?([\w.]+)")),
    ("view", re.compile(r"(?is)\bcreate\s+(?:or\s+replace\s+)?(?:materialized\s+)?view\s+([\w.]+)")),
    ("policy", re.compile(r'(?is)\bcreate\s+policy\s+(?:"([^"]+)"|([\w.]+))\s+on\s+([\w.]+)')),
    ("trigger", re.compile(r"(?is)\bcreate\s+(?:or\s+replace\s+)?trigger\s+([\w.]+)")),
    ("type", re.compile(r"(?is)\bcreate\s+type\s+([\w.]+)")),
]
SQL_CALL = re.compile(r"\b(public\.)?([a-zA-Z_][\w]*)\s*\(")
SQL_TABLE_REF = re.compile(r"(?is)\b(from|join|update|into|delete\s+from)\s+([\w.]+)")
RPC_CALL = re.compile(r"\.rpc\(\s*['\"]([^'\"]+)['\"]")
TABLE_CALL = re.compile(r"\.from\(\s*['\"]([^'\"]+)['\"]")
FUNCTION_INVOKE = re.compile(r"\.functions\.invoke\(\s*['\"]([^'\"]+)['\"]")
SQL_CALL_STOPWORDS = {
    "and", "any", "array", "as", "avg", "bool_or", "case", "check",
    "coalesce", "conflict", "count", "date_trunc", "exists", "extract",
    "filter", "from", "greatest", "if", "in", "json_agg", "json_build_object",
    "jsonb_agg", "jsonb_build_object", "key", "lateral", "least", "make_interval",
    "max", "min", "nullif", "numeric", "or", "over", "return", "round",
    "select", "sum", "table", "unique", "using", "values", "when", "where",
}
SQL_TABLE_STOPWORDS = {
    "anon", "lateral", "of", "on", "or", "public", "set", "to", "using",
}


def extractor_fingerprint() -> str:
    state = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("ts_structure.mjs")):
        raw = path.read_bytes()
        state.update(path.name.encode())
        state.update(len(raw).to_bytes(8, "big"))
        state.update(raw)
    return f"{EXTRACTOR_VERSION}:{state.hexdigest()[:20]}"


def ensure_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_authority (
            project TEXT NOT NULL, path TEXT NOT NULL, authority TEXT NOT NULL,
            reason TEXT NOT NULL, commit_hash TEXT NOT NULL,
            PRIMARY KEY(project,path)
        );
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY, project TEXT NOT NULL, path TEXT NOT NULL,
            kind TEXT NOT NULL, name TEXT NOT NULL, qualified_name TEXT NOT NULL,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            signature TEXT NOT NULL DEFAULT '', active INTEGER NOT NULL DEFAULT 1,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS symbols_lookup_idx
          ON symbols(project,name,kind,active);
        CREATE INDEX IF NOT EXISTS symbols_path_idx ON symbols(project,path);
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY, project TEXT NOT NULL, source_path TEXT NOT NULL,
            source_name TEXT NOT NULL DEFAULT '', relation TEXT NOT NULL,
            target_name TEXT NOT NULL, target_path TEXT, line INTEGER NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS edges_target_idx
          ON edges(project,target_name,relation);
        CREATE INDEX IF NOT EXISTS edges_source_idx
          ON edges(project,source_path,relation);
        CREATE TABLE IF NOT EXISTS structure_snapshots (
            project TEXT PRIMARY KEY,
            commit_hash TEXT NOT NULL,
            graph_digest TEXT NOT NULL,
            content_digest TEXT NOT NULL DEFAULT '',
            extractor_version TEXT NOT NULL DEFAULT '',
            generation_id TEXT NOT NULL DEFAULT '',
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    symbol_columns = {row[1] for row in db.execute("PRAGMA table_info(symbols)")}
    if "commit_hash" not in symbol_columns:
        db.execute("ALTER TABLE symbols ADD COLUMN commit_hash TEXT NOT NULL DEFAULT ''")
    if "generation_id" not in symbol_columns:
        db.execute("ALTER TABLE symbols ADD COLUMN generation_id TEXT NOT NULL DEFAULT ''")
    edge_columns = {row[1] for row in db.execute("PRAGMA table_info(edges)")}
    for name, declaration in (
        ("source_qualified_name", "TEXT NOT NULL DEFAULT ''"),
        ("target_qualified_name", "TEXT"),
        ("resolution_confidence", "TEXT NOT NULL DEFAULT 'unresolved'"),
        ("commit_hash", "TEXT NOT NULL DEFAULT ''"),
        ("generation_id", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in edge_columns:
            db.execute(f"ALTER TABLE edges ADD COLUMN {name} {declaration}")
    authority_columns = {row[1] for row in db.execute("PRAGMA table_info(source_authority)")}
    if "generation_id" not in authority_columns:
        db.execute("ALTER TABLE source_authority ADD COLUMN generation_id TEXT NOT NULL DEFAULT ''")
    snapshot_columns = {row[1] for row in db.execute("PRAGMA table_info(structure_snapshots)")}
    for name in ("content_digest", "extractor_version", "generation_id"):
        if name not in snapshot_columns:
            db.execute(
                f"ALTER TABLE structure_snapshots ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
            )


def classify_source(path: str) -> tuple[str, str]:
    lower = path.lower()
    parts = tuple(part for part in lower.split("/") if part)
    name = parts[-1] if parts else lower
    if "/history/" in lower or lower.startswith("docs/history/"):
        return "historical", "File inside historical documentation"
    if lower.endswith("src/integrations/supabase/types.ts"):
        return "generated", "Types generated from the Supabase schema"
    if lower.startswith("migrations/") or "/migrations/" in lower:
        return "schema_history", "SQL migration; only the latest same-name definition is active"
    if lower.endswith("agents.md"):
        return "canonical", "Explicit repository instructions"
    if lower.startswith("docs/architecture/") or lower.startswith("docs/reference/"):
        return "canonical", "Architecture or reference document"
    if lower.startswith("docs/"):
        return "active", "Active project documentation"
    if (
        name in {
            "package.json", "pyproject.toml", "cargo.toml", "go.mod",
            "pom.xml", "build.gradle", "build.gradle.kts", "makefile",
        }
        or name.startswith(("tsconfig", "jest.config", "vitest.config", "vite.config"))
    ):
        return "config", "Executable configuration or project manifest"
    if name in {"readme.md", "contributing.md", "code_of_conduct.md"}:
        return "active", "Active documentation at the repository root"
    if name in {"changelog.md", "history.md", "changes.md"}:
        return "historical", "Repository change history"
    if name.endswith((".md", ".rst")) or (parts and parts[0] == ".github"):
        return "active", "Documentation or collaboration metadata"
    if (
        any(part in {"test", "tests", "spec", "specs", "__tests__"} for part in parts[:-1])
        or any(marker in name for marker in (".test.", ".spec.", "_test.", "_spec."))
    ):
        return "test", "Executable test that constrains behavior"
    return "code", "Current source code"



def add_symbol(
    db, project, path, kind, name, start, end, signature="", metadata=None,
    qualified_name=None, commit_hash="", generation_id="",
):
    if not name:
        return
    db.execute(
        """INSERT INTO symbols(
               project,path,kind,name,qualified_name,start_line,end_line,
               signature,metadata,commit_hash,generation_id
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (project, path, kind, name, qualified_name or name, start, end,
         signature[:500], json.dumps(metadata or {}, ensure_ascii=False), commit_hash,
         generation_id),
    )


def add_edge(
    db, project, path, source, relation, target, line, target_path=None,
    metadata=None, source_qualified_name="", target_qualified_name=None,
    confidence="unresolved", commit_hash="", generation_id="",
):
    if not target:
        return
    db.execute(
        """INSERT INTO edges(
               project,source_path,source_name,relation,target_name,target_path,
               line,metadata,source_qualified_name,target_qualified_name,
               resolution_confidence,commit_hash,generation_id
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (project, path, source, relation, target, target_path, line,
         json.dumps(metadata or {}, ensure_ascii=False),
         source_qualified_name or source, target_qualified_name, confidence, commit_hash,
         generation_id),
    )


def extract_ts_compiler(
    db: sqlite3.Connection, project: str, repo: Path, commit: str,
    generation_id: str = "",
) -> tuple[int, dict]:
    """Extract TS/JS structure in one stable pass using the TypeScript compiler."""
    script = Path(__file__).with_name("ts_structure.mjs")
    result = subprocess.run(
        ["node", str(script), str(repo), project],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    for symbol in payload["symbols"]:
        add_symbol(
            db, project, symbol["path"], symbol["kind"], symbol["name"],
            symbol["start_line"], symbol["end_line"], symbol.get("signature", ""),
            symbol.get("metadata"), symbol.get("qualified_name"), commit, generation_id,
        )
    for edge in payload["edges"]:
        add_edge(
            db, project, edge["source_path"], edge.get("source_name", ""),
            edge["relation"], edge["target_name"], edge["line"],
            edge.get("target_path"), edge.get("metadata"),
            edge.get("source_qualified_name", ""),
            edge.get("target_qualified_name"),
            edge.get("resolution_confidence", "unresolved"), commit, generation_id,
        )
    touched = {item["path"] for item in payload["symbols"]}
    touched.update(item["source_path"] for item in payload["edges"])
    return len(touched), payload.get("coverage", {})



def _sql_definition_end(text: str, start: int, kind: str, fallback: int) -> int:
    """Find a SQL definition terminator without stopping at PL/pgSQL semicolons."""
    if kind == "function":
        opener = re.search(r"\$[A-Za-z_0-9]*\$", text[fallback:])
        if opener:
            delimiter = opener.group(0)
            body_start = fallback + opener.end()
            body_end = text.find(delimiter, body_start)
            if body_end >= 0:
                terminator = text.find(";", body_end + len(delimiter))
                return terminator + 1 if terminator >= 0 else body_end + len(delimiter)
    terminator = text.find(";", start)
    return terminator + 1 if terminator >= 0 else fallback


def extract_sql(
    db, project: str, path: Path, text: str, commit: str = "",
    generation_id: str = "",
) -> None:
    path_text = str(path)
    definitions: list[tuple[int, int, str]] = []
    for kind, pattern in SQL_DEFINITION_PATTERNS:
        for match in pattern.finditer(text):
            if kind == "policy":
                name = match.group(1) or match.group(2)
                metadata = {"table": match.group(3)}
            else:
                name = match.group(1)
                metadata = {}
            line = text.count("\n", 0, match.start()) + 1
            end = _sql_definition_end(text, match.start(), kind, match.end())
            end_line = text.count("\n", 0, end) + 1
            add_symbol(db, project, path_text, kind, name, line, end_line,
                       text[match.start():match.end()], metadata,
                       qualified_name=name, commit_hash=commit,
                       generation_id=generation_id)
            definitions.append((match.start(), end, name))
    definitions.sort()

    def owner(offset: int) -> str:
        result = ""
        for start, end, name in definitions:
            if start > offset:
                break
            result = name if offset <= end else ""
        return result

    for match in SQL_CALL.finditer(text):
        name = ("public." if match.group(1) else "") + match.group(2)
        if match.group(2).lower() in SQL_CALL_STOPWORDS:
            continue
        prefix = text[max(0, match.start() - 40):match.start()].lower()
        if "function" in prefix:
            continue
        add_edge(db, project, path_text, owner(match.start()), "calls_sql", name,
                 text.count("\n", 0, match.start()) + 1,
                 target_qualified_name=name, confidence="lexical", commit_hash=commit,
                 generation_id=generation_id)
    for match in SQL_TABLE_REF.finditer(text):
        prefix = text[max(0, match.start() - 30):match.start()].lower()
        if match.group(1).lower() == "from" and "extract" in prefix:
            continue
        if match.group(2).lower() in SQL_TABLE_STOPWORDS:
            continue
        add_edge(db, project, path_text, owner(match.start()), "accesses_table", match.group(2),
                 text.count("\n", 0, match.start()) + 1,
                 metadata={"operation": match.group(1).lower()},
                 target_qualified_name=match.group(2), confidence="lexical",
                 commit_hash=commit, generation_id=generation_id)


def graph_digest(db: sqlite3.Connection, project: str) -> str:
    edges = [dict(row) for row in db.execute(
        """SELECT source_path,source_name,source_qualified_name,relation,target_name,
                  target_path,target_qualified_name,line,resolution_confidence,metadata
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
    payload = json.dumps(
        {"edges": edges, "symbols": symbols},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def mark_latest_sql_definitions(db: sqlite3.Connection, project: str) -> None:
    rows = db.execute(
        """SELECT id,name,path FROM symbols
           WHERE project=? AND kind IN ('function','view','policy','trigger','type')
             AND path LIKE 'supabase/migrations/%'
           ORDER BY path DESC,id DESC""",
        (project,),
    ).fetchall()
    seen = set()
    for row in rows:
        key = row["name"].lower()
        active = 0 if key in seen else 1
        db.execute("UPDATE symbols SET active=? WHERE id=?", (active, row["id"]))
        seen.add(key)


def resolve_sql_targets(db: sqlite3.Connection, project: str) -> None:
    """Link RPC/SQL calls to the current active SQL definition when unambiguous."""
    active: dict[str, list[sqlite3.Row]] = {}
    for row in db.execute(
        """SELECT path,name,qualified_name FROM symbols
           WHERE project=? AND kind='function' AND active=1""", (project,),
    ):
        for key in {row["name"].casefold(), row["qualified_name"].casefold(),
                    row["qualified_name"].split(".")[-1].casefold()}:
            active.setdefault(key, []).append(row)
    for edge in db.execute(
        """SELECT id,target_name FROM edges WHERE project=?
           AND relation IN ('calls_rpc','calls_sql') AND target_path IS NULL""", (project,),
    ).fetchall():
        candidates = {row["path"]: row for row in active.get(edge["target_name"].casefold(), [])}
        if len(candidates) == 1:
            row = next(iter(candidates.values()))
            db.execute(
                "UPDATE edges SET target_path=?,target_qualified_name=?,"
                "resolution_confidence='active_sql' WHERE id=?",
                (row["path"], row["qualified_name"], edge["id"]),
            )


def index_structure(
    db: sqlite3.Connection, project: str, repo: Path, commit: str,
    *, content_digest: str = "", generation_id: str = "", manage_transaction: bool = True,
) -> dict:
    if manage_transaction:
        ensure_schema(db)
    files = 0
    result: tuple[dict, str] | None = None

    def publish() -> None:
        nonlocal files, result
        db.execute("DELETE FROM source_authority WHERE project=?", (project,))
        db.execute("DELETE FROM symbols WHERE project=?", (project,))
        db.execute("DELETE FROM edges WHERE project=?", (project,))
        tracked = subprocess_files(repo)
        files, ts_coverage = extract_ts_compiler(
            db, project, repo, commit, generation_id,
        )
        for path in tracked:
            authority, reason = classify_source(str(path))
            db.execute(
                "INSERT INTO source_authority(project,path,authority,reason,commit_hash,generation_id) VALUES(?,?,?,?,?,?)",
                (project, str(path), authority, reason, commit, generation_id),
            )
            absolute = repo / path
            if path.suffix in SQL_EXTENSIONS:
                extract_sql(
                    db, project, path,
                    absolute.read_text("utf-8", errors="ignore"), commit, generation_id,
                )
                files += 1
        mark_latest_sql_definitions(db, project)
        resolve_sql_targets(db, project)
        digest = graph_digest(db, project)
        db.execute(
            """INSERT INTO structure_snapshots(
                   project,commit_hash,graph_digest,content_digest,
                   extractor_version,generation_id
               ) VALUES(?,?,?,?,?,?)
               ON CONFLICT(project) DO UPDATE SET
                 commit_hash=excluded.commit_hash,
                 graph_digest=excluded.graph_digest,
                 content_digest=excluded.content_digest,
                 extractor_version=excluded.extractor_version,
                 generation_id=excluded.generation_id,
                 indexed_at=CURRENT_TIMESTAMP""",
            (project, commit, digest, content_digest, extractor_fingerprint(), generation_id),
        )
        result = (ts_coverage, digest)

    if manage_transaction:
        with db:
            publish()
    else:
        publish()
    assert result is not None
    ts_coverage, digest = result
    symbols = db.execute("SELECT count(*) FROM symbols WHERE project=?", (project,)).fetchone()[0]
    edges = db.execute("SELECT count(*) FROM edges WHERE project=?", (project,)).fetchone()[0]
    unresolved = db.execute(
        "SELECT count(*) FROM edges WHERE project=? AND resolution_confidence='unresolved'",
        (project,),
    ).fetchone()[0]
    return {
        "project": project, "files": files, "symbols": symbols, "edges": edges,
        "unresolved_edges": unresolved, "graph_digest": digest,
        "extractor_version": extractor_fingerprint(),
        "extractor_coverage": ts_coverage,
    }


def subprocess_files(repo: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    )
    return [Path(line) for line in output.splitlines() if line and (repo / line).is_file()]


def related(
    db: sqlite3.Connection, project: str, name: str, limit: int = 50,
    path: str | None = None,
) -> dict:
    ensure_schema(db)
    path_sql = " AND s.path=?" if path else ""
    symbol_params: list[object] = [project, f"%{name}%", f"%{name}%"]
    if path:
        symbol_params.append(path)
    symbol_params.extend([name, name, f"{name}%", limit])
    symbols = [dict(row) for row in db.execute(
        """SELECT s.*,a.authority,a.reason FROM symbols s
           LEFT JOIN source_authority a USING(project,path)
           WHERE s.project=? AND (
             lower(s.name) LIKE lower(?) OR lower(s.qualified_name) LIKE lower(?)
           )""" + path_sql + """
           ORDER BY
             CASE
               WHEN lower(s.qualified_name)=lower(?) THEN 0
               WHEN lower(s.name)=lower(?) THEN 1
               WHEN lower(s.qualified_name) LIKE lower(?) THEN 2
               ELSE 3
             END,
             s.active DESC,s.path,s.start_line
           LIMIT ?""",
        symbol_params,
    )]
    suggestions = []
    if not symbols:
        needle = name.casefold()
        candidates = [dict(row) for row in db.execute(
            """SELECT s.*,a.authority,a.reason FROM symbols s
               LEFT JOIN source_authority a USING(project,path)
               WHERE s.project=? AND s.active=1""" +
            (" AND s.path=?" if path else "") +
            " ORDER BY s.path,s.start_line",
            ([project, path] if path else [project]),
        )]
        scored = []
        for item in candidates:
            short = str(item.get("name") or "").casefold()
            qualified = str(item.get("qualified_name") or "").casefold()
            score = max(
                difflib.SequenceMatcher(None, needle, short).ratio(),
                difflib.SequenceMatcher(None, needle, qualified).ratio(),
            )
            if score >= 0.48:
                scored.append((score, item))
        suggestions = [item | {"match_score": round(score, 3)} for score, item in sorted(
            scored, key=lambda pair: (-pair[0], pair[1]["path"], pair[1]["start_line"]),
        )[:5]]
    edge_path_sql = " AND (source_path=? OR target_path=?)" if path else ""
    edge_params: list[object] = [project, f"%{name}%", f"%{name}%"]
    if path:
        edge_params.extend([path, path])
    edge_params.extend([name, name, name, name, limit])
    edges = [dict(row) for row in db.execute(
        """SELECT * FROM edges WHERE project=?
           AND (lower(source_name) LIKE lower(?) OR lower(target_name) LIKE lower(?))
        """ + edge_path_sql + """
           ORDER BY
             CASE
               WHEN lower(source_qualified_name)=lower(?) THEN 0
               WHEN lower(source_name)=lower(?) THEN 1
               WHEN lower(target_qualified_name)=lower(?) THEN 2
               WHEN lower(target_name)=lower(?) THEN 3
               ELSE 4
             END,
             CASE WHEN target_path IS NOT NULL THEN 0 ELSE 1 END,
             relation,source_path,line
           LIMIT ?""",
        edge_params,
    )]
    return {
        "query": name, "path_filter": path, "symbols": symbols,
        "suggestions": suggestions, "edges": edges,
    }


FLOW_RELATIONS = {
    "calls", "imports", "calls_rpc", "invokes_edge_function",
    "reads_table", "writes_table", "accesses_table", "extends", "overrides",
}
RESOURCE_RELATIONS = {"calls_rpc", "reads_table", "writes_table", "accesses_table"}


def diagnose_retrieval(flow: dict) -> dict:
    """Describe the shape of a flow result without pretending presence is completeness."""
    evidence = flow.get("evidence", [])
    reasons = [reason for item in evidence for reason in item.get("reasons", [])]
    confidence_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for reason in reasons:
        confidence = reason.get("confidence", "not_reported")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        kind = reason.get("kind", "unknown")
        reason_counts[kind] = reason_counts.get(kind, 0) + 1
    lane_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    authority_counts: dict[str, int] = {}
    anchor_authority_counts: dict[str, int] = {}
    for item in evidence:
        lane_counts[item["lane"]] = lane_counts.get(item["lane"], 0) + 1
        authority = item.get("authority", "unknown")
        authority_counts[authority] = authority_counts.get(authority, 0) + 1
        for role in item.get("roles", []):
            role_counts[role] = role_counts.get(role, 0) + 1
        if any(reason.get("kind") == "search_anchor" for reason in item.get("reasons", [])):
            anchor_authority_counts[authority] = anchor_authority_counts.get(authority, 0) + 1

    lexical = sum(
        count for confidence, count in confidence_counts.items()
        if "lexical" in confidence
    )
    reported = sum(confidence_counts.values()) - confidence_counts.get("not_reported", 0)
    lexical_ratio = lexical / reported if reported else 0.0
    anchor_files = len(flow.get("seed_paths", []))
    total = len(evidence)
    dominant_lane = max(lane_counts.values(), default=0)
    concentration = dominant_lane / total if total else 0.0
    rpc_map = flow.get("rpc_consumer_map", {})
    consumer_count = len({path for paths in rpc_map.values() for path in paths})
    lineage = flow.get("sql_lineage", {})
    lineage_entries = sum(len(items) for items in lineage.values())
    unresolved_cache = flow.get("potentially_uninvalidated_query_keys", [])
    reverse_dependencies = flow.get("sql_reverse_dependency_map", {})
    facet_coverage = flow.get("facet_coverage", {})

    warnings = []
    strengths = []
    if total == 0:
        warnings.append({"code": "no_evidence", "message": "No evidence was returned."})
    if concentration >= 0.60 and total >= 5:
        warnings.append({
            "code": "lane_concentration", "message": (
                f"{dominant_lane}/{total} results are concentrated in one lane; "
                "category presence may hide missing topology."
            ),
        })
    if lexical_ratio >= 0.50 and reported >= 4:
        warnings.append({
            "code": "lexical_dominance", "message": (
                f"{lexical}/{reported} reported relations are lexical; verify them directly."
            ),
        })
    anchor_count = sum(anchor_authority_counts.values())
    if anchor_count >= 2 and anchor_authority_counts.get("test", 0) == anchor_count:
        warnings.append({
            "code": "test_anchor_dominance", "message": (
                "Every search anchor is a test file; production implementation may be "
                "missing even when graph expansion later reaches generic code dependencies."
            ),
        })
    thin_roles = sorted(role for role in flow.get("roles_present", []) if role_counts.get(role, 0) == 1)
    if thin_roles:
        warnings.append({
            "code": "thin_role_coverage", "roles": thin_roles,
            "message": "These roles are represented by only one evidence item, not exhaustively covered.",
        })
    if flow.get("roles_missing"):
        warnings.append({
            "code": "missing_roles", "roles": flow["roles_missing"],
            "message": "Question-requested roles are missing.",
        })
    if unresolved_cache:
        warnings.append({
            "code": "cache_review_required", "keys": unresolved_cache,
            "message": "Consumer query keys were not found in the scoped invalidation set.",
        })
    if lineage and not all(lineage.values()):
        warnings.append({
            "code": "incomplete_sql_lineage",
            "symbols": sorted(name for name, items in lineage.items() if not items),
            "message": "At least one relevant SQL symbol has no definition/patch lineage.",
        })
    missing_facets = sorted(
        name for name, detail in facet_coverage.items() if detail.get("status") == "missing"
    )
    if missing_facets:
        warnings.append({
            "code": "missing_question_facets", "facets": missing_facets,
            "message": "Explicitly requested domains have no mapped SQL symbol or code candidate.",
        })
    if consumer_count:
        strengths.append({
            "code": "consumer_fanout", "message": f"Mapped {consumer_count} distinct consumers outside top-k ranking."
        })
    if lineage_entries:
        strengths.append({
            "code": "sql_lineage", "message": f"Mapped {lineage_entries} SQL definition/patch entries."
        })
    if all(flow.get(key) == [] for key in ("roles_missing", "lanes_missing")):
        strengths.append({
            "code": "nominal_coverage", "message": "All requested roles and standard lanes have at least one representative."
        })

    severe_codes = {"no_evidence", "missing_roles", "incomplete_sql_lineage"}
    quality = (
        "weak" if any(item["code"] in severe_codes for item in warnings)
        else "review" if warnings
        else "healthy"
    )
    return {
        "quality": quality,
        "shape": {
            "evidence_items": total,
            "anchor_files": anchor_files,
            "lane_counts": lane_counts,
            "role_counts": role_counts,
            "authority_counts": authority_counts,
            "anchor_authority_counts": anchor_authority_counts,
            "reason_counts": reason_counts,
            "confidence_counts": confidence_counts,
            "lexical_relation_ratio": round(lexical_ratio, 3),
            "dominant_lane_ratio": round(concentration, 3),
            "distinct_rpc_consumers": consumer_count,
            "sql_lineage_entries": lineage_entries,
            "shared_dependency_consumers": len({
                item["source"] for items in reverse_dependencies.values() for item in items
                if item.get("source_active")
            }),
            "facet_counts": {
                name: len(detail.get("sql_symbols", [])) + len(detail.get("code_paths", []))
                for name, detail in facet_coverage.items()
            },
        },
        "warnings": warnings,
        "strengths": strengths,
        "interpretation": (
            "This diagnoses retrieval shape, not code correctness or exhaustive repository coverage."
        ),
    }


def investigate_flow(
    db: sqlite3.Connection, project: str, seed_paths: list[str], limit: int = 16,
    question: str = "",
) -> dict:
    """Expand search anchors into a bounded, provenance-rich evidence map."""
    ensure_schema(db)
    scores: dict[str, float] = {}
    reasons: dict[str, list[dict]] = {}

    def add(path: str | None, score: float, reason: dict) -> None:
        if not path:
            return
        scores[path] = max(scores.get(path, 0.0), score)
        bucket = reasons.setdefault(path, [])
        key = json.dumps(reason, sort_keys=True)
        if all(json.dumps(item, sort_keys=True) != key for item in bucket):
            bucket.append(reason)

    def normalize_query_token(token: str) -> str:
        if token.endswith("ies") and len(token) > 5:
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            return token[:-1]
        return token

    ignored_affinity_terms = {
        "affected", "behavior", "change", "code", "every", "find", "fix",
        "identify", "implementation", "investigate", "production", "regression",
        "repository", "should", "test", "when", "while", "with",
    }
    affinity_terms = {
        normalize_query_token(token)
        for token in re.findall(r"[a-z0-9_]+", question.lower())
        if len(token) > 3
    } - ignored_affinity_terms
    affinity_cache: dict[str, tuple[int, list[str]]] = {}

    def path_question_affinity(path: str) -> tuple[int, list[str]]:
        cached = affinity_cache.get(path)
        if cached is not None:
            return cached
        row = db.execute(
            "SELECT lower(group_concat(content, '\n')) FROM chunks WHERE project=? AND path=?",
            (project, path),
        ).fetchone()
        haystack = f"{path.lower()}\n{str(row[0] or '') if row else ''}"
        matched = sorted(term for term in affinity_terms if term in haystack)
        result = (len(matched), matched[:8])
        affinity_cache[path] = result
        return result

    seeds = list(dict.fromkeys(seed_paths))[:6]
    for rank, path in enumerate(seeds, 1):
        add(path, 100 - rank, {"kind": "search_anchor", "rank": rank})

    frontier = set(seeds)
    expanded: set[str] = set()
    resources: set[str] = set()
    resource_mentions: dict[str, set[str]] = {}
    for depth in range(2):
        if not frontier:
            break
        current = set(frontier)
        expanded.update(current)
        placeholders = ",".join("?" for _ in current)
        outgoing = db.execute(
            f"""SELECT * FROM edges WHERE project=? AND source_path IN ({placeholders})
                AND relation IN ({','.join('?' for _ in FLOW_RELATIONS)})""",
            [project, *current, *FLOW_RELATIONS],
        ).fetchall()
        incoming = db.execute(
            f"""SELECT * FROM edges WHERE project=? AND target_path IN ({placeholders})
                AND relation IN ({','.join('?' for _ in FLOW_RELATIONS)})""",
            [project, *current, *FLOW_RELATIONS],
        ).fetchall()
        named_dispatch_targets = {
            row["target_qualified_name"]
            for row in outgoing
            if row["relation"] == "calls"
            and row["target_qualified_name"]
            and row["source_name"].lower() in affinity_terms
        }
        next_frontier: set[str] = set()
        for row in outgoing:
            if row["relation"] in RESOURCE_RELATIONS:
                resources.add(row["target_name"])
                resource_mentions.setdefault(row["target_name"], set()).add(row["source_path"])
            if row["relation"] == "invokes_edge_function" and not row["target_path"]:
                target_path = f"supabase/functions/{row['target_name']}/index.ts"
                exists = db.execute(
                    "SELECT 1 FROM source_authority WHERE project=? AND path=?",
                    (project, target_path),
                ).fetchone()
                if exists:
                    add(target_path, 78 - depth * 8, {
                        "kind": "resolved_edge_function", "from": row["source_path"],
                        "function": row["target_name"], "line": row["line"],
                        "confidence": "convention_verified",
                    })
                    next_frontier.add(target_path)
            if row["target_path"] and row["resolution_confidence"] != "unresolved":
                inheritance = row["relation"] in {"extends", "overrides"}
                add(row["target_path"], 62 - depth * 12, {
                    "kind": "inheritance_chain" if inheritance else "outgoing_edge",
                    "from": row["source_path"],
                    "relation": row["relation"], "symbol": row["target_name"],
                    "line": row["line"], "depth": depth,
                    "confidence": row["resolution_confidence"],
                })
                next_frontier.add(row["target_path"])
        for row in incoming:
            if row["resolution_confidence"] != "unresolved":
                inheritance = row["relation"] in {"extends", "overrides"}
                dispatched_override = (
                    row["relation"] == "overrides"
                    and row["target_qualified_name"] in named_dispatch_targets
                )
                affinity, matched_terms = path_question_affinity(row["source_path"])
                add(row["source_path"], 58 - depth * 12, {
                    "kind": (
                        "dispatch_override_chain" if dispatched_override
                        else "inheritance_chain" if inheritance
                        else "incoming_edge"
                    ),
                    "to": row["target_path"],
                    "relation": row["relation"], "symbol": row["source_name"],
                    "line": row["line"], "depth": depth,
                    "confidence": row["resolution_confidence"],
                    **({"query_terms": matched_terms} if inheritance and affinity else {}),
                })
                next_frontier.add(row["source_path"])
        frontier = next_frontier - expanded

    for resource in sorted(resources):
        resource_tokens = {
            token for token in re.split(r"[^a-z0-9]+", resource.lower()) if len(token) > 2
        }
        seed_tokens = {
            token for path in seeds
            for token in re.split(r"[^a-z0-9]+", path.lower()) if len(token) > 2
        }
        resource_affinity = len(resource_tokens & seed_tokens)
        for row in db.execute(
            """SELECT * FROM edges WHERE project=? AND target_name=?
               AND relation IN ('reads_table','writes_table','accesses_table','calls_rpc')""",
            (project, resource),
        ):
            add(row["source_path"], 48, {
                "kind": "shared_resource", "resource": resource,
                "relation": row["relation"], "line": row["line"],
            })

        # An RPC is both a dependency and an impact boundary. Keep every known
        # caller visible instead of relying on the bounded graph walk to happen
        # across each UI consumer.
        for row in db.execute(
            """SELECT * FROM edges WHERE project=? AND target_name=?
               AND relation='calls_rpc'""",
            (project, resource),
        ):
            add(row["source_path"], 57, {
                "kind": "rpc_consumer", "rpc": resource,
                "line": row["line"], "confidence": row["resolution_confidence"],
            })

        # Later migrations can patch a function through pg_get_functiondef and
        # EXECUTE without declaring a new CREATE FUNCTION symbol. Surface that
        # override chain so an older full definition is not mistaken for the
        # complete active implementation.
        rpc_name = resource.split(".")[-1]
        for row in db.execute(
            """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='schema_history'
                 AND lower(c.content) LIKE lower(?)
                 AND (lower(c.content) LIKE '%pg_get_functiondef%'
                      OR lower(c.content) LIKE '%create or replace function%')
               GROUP BY c.path ORDER BY c.path DESC LIMIT 6""",
            (project, f"%{rpc_name}%"),
        ):
            add(row["path"], 72, {
                "kind": "sql_override_chain", "symbol": resource,
                "lines": [row["start_line"], row["end_line"]],
                "confidence": "definition_or_patch_lexical",
            })
        pattern = f"%{resource.split('.')[-1]}%"
        for row in db.execute(
            """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='schema_history'
                 AND lower(c.content) LIKE lower(?)
                 AND (lower(c.content) LIKE '%unique%'
                      OR lower(c.content) LIKE '%constraint%'
                      OR lower(c.content) LIKE '%create index%')
               GROUP BY c.path LIMIT 8""",
            (project, pattern),
        ):
            add(row["path"], 54 + min(len(resource_mentions.get(resource, set())), 3) * 2 + resource_affinity * 6, {
                "kind": "constraint_candidate", "resource": resource,
                "lines": [row["start_line"], row["end_line"]],
                "confidence": "lexical",
            })

    # Seed SQL files are often historical definitions while callers have been
    # resolved to a later active definition. Expand by function identity, not
    # only by target_path, so the full consumer fan-out remains discoverable.
    sql_paths = list(scores)
    placeholders = ",".join("?" for _ in sql_paths)
    rpc_names: set[str] = set()
    if sql_paths:
        for row in db.execute(
            f"""SELECT DISTINCT s.name,s.qualified_name FROM symbols s
                JOIN source_authority a USING(project,path)
                WHERE s.project=? AND s.path IN ({placeholders})
                  AND s.kind='function' AND a.authority='schema_history'""",
            [project, *sql_paths],
        ):
            rpc_names.update((row["name"], row["qualified_name"], row["qualified_name"].split(".")[-1]))
    for rpc_name in sorted(name for name in rpc_names if name):
        short_name = rpc_name.split(".")[-1]
        question_words = {
            token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 2
        }
        question_words -= {
            "active", "all", "cache", "consumer", "consumers", "identify",
            "query", "rpc", "sql", "test", "tests", "type", "types", "ui",
        }
        rpc_relevant = any(token in short_name.lower() for token in question_words)
        rpc_score = 59 + (8 if rpc_relevant else 0)
        if "kpi" in question.lower() and "kpi" in short_name.lower():
            rpc_score += 12
        for row in db.execute(
            """SELECT * FROM edges WHERE project=? AND relation='calls_rpc'
               AND (target_name=? OR target_name=?)""",
            (project, rpc_name, short_name),
        ) if rpc_relevant else ():
            add(row["source_path"], 64, {
                "kind": "rpc_consumer", "rpc": short_name,
                "line": row["line"], "confidence": row["resolution_confidence"],
            })
        for row in db.execute(
            """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='code'
                 AND c.path LIKE 'src/%'
                 AND (c.path LIKE '%.ts' OR c.path LIKE '%.tsx')
                 AND lower(c.content) LIKE lower(?)
               GROUP BY c.path LIMIT 12""",
            (project, f"%{short_name}%"),
        ) if rpc_relevant else ():
            add(row["path"], rpc_score, {
                "kind": "rpc_consumer_lexical", "rpc": short_name,
                "lines": [row["start_line"], row["end_line"]],
                "confidence": "lexical",
            })
        for row in db.execute(
            """SELECT DISTINCT c.path FROM chunks c
               JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='generated'
                 AND lower(c.content) LIKE lower(?) LIMIT 6""",
            (project, f"%{short_name}%"),
        ):
            add(row["path"], 50, {
                "kind": "generated_contract_consumer", "rpc": short_name,
                "confidence": "lexical",
            })
    # A scheduled Edge Function is linked by its directory name, even when SQL
    # extraction cannot resolve a concrete target path.
    for path in list(scores):
        match = re.match(r"supabase/functions/([^/]+)/", path)
        if not match:
            continue
        function_name = match.group(1)
        for row in db.execute(
            "SELECT * FROM edges WHERE project=? AND relation='invokes_edge_function' AND target_name=?",
            (project, function_name),
        ):
            add(row["source_path"], 56, {
                "kind": "edge_function_invoker", "target": function_name,
                "line": row["line"], "confidence": row["resolution_confidence"],
            })
        schedule_pattern = f"%{function_name}%"
        for row in db.execute(
            """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='schema_history'
                 AND lower(c.content) LIKE lower(?)
                 AND (lower(c.content) LIKE '%cron.schedule%'
                      OR lower(c.content) LIKE '%net.http_post%')
               GROUP BY c.path LIMIT 4""",
            (project, schedule_pattern),
        ):
            add(row["path"], 60 + (10 if "retry" in function_name.lower() else 0), {
                "kind": "scheduler_lexical", "target": function_name,
                "lines": [row["start_line"], row["end_line"]],
                "confidence": "lexical",
            })

    # Preserve query-related UI even when the structural graph does not connect
    # it within the bounded traversal (for example, a second RPC in the same workflow).
    impact_terms = {
        token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 4
    }
    if any(token.startswith("ausenc") for token in impact_terms): impact_terms.add("absence")
    if any(token.startswith("vacacion") for token in impact_terms): impact_terms.add("vacation")
    for row in db.execute(
        """SELECT path FROM source_authority WHERE project=? AND authority='code'
           AND (path LIKE 'src/pages/%' OR path LIKE 'src/components/%')""",
        (project,),
    ):
        lower = row["path"].lower()
        matched = sorted(term for term in impact_terms if term in lower)
        if matched:
            add(row["path"], 52, {
                "kind": "query_related_ui", "terms": matched,
                "confidence": "lexical",
            })

    # Mutation-side cache invalidation is not normally connected to an RPC
    # reader in the symbol graph. Preserve domain-matching invalidators as
    # review candidates for changes whose backend value is cached in the UI.
    for row in db.execute(
        """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line,
                  lower(group_concat(c.content, ' ')) content
           FROM chunks c JOIN source_authority a USING(project,path)
           WHERE c.project=? AND a.authority='code'
           GROUP BY c.path
           HAVING content LIKE '%invalidatequeries%'""",
        (project,),
    ):
        lower_path = row["path"].lower()
        content = row["content"] or ""
        matched = sorted(term for term in impact_terms if term in lower_path or term in content)
        if matched:
            add(row["path"], 55, {
                "kind": "cache_invalidator", "terms": matched[:8],
                "lines": [row["start_line"], row["end_line"]],
                "confidence": "domain_lexical",
            })

    def lane(path: str, authority: str) -> str:
        lower = path.lower()
        path_reasons = reasons.get(path, [])
        if authority == "test" or any(marker in lower for marker in (".test.", ".spec.", "_test.", "_spec.")):
            return "tests"
        if (authority == "schema_history" and any(
            item["kind"] in {"edge_function_invoker", "scheduler_lexical"}
            for item in path_reasons
        )) or "schedule" in lower or "cron" in lower:
            return "scheduling"
        if any(item["kind"] == "constraint_candidate" for item in path_reasons):
            return "constraints"
        if "retry" in lower or "/workers/" in lower or "/jobs/" in lower:
            return "retries"
        if any(item["kind"] == "shared_resource" for item in path_reasons):
            return "persistence"
        return "implementation"

    authority_by_path = {}
    lane_by_path = {}
    for path in scores:
        authority_row = db.execute(
            "SELECT authority FROM source_authority WHERE project=? AND path=?",
            (project, path),
        ).fetchone()
        authority_by_path[path] = authority_row["authority"] if authority_row else "unknown"
        lane_by_path[path] = lane(path, authority_by_path[path])
    ordered = sorted(scores, key=lambda path: (-scores[path], path))
    ranked = []
    # Preserve scarce evidence classes before filling by global score; otherwise
    # high-degree implementation nodes can crowd out tests and constraints.
    for wanted_lane in ("implementation", "persistence", "retries", "scheduling", "constraints", "tests"):
        for path in (item for item in ordered if lane_by_path[item] == wanted_lane):
            if path not in ranked:
                ranked.append(path)
                break
    sql_overrides = sorted((path for path in ordered if any(
        reason["kind"] == "sql_override_chain" for reason in reasons[path]
    )), reverse=True)
    for path in sql_overrides[:4]:
        if path not in ranked:
            ranked.append(path)
    dispatch_candidates = [path for path in ordered if any(
        reason["kind"] == "dispatch_override_chain" for reason in reasons[path]
    )]
    dispatch_candidates.sort(key=lambda path: (
        path.startswith(("build/", "dist/", "vendor/")),
        min(
            reason.get("depth", 99) for reason in reasons[path]
            if reason["kind"] == "dispatch_override_chain"
        ),
        -path_question_affinity(path)[0], ordered.index(path), path,
    ))
    exhaustive_request = bool({"all", "cada", "every", "todas", "todos"} & {
        token for token in re.findall(r"[a-z0-9_]+", question.lower())
    })
    for path in dispatch_candidates[:10 if exhaustive_request else 4]:
        if path not in ranked:
            ranked.append(path)
    inheritance_candidates = [path for path in ordered if any(
        reason["kind"] == "inheritance_chain" for reason in reasons[path]
    )]
    inheritance_candidates.sort(key=lambda path: (
        min(
            reason.get("depth", 99) for reason in reasons[path]
            if reason["kind"] == "inheritance_chain"
        ),
        -path_question_affinity(path)[0], ordered.index(path), path,
    ))
    for path in inheritance_candidates[:4]:
        if path not in ranked:
            ranked.append(path)
    # Reserve impact slots independently from causal lanes. UI consumers and
    # cache owners are often peripheral to an explanation but central to a safe edit.
    query_tokens = set(impact_terms)
    ui_candidates = [path for path in ordered if (
        ("/pages/" in path.lower() or "/components/" in path.lower())
        and "/components/ui/" not in path.lower()
        and (
            any(token in path.lower() for token in query_tokens)
            or any(reason["kind"] in {
                "query_related_ui", "rpc_consumer", "rpc_consumer_lexical",
                "cache_invalidator",
            } for reason in reasons[path])
        )
    )]
    ui_candidates.sort(key=lambda path: (
        -sum(token in path.lower() for token in query_tokens),
        ordered.index(path),
    ))
    for path in ui_candidates[:5]:
        if path not in ranked:
            ranked.append(path)
    rpc_consumers = [path for path in ordered if any(
        reason["kind"] in {"rpc_consumer", "rpc_consumer_lexical"} for reason in reasons[path]
    )]
    reserved_rpc = 0
    for path in rpc_consumers:
        if path not in ranked:
            ranked.append(path)
            reserved_rpc += 1
        if reserved_rpc >= 5:
            break
    generated_contracts = [path for path in ordered if any(
        reason["kind"] == "generated_contract_consumer" for reason in reasons[path]
    )]
    for path in generated_contracts[:2]:
        if path not in ranked:
            ranked.append(path)
    lane_caps = {
        "implementation": 4, "persistence": 4, "retries": 3,
        "scheduling": 2, "constraints": 2, "tests": 3,
    }
    for path in ordered:
        if path not in ranked:
            path_lane = lane_by_path[path]
            if sum(lane_by_path[item] == path_lane for item in ranked) >= lane_caps[path_lane]:
                continue
            ranked.append(path)
        if len(ranked) >= max(1, limit):
            break
    ranked = ranked[:max(1, limit)]
    evidence = []
    for path in ranked:
        authority = authority_by_path[path]
        symbol = db.execute(
            """SELECT start_line,end_line,name,kind FROM symbols
               WHERE project=? AND path=? AND active=1
               ORDER BY CASE WHEN kind='module' THEN 0 ELSE 1 END,start_line LIMIT 1""",
            (project, path),
        ).fetchone()
        lines = [symbol["start_line"], symbol["end_line"]] if symbol else None
        path_reasons = reasons[path]
        roles = set()
        relations = {item.get("relation") for item in path_reasons}
        relations.update(row[0] for row in db.execute(
            """SELECT DISTINCT relation FROM edges WHERE project=? AND source_path=?
               AND relation IN ('reads_table','writes_table','accesses_table','calls_rpc')""",
            (project, path),
        ))
        has_write = db.execute(
            """SELECT 1 FROM edges WHERE project=? AND source_path=? AND (
                 relation='writes_table' OR (
                   relation='accesses_table'
                   AND json_extract(metadata,'$.operation') IN ('update','into','delete from')
                 )
               ) LIMIT 1""",
            (project, path),
        ).fetchone()
        if authority == "test": roles.add("tests")
        if authority == "schema_history": roles.add("sql")
        if "writes_table" in relations or has_write: roles.add("writers")
        if relations & {"reads_table", "accesses_table"}: roles.add("readers")
        if "calls_rpc" in relations or any(item.get("confidence") == "active_sql" for item in path_reasons): roles.add("rpc")
        lower, query = path.lower(), question.lower()
        if any(term in query for term in ("calcul", "saldo", "balance", "entitlement", "prorrat")) and any(term in lower for term in ("payroll", "vacation", "balance", "entitlement", "absence")): roles.add("calculation")
        if any(term in query for term in ("crea", "modif", "aprob", "cancel", "lifecycle", "status")) and any(term in lower for term in ("absence", "workflow", "status")): roles.add("lifecycle")
        evidence.append({
            "path": path, "lane": lane(path, authority), "authority": authority,
            "roles": sorted(roles),
            "score": round(scores[path], 2), "lines": lines,
            "citation": f"{project}:{path}:{lines[0]}-{lines[1]}" if lines else f"{project}:{path}",
            "reasons": reasons[path][:5],
        })
    present = {item["lane"] for item in evidence}
    roles_present = {role for item in evidence for role in item["roles"]}
    requested_roles = set()
    normalized_question = question.lower()
    if any(term in normalized_question for term in ("calcul", "saldo", "balance", "entitlement", "prorrat")): requested_roles.add("calculation")
    if any(term in normalized_question for term in ("writer", "escrib", "crea", "modif", "aprob", "cancel")): requested_roles.add("writers")
    if "rpc" in normalized_question or "sql" in normalized_question: requested_roles.update(("rpc", "sql"))
    if "test" in normalized_question or "prueba" in normalized_question: requested_roles.add("tests")
    expected = {"implementation", "persistence", "retries", "scheduling", "constraints", "tests"}
    causal_roles = {"calculation", "lifecycle", "rpc", "sql", "writers", "readers"}
    causal_evidence = [
        item for item in evidence
        if causal_roles & set(item["roles"])
        or any(reason["kind"] == "search_anchor" for reason in item["reasons"])
    ]
    impact_candidates = []
    for item in evidence:
        lower = item["path"].lower()
        role_set = set(item["roles"])
        content_row = db.execute(
            "SELECT group_concat(content, '\n') FROM chunks WHERE project=? AND path=?",
            (project, item["path"]),
        ).fetchone()
        content = str(content_row[0] or "").lower() if content_row else ""
        categories = set()
        if item["authority"] == "test": categories.add("test")
        if ("/components/" in lower or "/pages/" in lower) and "/components/ui/" not in lower:
            categories.add("ui")
        if "invalidatequeries" in content or "querykey" in content: categories.add("cache")
        if item["authority"] == "generated" or lower.endswith("types.ts"): categories.add("types")
        if item["authority"] == "schema_history": categories.add("schema")
        if "writers" in role_set: categories.add("writer")
        if "readers" in role_set: categories.add("reader")
        if "rpc" in role_set: categories.add("integration")
        if "retry" in lower or "/workers/" in lower or "/jobs/" in lower: categories.add("async")
        impact_reason = {reason["kind"] for reason in item["reasons"]} & {
            "incoming_edge", "outgoing_edge", "shared_resource", "resolved_edge_function",
            "rpc_consumer", "sql_override_chain", "cache_invalidator",
            "generated_contract_consumer", "rpc_consumer_lexical", "inheritance_chain",
            "dispatch_override_chain",
        }
        if categories or impact_reason:
            likely_modify = bool(
                role_set & {"calculation", "lifecycle", "writers", "rpc"}
                or categories & {"ui", "cache", "test"}
            )
            impact_candidates.append({
                **item,
                "impact_categories": sorted(categories or {"dependency"}),
                "suggested_action": "likely_modify" if likely_modify else "review",
                "impact_basis": sorted(impact_reason) or ["role_match"],
            })
    coverage_warnings = []
    if roles_present:
        coverage_warnings.append(
            "roles_present proves that at least one evidence item exists for a role; "
            "it does not prove exhaustive writer, reader, consumer, or override coverage."
        )
    if any(reason["kind"] == "sql_override_chain" for values in reasons.values() for reason in values):
        coverage_warnings.append(
            "SQL override chains may be lexical patches; inspect migrations in order and verify "
            "the deployed pg_get_functiondef before editing."
        )

    # Exhaustive maps are kept outside the ranked evidence budget. They are
    # deliberately lexical/structural inventories for verification, not proof
    # that each file participates in the runtime branch under investigation.
    domain_words = {
        token for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 2 and token not in {
            "active", "cache", "consumer", "consumers", "identify", "query",
            "requested", "review", "tests", "types",
        }
    }
    technical_terms = {
        token for token in domain_words
        if token in {"api", "kpi", "rpc", "webhook", "worker"}
        or "_" in token
    }
    relevance_terms = technical_terms or domain_words
    relevant_rpcs = {
        name.split(".")[-1] for name in rpc_names
        if any(token in name.lower() for token in relevance_terms)
    }
    # Include direct SQL wrappers only when they share the same technical term;
    # unrestricted reverse closure quickly drifts through authorization helpers
    # and policies that are not consumers of the business result.
    for target in list(relevant_rpcs):
        for row in db.execute(
            """SELECT source_name FROM edges WHERE project=? AND relation='calls_sql'
               AND (target_name=? OR target_name=?)""",
            (project, target, f"public.{target}"),
        ):
            source = (row["source_name"] or "").split(".")[-1]
            if source and any(term in source.lower() for term in relevance_terms):
                relevant_rpcs.add(source)
    relevant_rpcs = {
        rpc for rpc in relevant_rpcs
        if rpc.startswith("get_") or db.execute(
            """SELECT 1 FROM edges WHERE project=? AND relation='calls_rpc'
               AND (target_name=? OR target_name=?) LIMIT 1""",
            (project, rpc, f"public.{rpc}"),
        ).fetchone()
    }

    rpc_consumer_map = {}
    rpc_consumer_query_keys: dict[str, set[str]] = {}
    sql_lineage = {}
    consumer_paths: set[str] = set()
    for rpc in sorted(relevant_rpcs):
        consumers = set()
        for row in db.execute(
            """SELECT c.path,group_concat(c.content, '\n') content
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='code' AND c.path LIKE 'src/%'
                 AND lower(c.content) LIKE lower(?) GROUP BY c.path""",
            (project, f"%{rpc}%"),
        ):
            consumers.add(row["path"])
            keys = set(re.findall(
                r"queryKey\s*:\s*\[\s*['\"]([^'\"]+)['\"]", row["content"] or "", re.I,
            ))
            rpc_consumer_query_keys.setdefault(row["path"], set()).update(keys)
        consumer_paths.update(consumers)
        rpc_consumer_map[rpc] = sorted(consumers)
        lineage = []
        for row in db.execute(
            """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line,
                      lower(group_concat(c.content, ' ')) content
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='schema_history'
                 AND lower(c.content) LIKE lower(?)
               GROUP BY c.path ORDER BY c.path""",
            (project, f"%{rpc}%"),
        ):
            content = row["content"] or ""
            if "create or replace function" not in content and "pg_get_functiondef" not in content:
                continue
            lineage.append({
                "path": row["path"],
                "lines": [row["start_line"], row["end_line"]],
                "kind": "dynamic_patch" if "pg_get_functiondef" in content else "definition",
                "confidence": "lexical",
            })
        sql_lineage[rpc] = lineage

    facet_terms = {
        "costs": ("cost", "coste", "kosten"),
        "overtime": ("overtime", "überstunden"),
        "coverage": ("coverage", "cobertura"),
        "planner": ("planner", "planning", "planificador"),
        "payroll": ("payroll", "nómina", "nomina"),
        "reports_exports": ("report", "reporte", "export"),
        "absences": ("absence", "ausencia"),
    }
    requested_facets = {
        facet for facet, aliases in facet_terms.items()
        if any(alias in normalized_question for alias in aliases)
    }
    facet_coverage = {}
    facet_sql_symbols: set[str] = set()
    for facet in sorted(requested_facets):
        aliases = facet_terms[facet]
        sql_symbols = []
        for row in db.execute(
            """SELECT DISTINCT s.name,s.qualified_name,s.path FROM symbols s
               JOIN source_authority a USING(project,path)
               WHERE s.project=? AND s.kind='function' AND s.active=1
                 AND a.authority='schema_history'""",
            (project,),
        ):
            symbol_name = f"{row['name']} {row['qualified_name']}".lower()
            if any(alias in symbol_name for alias in aliases):
                short = row["qualified_name"].split(".")[-1]
                facet_sql_symbols.add(short)
                sql_symbols.append({
                    "name": row["qualified_name"], "path": row["path"],
                    "confidence": "active_sql_name",
                })
        code_paths = []
        for row in db.execute(
            """SELECT c.path,lower(group_concat(c.content, ' ')) content
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='code' AND c.path LIKE 'src/%'
               GROUP BY c.path""",
            (project,),
        ):
            lower_path = row["path"].lower()
            content = row["content"] or ""
            if facet == "reports_exports":
                matched = (
                    any(alias in lower_path for alias in aliases)
                    or bool(re.search(r"handleexport|export[A-Z_]|\.pdf\b|\.csv\b|download", content, re.I))
                )
            else:
                matched = any(alias in lower_path or alias in content for alias in aliases)
            if matched:
                code_paths.append(row["path"])
        facet_coverage[facet] = {
            "status": "mapped" if sql_symbols or code_paths else "missing",
            "sql_symbols": sql_symbols[:12], "code_paths": sorted(code_paths)[:12],
        }

    # Follow active SQL callees separately from ranked evidence. This exposes
    # shared calculators whose later redefinition can invalidate upstream KPI
    # guards even when their names do not resemble the user's primary RPC.
    dependency_sources = set(relevant_rpcs) | facet_sql_symbols
    sql_dependency_map: dict[str, list[dict]] = {}
    frontier = set(dependency_sources)
    seen_sources: set[str] = set()
    dependency_symbols: set[str] = set()
    for _depth in range(2):
        next_frontier: set[str] = set()
        for source in sorted(frontier - seen_sources):
            seen_sources.add(source)
            dependencies = []
            for row in db.execute(
                """SELECT target_name,target_path,line,resolution_confidence FROM edges
                   WHERE project=? AND relation='calls_sql'
                     AND (source_name=? OR source_name=?)
                     AND target_path IS NOT NULL
                     AND resolution_confidence='active_sql'""",
                (project, source, f"public.{source}"),
            ):
                target = row["target_name"].split(".")[-1]
                dependencies.append({
                    "symbol": row["target_name"], "path": row["target_path"],
                    "line": row["line"], "confidence": row["resolution_confidence"],
                })
                dependency_symbols.add(target)
                next_frontier.add(target)
            if dependencies:
                unique = {(item["symbol"], item["path"], item["line"]): item for item in dependencies}
                sql_dependency_map[source] = list(unique.values())
        frontier = next_frontier

    facet_aliases = {
        alias for facet in requested_facets for alias in facet_terms[facet]
        if alias != "export"
    }
    lineage_dependencies = {
        symbol for symbol in dependency_symbols
        if symbol in relevant_rpcs
        or any(alias in symbol.lower() for alias in facet_aliases)
        or "employee_in_kpis" in symbol.lower()
    }
    for symbol_name in sorted(lineage_dependencies | facet_sql_symbols):
        if symbol_name in sql_lineage:
            continue
        lineage = []
        for row in db.execute(
            """SELECT c.path,min(c.start_line) start_line,max(c.end_line) end_line,
                      lower(group_concat(c.content, ' ')) content
               FROM chunks c JOIN source_authority a USING(project,path)
               WHERE c.project=? AND a.authority='schema_history'
                 AND lower(c.content) LIKE lower(?)
               GROUP BY c.path ORDER BY c.path""",
            (project, f"%{symbol_name}%"),
        ):
            content = row["content"] or ""
            if "create or replace function" in content or "pg_get_functiondef" in content:
                lineage.append({
                    "path": row["path"], "lines": [row["start_line"], row["end_line"]],
                    "kind": "dynamic_patch" if "pg_get_functiondef" in content else "definition",
                    "confidence": "lexical",
                })
        sql_lineage[symbol_name] = lineage

    sql_reverse_dependency_map: dict[str, list[dict]] = {}
    for dependency in sorted(lineage_dependencies | facet_sql_symbols):
        callers = []
        for row in db.execute(
            """SELECT source_name,source_path,line,resolution_confidence FROM edges
               WHERE project=? AND relation='calls_sql'
                 AND (target_name=? OR target_name=?)
                 AND resolution_confidence='active_sql'""",
            (project, dependency, f"public.{dependency}"),
        ):
            if not row["source_name"]:
                continue
            source_active = bool(db.execute(
                """SELECT 1 FROM symbols WHERE project=? AND path=?
                   AND qualified_name=? AND active=1 LIMIT 1""",
                (project, row["source_path"], row["source_name"]),
            ).fetchone())
            callers.append({
                "source": row["source_name"], "path": row["source_path"],
                "line": row["line"], "confidence": row["resolution_confidence"],
                "source_active": source_active,
            })
        if callers:
            unique = {(item["source"], item["path"], item["line"]): item for item in callers}
            sql_reverse_dependency_map[dependency] = list(unique.values())

    query_key_pattern = re.compile(r"queryKey\s*:\s*\[\s*['\"]([^'\"]+)['\"]", re.I)
    invalidation_pattern = re.compile(
        r"invalidateQueries\s*\(\s*\{\s*queryKey\s*:\s*\[\s*['\"]([^'\"]+)['\"]", re.I,
    )
    cache_invalidation_candidates = []
    all_invalidated_keys: set[str] = set()
    consumer_query_keys: set[str] = set()
    cache_paths = consumer_paths | {
        item["path"] for item in impact_candidates if "cache" in item["impact_categories"]
    }
    invalidation_scope_paths = set(seeds) | {
        path for path in cache_paths if path not in consumer_paths and any(
            reason["kind"] == "cache_invalidator" for reason in reasons.get(path, [])
        )
    }
    delegated_invalidation_paths: set[str] = set()
    for path in list(invalidation_scope_paths):
        row = db.execute(
            "SELECT group_concat(content, '\n') content FROM chunks WHERE project=? AND path=?",
            (project, path),
        ).fetchone()
        if not row or "invalidateQueries" not in (row["content"] or ""):
            continue
        source_content = row["content"] or ""
        for edge in db.execute(
            """SELECT target_path FROM edges WHERE project=? AND source_path=?
               AND relation='imports' AND target_path IS NOT NULL""",
            (project, path),
        ):
            delegated_invalidation_paths.add(edge["target_path"])
        for specifier in re.findall(r"from\s+['\"](@/[^'\"]+|\.[^'\"]+)['\"]", source_content):
            if specifier.startswith("@/"):
                base = "src/" + specifier[2:]
            else:
                base = str((Path(path).parent / specifier).as_posix())
            for candidate in (base, f"{base}.ts", f"{base}.tsx", f"{base}/index.ts"):
                if db.execute(
                    "SELECT 1 FROM source_authority WHERE project=? AND path=?",
                    (project, candidate),
                ).fetchone():
                    delegated_invalidation_paths.add(candidate)
                    break
    cache_paths.update(delegated_invalidation_paths)
    for path in sorted(cache_paths):
        row = db.execute(
            "SELECT group_concat(content, '\n') content FROM chunks WHERE project=? AND path=?",
            (project, path),
        ).fetchone()
        content = row["content"] or "" if row else ""
        query_keys = sorted(set(query_key_pattern.findall(content)))
        invalidated_keys = sorted(set(invalidation_pattern.findall(content)))
        if path in delegated_invalidation_paths:
            invalidated_keys = sorted(set(invalidated_keys) | set(re.findall(
                r"['\"]([a-z0-9][a-z0-9-]*kpi[a-z0-9-]*)['\"]", content, re.I,
            )))
        if path in invalidation_scope_paths or path in delegated_invalidation_paths:
            all_invalidated_keys.update(invalidated_keys)
        if path in consumer_paths:
            consumer_query_keys.update(
                key for key in rpc_consumer_query_keys.get(path, set()) if "kpi" in key.lower()
            )
        if query_keys or invalidated_keys:
            cache_invalidation_candidates.append({
                "path": path, "query_keys": query_keys,
                "invalidated_keys": invalidated_keys, "confidence": "lexical",
            })
    potentially_uninvalidated_query_keys = sorted(consumer_query_keys - all_invalidated_keys)
    result = {
        "seed_paths": seeds,
        "resources": sorted(resources),
        "evidence": evidence,
        "lanes_present": sorted(present),
        "lanes_missing": sorted(expected - present),
        "roles_present": sorted(roles_present),
        "roles_missing": sorted(requested_roles - roles_present),
        "causal_evidence": causal_evidence,
        "impact_candidates": impact_candidates,
        "coverage_warnings": coverage_warnings,
        "sql_lineage": sql_lineage,
        "sql_dependency_map": sql_dependency_map,
        "sql_reverse_dependency_map": sql_reverse_dependency_map,
        "facet_coverage": facet_coverage,
        "rpc_consumer_map": rpc_consumer_map,
        "cache_invalidation_candidates": cache_invalidation_candidates,
        "potentially_uninvalidated_query_keys": potentially_uninvalidated_query_keys,
        "guidance": (
            "causal_evidence supports understanding; impact_candidates preserves surrounding "
            "consumers that may need review or modification. Neither is a proven causal trace. "
            "Open cited files and verify branch conditions before conclusions or edits."
        ),
    }
    result["retrieval_diagnostic"] = diagnose_retrieval(result)
    return result
