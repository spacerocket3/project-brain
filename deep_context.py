"""Build one large, cited repository context package for a coding agent.

This module does no generation.  It combines already-ranked retrieval, the
structural flow map, and exact source from one indexed working-tree generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

import project_brain as brain


DEFAULT_CONTEXT_TOKENS = 36_000
MIN_CONTEXT_TOKENS = 16_000
MAX_CONTEXT_TOKENS = 60_000
MAX_SOURCE_ITEMS = 48
MAX_ITEMS_PER_PATH = 4
MAX_SOURCE_CHARS = 18_000
SOURCE_MARGIN_LINES = 24
TOKEN_ESTIMATE_METHOD = "compact_json_chars_div_4"


@dataclass
class SourceCandidate:
    path: str
    start_line: int = 1
    end_line: int = 1
    priority: int = 0
    authority: str = "unknown"
    reasons: set[str] = field(default_factory=set)


def estimate_tokens(value: Any) -> int:
    """Return a deterministic transport-size estimate for an MCP JSON result.

    The client model tokenizer is not available to the MCP process.  Four JSON
    characters per token is deliberately reported as an estimate, never as an
    exact Codex token count.
    """
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(encoded) / 4))


def _safe_range(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start, end = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return 1, 1
        if start >= 1 and end >= start:
            return start, end
    return 1, 1


def _add_candidate(
    candidates: dict[tuple[str, int, int], SourceCandidate],
    path: Any,
    lines: Any,
    priority: int,
    reason: str,
    authority: Any = "unknown",
) -> None:
    if not isinstance(path, str) or not path or path.startswith("/"):
        return
    if ".." in Path(path).parts:
        return
    start, end = _safe_range(lines)
    key = (path, start, end)
    item = candidates.setdefault(
        key,
        SourceCandidate(
            path=path,
            start_line=start,
            end_line=end,
            priority=priority,
            authority=str(authority or "unknown"),
        ),
    )
    item.priority = max(item.priority, priority)
    if item.authority == "unknown" and authority:
        item.authority = str(authority)
    item.reasons.add(reason)


def collect_source_candidates(
    hits: Iterable[brain.Hit],
    flow: dict[str, Any],
    relationship_hints: list[dict[str, Any]],
    anchor_relationships: list[dict[str, Any]],
    contract_differences: list[dict[str, Any]],
    structural_neighbors: list[SourceCandidate] | None = None,
    question: str = "",
) -> list[SourceCandidate]:
    """Fuse every current non-generative evidence surface into source requests."""
    candidates: dict[tuple[str, int, int], SourceCandidate] = {}

    for rank, hit in enumerate(hits):
        _add_candidate(
            candidates, hit.path, [hit.start_line, hit.end_line],
            2_000 - min(rank, 80) * 20, "semantic_or_hybrid_rank", hit.authority,
        )

    for item in flow.get("causal_evidence", []):
        _add_candidate(
            candidates, item.get("path"), item.get("lines"),
            1_000 + int(item.get("score") or 0), "causal_evidence",
            item.get("authority"),
        )
    for item in flow.get("impact_candidates", []):
        action_bonus = 80 if item.get("suggested_action") == "likely_modify" else 0
        _add_candidate(
            candidates, item.get("path"), item.get("lines"),
            950 + action_bonus + int(item.get("score") or 0),
            f"impact_{item.get('suggested_action') or 'review'}", item.get("authority"),
        )
    for item in flow.get("evidence", []):
        _add_candidate(
            candidates, item.get("path"), item.get("lines"),
            900 + int(item.get("score") or 0), "flow_evidence", item.get("authority"),
        )

    for entries in flow.get("sql_lineage", {}).values():
        for item in entries:
            _add_candidate(
                candidates, item.get("path"), item.get("lines"), 1_500,
                f"sql_lineage_{item.get('kind') or 'entry'}", "schema_history",
            )
    for mapping_name in ("sql_dependency_map", "sql_reverse_dependency_map"):
        for entries in flow.get(mapping_name, {}).values():
            for item in entries:
                line = int(item.get("line") or 1)
                _add_candidate(
                    candidates, item.get("path"), [line, line], 1_450,
                    mapping_name, "schema_history",
                )
    for paths in flow.get("rpc_consumer_map", {}).values():
        for path in paths:
            _add_candidate(candidates, path, [1, 1], 1_350, "rpc_consumer")
    for facet in flow.get("facet_coverage", {}).values():
        for item in facet.get("sql_symbols", []):
            _add_candidate(
                candidates, item.get("path"), [1, 1], 1_380,
                "question_facet_sql", "schema_history",
            )
        for path in facet.get("code_paths", []):
            _add_candidate(candidates, path, [1, 1], 1_370, "question_facet_code")
    for item in flow.get("cache_invalidation_candidates", []):
        _add_candidate(
            candidates, item.get("path"), [1, 1], 1_330,
            "cache_invalidation_candidate",
        )

    for item in relationship_hints:
        _add_candidate(
            candidates, item.get("path"), item.get("indexed_range") or item.get("lines"),
            1_800, "reverse_relationship_hint", item.get("authority"),
        )
    for item in anchor_relationships:
        _add_candidate(
            candidates, item.get("path"), item.get("lines"),
            1_820, "anchor_call_graph", "code",
        )
        for call in item.get("direct_calls", []):
            _add_candidate(
                candidates, call.get("path"), [1, 1], 1_600,
                "anchor_direct_callee", "code",
            )
    for item in contract_differences:
        for side in ("anchor", "related"):
            value = item.get(side, {})
            _add_candidate(
                candidates, value.get("path"), value.get("lines"),
                1_850, "possible_contract_difference", "code",
            )

    for neighbor in structural_neighbors or []:
        for reason in neighbor.reasons or {"structured_neighbor"}:
            _add_candidate(
                candidates, neighbor.path,
                [neighbor.start_line, neighbor.end_line], neighbor.priority,
                reason, neighbor.authority,
            )

    path_terms = {
        "".join(character for character in term.lower() if character.isalnum())
        for term in brain.query_terms(question)
        if len(term) >= 4
    }
    for item in candidates.values():
        normalized_path = "".join(
            character for character in item.path.lower() if character.isalnum()
        )
        matches = sum(term in normalized_path for term in path_terms if term)
        item.priority += min(matches * 40, 120)

    return sorted(
        candidates.values(),
        key=lambda item: (
            -item.priority,
            0 if item.authority in {"code", "schema_history"} else 1,
            item.path,
            item.start_line,
        ),
    )


def collect_structural_neighbors(
    db: Any, project: str, hits: list[brain.Hit], question: str,
    limit: int = 36,
) -> list[SourceCandidate]:
    """Follow one resolved hop from top ranked ranges before lexical fan-out."""
    query_terms = {
        term.replace("_", " ") for term in brain.query_terms(question)
        if len(term) >= 4
    }
    candidates: dict[tuple[str, int, int], SourceCandidate] = {}
    for rank, hit in enumerate(hits[:18]):
        source_symbol = _covering_symbol(
            db, project,
            SourceCandidate(hit.path, hit.start_line, hit.end_line),
            question,
        )
        edge_start = int(source_symbol["start_line"]) if source_symbol else hit.start_line
        edge_end = int(source_symbol["end_line"]) if source_symbol else hit.end_line
        rows = db.execute(
            """SELECT relation,target_name,target_path,target_qualified_name,
                      resolution_confidence
               FROM edges WHERE project=? AND source_path=?
                 AND line BETWEEN ? AND ?
                 AND resolution_confidence!='unresolved'
                 AND relation IN ('imports','calls','calls_rpc','calls_sql',
                                  'invokes_edge_function','overrides','reads_table',
                                  'writes_table','accesses_table')
               ORDER BY line,relation,target_name LIMIT 80""",
            (project, hit.path, edge_start, edge_end),
        ).fetchall()
        for row in rows:
            target_path = str(row["target_path"] or "")
            target_name = str(row["target_qualified_name"] or row["target_name"] or "")
            if target_path:
                symbol = db.execute(
                    """SELECT s.path,s.start_line,s.end_line,a.authority FROM symbols s
                       JOIN source_authority a USING(project,path)
                       WHERE s.project=? AND s.path=? AND s.active=1
                         AND (s.name=? OR s.qualified_name=? OR s.qualified_name LIKE ?)
                       ORDER BY (s.end_line-s.start_line),s.start_line LIMIT 1""",
                    (
                        project, target_path, row["target_name"], target_name,
                        f"%.{row['target_name']}",
                    ),
                ).fetchone()
            else:
                symbol = db.execute(
                    """SELECT s.path,s.start_line,s.end_line,a.authority FROM symbols s
                       JOIN source_authority a USING(project,path)
                       WHERE s.project=? AND s.active=1
                         AND (s.name=? OR s.qualified_name=?)
                       ORDER BY CASE a.authority WHEN 'schema_history' THEN 0
                                                  WHEN 'code' THEN 1 ELSE 2 END,
                                (s.end_line-s.start_line),s.path LIMIT 1""",
                    (project, row["target_name"], target_name),
                ).fetchone()
                if symbol:
                    target_path = str(symbol["path"])
            if not target_path or target_path == hit.path:
                continue
            target_text = f"{target_path} {target_name}".lower().replace("_", " ")
            contract_relation = row["relation"] in {
                "calls_rpc", "calls_sql", "invokes_edge_function", "overrides",
            }
            test_implementation_neighbor = bool(
                hit.authority == "test"
                and row["relation"] in {"imports", "calls"}
                and symbol
                and symbol["authority"] == "code"
            )
            if not (
                test_implementation_neighbor
                or contract_relation
                or any(term in target_text for term in query_terms)
            ):
                continue
            lines = [symbol["start_line"], symbol["end_line"]] if symbol else [1, 1]
            _add_candidate(
                candidates, target_path, lines, 1_950 - rank * 20,
                f"structured_neighbor_{row['relation']}",
                symbol["authority"] if symbol else "code",
            )
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return sorted(
        candidates.values(),
        key=lambda item: (-item.priority, item.path, item.start_line),
    )


def _verified_snapshot(project: str) -> tuple[Any, Path, dict[str, Any], dict[str, str]]:
    db = brain.connect()
    repo = brain.require_registered_project(project)
    indexed = db.execute(
        "SELECT commit_hash,content_digest,extractor_version,generation_id "
        "FROM repos WHERE project=?", (project,),
    ).fetchone()
    structured = db.execute(
        "SELECT commit_hash,graph_digest,content_digest,extractor_version,generation_id "
        "FROM structure_snapshots WHERE project=?", (project,),
    ).fetchone()
    if not indexed or not structured:
        raise RuntimeError(f"{project} has no complete Project Brain generation")
    current_commit = brain.git(repo, "rev-parse", "HEAD")
    current_digest = brain.repository_content_digest(repo)
    fingerprint = brain.extractor_fingerprint()
    comparable = (
        indexed["commit_hash"], indexed["content_digest"],
        indexed["extractor_version"], indexed["generation_id"],
    )
    structural_comparable = (
        structured["commit_hash"], structured["content_digest"],
        structured["extractor_version"], structured["generation_id"],
    )
    expected = (current_commit, current_digest, fingerprint, indexed["generation_id"])
    if comparable != expected or structural_comparable != expected:
        raise RuntimeError(
            "Project Brain snapshot is stale or mixed; run refresh_repository before "
            "load_deep_context"
        )
    authorities = {
        row["path"]: row["authority"]
        for row in db.execute(
            "SELECT path,authority FROM source_authority WHERE project=?", (project,),
        )
    }
    snapshot = {
        "commit": current_commit,
        "content_digest": current_digest,
        "graph_digest": structured["graph_digest"],
        "extractor_version": fingerprint,
        "generation_id": indexed["generation_id"],
        "working_tree_dirty": bool(
            brain.git(repo, "status", "--porcelain", "--untracked-files=all")
        ),
        **brain.semantic_status(db, project),
    }
    return db, repo, snapshot, authorities


def _best_chunk_range(db: Any, project: str, path: str, question: str) -> tuple[int, int]:
    terms = brain.query_terms(question)
    rows = db.execute(
        "SELECT start_line,end_line,content FROM chunks WHERE project=? AND path=? "
        "ORDER BY start_line", (project, path),
    ).fetchall()
    if not rows:
        return 1, 1
    return max(
        rows,
        key=lambda row: sum(
            min(str(row["content"]).lower().count(term), 4) for term in terms
        ),
    )["start_line"], max(
        rows,
        key=lambda row: sum(
            min(str(row["content"]).lower().count(term), 4) for term in terms
        ),
    )["end_line"]


def _covering_symbol(
    db: Any, project: str, candidate: SourceCandidate, question: str,
) -> Any:
    rows = db.execute(
        """SELECT kind,name,qualified_name,start_line,end_line,active FROM symbols
           WHERE project=? AND path=? AND start_line<=? AND end_line>=?
           ORDER BY active DESC,
                    CASE kind WHEN 'function' THEN 0 WHEN 'method' THEN 1
                              WHEN 'class' THEN 2 ELSE 3 END,
                    (end_line-start_line),start_line""",
        (
            project, candidate.path, candidate.end_line, candidate.start_line,
        ),
    ).fetchall()
    containing = [
        row for row in rows
        if row["start_line"] <= candidate.start_line
        and row["end_line"] >= candidate.end_line
    ]
    if containing:
        return containing[0]
    candidate_size = max(1, candidate.end_line - candidate.start_line + 1)
    overlapping = sorted(
        rows,
        key=lambda row: (
            -(
                min(row["end_line"], candidate.end_line)
                - max(row["start_line"], candidate.start_line) + 1
            ),
            -int(row["active"]),
            row["end_line"] - row["start_line"],
        ),
    )
    if not overlapping:
        return None
    best = overlapping[0]
    overlap = (
        min(best["end_line"], candidate.end_line)
        - max(best["start_line"], candidate.start_line) + 1
    )
    symbol_text = f"{best['name']} {best['qualified_name']}".lower().replace("_", " ")
    query_terms = {
        term.replace("_", " ") for term in brain.query_terms(question)
        if len(term) >= 4
    }
    name_matches_question = any(term in symbol_text for term in query_terms)
    return best if (
        bool(best["active"])
        and overlap / candidate_size >= 0.45
        and name_matches_question
    ) else None


def compact_flow_map(flow: dict[str, Any]) -> dict[str, Any]:
    """Preserve flow conclusions and relationships without repeating full items."""
    reason_keys = {
        "kind", "rank", "relation", "symbol", "from", "to", "line", "depth",
        "confidence", "resource", "rpc",
    }

    def compact_reasons(reasons: Any) -> list[dict[str, Any]]:
        result = []
        seen = set()
        for reason in reasons or []:
            compact = {key: reason[key] for key in reason_keys if key in reason}
            marker = json.dumps(compact, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(compact)
            if len(result) >= 10:
                break
        return result

    evidence = [{
        key: item.get(key)
        for key in ("path", "lane", "authority", "roles", "score", "lines", "citation")
    } | {"reasons": compact_reasons(item.get("reasons"))}
        for item in flow.get("evidence", [])]
    causal = [{
        key: item.get(key)
        for key in ("path", "lane", "authority", "roles", "score", "lines", "citation")
    } for item in flow.get("causal_evidence", [])]
    impact = [{
        key: item.get(key)
        for key in (
            "path", "lane", "authority", "roles", "score", "lines", "citation",
            "impact_categories", "suggested_action", "impact_basis",
        )
    } for item in flow.get("impact_candidates", [])]
    retained = {
        key: flow.get(key)
        for key in (
            "seed_paths", "resources", "lanes_present", "lanes_missing",
            "roles_present", "roles_missing", "coverage_warnings", "sql_lineage",
            "sql_dependency_map", "sql_reverse_dependency_map", "facet_coverage",
            "rpc_consumer_map", "cache_invalidation_candidates",
            "potentially_uninvalidated_query_keys", "retrieval_diagnostic", "guidance",
        )
    }
    return {
        **retained,
        "evidence": evidence,
        "causal_evidence": causal,
        "impact_candidates": impact,
        "compaction": {
            "reason_limit_per_evidence": 10,
            "duplicated_evidence_fields_removed": True,
        },
    }


def _bounded_source(
    content: str, start: int, end: int, focus_start: int, focus_end: int,
) -> tuple[int, int, str, bool]:
    lines = content.splitlines()
    if not lines:
        return 1, 1, "", True
    start = max(1, min(start, len(lines)))
    end = max(start, min(end, len(lines)))
    selected = "\n".join(lines[start - 1:end])
    if len(selected) <= MAX_SOURCE_CHARS:
        return start, end, selected, True

    center = max(start, min((focus_start + focus_end) // 2, end))
    window_start = max(start, center - 160)
    window_end = min(end, center + 160)
    while window_start < window_end:
        selected = "\n".join(lines[window_start - 1:window_end])
        if len(selected) <= MAX_SOURCE_CHARS:
            return window_start, window_end, selected, False
        if window_end - center >= center - window_start:
            window_end -= 20
        else:
            window_start += 20
    selected = "\n".join(lines[center - 1:center])[:MAX_SOURCE_CHARS]
    return center, center, selected, False


def _source_excerpt(
    db: Any,
    repo: Path,
    project: str,
    candidate: SourceCandidate,
    authorities: dict[str, str],
    question: str,
) -> dict[str, Any] | None:
    if candidate.path not in authorities:
        return None
    absolute = repo / candidate.path
    if not absolute.is_file():
        return None
    try:
        content = absolute.read_text("utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    total_lines = max(1, len(content.splitlines()))
    if candidate.start_line == candidate.end_line == 1:
        candidate.start_line, candidate.end_line = _best_chunk_range(
            db, project, candidate.path, question,
        )
    symbol = _covering_symbol(db, project, candidate, question)
    if symbol:
        start, end = int(symbol["start_line"]), int(symbol["end_line"])
        if start <= end and start <= total_lines:
            end = min(end, total_lines)
            symbol_name = symbol["qualified_name"] or symbol["name"]
            active = bool(symbol["active"])
        else:
            symbol = None
    if not symbol:
        start = max(1, min(candidate.start_line - SOURCE_MARGIN_LINES, total_lines))
        end = max(start, min(total_lines, candidate.end_line + SOURCE_MARGIN_LINES))
        if total_lines <= 240:
            start, end = 1, total_lines
        symbol_name = ""
        active = None
    selected_start, selected_end, selected, complete = _bounded_source(
        content, start, end, candidate.start_line, candidate.end_line,
    )
    if not selected.strip():
        return None
    return {
        "path": candidate.path,
        "lines": [selected_start, selected_end],
        "full_range": [start, end],
        "citation": f"{project}:{candidate.path}:{selected_start}-{selected_end}",
        "authority": authorities[candidate.path],
        "symbol": symbol_name,
        "active": active,
        "complete": complete,
        "priority": candidate.priority,
        "reasons": sorted(candidate.reasons),
        "content": selected,
    }


def build_deep_context(
    *,
    project: str,
    question: str,
    hits: list[brain.Hit],
    flow: dict[str, Any],
    relationship_hints: list[dict[str, Any]],
    anchor_relationships: list[dict[str, Any]],
    contract_differences: list[dict[str, Any]],
    semantic_mode: str,
    reranker_fallback: str | None,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> dict[str, Any]:
    max_context_tokens = max(
        MIN_CONTEXT_TOKENS, min(int(max_context_tokens), MAX_CONTEXT_TOKENS),
    )
    db, repo, snapshot, authorities = _verified_snapshot(project)
    structural_neighbors = collect_structural_neighbors(
        db, project, hits, question,
    )
    candidates = collect_source_candidates(
        hits, flow, relationship_hints, anchor_relationships, contract_differences,
        structural_neighbors, question,
    )
    ranked_candidates = [
        {
            "path": item.path,
            "lines": [item.start_line, item.end_line],
            "authority": authorities.get(item.path, item.authority),
            "priority": item.priority,
            "reasons": sorted(item.reasons),
        }
        for item in candidates[:80]
    ]
    result: dict[str, Any] = {
        "project": project,
        "question": question,
        "snapshot": snapshot,
        "retrieval": {
            "semantic_mode": semantic_mode,
            "reranker_fallback": reranker_fallback,
            "ranked_candidates_considered": len(hits),
            "source_candidates": len(candidates),
            "relationship_hints": relationship_hints,
            "anchor_relationships": anchor_relationships,
            "contract_differences": contract_differences,
            "structural_neighbors": [
                {
                    "path": item.path, "lines": [item.start_line, item.end_line],
                    "priority": item.priority, "reasons": sorted(item.reasons),
                }
                for item in structural_neighbors
            ],
            "ranked_source_candidates": ranked_candidates,
        },
        "flow": compact_flow_map(flow),
        "source_bundle": [],
        "omitted_source_candidates": [],
        "guidance": (
            "Large one-shot repository context for heavy development. Source excerpts are "
            "navigation evidence from one verified indexed working-tree generation. Verify "
            "decisive branch behavior and tests in the checkout before editing."
        ),
    }
    base_tokens = estimate_tokens(result)
    per_path: dict[str, int] = {}
    included_ranges: dict[str, list[tuple[int, int]]] = {}
    omitted: list[dict[str, Any]] = []
    reserve = 2_000
    for candidate in candidates:
        if len(result["source_bundle"]) >= MAX_SOURCE_ITEMS:
            omitted.append({
                "path": candidate.path,
                "lines": [candidate.start_line, candidate.end_line],
                "reason": "source_item_limit",
            })
            continue
        if per_path.get(candidate.path, 0) >= MAX_ITEMS_PER_PATH:
            continue
        excerpt = _source_excerpt(
            db, repo, project, candidate, authorities, question,
        )
        if not excerpt:
            omitted.append({
                "path": candidate.path,
                "lines": [candidate.start_line, candidate.end_line],
                "reason": "not_available_in_verified_text_snapshot",
            })
            continue
        start, end = excerpt["full_range"]
        if any(
            start <= existing_end and existing_start <= end
            for existing_start, existing_end in included_ranges.get(candidate.path, [])
        ):
            continue
        result["source_bundle"].append(excerpt)
        if estimate_tokens(result) + reserve > max_context_tokens:
            result["source_bundle"].pop()
            omitted.append({
                "path": candidate.path,
                "lines": [candidate.start_line, candidate.end_line],
                "symbol": excerpt["symbol"],
                "reason": "token_budget",
                "estimated_item_tokens": estimate_tokens(excerpt),
            })
            continue
        per_path[candidate.path] = per_path.get(candidate.path, 0) + 1
        included_ranges.setdefault(candidate.path, []).append((start, end))

    if brain.repository_content_digest(repo) != snapshot["content_digest"]:
        raise RuntimeError("Repository changed while load_deep_context was building its package")
    omission_reasons: dict[str, int] = {}
    for item in omitted:
        reason = item["reason"]
        omission_reasons[reason] = omission_reasons.get(reason, 0) + 1
    result["omission_summary"] = {
        "total": len(omitted),
        "by_reason": omission_reasons,
        "representative_candidates_returned": 0,
    }
    for item in omitted:
        result["omitted_source_candidates"].append(item)
        if estimate_tokens(result) + 700 > max_context_tokens:
            result["omitted_source_candidates"].pop()
            break
    result["omission_summary"]["representative_candidates_returned"] = len(
        result["omitted_source_candidates"]
    )
    final_tokens = estimate_tokens(result)
    result["budget"] = {
        "requested_estimated_tokens": max_context_tokens,
        "base_map_estimated_tokens": base_tokens,
        "source_bundle_estimated_tokens": estimate_tokens(result["source_bundle"]),
        "total_estimated_tokens": final_tokens,
        "estimate_method": TOKEN_ESTIMATE_METHOD,
        "exact_client_tokens_available": False,
        "source_items_included": len(result["source_bundle"]),
        "source_items_omitted": len(omitted),
        "omitted_candidates_described": len(result["omitted_source_candidates"]),
        "unique_source_files": len({item["path"] for item in result["source_bundle"]}),
    }
    result["budget"]["total_estimated_tokens"] = estimate_tokens(result)
    while result["budget"]["total_estimated_tokens"] > max_context_tokens:
        if result["omitted_source_candidates"]:
            result["omitted_source_candidates"].pop()
        elif result["source_bundle"]:
            result["source_bundle"].pop()
        else:
            break
        result["omission_summary"]["representative_candidates_returned"] = len(
            result["omitted_source_candidates"]
        )
        result["budget"]["source_items_included"] = len(result["source_bundle"])
        result["budget"]["omitted_candidates_described"] = len(
            result["omitted_source_candidates"]
        )
        result["budget"]["unique_source_files"] = len({
            item["path"] for item in result["source_bundle"]
        })
        result["budget"]["source_bundle_estimated_tokens"] = estimate_tokens(
            result["source_bundle"]
        )
        result["budget"]["total_estimated_tokens"] = estimate_tokens(result)
    return result
