#!/usr/bin/env python3
"""MCP stdio adapter that exposes Project Brain to Codex."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback

import deep_context
import project_brain as brain
from structural_index import diagnose_retrieval, investigate_flow, related


SERVER_INSTRUCTIONS = (
    "Project Brain is a repository cartographer. It reads only explicitly registered local "
    "Git checkouts and returns compact navigation evidence; it does not replace direct file "
    "inspection, rg, or tests. A project name identifies the exact path shown by "
    "project_brain_status and never syncs another machine or worktree. Keep project scopes "
    "isolated and verify decisive claims against the checkout being edited."
)
PROJECT = {
    "type": "string",
    "description": (
        "Exactly one repository name previously authorized with the local Project Brain CLI. "
        "Use project_brain_status to list registered names and exact checkout paths."
    ),
    }
MAX_SEARCH_RESULTS = 6
MAX_HITS_PER_PATH = 2
MAX_SNIPPET_LINES = 14
MAX_SNIPPET_CHARS = 1_800
CONTRACT_CLASSIFIER_RE = re.compile(
    r"^(?:is|has|can|supports|accepts|allows|validates?)(?:[A-Z0-9_]|$)"
)
IMPLEMENTATION_INTENT_RE = re.compile(
    r"\b(?:fix|causal|implement(?:ation)?|implementaci[oó]n|"
    r"modif(?:y|ication|icar|icaci[oó]n)|production code|c[oó]digo productivo)\b",
    re.IGNORECASE,
)
TEST_INTENT_RE = re.compile(r"\b(?:tests?|pruebas?|regressions?)\b", re.IGNORECASE)
IMPLEMENTATION_AUTHORITIES = {"code", "schema_history"}


def requests_mixed_implementation_and_tests(question: str) -> bool:
    return bool(
        IMPLEMENTATION_INTENT_RE.search(question)
        and TEST_INTENT_RE.search(question)
    )


def _query_terms(question: str) -> list[str]:
    return brain.query_terms(question)


def _hit_match_score(hit: brain.Hit, terms: list[str]) -> int:
    haystack = f"{hit.path}\n{hit.content}".lower()
    return sum(min(haystack.count(term), 4) for term in terms)


def _compact_hit(hit: brain.Hit, terms: list[str]) -> dict:
    lines = hit.content.splitlines()
    if not lines:
        offset = 0
        chosen = []
    else:
        scores = [
            sum(1 for term in terms if term in line.lower()) for line in lines
        ]
        center = max(range(len(lines)), key=lambda index: scores[index]) if terms else 0
        offset = max(0, min(center - MAX_SNIPPET_LINES // 2, len(lines) - MAX_SNIPPET_LINES))
        chosen = lines[offset:offset + MAX_SNIPPET_LINES]
    content = "\n".join(chosen)
    if len(content) > MAX_SNIPPET_CHARS:
        content = content[:MAX_SNIPPET_CHARS].rsplit("\n", 1)[0]
        chosen = content.splitlines()
    start = hit.start_line + offset
    end = start + max(0, len(chosen) - 1)
    return {
        "path": hit.path,
        "lines": [start, end],
        "indexed_range": [hit.start_line, hit.end_line],
        "citation": f"{hit.project}:{hit.path}:{start}-{end}",
        "commit": hit.commit,
        "authority": hit.authority,
        "score": hit.score,
        "content": content,
    }


def compact_search_response(
    question: str, hits: list[brain.Hit], limit: int = MAX_SEARCH_RESULTS,
    relationship_hints: list[dict] | None = None,
    anchor_relationships: list[dict] | None = None,
    contract_differences: list[dict] | None = None,
) -> dict:
    terms = _query_terms(question)
    selected: list[brain.Hit] = []
    path_counts: dict[str, int] = {}
    ranges: dict[str, list[tuple[int, int]]] = {}
    selected_citations: set[str] = set()
    output_limit = max(1, min(limit, MAX_SEARCH_RESULTS))

    def append_hit(hit: brain.Hit, per_path_limit: int = 1) -> bool:
        if hit.citation in selected_citations:
            return False
        if path_counts.get(hit.path, 0) >= per_path_limit:
            return False
        if any(
            hit.start_line <= end and start <= hit.end_line
            for start, end in ranges.get(hit.path, [])
        ):
            return False
        selected.append(hit)
        selected_citations.add(hit.citation)
        path_counts[hit.path] = path_counts.get(hit.path, 0) + 1
        ranges.setdefault(hit.path, []).append((hit.start_line, hit.end_line))
        return True

    # When the user explicitly asks for both implementation and regression tests,
    # preserve both evidence classes. Test suites often repeat the issue vocabulary
    # much more densely than production code and can otherwise occupy every compact
    # result even when implementation candidates exist deeper in the fused ranking.
    if (
        output_limit >= 2
        and requests_mixed_implementation_and_tests(question)
    ):
        implementation_quota = min(max(1, output_limit // 2), output_limit - 1)
        for hit in hits:
            if hit.authority in IMPLEMENTATION_AUTHORITIES and append_hit(hit):
                implementation_quota -= 1
                if implementation_quota == 0:
                    break
        for hit in hits:
            if hit.authority == "test" and append_hit(hit):
                break

    # `brain.search` already fused semantic, lexical, authority and intent
    # signals. First take one anchor per file, then fill with a second
    # non-overlapping range only when the result would otherwise be sparse.
    for per_path_limit in (1, MAX_HITS_PER_PATH):
        for hit in hits:
            if not append_hit(hit, per_path_limit):
                continue
            if len(selected) >= output_limit:
                break
        if len(selected) >= output_limit:
            break
    rank_by_citation = {hit.citation: rank for rank, hit in enumerate(hits)}
    selected.sort(key=lambda hit: rank_by_citation[hit.citation])
    compact_hits = [_compact_hit(hit, terms) for hit in selected]
    return {
        "query": question,
        "hits": compact_hits,
        "summary": {
            "candidates": len(hits),
            "returned": len(compact_hits),
            "unique_files": len({hit["path"] for hit in compact_hits}),
            "snippet_line_limit": MAX_SNIPPET_LINES,
        },
        "relationship_hints": relationship_hints or [],
        "anchor_relationships": anchor_relationships or [],
        "contract_differences": contract_differences or [],
        "guidance": (
            "Navigation anchors only. Trace the producer/transformer/consumer contract, "
            "including adjacent variants handled by the same validation branch. Contract "
            "differences are structural review candidates, not proven bugs. Open current "
            "files or inspect a named symbol before making a decisive claim."
        ),
    }


def retrieve_search_hits(
    question: str, project: str, semantic_mode: str,
) -> tuple[list[brain.Hit], bool, str | None]:
    """Run the same bounded first stage and optional code reranker for every MCP path."""
    use_code_reranker = brain.should_code_rerank(question, semantic_mode)
    retrieval_limit = 60 if (
        use_code_reranker or requests_mixed_implementation_and_tests(question)
    ) else 24
    hits = brain.search(question, project, retrieval_limit)
    reranker_fallback = None
    if use_code_reranker:
        try:
            hits = brain.rerank_code_hits(question, hits)
        except RuntimeError as exc:
            if semantic_mode == "code":
                raise
            use_code_reranker = False
            reranker_fallback = str(exc)
    return hits, use_code_reranker, reranker_fallback


def deep_seed_paths(hits: list[brain.Hit], limit: int = 14) -> list[str]:
    """Keep broad implementation/test coverage for the one-shot deep flow map."""
    selected: list[str] = []
    for authorities, quota in ((IMPLEMENTATION_AUTHORITIES, 8), ({"test"}, 3)):
        for hit in hits:
            if hit.authority not in authorities or hit.path in selected:
                continue
            selected.append(hit.path)
            quota -= 1
            if quota == 0:
                break
    for hit in hits:
        if hit.path not in selected:
            selected.append(hit.path)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _symbol_covering_range(
    db, project: str, path: str, start_line: int, end_line: int,
):
    return db.execute(
        """SELECT kind,name,qualified_name,start_line,end_line FROM symbols
           WHERE project=? AND path=? AND active=1
             AND start_line<=? AND end_line>=?
           ORDER BY CASE kind WHEN 'function' THEN 0 WHEN 'method' THEN 1 ELSE 2 END,
                    (MIN(end_line,?)-MAX(start_line,?)) DESC,
                    (end_line-start_line),start_line LIMIT 1""",
        (project, path, end_line, start_line, end_line, start_line),
    ).fetchone()


def _direct_resolved_calls(db, project: str, path: str, symbol) -> list[dict]:
    calls = db.execute(
        """SELECT target_name,target_path,resolution_confidence FROM edges
           WHERE project=? AND source_path=? AND source_name=? AND relation='calls'
             AND line BETWEEN ? AND ? AND target_path IS NOT NULL
             AND resolution_confidence!='unresolved'
           ORDER BY line LIMIT 80""",
        (
            project, path, symbol["name"],
            symbol["start_line"], symbol["end_line"],
        ),
    ).fetchall()
    result = []
    seen = set()
    for call in calls:
        key = (call["target_name"], call["target_path"])
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "name": call["target_name"],
            "path": call["target_path"],
            "confidence": call["resolution_confidence"],
        })
    return result


def compact_anchor_relationships(hits: list[brain.Hit], limit: int = 3) -> list[dict]:
    """Summarize direct resolved calls for a few retrieved implementation symbols."""
    if not hits or limit <= 0:
        return []
    db = brain.connect()
    result = []
    seen = set()
    for hit in hits[:12]:
        if hit.authority != "code":
            continue
        symbol = _symbol_covering_range(
            db, hit.project, hit.path, hit.start_line, hit.end_line,
        )
        if not symbol:
            continue
        key = (hit.path, symbol["qualified_name"])
        if key in seen:
            continue
        direct_calls = _direct_resolved_calls(
            db, hit.project, hit.path, symbol,
        )[:12]
        if len(direct_calls) < 2:
            continue
        seen.add(key)
        result.append({
            "path": hit.path,
            "symbol": symbol["qualified_name"],
            "lines": [symbol["start_line"], symbol["end_line"]],
            "direct_calls": direct_calls,
        })
        if len(result) >= limit:
            break
    return result


def _contract_classifiers(calls: list[dict]) -> dict[str, dict]:
    """Return predicate-like calls that commonly encode accepted variants."""
    result = {}
    for call in calls:
        name = str(call.get("name") or "")
        if CONTRACT_CLASSIFIER_RE.match(name):
            result.setdefault(name.casefold(), call)
    return result


def compact_contract_differences(
    hits: list[brain.Hit], relationship_hints: list[dict], limit: int = 3,
) -> list[dict]:
    """Compare predicate sets across structurally related implementation stages.

    The result is intentionally labelled as a review candidate. Different layers
    often have legitimate classifier differences; the purpose is to prevent an
    adjacent accepted variant from disappearing silently during implementation.
    """
    if not hits or not relationship_hints or limit <= 0:
        return []
    db = brain.connect()
    anchors_by_path: dict[str, brain.Hit] = {}
    for hit in hits[:24]:
        anchors_by_path.setdefault(hit.path, hit)
    result = []
    seen_pairs = set()
    for hint in relationship_hints:
        anchor = anchors_by_path.get(str(hint.get("from_anchor") or ""))
        related_path = str(hint.get("path") or "")
        related_lines = hint.get("indexed_range") or hint.get("lines") or []
        if not anchor or not related_path or len(related_lines) != 2:
            continue
        pair_key = (anchor.path, related_path)
        if pair_key in seen_pairs:
            continue
        anchor_symbol = _symbol_covering_range(
            db, anchor.project, anchor.path, anchor.start_line, anchor.end_line,
        )
        related_symbol = _symbol_covering_range(
            db, anchor.project, related_path,
            int(related_lines[0]), int(related_lines[1]),
        )
        if not anchor_symbol or not related_symbol:
            continue
        anchor_classifiers = _contract_classifiers(_direct_resolved_calls(
            db, anchor.project, anchor.path, anchor_symbol,
        ))
        related_classifiers = _contract_classifiers(_direct_resolved_calls(
            db, anchor.project, related_path, related_symbol,
        ))
        shared = set(anchor_classifiers) & set(related_classifiers)
        only_anchor = set(anchor_classifiers) - set(related_classifiers)
        only_related = set(related_classifiers) - set(anchor_classifiers)
        if not shared or (not only_anchor and not only_related):
            continue
        seen_pairs.add(pair_key)
        result.append({
            "kind": "possible_contract_difference",
            "status": "review_candidate_not_proven_bug",
            "shared_classifier": hint.get("via_symbol"),
            "anchor": {
                "path": anchor.path,
                "symbol": anchor_symbol["qualified_name"],
                "lines": [anchor_symbol["start_line"], anchor_symbol["end_line"]],
            },
            "related": {
                "path": related_path,
                "symbol": related_symbol["qualified_name"],
                "lines": [related_symbol["start_line"], related_symbol["end_line"]],
            },
            "shared": [anchor_classifiers[key]["name"] for key in sorted(shared)],
            "only_in_anchor": [
                anchor_classifiers[key]["name"] for key in sorted(only_anchor)
            ][:12],
            "only_in_related": [
                related_classifiers[key]["name"] for key in sorted(only_related)
            ][:12],
            "interpretation": (
                "Check whether variants accepted or classified in one stage must remain "
                "supported in the related stage; asymmetry alone does not prove a defect."
            ),
        })
        if len(result) >= limit:
            break
    return result


def compact_relationship_hints(
    question: str, hits: list[brain.Hit], limit: int = 2,
) -> list[dict]:
    """Expose bounded cross-file callers of query-relevant callees in top anchors."""
    if not hits or limit <= 0:
        return []
    terms = {
        re.sub(r"[^a-z0-9]", "", term.casefold())
        for term in _query_terms(question)
        if len(re.sub(r"[^a-z0-9]", "", term.casefold())) >= 4
    }
    if not terms:
        return []
    db = brain.connect()
    candidates: dict[str, tuple[brain.Hit, str, str, int]] = {}
    for anchor in hits[:24]:
        anchor_symbol = _symbol_covering_range(
            db, anchor.project, anchor.path, anchor.start_line, anchor.end_line,
        ) if anchor.authority == "code" else None
        edge_start = anchor_symbol["start_line"] if anchor_symbol else anchor.start_line
        edge_end = anchor_symbol["end_line"] if anchor_symbol else anchor.end_line
        edges = db.execute(
            """SELECT target_name,target_path,relation FROM edges
               WHERE project=? AND source_path=? AND line BETWEEN ? AND ?
                 AND relation='calls' AND target_path IS NOT NULL
                 AND resolution_confidence!='unresolved'
               ORDER BY line LIMIT 40""",
            (anchor.project, anchor.path, edge_start, edge_end),
        ).fetchall()
        for edge in edges:
            normalized = re.sub(r"[^a-z0-9]", "", str(edge["target_name"]).casefold())
            if not normalized or not any(
                term in normalized or normalized in term for term in terms
            ):
                continue
            incoming = db.execute(
                """SELECT e.source_path,e.source_name,e.line,c.start_line,c.end_line,
                          c.commit_hash,c.content,a.authority
                   FROM edges e
                   JOIN chunks c ON c.project=e.project AND c.path=e.source_path
                     AND e.line BETWEEN c.start_line AND c.end_line
                   JOIN source_authority a ON a.project=c.project AND a.path=c.path
                   WHERE e.project=? AND e.relation='calls' AND e.target_name=?
                     AND e.target_path=? AND e.resolution_confidence!='unresolved'
                   ORDER BY CASE a.authority WHEN 'code' THEN 0 WHEN 'test' THEN 1 ELSE 2 END,
                            e.source_path,e.line
                   LIMIT 30""",
                (anchor.project, edge["target_name"], edge["target_path"]),
            ).fetchall()
            for row in incoming:
                if row["source_path"] == anchor.path:
                    continue
                related_hit = brain.Hit(
                    anchor.project, row["source_path"], row["start_line"], row["end_line"],
                    row["commit_hash"][:12], anchor.score, row["content"], row["authority"],
                )
                citation = related_hit.citation
                source_name = str(row["source_name"] or "")
                name_score = sum(term in source_name.casefold() for term in terms)
                match_score = _hit_match_score(related_hit, list(terms))
                authority_score = 2 if row["authority"] == "code" else 1
                path_lower = str(row["source_path"]).casefold()
                role_score = 0
                if terms & {"dispatch", "dispatched", "request", "requests", "send"}:
                    role_score += 40 if "/adapters/" in path_lower else 0
                if "node" in terms and (
                    "/node/" in path_lower or path_lower.endswith("/http.js")
                ):
                    role_score += 30
                anchor_score = 30 if anchor.authority == "code" else 0
                anchor_score += min(_hit_match_score(anchor, list(terms)), 20)
                candidate_score = (
                    authority_score * 100 + role_score + anchor_score
                    + name_score * 10 + match_score
                )
                previous = candidates.get(citation)
                if previous is None or candidate_score > previous[3]:
                    candidates[citation] = (
                        related_hit, str(edge["target_name"]), anchor.path, candidate_score,
                    )
    selected = []
    selected_paths = set()
    for candidate in sorted(candidates.values(), key=lambda item: (-item[3], item[0].path)):
        if candidate[0].path in selected_paths:
            continue
        selected.append(candidate)
        selected_paths.add(candidate[0].path)
        if len(selected) >= limit:
            break
    result = []
    for hit, via_symbol, anchor_path, _score in selected:
        result.append({
            **_compact_hit(hit, list(terms)),
            "relation": "shares_query_relevant_callee",
            "via_symbol": via_symbol,
            "from_anchor": anchor_path,
        })
    return result


def compact_related_response(result: dict, name: str, limit: int) -> dict:
    needle = name.casefold()
    symbols = result.get("symbols", [])
    qualified_exact = [
        item for item in symbols
        if str(item.get("qualified_name", "")).casefold() == needle
    ]
    name_exact = [
        item for item in symbols
        if str(item.get("name", "")).casefold() == needle
    ]
    primary = qualified_exact or name_exact or symbols[:1]
    definitions = []
    for item in primary[:6]:
        definitions.append({
            "path": item["path"], "kind": item["kind"],
            "name": item["name"], "qualified_name": item["qualified_name"],
            "lines": [item["start_line"], item["end_line"]],
            "signature": str(item.get("signature") or "")[:500],
            "active": bool(item.get("active", 1)),
            "authority": item.get("authority") or "unknown",
            "citation": (
                f"{item['project']}:{item['path']}:"
                f"{item['start_line']}-{item['end_line']}"
            ),
        })
    alternatives = []
    primary_keys = {
        (item["path"], item["qualified_name"], item["start_line"])
        for item in primary
    }
    for item in symbols:
        key = (item["path"], item["qualified_name"], item["start_line"])
        if key in primary_keys:
            continue
        alternatives.append({
            "path": item["path"], "kind": item["kind"],
            "qualified_name": item["qualified_name"],
            "lines": [item["start_line"], item["end_line"]],
            "active": bool(item.get("active", 1)),
        })
        if len(alternatives) >= 3:
            break
    if not definitions:
        alternatives = [{
            "path": item["path"], "kind": item["kind"],
            "name": item["name"], "qualified_name": item["qualified_name"],
            "lines": [item["start_line"], item["end_line"]],
            "active": bool(item.get("active", 1)),
            "match_score": item.get("match_score"),
        } for item in result.get("suggestions", [])]

    primary_paths = {item["path"] for item in primary}
    relevant_edges = []
    for edge in result.get("edges", []):
        source_exact = (
            str(edge.get("source_name", "")).casefold() == needle
            or str(edge.get("source_qualified_name", "")).casefold() == needle
        )
        target_exact = (
            str(edge.get("target_name", "")).casefold() == needle
            or str(edge.get("target_qualified_name", "")).casefold() == needle
        )
        if (qualified_exact or name_exact) and not (
            (source_exact and edge.get("source_path") in primary_paths)
            or (target_exact and (edge.get("target_path") in primary_paths or not edge.get("target_path")))
        ):
            continue
        relevant_edges.append((edge, source_exact, target_exact))

    direct_calls = []
    seen_calls = set()
    for edge, source_exact, _target_exact in relevant_edges:
        if not source_exact or edge.get("relation") != "calls":
            continue
        call_name = edge.get("target_qualified_name") or edge.get("target_name")
        key = (call_name, edge.get("target_path"))
        if not call_name or key in seen_calls:
            continue
        seen_calls.add(key)
        direct_calls.append({
            "name": call_name,
            "path": edge.get("target_path"),
            "confidence": edge.get("resolution_confidence") or "unknown",
        })
        if len(direct_calls) >= 24:
            break

    relationships = []
    seen = set()
    for edge, source_exact, target_exact in relevant_edges:
        direction = "outgoing" if source_exact else "incoming" if target_exact else "related"
        key = (
            direction, edge.get("relation"), edge.get("source_path"),
            edge.get("source_qualified_name"), edge.get("target_path"),
            edge.get("target_qualified_name") or edge.get("target_name"),
        )
        if key in seen:
            continue
        seen.add(key)
        relationships.append({
            "direction": direction,
            "relation": edge.get("relation"),
            "source": {
                "path": edge.get("source_path"),
                "qualified_name": edge.get("source_qualified_name") or edge.get("source_name"),
                "line": edge.get("line"),
            },
            "target": {
                "path": edge.get("target_path"),
                "qualified_name": edge.get("target_qualified_name") or edge.get("target_name"),
            },
            "confidence": edge.get("resolution_confidence") or "unknown",
        })
        if len(relationships) >= limit:
            break
    return {
        "query": name,
        "path_filter": result.get("path_filter"),
        "definitions": definitions,
        "alternatives": alternatives,
        "direct_calls": direct_calls,
        "relationships": relationships,
        "summary": {
            "symbol_candidates": len(symbols),
            "suggestions_returned": len(result.get("suggestions", [])),
            "edge_candidates": len(result.get("edges", [])),
            "relationships_returned": len(relationships),
        },
        "guidance": (
            "Definitions and graph links are navigation evidence. Open current source "
            "and tests before asserting runtime behavior."
        ),
    }
TOOLS = [
    {
        "name": "project_brain_status",
        "description": "Check index freshness, counts, and active local models.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "search_repository",
        "description": (
            "Return a small set of cited repository navigation anchors: at most six "
            "short snippets, plus bounded relationship hints. Auto mode reuses an already "
            "cached code-aware reranker for complex questions; choose fast for low latency or "
            "code to require the deeper pass. Open current files before decisive claims."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 12},
                "semantic_mode": {
                    "type": "string", "enum": ["auto", "fast", "code"], "default": "auto",
                },
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "inspect_symbol",
        "description": (
            "Find exact definitions first and return compact direct callers/callees for a "
            "named symbol. Use path to disambiguate homonyms. SQL active=0 is superseded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "name": {"type": "string", "minLength": 1},
                "path": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 12},
            },
            "required": ["project", "name"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "load_deep_context",
        "description": (
            "Load one large, token-budgeted repository dossier for heavy development. "
            "Runs hybrid/code retrieval, structural flow expansion, relationship and "
            "contract checks, then includes exact complete source bodies when they fit. "
            "Reports the verified working-tree generation, estimated transport tokens and "
            "omitted evidence. It does not invoke Nemotron or Gemini."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "max_context_tokens": {
                    "type": "integer", "minimum": deep_context.MIN_CONTEXT_TOKENS,
                    "maximum": deep_context.MAX_CONTEXT_TOKENS,
                    "default": deep_context.DEFAULT_CONTEXT_TOKENS,
                    "description": (
                        "Estimated MCP result budget. Exact Codex tokenization is unavailable "
                        "inside the server; the response reports the estimation method."
                    ),
                },
                "semantic_mode": {
                    "type": "string", "enum": ["auto", "fast", "code"], "default": "code",
                },
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "investigate_flow",
        "description": (
            "Start from hybrid search anchors and expand a bounded structural evidence map "
            "across implementation, persistence, retries, scheduling, constraints, and tests. "
            "Separates causal evidence from UI/tests/cache/types and other impact candidates. "
            "Returns provenance and missing coverage; it does not claim a proven causal trace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 6, "maximum": 30, "default": 16},
                "semantic_mode": {
                    "type": "string", "enum": ["auto", "fast", "code"], "default": "auto",
                },
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "diagnose_retrieval",
        "description": (
            "Run the same flow investigation and diagnose the shape of its evidence: lane and "
            "role concentration, lexical dependence, SQL lineage, consumer fan-out, cache gaps, "
            "and thin nominal coverage. Returns transparent counts and warnings, not a 1-10 score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 6, "maximum": 30, "default": 16},
                "semantic_mode": {
                    "type": "string", "enum": ["auto", "fast", "code"], "default": "auto",
                },
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "consult_nemotron",
        "description": (
            "Run the exhaustive graph-guided Nemotron loop. It accounts for 100% of the "
            "frozen graph, progressively opens complete code/SQL/tests, preserves "
            "contradictions and reports why it stopped. Use for difficult architecture, "
            "implementation, invariant, or calculation questions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 10},
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "consult_gemini",
        "description": (
            "Run the same exhaustive graph-guided snapshot and evidence protocol with "
            "Gemini. Use to benchmark the loop against Nemotron or when the 1M-token "
            "context materially improves a difficult repository-wide investigation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 10},
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "consult_nemotron_fast",
        "description": (
            "Fast top-k Nemotron orientation. It is intentionally non-exhaustive; use only "
            "when low latency matters more than repository-wide coverage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "consult_nemotron_max",
        "description": (
            "Compatibility alias for consult_nemotron. Nemotron stays hot; no model switching."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "question": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 10},
            },
            "required": ["project", "question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "refresh_repository",
        "description": (
            "Refresh text, authority, symbols, and dependencies from the current "
            "registered working tree. It cannot read or synchronize a different checkout. "
            "Updates only the local index and never edits the registered repository. Includes tracked "
            "changes and untracked non-ignored source files without requiring a commit. "
            "Does not start the embedding encoder; reports semantic debt separately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"project": PROJECT},
            "required": ["project"],
            "additionalProperties": False,
        },
    },
    {
        "name": "refresh_embeddings",
        "description": (
            "Embed only missing chunks for one project after semantic debt accumulates, a "
            "large refactor, or at task/session completion. This may start the encoder and is "
            "intentionally separate from the fast structural refresh."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": PROJECT,
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
            },
            "required": ["project"],
            "additionalProperties": False,
        },
    },
]

# Generation-model integrations were useful during development, but they distract from the
# public product's purpose: selecting repository evidence for an agent that already has a
# model. Keep that experimental surface opt-in and expose only cartography by default.
if os.environ.get("PROJECT_BRAIN_ENABLE_GENERATION_TOOLS", "").casefold() not in {
    "1", "true", "yes",
}:
    TOOLS = [tool for tool in TOOLS if not tool["name"].startswith("consult_")]


def require_project(arguments: dict) -> str:
    project = arguments.get("project")
    if not isinstance(project, str):
        raise ValueError("project must be a registered project name")
    brain.require_registered_project(project)
    return project


def scope_note(project: str) -> str:
    return brain.REPOS.note(project)


def call_tool(name: str, arguments: dict) -> object:
    if name == "project_brain_status":
        return brain.status() | {
            "scope_notes": {project: brain.REPOS.note(project) for project in brain.REPOS},
        }
    project = require_project(arguments)
    if name == "search_repository":
        output_limit = max(
            1, min(int(arguments.get("limit", 12)), MAX_SEARCH_RESULTS),
        )
        semantic_mode = arguments.get("semantic_mode", "auto")
        hits, use_code_reranker, reranker_fallback = retrieve_search_hits(
            arguments["question"], project, semantic_mode,
        )
        relationship_hints = compact_relationship_hints(
            arguments["question"], hits, limit=2,
        )
        anchor_relationships = compact_anchor_relationships(hits, limit=3)
        contract_differences = compact_contract_differences(
            hits, relationship_hints, limit=3,
        )
        compact = compact_search_response(
            arguments["question"], hits, output_limit, relationship_hints,
            anchor_relationships, contract_differences,
        )
        compact["summary"]["semantic_mode"] = (
            "code" if use_code_reranker else "fast"
        )
        if reranker_fallback:
            compact["summary"]["reranker_fallback"] = reranker_fallback
        return {
            "project": project,
            "scope_note": scope_note(project),
            **compact,
        }
    if name == "inspect_symbol":
        output_limit = int(arguments.get("limit", 12))
        result = related(
            brain.connect(), project, arguments["name"],
            max(60, output_limit * 6), arguments.get("path"),
        )
        return {
            "project": project,
            "scope_note": scope_note(project),
            **compact_related_response(result, arguments["name"], output_limit),
        }
    if name == "load_deep_context":
        question = arguments["question"]
        semantic_mode = arguments.get("semantic_mode", "code")
        hits, use_code_reranker, reranker_fallback = retrieve_search_hits(
            question, project, semantic_mode,
        )
        relationship_hints = compact_relationship_hints(question, hits, limit=12)
        anchor_relationships = compact_anchor_relationships(hits, limit=12)
        contract_differences = compact_contract_differences(
            hits, relationship_hints, limit=12,
        )
        flow = investigate_flow(
            brain.connect(), project, deep_seed_paths(hits), 30, question,
        )
        context = deep_context.build_deep_context(
            project=project,
            question=question,
            hits=hits,
            flow=flow,
            relationship_hints=relationship_hints,
            anchor_relationships=anchor_relationships,
            contract_differences=contract_differences,
            semantic_mode="code" if use_code_reranker else "fast",
            reranker_fallback=reranker_fallback,
            max_context_tokens=int(arguments.get(
                "max_context_tokens", deep_context.DEFAULT_CONTEXT_TOKENS,
            )),
        )
        return {"scope_note": scope_note(project), **context}
    if name == "investigate_flow":
        output_limit = int(arguments.get("limit", 16))
        semantic_mode = arguments.get("semantic_mode", "auto")
        hits, use_code_reranker, reranker_fallback = retrieve_search_hits(
            arguments["question"], project, semantic_mode,
        )
        anchors = compact_search_response(arguments["question"], hits)
        anchors["summary"]["semantic_mode"] = (
            "code" if use_code_reranker else "fast"
        )
        if reranker_fallback:
            anchors["summary"]["reranker_fallback"] = reranker_fallback
        seed_paths = list(dict.fromkeys(hit["path"] for hit in anchors["hits"]))
        return {
            "project": project,
            "scope_note": scope_note(project),
            "question": arguments["question"],
            "anchors": anchors,
            "flow": investigate_flow(
                brain.connect(), project, seed_paths, output_limit, arguments["question"],
            ),
        }
    if name == "diagnose_retrieval":
        output_limit = int(arguments.get("limit", 16))
        semantic_mode = arguments.get("semantic_mode", "auto")
        hits, use_code_reranker, reranker_fallback = retrieve_search_hits(
            arguments["question"], project, semantic_mode,
        )
        anchors = compact_search_response(arguments["question"], hits)
        anchors["summary"]["semantic_mode"] = (
            "code" if use_code_reranker else "fast"
        )
        if reranker_fallback:
            anchors["summary"]["reranker_fallback"] = reranker_fallback
        seed_paths = list(dict.fromkeys(hit["path"] for hit in anchors["hits"]))
        flow = investigate_flow(
            brain.connect(), project, seed_paths, output_limit, arguments["question"],
        )
        return {
            "project": project,
            "scope_note": scope_note(project),
            "question": arguments["question"],
            "anchors": anchors,
            "diagnostic": flow["retrieval_diagnostic"],
            "flow_summary": {
                key: flow[key] for key in (
                    "lanes_present", "lanes_missing", "roles_present", "roles_missing",
                    "rpc_consumer_map", "sql_lineage", "potentially_uninvalidated_query_keys",
                    "sql_dependency_map", "facet_coverage",
                    "sql_reverse_dependency_map",
                )
            },
        }
    if name == "consult_nemotron":
        return {
            "project": project,
            "scope_note": scope_note(project),
            **brain.ask(arguments["question"], project, int(arguments.get("limit", 10))),
        }
    if name == "consult_gemini":
        return {
            "project": project,
            "scope_note": scope_note(project),
            **brain.ask_gemini(
                arguments["question"], project, int(arguments.get("limit", 10)),
            ),
        }
    if name == "consult_nemotron_fast":
        return {
            "project": project,
            "scope_note": scope_note(project),
            **brain.ask_fast(arguments["question"], project, int(arguments.get("limit", 10))),
        }
    if name == "consult_nemotron_max":
        return {
            "project": project,
            "scope_note": scope_note(project),
            **brain.ask_max(arguments["question"], project, int(arguments.get("limit", 10))),
        }
    if name == "refresh_repository":
        return brain.index_project(project, force=True)
    if name == "refresh_embeddings":
        return brain.embed_projects(project, int(arguments.get("batch_size", 8)))
    raise ValueError(f"Unknown tool: {name}")


def tool_result(value: object, is_error: bool = False) -> dict:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def respond(message_id: object, result: object = None, error: dict | None = None) -> None:
    message = {"jsonrpc": "2.0", "id": message_id}
    message["error" if error else "result"] = error if error else result
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle(message: dict) -> None:
    method = message.get("method")
    message_id = message.get("id")
    if method == "initialize":
        protocol = message.get("params", {}).get("protocolVersion", "2025-06-18")
        respond(
            message_id,
            {
                "protocolVersion": protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "project-brain", "version": "1.0.0"},
                "instructions": SERVER_INSTRUCTIONS,
            },
        )
    elif method == "tools/list":
        respond(message_id, {"tools": TOOLS})
    elif method == "tools/call":
        params = message.get("params", {})
        try:
            respond(
                message_id,
                tool_result(call_tool(params.get("name", ""), params.get("arguments") or {})),
            )
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            respond(message_id, tool_result(str(exc), is_error=True))
    elif method == "ping":
        respond(message_id, {})
    elif message_id is not None:
        respond(message_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            handle(json.loads(line))
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            respond(None, error={"code": -32700, "message": str(exc)})


if __name__ == "__main__":
    main()
