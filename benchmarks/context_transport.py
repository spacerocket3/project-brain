#!/usr/bin/env python3
"""Measure dossier transport against blindly sending all eligible repository text.

This is a transport diagnostic, not an agent-quality benchmark. The paired agent protocol
in docs/EVALUATION.md is required for correctness claims.
"""

from __future__ import annotations

import argparse
import json
import math

import mcp_server
import project_brain as brain


def estimated_file_tokens(project: str) -> tuple[int, int, int]:
    root = brain.require_registered_project(project)
    characters = 0
    files = 0
    skipped = 0
    for relative in brain.working_tree_files(root):
        try:
            text = (root / relative).read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            skipped += 1
            continue
        characters += len(text)
        files += 1
    return files, skipped, max(1, math.ceil(characters / 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="Registered Project Brain project name")
    parser.add_argument("question", help="Complete engineering question")
    parser.add_argument(
        "--semantic-mode", choices=("fast", "auto", "code"), default="fast",
    )
    parser.add_argument("--max-context-tokens", type=int, default=36_000)
    args = parser.parse_args()

    file_count, skipped, full_tokens = estimated_file_tokens(args.project)
    dossier = mcp_server.call_tool("load_deep_context", {
        "project": args.project,
        "question": args.question,
        "semantic_mode": args.semantic_mode,
        "max_context_tokens": args.max_context_tokens,
    })
    dossier_tokens = int(dossier["budget"]["total_estimated_tokens"])
    reduction = 1 - min(dossier_tokens / full_tokens, 1.0)
    print(json.dumps({
        "comparison": "deep dossier vs all eligible repository text",
        "warning": "Transport diagnostic only; it does not measure answer correctness.",
        "project": args.project,
        "question": args.question,
        "semantic_mode": dossier["semantic_mode"],
        "eligible_files": file_count,
        "unreadable_files_skipped": skipped,
        "all_source_estimated_tokens": full_tokens,
        "dossier_estimated_tokens": dossier_tokens,
        "estimated_transport_reduction": round(reduction, 4),
        "included_source_items": dossier["budget"]["included_source_items"],
        "omitted_source_items": dossier["budget"]["omitted_source_items"],
        "estimate_method": dossier["budget"]["estimate_method"],
    }, indent=2))


if __name__ == "__main__":
    main()
