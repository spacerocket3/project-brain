import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import deep_context
import mcp_server as mcp
import project_brain as brain
from project_registry import ProjectRegistry


class DeepContextTests(unittest.TestCase):
    def indexed_fixture(self, root: Path) -> tuple[Path, ProjectRegistry, Path]:
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "lib").mkdir()
        (repo / "test").mkdir()
        (repo / "lib" / "adapter.js").write_text(
            "export function httpAdapter(config) {\n"
            "  if (config.data instanceof FormData) {\n"
            "    return sendMultipart(config.data);\n"
            "  }\n"
            "  return send(config.data);\n"
            "}\n",
            encoding="utf-8",
        )
        (repo / "test" / "adapter.test.js").write_text(
            "import {httpAdapter} from '../lib/adapter.js';\n"
            "describe('httpAdapter', () => {\n"
            "  it('streams standards FormData', () => httpAdapter({data: new FormData()}));\n"
            "});\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git", "-C", str(repo), "-c", "user.name=Project Brain",
                "-c", "user.email=project-brain@example.invalid", "commit", "-qm", "fixture",
            ],
            check=True,
        )
        registry = ProjectRegistry(root / "projects.json")
        registry.add("sample", repo, "fixture")
        return repo, registry, root / "index.sqlite3"

    def build_fixture_context(
        self, repo: Path, registry: ProjectRegistry, database: Path,
    ) -> dict:
        with patch.object(brain, "REPOS", registry), patch.object(
            brain, "DB_PATH", database,
        ):
            brain.index_project("sample")
            commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            hits = [
                brain.Hit(
                    "sample", "lib/adapter.js", 1, 6, commit[:12], 1.0,
                    (repo / "lib" / "adapter.js").read_text(), "code",
                ),
                brain.Hit(
                    "sample", "test/adapter.test.js", 1, 4, commit[:12], 0.8,
                    (repo / "test" / "adapter.test.js").read_text(), "test",
                ),
            ]
            flow = {
                "evidence": [],
                "causal_evidence": [{
                    "path": "lib/adapter.js", "lines": [1, 6], "score": 99,
                    "authority": "code",
                }],
                "impact_candidates": [{
                    "path": "test/adapter.test.js", "lines": [1, 4], "score": 90,
                    "authority": "test", "suggested_action": "likely_modify",
                }],
                "sql_lineage": {},
                "sql_dependency_map": {},
                "sql_reverse_dependency_map": {},
                "rpc_consumer_map": {},
                "facet_coverage": {},
                "cache_invalidation_candidates": [],
            }
            return deep_context.build_deep_context(
                project="sample",
                question="Implement standards FormData in httpAdapter and add regression tests",
                hits=hits,
                flow=flow,
                relationship_hints=[],
                anchor_relationships=[],
                contract_differences=[],
                semantic_mode="code",
                reranker_fallback=None,
                max_context_tokens=deep_context.MIN_CONTEXT_TOKENS,
            )

    def test_deep_context_loads_complete_decisive_symbols_under_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, registry, database = self.indexed_fixture(Path(temporary))
            result = self.build_fixture_context(repo, registry, database)

        self.assertRegex(result["snapshot"]["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(result["source_bundle"][0]["citation"].startswith("sample:"))
        by_path = {item["path"]: item for item in result["source_bundle"]}
        self.assertIn("lib/adapter.js", by_path)
        self.assertIn("test/adapter.test.js", by_path)
        self.assertIn("sendMultipart", by_path["lib/adapter.js"]["content"])
        self.assertTrue(by_path["lib/adapter.js"]["complete"])
        self.assertLessEqual(
            result["budget"]["total_estimated_tokens"],
            result["budget"]["requested_estimated_tokens"],
        )
        self.assertEqual(
            result["budget"]["estimate_method"], "compact_json_chars_div_4",
        )

    def test_deep_context_fails_closed_when_working_tree_is_newer_than_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, registry, database = self.indexed_fixture(root)
            with patch.object(brain, "REPOS", registry), patch.object(
                brain, "DB_PATH", database,
            ):
                brain.index_project("sample")
                (repo / "lib" / "adapter.js").write_text("export const changed = true;\n")
                with self.assertRaisesRegex(RuntimeError, "refresh_repository"):
                    deep_context.build_deep_context(
                        project="sample", question="inspect adapter", hits=[], flow={},
                        relationship_hints=[], anchor_relationships=[],
                        contract_differences=[], semantic_mode="fast",
                        reranker_fallback=None,
                    )

    def test_deep_context_follows_test_calls_to_production_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, registry, database = self.indexed_fixture(root)
            with patch.object(brain, "REPOS", registry), patch.object(
                brain, "DB_PATH", database,
            ):
                brain.index_project("sample")
                commit = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
                neighbors = deep_context.collect_structural_neighbors(
                    brain.connect(), "sample", [brain.Hit(
                        "sample", "test/adapter.test.js", 1, 4, commit[:12], 1.0,
                        (repo / "test" / "adapter.test.js").read_text(), "test",
                    )],
                    "Implement standards FormData in httpAdapter",
                )

        self.assertTrue(any(
            item.path == "lib/adapter.js"
            and "structured_neighbor_calls" in item.reasons
            for item in neighbors
        ))

    def test_candidate_fusion_prioritizes_contract_and_sql_evidence(self):
        candidates = deep_context.collect_source_candidates(
            [],
            {
                "causal_evidence": [], "impact_candidates": [], "evidence": [],
                "sql_lineage": {"rpc": [{
                    "path": "migrations/active.sql", "lines": [10, 40],
                    "kind": "definition",
                }]},
                "sql_dependency_map": {}, "sql_reverse_dependency_map": {},
                "rpc_consumer_map": {}, "facet_coverage": {},
                "cache_invalidation_candidates": [],
            },
            [], [],
            [{
                "anchor": {"path": "src/producer.ts", "lines": [1, 20]},
                "related": {"path": "src/consumer.ts", "lines": [5, 30]},
            }],
        )

        self.assertEqual(candidates[0].path, "src/consumer.ts")
        self.assertEqual(candidates[1].path, "src/producer.ts")
        self.assertTrue(any(item.path == "migrations/active.sql" for item in candidates))

    @patch("mcp_server.deep_context.build_deep_context")
    @patch("mcp_server.investigate_flow", return_value={})
    @patch("mcp_server.compact_contract_differences", return_value=[])
    @patch("mcp_server.compact_anchor_relationships", return_value=[])
    @patch("mcp_server.compact_relationship_hints", return_value=[])
    @patch("mcp_server.retrieve_search_hits", return_value=([], True, None))
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_deep_context_runs_one_composed_pipeline(
        self, _scope, _registered, _retrieve, _relationships, _anchors,
        _contracts, investigate, build,
    ):
        build.return_value = {"project": "sample", "budget": {"total_estimated_tokens": 10}}

        result = mcp.call_tool("load_deep_context", {
            "project": "sample", "question": "heavy change", "max_context_tokens": 20_000,
        })

        self.assertEqual(result["scope_note"], "fixture")
        investigate.assert_called_once()
        self.assertEqual(build.call_args.kwargs["semantic_mode"], "code")
        self.assertEqual(build.call_args.kwargs["max_context_tokens"], 20_000)

    def test_mcp_schema_exposes_explicit_heavy_context_budget(self):
        tool = next(item for item in mcp.TOOLS if item["name"] == "load_deep_context")
        properties = tool["inputSchema"]["properties"]
        self.assertEqual(properties["semantic_mode"]["default"], "code")
        self.assertEqual(
            properties["max_context_tokens"]["default"],
            deep_context.DEFAULT_CONTEXT_TOKENS,
        )


if __name__ == "__main__":
    unittest.main()
