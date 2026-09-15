import tempfile
import io
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import project_brain as brain
import structural_index as structure
import sqlite3
import numpy as np
import vector_search as vectors
import exhaustive_loop as loop
import mcp_server as mcp
from project_registry import ProjectRegistry


class ProjectRegistryTests(unittest.TestCase):
    def make_repo(self, root: Path, name: str = "repository") -> Path:
        repo = root / name
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        return repo

    def test_registry_adds_any_git_root_and_reloads_dynamically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self.make_repo(root)
            registry_path = root / "projects.json"
            reader = ProjectRegistry(registry_path)
            writer = ProjectRegistry(registry_path)

            added = writer.add("example", repo, "Public example checkout")

            self.assertEqual(added["project"], "example")
            self.assertEqual(reader["example"], repo.resolve())
            self.assertEqual(reader.note("example"), "Public example checkout")

    def test_registry_rejects_invalid_names_subdirectories_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self.make_repo(root)
            nested = repo / "src"
            nested.mkdir()
            registry = ProjectRegistry(root / "projects.json")

            with self.assertRaisesRegex(ValueError, "Project names"):
                registry.add("Not Valid", repo)
            with self.assertRaisesRegex(ValueError, "repository root"):
                registry.add("nested", nested)
            registry.add("example", repo)
            with self.assertRaisesRegex(ValueError, "path already registered"):
                registry.add("duplicate", repo)

    def test_unregister_purges_only_memory_and_preserves_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self.make_repo(root)
            registry = ProjectRegistry(root / "projects.json")
            registry.add("example", repo)
            database = root / "index.sqlite3"
            with patch.object(brain, "REPOS", registry), patch.object(
                brain, "DB_PATH", database,
            ):
                db = brain.connect()
                db.execute(
                    "INSERT INTO repos(project,root,commit_hash) VALUES(?,?,?)",
                    ("example", str(repo), "abc"),
                )
                db.execute(
                    "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,content) "
                    "VALUES('example','README.md',1,1,'abc','hello')"
                )
                db.commit()

                result = brain.unregister_project("example")

                self.assertFalse(result["checkout_deleted"])
                self.assertTrue(repo.is_dir())
                self.assertNotIn("example", registry)
                self.assertEqual(
                    db.execute(
                        "SELECT count(*) FROM chunks WHERE project='example'"
                    ).fetchone()[0],
                    0,
                )

    def test_mcp_project_argument_is_dynamic_not_a_compiled_enum(self):
        self.assertNotIn("enum", mcp.PROJECT)
        self.assertIn("previously authorized", mcp.PROJECT["description"])


class ProjectBrainTests(unittest.TestCase):
    def test_chunks_keep_line_numbers_and_overlap(self):
        text = "\n".join(f"line {n}" for n in range(1, 121))
        chunks = list(brain.chunks_for(text, size=50, overlap=10))
        self.assertEqual((chunks[0][0], chunks[0][1]), (1, 50))
        self.assertEqual((chunks[1][0], chunks[1][1]), (41, 90))
        self.assertEqual((chunks[2][0], chunks[2][1]), (81, 120))

    def test_fts_query_is_safe_and_useful(self):
        query = brain.fts_query('¿Cómo se calcula el saldo de vacaciones?')
        self.assertNotIn('?', query)
        self.assertIn('"vacation"', query)
        self.assertIn('"balance"', query)

    def test_fts_query_expands_partial_external_failure_concepts(self):
        query = brain.fts_query(
            "El proveedor recibió la operación pero falló la conexión antes de responder"
        )
        self.assertIn('"reconcile"', query)
        self.assertIn('"idempotency"', query)
        self.assertIn('"command"', query)

    def test_fts_query_preserves_late_code_terms_in_long_agent_question(self):
        question = " ".join(
            [f"generic{index}" for index in range(55)]
            + ["FormData", "Blob", "ArrayBuffer", "Content-Length"]
        )
        query = brain.fts_query(question)
        self.assertIn('"formdata"', query)
        self.assertIn('"blob"', query)
        self.assertIn('"arraybuffer"', query)
        self.assertIn('"content-length"', query)

    def test_operational_intent_prefers_backend_command_over_marketing(self):
        question = "el proveedor falló después de recibir la operación"
        backend = brain.Hit(
            "sample", "functions/retry-commands/index.ts", 1, 2,
            "abc", 1.0, "reconcile command error", "code",
        )
        marketing = brain.Hit(
            "sample", "src/data/competitors.ts", 1, 2,
            "abc", 1.0, "external provider error", "code",
        )
        self.assertGreater(
            brain.retrieval_multiplier(question, backend),
            brain.retrieval_multiplier(question, marketing),
        )

    def test_secret_paths_are_excluded(self):
        self.assertTrue(brain.SECRET_RE.search(".env"))
        self.assertTrue(brain.SECRET_RE.search("config/private.key"))
        self.assertFalse(brain.SECRET_RE.search("src/config.ts"))

    @patch("project_brain.semantic_rows", return_value=[])
    @patch("project_brain.keyword_search")
    def test_search_uses_code_without_embeddings(self, keyword, semantic):
        keyword.return_value = [
            brain.Hit("workforce", "src/a.ts", 1, 4, "abc", 2.0, "vacation balance", "code")
        ]
        with patch("project_brain.connect") as connect:
            db = connect.return_value
            db.execute.return_value.fetchone.return_value = {"authority": "code"}
            hits = brain.search("saldo de vacaciones", "workforce", 5)

        self.assertEqual({hit.project for hit in hits}, {"workforce"})
        keyword.assert_called_once_with("saldo de vacaciones", "workforce", 20)
        semantic.assert_called_once()

    def test_query_embedding_is_cached(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY)")
        vectors.ensure_schema(db)
        expected = np.asarray([[1.0, 0.0]], dtype=np.float32)
        with patch("vector_search._embed", return_value=expected) as embed:
            first = vectors.query_vector(db, "same question")
            second = vectors.query_vector(db, "same question")
        np.testing.assert_array_equal(first, second)
        embed.assert_called_once()

    def test_code_reranker_caches_query_and_document_vectors(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY)")
        vectors.ensure_schema(db)
        expected = np.asarray([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
        ], dtype=np.float32)
        with patch("vector_search._embed_code", return_value=expected) as embed:
            first = vectors.code_rerank_scores(db, "request form", ["adapter", "docs"])
            second = vectors.code_rerank_scores(db, "request form", ["adapter", "docs"])
        self.assertGreater(first[0], first[1])
        self.assertEqual(first, second)
        embed.assert_called_once()

    def test_remote_code_embeddings_use_openai_contract_and_normalize_vectors(self):
        payload = {
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 0.0]},
            ]
        }
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch.object(vectors, "CODE_EMBED_URL", "http://gpu/v1/embeddings"), patch.object(
            vectors, "CODE_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B",
        ), patch.object(vectors, "urlopen", return_value=response) as open_url:
            embedded = vectors._embed_code_remote(["query", "document"])
        np.testing.assert_array_equal(
            embedded, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        request = open_url.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], "Qwen/Qwen3-Embedding-4B")
        self.assertEqual(sent["input"], ["query", "document"])

    def test_configured_remote_backend_is_available_without_local_model_cache(self):
        with patch.object(vectors, "CODE_EMBED_URL", "http://gpu/v1/embeddings"):
            self.assertEqual(vectors.code_reranker_backend(), "remote")
            self.assertTrue(vectors.code_reranker_cached())

    @patch("project_brain.code_reranker_cached", return_value=True)
    def test_auto_code_reranker_is_reserved_for_complex_queries(self, _cached):
        self.assertFalse(brain.should_code_rerank("find httpAdapter", "auto"))
        self.assertTrue(brain.should_code_rerank(
            "map request data transform dispatch node formdata detection headers tests",
            "auto",
        ))
        self.assertTrue(brain.should_code_rerank("short", "code"))
        self.assertFalse(brain.should_code_rerank("long enough words for code search", "fast"))

    @patch("project_brain.code_rerank_scores", return_value=[0.1, 0.9])
    @patch("project_brain.connect")
    def test_code_reranking_promotes_code_semantics_without_discarding_base_rank(
        self, _connect, _scores,
    ):
        hits = [
            brain.Hit("sample", "docs.md", 1, 2, "abc", 1.0, "generic docs", "config"),
            brain.Hit("sample", "lib/http.js", 1, 2, "abc", 0.5, "request adapter", "code"),
        ]
        reranked = brain.rerank_code_hits("request formdata adapter", hits)
        self.assertEqual(reranked[0].path, "lib/http.js")

    def test_latest_sql_definition_is_authoritative(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        values = [
            ("workforce", "supabase/migrations/20260101_old.sql", "function", "public.hours", "public.hours", 1, 2),
            ("workforce", "supabase/migrations/20260201_new.sql", "function", "public.hours", "public.hours", 1, 2),
        ]
        db.executemany(
            "INSERT INTO symbols(project,path,kind,name,qualified_name,start_line,end_line) VALUES(?,?,?,?,?,?,?)",
            values,
        )
        structure.mark_latest_sql_definitions(db, "workforce")
        rows = db.execute("SELECT path,active FROM symbols ORDER BY path").fetchall()
        self.assertEqual([row["active"] for row in rows], [0, 1])

    def test_source_authority_penalizes_history(self):
        self.assertEqual(structure.classify_source("docs/history/old.md")[0], "historical")
        self.assertEqual(structure.classify_source("functions/helper_test.ts")[0], "test")
        self.assertGreater(brain.AUTHORITY_WEIGHTS["canonical"], brain.AUTHORITY_WEIGHTS["historical"])

    def test_generic_source_authority_distinguishes_docs_tests_and_manifests(self):
        self.assertEqual(structure.classify_source("README.md")[0], "active")
        self.assertEqual(structure.classify_source("CHANGELOG.md")[0], "historical")
        self.assertEqual(structure.classify_source("test/unit/adapters/http.js")[0], "test")
        self.assertEqual(structure.classify_source("test/module/package.json")[0], "config")
        self.assertEqual(structure.classify_source("package.json")[0], "config")
        self.assertEqual(structure.classify_source("lib/adapters/http.js")[0], "code")

    def test_mcp_search_is_compact_and_diverse(self):
        hits = []
        for index in range(10):
            path = "src/crowded.ts" if index < 6 else f"src/file{index}.ts"
            content = "\n".join(
                f"line {line} {'Immediate ARI' if line == 25 else ''}"
                for line in range(40)
            )
            hits.append(brain.Hit(
                "sample", path, index * 100 + 1, index * 100 + 40,
                "abc", 1 / (index + 1), content, "code",
            ))
        result = mcp.compact_search_response("Immediate ARI", hits)
        self.assertLessEqual(len(result["hits"]), 6)
        self.assertLessEqual(
            sum(hit["path"] == "src/crowded.ts" for hit in result["hits"]), 2,
        )
        self.assertTrue(all(
            len(hit["content"].splitlines()) <= mcp.MAX_SNIPPET_LINES
            for hit in result["hits"]
        ))
        self.assertTrue(all("Immediate ARI" in hit["content"] for hit in result["hits"]))

    def test_mcp_search_preserves_hybrid_rank_over_literal_frequency(self):
        semantic = brain.Hit(
            "sample", "backend/outbox.ts", 1, 2, "abc", 0.9,
            "claim command and reconcile ambiguous result", "code",
        )
        lexical = brain.Hit(
            "sample", "docs/noise.ts", 1, 2, "abc", 0.1,
            "failure failure failure failure failure", "code",
        )
        result = mcp.compact_search_response("failure", [semantic, lexical])
        self.assertEqual(result["hits"][0]["path"], "backend/outbox.ts")

    def test_mcp_search_balances_implementation_and_tests_when_both_requested(self):
        hits = [
            brain.Hit(
                "sample", f"test/case{index}.spec.ts", 1, 8, "abc",
                1 / (index + 1), "hydration regression test", "test",
            )
            for index in range(8)
        ] + [
            brain.Hit(
                "sample", f"src/implementation{index}.ts", 1, 8, "abc",
                0.01 / (index + 1), "runtime hydration implementation", "code",
            )
            for index in range(4)
        ]

        result = mcp.compact_search_response(
            "Fix the causal implementation and identify regression tests", hits, 6,
        )

        self.assertEqual(len(result["hits"]), 6)
        self.assertGreaterEqual(
            sum(hit["authority"] == "code" for hit in result["hits"]), 3,
        )
        self.assertGreaterEqual(
            sum(hit["authority"] == "test" for hit in result["hits"]), 1,
        )

    @patch("mcp_server.brain.search")
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_search_overfetches_before_file_diversification(
        self, _scope, _registered, search,
    ):
        search.return_value = [
            brain.Hit(
                "sample", "test/crowded.js" if index < 8 else f"src/file{index}.js",
                index * 10 + 1, index * 10 + 5, "abc", 1 / (index + 1),
                f"body classification {index}", "test" if index < 8 else "code",
            )
            for index in range(24)
        ]

        result = mcp.call_tool(
            "search_repository",
            {"project": "sample", "question": "body classification", "limit": 6},
        )

        search.assert_called_once_with("body classification", "sample", 24)
        self.assertEqual(len(result["hits"]), 6)
        self.assertEqual(result["summary"]["unique_files"], 6)
        self.assertEqual(
            sum(hit["path"] == "test/crowded.js" for hit in result["hits"]), 1,
        )

    @patch("mcp_server.compact_relationship_hints", return_value=[])
    @patch("mcp_server.brain.should_code_rerank", return_value=False)
    @patch("mcp_server.brain.search")
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_fast_search_expands_pool_for_implementation_and_tests(
        self, _scope, _registered, search, _should, _relationships,
    ):
        search.return_value = [
            brain.Hit(
                "sample", f"src/file{index}.ts", 1, 5, "abc", 1.0,
                "runtime implementation", "code",
            )
            for index in range(60)
        ]

        mcp.call_tool(
            "search_repository",
            {
                "project": "sample",
                "question": "fix causal implementation and regression tests",
                "semantic_mode": "fast",
            },
        )

        search.assert_called_once_with(
            "fix causal implementation and regression tests", "sample", 60,
        )

    @patch("mcp_server.compact_relationship_hints", return_value=[])
    @patch("mcp_server.brain.rerank_code_hits")
    @patch("mcp_server.brain.should_code_rerank", return_value=True)
    @patch("mcp_server.brain.search")
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_complex_search_uses_bounded_code_reranker(
        self, _scope, _registered, search, _should, rerank, _relationships,
    ):
        candidates = [
            brain.Hit("sample", f"src/file{index}.js", 1, 2, "abc", 1.0,
                      "request formdata adapter", "code")
            for index in range(60)
        ]
        search.return_value = candidates
        rerank.return_value = list(reversed(candidates))

        result = mcp.call_tool(
            "search_repository",
            {"project": "sample", "question": "complex request formdata flow"},
        )

        search.assert_called_once_with("complex request formdata flow", "sample", 60)
        rerank.assert_called_once_with("complex request formdata flow", candidates)
        self.assertEqual(result["summary"]["semantic_mode"], "code")
        self.assertEqual(result["hits"][0]["path"], "src/file59.js")

    @patch("mcp_server.investigate_flow")
    @patch("mcp_server.brain.rerank_code_hits")
    @patch("mcp_server.brain.should_code_rerank", return_value=True)
    @patch("mcp_server.brain.search")
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_investigate_flow_uses_the_requested_code_reranker(
        self, _scope, _registered, search, _should, rerank, investigate,
    ):
        candidates = [
            brain.Hit(
                "sample", f"src/file{index}.ts", 1, 10, "abc", 1.0,
                "runtime hydration implementation", "code",
            )
            for index in range(60)
        ]
        search.return_value = candidates
        rerank.return_value = list(reversed(candidates))
        investigate.return_value = {"retrieval_diagnostic": {"quality": "healthy"}}

        result = mcp.call_tool(
            "investigate_flow",
            {
                "project": "sample",
                "question": "trace complex runtime hydration implementation",
                "semantic_mode": "code",
            },
        )

        search.assert_called_once_with(
            "trace complex runtime hydration implementation", "sample", 60,
        )
        rerank.assert_called_once_with(
            "trace complex runtime hydration implementation", candidates,
        )
        self.assertEqual(result["anchors"]["summary"]["semantic_mode"], "code")
        self.assertEqual(result["anchors"]["hits"][0]["path"], "src/file59.ts")
        self.assertEqual(investigate.call_args.args[2][0], "src/file59.ts")

    @patch("mcp_server.compact_relationship_hints", return_value=[])
    @patch("mcp_server.brain.rerank_code_hits", side_effect=RuntimeError("gpu offline"))
    @patch("mcp_server.brain.should_code_rerank", return_value=True)
    @patch("mcp_server.brain.search")
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_auto_search_falls_back_when_remote_reranker_is_unavailable(
        self, _scope, _registered, search, _should, _rerank, _relationships,
    ):
        search.return_value = [
            brain.Hit("sample", "src/http.js", 1, 2, "abc", 1.0, "adapter", "code")
        ]
        result = mcp.call_tool(
            "search_repository",
            {"project": "sample", "question": "complex request formdata flow"},
        )
        self.assertEqual(result["summary"]["semantic_mode"], "fast")
        self.assertIn("gpu offline", result["summary"]["reranker_fallback"])

    @patch("mcp_server.brain.rerank_code_hits", side_effect=RuntimeError("gpu offline"))
    @patch("mcp_server.brain.should_code_rerank", return_value=True)
    @patch("mcp_server.brain.search", return_value=[])
    @patch("mcp_server.brain.require_registered_project")
    @patch("mcp_server.scope_note", return_value="fixture")
    def test_mcp_explicit_code_mode_reports_remote_failure(
        self, _scope, _registered, _search, _should, _rerank,
    ):
        with self.assertRaisesRegex(RuntimeError, "gpu offline"):
            mcp.call_tool(
                "search_repository",
                {
                    "project": "sample", "question": "complex request formdata flow",
                    "semantic_mode": "code",
                },
            )

    def test_relationship_hints_follow_query_relevant_shared_callee_to_consumer(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            """CREATE TABLE chunks(
                 id INTEGER PRIMARY KEY,project TEXT,path TEXT,start_line INTEGER,
                 end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT
               )"""
        )
        for path in ("lib/defaults.js", "lib/http.js", "test/utils.js"):
            db.execute(
                "INSERT INTO source_authority VALUES(?,?,?,?,?,?)",
                (
                    "sample", path, "test" if path.startswith("test/") else "code",
                    "fixture", "abc", "gen",
                ),
            )
        db.execute(
            "INSERT INTO chunks VALUES(1,'sample','lib/http.js',1,80,'abc','gen',?)",
            ("function dispatchHttpRequest(data) { return isFormData(data); }",),
        )
        db.executemany(
            """INSERT INTO edges(
                 project,source_path,source_name,relation,target_name,target_path,line,
                 resolution_confidence,commit_hash,generation_id
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            [
                ("sample", "lib/defaults.js", "transformRequest", "calls", "isFormData",
                 "lib/utils.js", 20, "import_resolved", "abc", "gen"),
                ("sample", "test/utils.js", "recognizesFormData", "calls", "isFormData",
                 "lib/utils.js", 5, "import_resolved", "abc", "gen"),
                ("sample", "lib/http.js", "dispatchHttpRequest", "calls", "isFormData",
                 "lib/utils.js", 30, "import_resolved", "abc", "gen"),
            ],
        )
        anchor = brain.Hit(
            "sample", "lib/defaults.js", 1, 50, "abc", 1.0,
            "function transformRequest(data) { return isFormData(data); }", "code",
        )
        test_anchor = brain.Hit(
            "sample", "test/utils.js", 1, 10, "abc", 1.1,
            "it('recognizes FormData', () => isFormData(value));", "test",
        )
        related_anchor = brain.Hit(
            "sample", "lib/http.js", 1, 80, "abc", 0.9,
            "function dispatchHttpRequest(data) { return isFormData(data); }", "code",
        )
        with patch("mcp_server.brain.connect", return_value=db):
            hints = mcp.compact_relationship_hints(
                "Map FormData request transformation",
                [test_anchor, anchor, related_anchor], limit=2,
            )
        self.assertEqual(hints[0]["path"], "lib/http.js")
        self.assertEqual(hints[0]["via_symbol"], "isFormData")
        self.assertEqual(hints[0]["from_anchor"], "lib/defaults.js")

    def test_anchor_relationships_expose_adjacent_input_classifiers(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            """INSERT INTO symbols(
                 project,path,kind,name,qualified_name,start_line,end_line,active
               ) VALUES('sample','lib/defaults.js','function','transformRequest',
                        'transformRequest',10,40,1)"""
        )
        db.executemany(
            """INSERT INTO edges(
                 project,source_path,source_name,relation,target_name,target_path,line,
                 resolution_confidence
               ) VALUES('sample','lib/defaults.js','transformRequest','calls',?,?,?,
                        'import_resolved')""",
            [("isFormData", "lib/utils.js", 20), ("isBlob", "lib/utils.js", 21)],
        )
        hit = brain.Hit(
            "sample", "lib/defaults.js", 15, 30, "abc", 1.0,
            "return isFormData(data) || isBlob(data)", "code",
        )
        with patch("mcp_server.brain.connect", return_value=db):
            relationships = mcp.compact_anchor_relationships([hit])
        self.assertEqual(relationships[0]["symbol"], "transformRequest")
        self.assertEqual(
            [call["name"] for call in relationships[0]["direct_calls"]],
            ["isFormData", "isBlob"],
        )

    def test_contract_differences_surface_variants_missing_from_related_stage(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.executemany(
            """INSERT INTO symbols(
                 project,path,kind,name,qualified_name,start_line,end_line,active
               ) VALUES('sample',?,'function',?,?,1,80,1)""",
            [
                ("lib/defaults.js", "transformRequest", "transformRequest"),
                ("lib/http.js", "httpAdapter", "httpAdapter"),
            ],
        )
        edges = []
        for path, source, calls in (
            ("lib/defaults.js", "transformRequest", ["isFormData", "isBlob", "isFile"]),
            ("lib/http.js", "httpAdapter", ["isFormData", "isStream"]),
        ):
            edges.extend(
                ("sample", path, source, "calls", call, "lib/utils.js", line,
                 "import_resolved")
                for line, call in enumerate(calls, 10)
            )
        db.executemany(
            """INSERT INTO edges(
                 project,source_path,source_name,relation,target_name,target_path,line,
                 resolution_confidence
               ) VALUES(?,?,?,?,?,?,?,?)""",
            edges,
        )
        anchor = brain.Hit(
            "sample", "lib/defaults.js", 1, 80, "abc", 1.0,
            "function transformRequest(data) {}", "code",
        )
        hints = [{
            "path": "lib/http.js", "indexed_range": [1, 80],
            "from_anchor": "lib/defaults.js", "via_symbol": "isFormData",
        }]
        with patch("mcp_server.brain.connect", return_value=db):
            differences = mcp.compact_contract_differences([anchor], hints)
        self.assertEqual(len(differences), 1)
        self.assertEqual(differences[0]["shared"], ["isFormData"])
        self.assertEqual(differences[0]["only_in_anchor"], ["isBlob", "isFile"])
        self.assertEqual(differences[0]["only_in_related"], ["isStream"])
        self.assertEqual(
            differences[0]["status"], "review_candidate_not_proven_bug",
        )

    def test_contract_differences_ignore_matching_classifier_sets(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.executemany(
            """INSERT INTO symbols(
                 project,path,kind,name,qualified_name,start_line,end_line,active
               ) VALUES('sample',?,'function',?,?,1,40,1)""",
            [
                ("src/producer.js", "produce", "produce"),
                ("src/consumer.js", "consume", "consume"),
            ],
        )
        for path, source in (("src/producer.js", "produce"), ("src/consumer.js", "consume")):
            db.executemany(
                """INSERT INTO edges(
                     project,source_path,source_name,relation,target_name,target_path,line,
                     resolution_confidence
                   ) VALUES('sample',?,?,'calls',?,'src/types.js',10,'import_resolved')""",
                [(path, source, call) for call in ("isBlob", "isStream")],
            )
        anchor = brain.Hit(
            "sample", "src/producer.js", 1, 40, "abc", 1.0, "produce", "code",
        )
        hints = [{
            "path": "src/consumer.js", "indexed_range": [1, 40],
            "from_anchor": "src/producer.js", "via_symbol": "isBlob",
        }]
        with patch("mcp_server.brain.connect", return_value=db):
            differences = mcp.compact_contract_differences([anchor], hints)
        self.assertEqual(differences, [])

    def test_related_suggests_nearby_symbol_when_name_is_approximate(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            "INSERT INTO symbols(project,path,kind,name,qualified_name,start_line,end_line,active) "
            "VALUES('sample','locks.ts','function','withGuestLock','withGuestLock',10,30,1)"
        )
        result = structure.related(db, "sample", "withGuestLok")
        self.assertFalse(result["symbols"])
        self.assertEqual(result["suggestions"][0]["name"], "withGuestLock")

    def test_investigate_flow_expands_workers_scheduling_constraints_and_tests(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY,project TEXT,path TEXT,"
            "start_line INTEGER,end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT)"
        )
        paths = {
            "supabase/functions/send/index.ts": "code",
            "supabase/functions/create/index.ts": "code",
            "supabase/functions/shared/commands.ts": "code",
            "supabase/functions/retry-commands/index.ts": "code",
            "supabase/functions/shared/commands_test.ts": "test",
            "migrations/schedule.sql": "schema_history",
            "migrations/commands.sql": "schema_history",
        }
        db.executemany(
            "INSERT INTO source_authority(project,path,authority,reason,commit_hash,generation_id) "
            "VALUES('sample',?,?,?, 'abc','gen')",
            [(path, authority, "fixture") for path, authority in paths.items()],
        )
        db.executemany(
            "INSERT INTO symbols(project,path,kind,name,qualified_name,start_line,end_line,active) "
            "VALUES('sample',?,'module',?,?,1,20,1)",
            [(path, Path(path).stem, Path(path).stem) for path in paths],
        )
        edge_values = [
            ("supabase/functions/create/index.ts", "create", "invokes_edge_function", "send", None, 15, "literal"),
            ("supabase/functions/send/index.ts", "send", "imports", "commands", "supabase/functions/shared/commands.ts", 2, "exact_path"),
            ("supabase/functions/send/index.ts", "send", "writes_table", "public.commands", None, 10, "literal"),
            ("supabase/functions/retry-commands/index.ts", "worker", "calls", "canRetry", "supabase/functions/shared/commands.ts", 20, "import_resolved"),
            ("supabase/functions/retry-commands/index.ts", "worker", "reads_table", "public.commands", None, 8, "literal"),
            ("supabase/functions/shared/commands_test.ts", "test", "calls", "canRetry", "supabase/functions/shared/commands.ts", 5, "import_resolved"),
            ("migrations/schedule.sql", "schedule", "invokes_edge_function", "retry-commands", None, 3, "literal"),
        ]
        db.executemany(
            "INSERT INTO edges(project,source_path,source_name,relation,target_name,target_path,line,resolution_confidence) "
            "VALUES('sample',?,?,?,?,?,?,?)",
            edge_values,
        )
        db.execute(
            "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
            "VALUES('sample','migrations/commands.sql',1,10,'abc','gen',"
            "'create unique index commands_key on public.commands(id)')"
        )

        result = structure.investigate_flow(
            db, "sample", ["supabase/functions/create/index.ts", "supabase/functions/shared/commands.ts"], 20,
        )
        self.assertTrue(
            {"implementation", "persistence", "retries", "scheduling", "constraints", "tests"}
            <= set(result["lanes_present"]), result,
        )
        self.assertFalse(result["lanes_missing"])
        executor = next(
            item for item in result["evidence"]
            if item["path"] == "supabase/functions/send/index.ts"
        )
        self.assertTrue(any(
            reason["kind"] == "resolved_edge_function" for reason in executor["reasons"]
        ))

    def test_investigate_flow_caps_constraint_noise(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY,project TEXT,path TEXT,"
            "start_line INTEGER,end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT)"
        )
        paths = {"supabase/functions/send/index.ts": "code"} | {
            f"migrations/noise{index}.sql": "schema_history" for index in range(8)
        }
        db.executemany(
            "INSERT INTO source_authority(project,path,authority,reason,commit_hash,generation_id) "
            "VALUES('sample',?,?,?, 'abc','gen')",
            [(path, authority, "fixture") for path, authority in paths.items()],
        )
        db.execute(
            "INSERT INTO edges(project,source_path,source_name,relation,target_name,line,resolution_confidence) "
            "VALUES('sample','supabase/functions/send/index.ts','send','writes_table','public.commands',5,'literal')"
        )
        db.executemany(
            "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
            "VALUES('sample',?,1,3,'abc','gen','create unique index on public.commands(id)')",
            [(f"migrations/noise{index}.sql",) for index in range(8)],
        )
        result = structure.investigate_flow(
            db, "sample", ["supabase/functions/send/index.ts"], 16,
        )
        self.assertLessEqual(
            sum(item["lane"] == "constraints" for item in result["evidence"]), 2,
        )

    def test_investigate_flow_reserves_question_relevant_inheritance_chains(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY,project TEXT,path TEXT,"
            "start_line INTEGER,end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT)"
        )
        paths = [
            "src/core/Object3D.js", "src/objects/Camera.js", "src/objects/Line.js",
            "src/objects/Mesh.js", "src/objects/Points.js", "src/objects/Sprite.js",
        ]
        db.executemany(
            "INSERT INTO source_authority(project,path,authority,reason,commit_hash,generation_id) "
            "VALUES('three',?,'code','fixture','abc','gen')",
            [(path,) for path in paths],
        )
        db.executemany(
            "INSERT INTO symbols(project,path,kind,name,qualified_name,start_line,end_line,active) "
            "VALUES('three',?,'module',?,?,1,20,1)",
            [(path, Path(path).stem, Path(path).stem) for path in paths],
        )
        db.executemany(
            "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
            "VALUES('three',?,1,20,'abc','gen',?)",
            [
                ("src/core/Object3D.js", "clone delegates to copy"),
                ("src/objects/Camera.js", "copy projection matrix"),
                ("src/objects/Line.js", "copy this.material from source material"),
                ("src/objects/Mesh.js", "copy this.material from source material"),
                ("src/objects/Points.js", "copy this.material from source material"),
                ("src/objects/Sprite.js", "copy this.material from source material"),
            ],
        )
        db.executemany(
            "INSERT INTO edges(project,source_path,source_name,source_qualified_name,relation,"
            "target_name,target_path,target_qualified_name,line,resolution_confidence) "
            "VALUES('three',?,? ,?,'overrides','copy','src/core/Object3D.js',"
            "'Object3D.copy',10,'import_resolved_inheritance')",
            [
                (path, f"{Path(path).stem}.copy", f"{Path(path).stem}.copy")
                for path in paths[1:]
            ],
        )
        db.execute(
            "INSERT INTO edges(project,source_path,source_name,source_qualified_name,relation,"
            "target_name,target_path,target_qualified_name,line,resolution_confidence) "
            "VALUES('three','src/core/Object3D.js','clone','Object3D.clone','calls',"
            "'copy','src/core/Object3D.js','Object3D.copy',5,'same_file')"
        )

        result = structure.investigate_flow(
            db, "three", ["src/core/Object3D.js"], 6,
            question="clone arrays of materials and preserve material references",
        )
        by_path = {item["path"]: item for item in result["evidence"]}
        self.assertTrue(
            {"src/objects/Line.js", "src/objects/Mesh.js", "src/objects/Points.js"}
            <= set(by_path),
            result,
        )
        self.assertTrue(any(
            reason["kind"] == "dispatch_override_chain"
            for reason in by_path["src/objects/Line.js"]["reasons"]
        ))

    def test_investigate_flow_surfaces_rpc_consumers_override_chain_and_cache(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        structure.ensure_schema(db)
        db.execute(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY,project TEXT,path TEXT,"
            "start_line INTEGER,end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT)"
        )
        paths = {
            "src/pages/Absences.tsx": "code",
            "src/pages/EmployeeDetail.tsx": "code",
            "src/lib/absence-query-keys.ts": "code",
            "supabase/migrations/001_kpi.sql": "schema_history",
            "supabase/migrations/003_patch_kpi.sql": "schema_history",
        }
        db.executemany(
            "INSERT INTO source_authority(project,path,authority,reason,commit_hash,generation_id) "
            "VALUES('workforce',?,?,?,'abc','gen')",
            [(path, authority, "fixture") for path, authority in paths.items()],
        )
        db.execute(
            "INSERT INTO symbols(project,path,kind,name,qualified_name,start_line,end_line,active) "
            "VALUES('workforce','supabase/migrations/001_kpi.sql','function',"
            "'get_employee_kpi_summary','public.get_employee_kpi_summary',1,30,1)"
        )
        db.execute(
            "INSERT INTO edges(project,source_path,source_name,relation,target_name,target_path,line,resolution_confidence) "
            "VALUES('workforce','src/pages/EmployeeDetail.tsx','loadKpi','calls_rpc',"
            "'get_employee_kpi_summary','supabase/migrations/001_kpi.sql',20,'active_sql')"
        )
        db.executemany(
            "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
            "VALUES('workforce',?,1,30,'abc','gen',?)",
            [
                ("src/pages/Absences.tsx", "import { KEYS } from '@/lib/absence-query-keys'; absence vacation queryClient.invalidateQueries({ queryKey: [key] })"),
                ("src/pages/EmployeeDetail.tsx", "queryKey: ['employee-detail-kpi']; get_employee_kpi_summary"),
                ("src/lib/absence-query-keys.ts", "export const KEYS = ['employee-detail-kpi']"),
                ("supabase/migrations/001_kpi.sql", "create or replace function get_employee_kpi_summary()"),
                ("supabase/migrations/003_patch_kpi.sql", "select pg_get_functiondef('get_employee_kpi_summary'); execute patched"),
            ],
        )

        result = structure.investigate_flow(
            db, "workforce", ["supabase/migrations/001_kpi.sql"], 20,
            question="KPI pending_absence_days saldo de vacaciones y ausencia",
        )
        by_path = {item["path"]: item for item in result["evidence"]}
        self.assertTrue(any(
            reason["kind"] == "rpc_consumer"
            for reason in by_path["src/pages/EmployeeDetail.tsx"]["reasons"]
        ))
        self.assertTrue(any(
            reason["kind"] == "sql_override_chain"
            for reason in by_path["supabase/migrations/003_patch_kpi.sql"]["reasons"]
        ))
        absence_impact = next(
            item for item in result["impact_candidates"]
            if item["path"] == "src/pages/Absences.tsx"
        )
        self.assertIn("cache", absence_impact["impact_categories"])
        self.assertTrue(result["coverage_warnings"])
        self.assertEqual(
            result["rpc_consumer_map"]["get_employee_kpi_summary"],
            ["src/pages/EmployeeDetail.tsx"],
        )
        self.assertEqual(
            [item["path"] for item in result["sql_lineage"]["get_employee_kpi_summary"]],
            ["supabase/migrations/001_kpi.sql", "supabase/migrations/003_patch_kpi.sql"],
        )
        self.assertNotIn("employee-detail-kpi", result["potentially_uninvalidated_query_keys"])

    def test_retrieval_diagnostic_exposes_thin_nominal_and_lexical_coverage(self):
        flow = {
            "evidence": [
                {
                    "path": "one.sql", "lane": "constraints", "authority": "schema_history",
                    "roles": ["sql", "rpc"],
                    "reasons": [
                        {"kind": "constraint_candidate", "confidence": "lexical"},
                        {"kind": "sql_override_chain", "confidence": "definition_or_patch_lexical"},
                    ],
                },
                {
                    "path": "two.ts", "lane": "implementation", "authority": "code",
                    "roles": ["writers"],
                    "reasons": [
                        {"kind": "query_related_ui", "confidence": "lexical"},
                        {"kind": "cache_invalidator", "confidence": "domain_lexical"},
                    ],
                },
            ],
            "seed_paths": ["one.sql"],
            "roles_present": ["rpc", "sql", "writers"],
            "roles_missing": [],
            "lanes_missing": [],
            "rpc_consumer_map": {},
            "sql_lineage": {"rpc": []},
            "potentially_uninvalidated_query_keys": ["station-kpi"],
        }
        diagnostic = structure.diagnose_retrieval(flow)
        codes = {item["code"] for item in diagnostic["warnings"]}
        self.assertEqual(diagnostic["quality"], "weak")
        self.assertIn("lexical_dominance", codes)
        self.assertIn("thin_role_coverage", codes)
        self.assertIn("incomplete_sql_lineage", codes)
        self.assertIn("cache_review_required", codes)

    def test_retrieval_diagnostic_counts_structured_fanout_outside_top_k(self):
        flow = {
            "evidence": [], "seed_paths": [], "roles_present": [],
            "roles_missing": [], "lanes_missing": [],
            "rpc_consumer_map": {
                "station": ["A.tsx", "B.tsx"],
                "overview": ["C.tsx", "A.tsx"],
            },
            "sql_lineage": {
                "station": [{"path": "1.sql"}, {"path": "2.sql"}],
                "overview": [{"path": "1.sql"}],
            },
            "potentially_uninvalidated_query_keys": [],
        }
        diagnostic = structure.diagnose_retrieval(flow)
        self.assertEqual(diagnostic["shape"]["distinct_rpc_consumers"], 3)
        self.assertEqual(diagnostic["shape"]["sql_lineage_entries"], 3)

    def test_retrieval_diagnostic_warns_when_every_anchor_is_a_test(self):
        flow = {
            "evidence": [
                {
                    "path": f"test/case{index}.spec.ts", "lane": "tests",
                    "authority": "test", "roles": ["tests"],
                    "reasons": [{"kind": "search_anchor"}],
                }
                for index in range(2)
            ] + [{
                "path": "src/generic.ts", "lane": "implementation",
                "authority": "code", "roles": [],
                "reasons": [{"kind": "outgoing_edge", "confidence": "exact_path"}],
            }],
            "seed_paths": ["test/case0.spec.ts", "test/case1.spec.ts"],
            "roles_present": ["tests"], "roles_missing": [],
            "lanes_missing": [], "rpc_consumer_map": {}, "sql_lineage": {},
            "potentially_uninvalidated_query_keys": [],
        }

        diagnostic = structure.diagnose_retrieval(flow)

        self.assertEqual(diagnostic["quality"], "review")
        self.assertIn(
            "test_anchor_dominance",
            {item["code"] for item in diagnostic["warnings"]},
        )
        self.assertEqual(diagnostic["shape"]["anchor_authority_counts"], {"test": 2})

    def test_retrieval_diagnostic_reports_explicit_missing_facets(self):
        flow = {
            "evidence": [{
                "path": "kpi.sql", "lane": "implementation", "authority": "schema_history",
                "roles": ["sql"], "reasons": [],
            }],
            "seed_paths": ["kpi.sql"], "roles_present": ["sql"],
            "roles_missing": [], "lanes_missing": [], "rpc_consumer_map": {},
            "sql_lineage": {}, "potentially_uninvalidated_query_keys": [],
            "facet_coverage": {
                "planner": {"status": "mapped", "sql_symbols": [{"name": "planner"}], "code_paths": []},
                "reports_exports": {"status": "missing", "sql_symbols": [], "code_paths": []},
            },
        }
        diagnostic = structure.diagnose_retrieval(flow)
        warning = next(item for item in diagnostic["warnings"] if item["code"] == "missing_question_facets")
        self.assertEqual(warning["facets"], ["reports_exports"])

    def test_mcp_symbol_prefers_exact_and_filters_homonym_edges(self):
        exact = {
            "project": "sample", "path": "server/create.ts", "kind": "function",
            "name": "createReservation", "qualified_name": "createReservation",
            "start_line": 10, "end_line": 90, "signature": "function createReservation()",
            "active": 1, "authority": "code",
        }
        alternative = {
            **exact, "path": "ui/useReservations.ts", "start_line": 20,
            "end_line": 40, "qualified_name": "useReservations.createReservation",
        }
        args = {
            **exact, "name": "CreateReservationArgs",
            "qualified_name": "CreateReservationArgs", "start_line": 1, "end_line": 8,
        }
        edges = [
            {
                "source_path": "server/create.ts", "source_name": "createReservation",
                "source_qualified_name": "createReservation", "relation": "calls",
                "target_name": "book", "target_qualified_name": "book",
                "target_path": "server/book.ts", "line": 80,
                "resolution_confidence": "resolved",
            },
            {
                "source_path": "ui/useReservations.ts", "source_name": "createReservation",
                "source_qualified_name": "useReservations.createReservation",
                "relation": "calls", "target_name": "insert",
                "target_qualified_name": None, "target_path": None, "line": 30,
                "resolution_confidence": "unresolved",
            },
        ]
        result = mcp.compact_related_response(
            {"symbols": [args, exact, alternative], "edges": edges, "path_filter": None},
            "createReservation", 12,
        )
        self.assertEqual(
            [item["path"] for item in result["definitions"]],
            ["server/create.ts"],
        )
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["relationships"][0]["target"]["qualified_name"], "book")

    def test_mcp_symbol_summarizes_direct_calls_beyond_relationship_limit(self):
        symbol = {
            "project": "sample", "path": "lib/defaults.js", "kind": "function",
            "name": "transformRequest", "qualified_name": "transformRequest",
            "start_line": 1, "end_line": 40, "signature": "function transformRequest()",
            "active": 1, "authority": "code",
        }
        callees = ["isFormData", "isBuffer", "isStream", "isFile", "isBlob"]
        edges = [{
            "source_path": "lib/defaults.js", "source_name": "transformRequest",
            "source_qualified_name": "transformRequest", "relation": "calls",
            "target_name": callee, "target_qualified_name": callee,
            "target_path": "lib/utils.js", "line": index + 2,
            "resolution_confidence": "import_resolved",
        } for index, callee in enumerate(callees)]

        result = mcp.compact_related_response(
            {"symbols": [symbol], "edges": edges, "path_filter": "lib/defaults.js"},
            "transformRequest", 2,
        )

        self.assertEqual(len(result["relationships"]), 2)
        self.assertEqual(
            [call["name"] for call in result["direct_calls"]], callees,
        )

    def test_sql_function_boundary_ignores_internal_semicolons(self):
        text = """create or replace function public.demo() returns void
language plpgsql as $$
begin
  perform 1;
  perform 2;
end;
$$;
create table public.afterward(id int);
"""
        start = text.lower().index("create or replace function")
        fallback = text.index("()") + 2
        end = structure._sql_definition_end(text, start, "function", fallback)
        self.assertIn("$$;", text[:end])
        self.assertNotIn("create table", text[:end].lower())


class FakeGateway:
    max_output_tokens = 200

    def __init__(self, char_divisor=4):
        self.char_divisor = char_divisor

    def token_count(self, messages):
        return max(1, len(loop.NemotronGateway.rendered_messages(messages)) // self.char_divisor)


class ProtocolGateway(FakeGateway):
    def __init__(self, invalid_first=None, char_divisor=100):
        super().__init__(char_divisor)
        self.invalid_first = invalid_first
        self.completions = []
        self.tokenized = []

    def token_count(self, messages):
        self.tokenized.append(messages)
        return super().token_count(messages)

    def complete(self, messages):
        self.completions.append(messages)
        if self.invalid_first is not None and len(self.completions) == 1:
            return self.invalid_first
        payload = json.loads(messages[-1]["content"])
        schema = payload["response_schema"]
        return json.dumps({
            "snapshot_id": schema["snapshot_id"],
            "call_id": schema["call_id"],
            "batch_id": schema["batch_id"],
            "batch_digest": schema["batch_digest"],
            "status": "need_more",
            "inspect_edge_ids": [], "claims": [], "requests": [],
            "contradictions": [], "answer": "", "answer_claim_ids": [],
        })


def fixture_snapshot(line_count=300):
    lines = [f"line {number}" for number in range(1, line_count + 1)]
    lines[244] = "DECISIVE_AGGREGATION"
    descriptor = loop.SnapshotDescriptor(
        project="sample", commit="abc", content_digest="content",
        graph_digest="graph", snapshot_id="snapshot",
    )
    symbol = loop.SymbolRecord(
        "S0", "src/service.ts", "function", "availability", "availability",
        50, 271, True, "code", {},
    )
    helper = loop.SymbolRecord(
        "S1", "src/helper.ts", "function", "helper", "helper",
        1, 4, True, "code", {},
    )
    edges = (
        loop.EdgeRecord(
            "E0", "src/service.ts", "availability", "availability", "calls",
            "helper", "src/helper.ts", "helper", 131, "import_resolved", {},
        ),
        loop.EdgeRecord(
            "E1", "src/helper.ts", "helper", "helper", "calls",
            "availability", "src/service.ts", "availability", 2, "import_resolved", {},
        ),
    )
    return loop.FrozenSnapshot(
        descriptor,
        {"src/service.ts": "\n".join(lines), "src/helper.ts": "a\nb\nc\nd"},
        (symbol, helper), edges,
        {"src/service.ts": "code", "src/helper.ts": "code"},
    )


class ExhaustiveLoopTests(unittest.TestCase):
    def test_atlas_ownership_requires_every_edge_exactly_once(self):
        snapshot = fixture_snapshot()
        valid = [loop.AtlasBatch("B0", {}, ("E0",)), loop.AtlasBatch("B1", {}, ("E1",))]
        loop.validate_batch_ownership(snapshot.edge_by_id, valid)
        with self.assertRaises(loop.CoverageError):
            loop.validate_batch_ownership(
                snapshot.edge_by_id,
                [loop.AtlasBatch("B0", {}, ("E0", "E0", "E1"))],
            )

    def test_complete_atlas_uses_one_global_batch(self):
        snapshot = fixture_snapshot()
        mode, batches, diagnostics = loop.partition_atlas(
            snapshot, "question", FakeGateway(char_divisor=1),
            context_window=20_000, completion_reserve=200, safety_margin=100,
        )
        self.assertEqual(mode, "global")
        self.assertEqual(len(batches), 1)
        self.assertEqual(set(batches[0].owned_edge_ids), {"E0", "E1"})
        self.assertGreater(diagnostics["global_input_tokens"], 0)

    def test_oversized_atlas_partitions_without_losing_edges(self):
        snapshot = fixture_snapshot()

        class EdgeAwareGateway(FakeGateway):
            def token_count(self, messages):
                rendered = loop.NemotronGateway.rendered_messages(messages)
                represented = sum(f'"E{number}"' in rendered for number in range(2))
                return 4_000 if represented > 1 else 1_000

        mode, batches, _ = loop.partition_atlas(
            snapshot, "question", EdgeAwareGateway(),
            context_window=2_500,
            completion_reserve=200, safety_margin=100,
        )
        self.assertEqual(mode, "partitioned")
        loop.validate_batch_ownership(snapshot.edge_by_id, batches)

    def test_partial_hit_expands_to_complete_enclosing_symbol(self):
        snapshot = fixture_snapshot()
        ledger = loop.EvidenceLedger(snapshot)
        evidence = ledger.add_range("src/service.ts", 131, 210)
        self.assertEqual(len(evidence), 1)
        self.assertEqual((evidence[0].start_line, evidence[0].end_line), (50, 271))
        self.assertTrue(evidence[0].complete)
        self.assertIn("DECISIVE_AGGREGATION", evidence[0].content)

    def test_explicit_question_seed_opens_complete_named_function(self):
        snapshot = fixture_snapshot()
        evidence = loop.EvidenceLedger(snapshot).seed_question(
            "Audit availability and logic after line 210",
        )
        selected = next(item for item in evidence if item.symbol == "availability")
        self.assertEqual((selected.start_line, selected.end_line), (50, 271))
        self.assertTrue(selected.complete)
        self.assertIn("DECISIVE_AGGREGATION", selected.content)

    def test_overlapping_hits_deduplicate_to_one_complete_symbol(self):
        snapshot = fixture_snapshot()
        ledger = loop.EvidenceLedger(snapshot)
        first = ledger.add_range("src/service.ts", 131, 210)[0]
        second = ledger.add_range("src/service.ts", 196, 260)[0]
        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(len(ledger.items), 1)

    def test_cycle_requests_stop_adding_evidence(self):
        snapshot = fixture_snapshot()
        ledger = loop.EvidenceLedger(snapshot)
        first = ledger.resolve_request({"op": "callees", "name": "availability"})
        second = ledger.resolve_request({"name": "availability", "op": "callees"})
        self.assertGreater(len(first), 0)
        self.assertEqual(second, [])

    def test_unknown_citations_preserve_hypothesis(self):
        snapshot = fixture_snapshot()
        ledger = loop.EvidenceLedger(snapshot)
        result = loop.validate_model_response({
            "snapshot_id": snapshot.snapshot_id,
            "status": "answer",
            "claims": [{
                "text": "possible systemic inference", "kind": "inference",
                "support": "support", "evidence_ids": ["V_UNKNOWN"],
                "edge_ids": ["E_UNKNOWN"],
            }],
        }, snapshot, ledger)
        self.assertEqual(result["claims"][0]["text"], "possible systemic inference")
        self.assertEqual(result["claims"][0]["evidence_ids"], [])
        self.assertTrue(result["claims"][0]["invalid_references"])

    def test_inference_is_not_rejected_for_lacking_one_line(self):
        snapshot = fixture_snapshot()
        result = loop.validate_model_response({
            "status": "need_more",
            "claims": [{
                "text": "system-wide hypothesis", "kind": "inference",
                "support": "unknown", "evidence_ids": [], "edge_ids": ["E0"],
            }],
        }, snapshot, loop.EvidenceLedger(snapshot))
        self.assertEqual(result["claims"][0]["kind"], "inference")
        self.assertEqual(result["claims"][0]["text"], "system-wide hypothesis")

    def test_oversized_symbol_is_not_silently_complete(self):
        snapshot = fixture_snapshot()
        evidence = loop.EvidenceLedger(snapshot, max_symbol_bytes=30).add_symbol(
            snapshot.symbol_by_id["S0"],
        )
        self.assertIsNotNone(evidence)
        self.assertFalse(evidence.complete)
        self.assertEqual(evidence.metadata["full_range"], [50, 271])

    def test_strict_json_rejects_duplicate_keys_and_nan(self):
        with self.assertRaises(ValueError):
            loop.strict_json_loads('{"status":"answer","status":"need_more"}')
        with self.assertRaises(ValueError):
            loop.strict_json_loads('{"score":NaN}')

    def test_atlas_preserves_edge_metadata(self):
        snapshot = fixture_snapshot()
        snapshot.edges = (
            loop.EdgeRecord(
                "E0", "src/service.ts", "availability", "availability", "calls",
                "helper", "src/helper.ts", "helper", 131, "import_resolved",
                {"expression": "helper", "argument_count": 2},
            ),
        )
        atlas = loop.compact_atlas(snapshot.edges)
        encoded_edge = atlas["g"][0][2][0]
        decoded_metadata = {atlas["m"][key]: value for key, value in encoded_edge[-1]}
        self.assertEqual(decoded_metadata["argument_count"], 2)

    def test_protocol_rejects_edge_from_another_map(self):
        snapshot = fixture_snapshot()
        value = {
            "snapshot_id": "snapshot", "call_id": "Q1", "batch_id": "B0",
            "batch_digest": "digest", "status": "need_more",
            "inspect_edge_ids": ["E1"], "claims": [], "requests": [],
            "contradictions": [], "answer": "", "answer_claim_ids": [],
        }
        with self.assertRaises(loop.ModelProtocolError):
            loop.validate_response_schema(
                value, snapshot_id="snapshot", call_id="Q1", batch_id="B0",
                batch_digest="digest", allowed_edge_ids={"E0"},
                allowed_evidence_ids=set(), allowed_claim_ids=set(),
            )

    def test_evidence_schema_requires_cited_claims(self):
        envelope = {
            "snapshot_id": "snapshot", "call_id": "Q1",
            "batch_id": "B0", "batch_digest": "digest",
        }
        payload = {
            "phase": "evidence",
            "evidence": [{"evidence_id": "V1", "content": "proof"}],
        }
        schema = loop.model_response_json_schema(envelope, payload)
        claims = schema["properties"]["claims"]
        evidence_ids = claims["items"]["properties"]["evidence_ids"]
        self.assertEqual(claims["minItems"], 1)
        self.assertEqual(evidence_ids["minItems"], 1)
        self.assertEqual(evidence_ids["items"]["enum"], ["V1"])
        final_schema = loop.model_response_json_schema(
            envelope, {
                "phase": "final_synthesis",
                "eligible_answer_claim_ids": ["C1"],
                "eligible_claims": [{
                    "claim_id": "C1", "evidence_ids": ["V1"],
                }],
            },
        )
        self.assertEqual(final_schema["properties"]["claims"]["minItems"], 1)
        self.assertEqual(final_schema["properties"]["status"], {"const": "answer"})
        self.assertEqual(final_schema["properties"]["answer_claim_ids"]["minItems"], 1)
        self.assertEqual(
            final_schema["properties"]["answer_claim_ids"]["items"]["enum"],
            ["C1"],
        )

    def test_protocol_rejects_incomplete_range_request(self):
        value = {
            "snapshot_id": "snapshot", "call_id": "Q1", "batch_id": "B0",
            "batch_digest": "digest", "status": "need_more",
            "inspect_edge_ids": [], "claims": [],
            "requests": [{"op": "range", "path": "src/a.ts", "reason": "inspect"}],
            "contradictions": [], "answer": "", "answer_claim_ids": [],
        }
        with self.assertRaisesRegex(loop.ModelProtocolError, "valid inclusive line range"):
            loop.validate_response_schema(
                value, snapshot_id="snapshot", call_id="Q1", batch_id="B0",
                batch_digest="digest", allowed_edge_ids=set(),
                allowed_evidence_ids=set(), allowed_claim_ids=set(),
            )

    def test_malformed_json_retries_once_with_same_protocol(self):
        snapshot = fixture_snapshot()
        gateway = ProtocolGateway(invalid_first="{broken")
        runner = loop.ExhaustiveLoop(None, Path("/unused"), gateway, context_window=10_000)
        with patch("exhaustive_loop.assert_snapshot_current"):
            result = runner._call(
                "plan", {"question": "q"}, snapshot, batch_id="B0",
                batch_digest="digest", allowed_edge_ids={"E0"},
            )
        self.assertEqual(result["batch_digest"], "digest")
        self.assertEqual(len(gateway.completions), 2)
        self.assertEqual(gateway.completions[0], gateway.tokenized[0])

    def test_invalid_cache_is_revalidated_before_use(self):
        snapshot = fixture_snapshot()
        gateway = ProtocolGateway()
        with tempfile.TemporaryDirectory() as directory, patch(
            "exhaustive_loop.assert_snapshot_current"
        ):
            runner = loop.ExhaustiveLoop(
                None, Path("/unused"), gateway, cache_dir=Path(directory),
                context_window=10_000,
            )
            arguments = dict(
                phase="plan", payload={"question": "q"}, snapshot=snapshot,
                batch_id="B0", batch_digest="digest", allowed_edge_ids={"E0"},
            )
            runner._call(**arguments)
            cache_file = next(Path(directory).glob("*.json"))
            cached = json.loads(cache_file.read_text())
            cached["batch_id"] = "WRONG"
            cache_file.write_text(json.dumps(cached))
            runner._call(**arguments)
        self.assertEqual(len(gateway.completions), 2)

    def test_evidence_does_not_reuse_legacy_cache_key(self):
        snapshot = fixture_snapshot()
        gateway = ProtocolGateway()
        payload = {"phase": "evidence", "question": "q", "evidence": []}
        prepared, call_id = loop.protocol_payload(
            "evidence", payload, "snapshot", "B0", "digest",
        )
        legacy_key = loop.digest([
            "snapshot", "evidence", prepared, loop.PROMPT_VERSION,
        ])
        stale = {
            "snapshot_id": "snapshot", "call_id": call_id, "batch_id": "B0",
            "batch_digest": "digest", "status": "need_more",
            "inspect_edge_ids": [], "claims": [], "requests": [],
            "contradictions": [], "answer": "", "answer_claim_ids": [],
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "exhaustive_loop.assert_snapshot_current"
        ):
            Path(directory, f"{legacy_key}.json").write_text(json.dumps(stale))
            runner = loop.ExhaustiveLoop(
                None, Path("/unused"), gateway, cache_dir=Path(directory),
                context_window=10_000,
            )
            runner._call(
                "evidence", payload, snapshot, batch_id="B0",
                batch_digest="digest", allowed_edge_ids=set(),
            )
        self.assertEqual(len(gateway.completions), 1)

    def test_reduce_never_creates_singleton_groups(self):
        snapshot = fixture_snapshot()
        gateway = ProtocolGateway(char_divisor=1000)
        runner = loop.ExhaustiveLoop(None, Path("/unused"), gateway, context_window=20_000)
        outputs = [{
            "status": "need_more", "answer": "", "claims": [], "requests": [],
            "inspect_edge_ids": [], "contradictions": [], "invalid_references": [],
            "answer_claim_ids": [],
        } for _ in range(7)]
        with patch("exhaustive_loop.assert_snapshot_current"):
            runner._reduce(
                "q", snapshot, outputs, "test", loop.EvidenceLedger(snapshot),
                loop.InvestigationLedger(),
            )
        reduce_payloads = [
            json.loads(messages[-1]["content"])
            for messages in gateway.completions
            if json.loads(messages[-1]["content"]).get("phase") == "test"
        ]
        self.assertTrue(reduce_payloads)
        self.assertTrue(all(len(payload["inputs"]) >= 2 for payload in reduce_payloads))

    def test_ledger_preserves_omitted_claims_and_contradictions(self):
        snapshot = fixture_snapshot()
        evidence = loop.EvidenceLedger(snapshot)
        ledger = loop.InvestigationLedger()
        first = loop.validate_model_response({
            "status": "need_more", "claims": [{
                "claim_id": "first", "text": "possible behavior", "kind": "inference",
                "support": "unknown", "evidence_ids": [], "edge_ids": ["E0"],
            }], "requests": [], "inspect_edge_ids": [], "contradictions": [],
            "answer_claim_ids": [],
        }, snapshot, evidence)
        ledger.merge(first)
        result = ledger.merge({
            "status": "need_more", "claims": [], "requests": [],
            "inspect_edge_ids": [], "contradictions": [{
                "text": "still uncertain", "claim_ids": [first["claims"][0]["claim_id"]],
            }], "invalid_references": [], "answer_claim_ids": [],
        })
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(len(result["contradictions"]), 1)

    def test_oversized_edge_opens_callsite_without_mislabeling_preview(self):
        base = fixture_snapshot()
        edge = loop.EdgeRecord(
            "E0", "src/service.ts", "availability", "availability", "calls",
            "helper", "src/helper.ts", "helper", 245, "import_resolved", {},
        )
        snapshot = loop.FrozenSnapshot(
            base.descriptor, base.files, base.symbols, (edge,), base.authorities,
        )
        evidence = loop.EvidenceLedger(snapshot, max_symbol_bytes=30).expand_edge("E0")
        preview = next(item for item in evidence if item.kind == "symbol" and item.path == "src/service.ts")
        callsite = next(item for item in evidence if item.kind == "callsite")
        self.assertNotIn("E0", preview.metadata.get("edge_ids", []))
        self.assertIn("E0", callsite.metadata["edge_ids"])
        self.assertLessEqual(callsite.start_line, 245)
        self.assertGreaterEqual(callsite.end_line, 245)

    def test_homonymous_target_uses_path_and_qualified_name(self):
        base = fixture_snapshot()
        other = loop.SymbolRecord(
            "S2", "src/other.ts", "function", "helper", "Other.helper",
            1, 2, True, "code", {},
        )
        snapshot = loop.FrozenSnapshot(
            base.descriptor,
            base.files | {"src/other.ts": "wrong\nhelper"},
            base.symbols + (other,), base.edges, base.authorities | {"src/other.ts": "code"},
        )
        opened = loop.EvidenceLedger(snapshot).expand_edge("E0")
        target_paths = {item.path for item in opened if "E0" in item.metadata.get("target_of_edge_ids", [])}
        self.assertEqual(target_paths, {"src/helper.ts"})

    def test_answer_requires_valid_grounding_claims(self):
        snapshot = fixture_snapshot()
        result = loop.validate_model_response({
            "status": "answer", "answer": "invented", "answer_claim_ids": ["fake"],
            "claims": [], "requests": [], "inspect_edge_ids": [], "contradictions": [],
        }, snapshot, loop.EvidenceLedger(snapshot))
        self.assertEqual(result["status"], "need_more")
        self.assertEqual(result["answer"], "")

    def test_invalid_contradiction_reference_is_preserved_and_flagged(self):
        snapshot = fixture_snapshot()
        raw = {
            "snapshot_id": "snapshot", "call_id": "Q1", "batch_id": "B0",
            "batch_digest": "digest", "status": "need_more",
            "inspect_edge_ids": [], "claims": [], "requests": [],
            "contradictions": [{"text": "conflict", "claim_ids": ["C_FAKE"]}],
            "answer": "", "answer_claim_ids": [],
        }
        loop.validate_response_schema(
            raw, snapshot_id="snapshot", call_id="Q1", batch_id="B0",
            batch_digest="digest", allowed_edge_ids={"E0"},
            allowed_evidence_ids=set(), allowed_claim_ids=set(),
        )
        result = loop.validate_model_response(
            raw, snapshot, loop.EvidenceLedger(snapshot),
        )
        self.assertEqual(result["contradictions"][0]["text"], "conflict")
        self.assertEqual(result["contradictions"][0]["claim_ids"], [])
        self.assertEqual(result["contradictions"][0]["invalid_claim_ids"], ["C_FAKE"])
        self.assertTrue(result["invalid_references"])

    def test_mixed_generation_snapshot_aborts(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute(
            "CREATE TABLE repos(project TEXT PRIMARY KEY,commit_hash TEXT,content_digest TEXT,extractor_version TEXT,generation_id TEXT)"
        )
        db.execute(
            "CREATE TABLE chunks(project TEXT,path TEXT,commit_hash TEXT,generation_id TEXT)"
        )
        structure.ensure_schema(db)
        fingerprint = structure.extractor_fingerprint()
        db.execute("INSERT INTO repos VALUES('sample','abc','bytes',?, 'g1')", (fingerprint,))
        db.execute("INSERT INTO chunks VALUES('sample','a.ts','abc','g1')")
        db.execute("INSERT INTO source_authority VALUES('sample','a.ts','code','x','abc','g1')")
        db.execute(
            "INSERT INTO symbols(project,path,kind,name,qualified_name,start_line,end_line,commit_hash,generation_id) "
            "VALUES('sample','a.ts','function','a','a',1,1,'abc','g1')"
        )
        db.execute(
            "INSERT INTO edges(project,source_path,source_name,relation,target_name,line,commit_hash,generation_id) "
            "VALUES('sample','a.ts','a','calls','b',1,'abc','g2')"
        )
        db.execute(
            "INSERT INTO structure_snapshots(project,commit_hash,graph_digest,content_digest,extractor_version,generation_id) "
            "VALUES('sample','abc','graph','bytes',?,'g1')", (fingerprint,),
        )
        with patch("exhaustive_loop.git", return_value="abc"), patch(
            "exhaustive_loop.require_clean_repository"
        ), patch("exhaustive_loop.repository_content_digest", return_value="bytes"):
            with self.assertRaises(loop.SnapshotChanged):
                loop.freeze_snapshot(db, "sample", Path("/repo"))

    def test_typescript_extractor_keeps_untracked_common_calls_and_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            source = repo / "sample.ts"
            source.write_text(
                "function resolve() {}\n"
                "class A { get() {} run(client:any) { this.get(); client.update(); resolve(); client.select(); } }\n"
                "class B { get() {} run() { this.get(); } }\n"
            )
            completed = subprocess.run(
                ["node", str(Path(loop.__file__).with_name("ts_structure.mjs")), str(repo), "test"],
                check=True, text=True, capture_output=True,
            )
        payload = json.loads(completed.stdout)
        calls = [edge for edge in payload["edges"] if edge["relation"] == "calls"]
        targets = {edge["target_name"] for edge in calls}
        self.assertTrue({"get", "update", "resolve", "select"} <= targets)
        self.assertTrue(all(edge["source_name"] != "" for edge in calls))
        class_gets = {
            edge["target_qualified_name"] for edge in calls
            if edge["target_name"] == "get" and edge["resolution_confidence"] == "same_file"
        }
        self.assertEqual(class_gets, {"A.get", "B.get"})

    def test_typescript_extractor_records_imported_inheritance_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "Object3D.js").write_text(
                "export class Object3D { copy(source) { return this; } clone() { return new this.constructor().copy(this); } }\n"
            )
            (repo / "Line.js").write_text(
                "import { Object3D } from './Object3D.js';\n"
                "export class Line extends Object3D { copy(source) { super.copy(source); return this; } }\n"
            )
            completed = subprocess.run(
                ["node", str(Path(loop.__file__).with_name("ts_structure.mjs")), str(repo), "test"],
                check=True, text=True, capture_output=True,
            )
        payload = json.loads(completed.stdout)
        extends = next(edge for edge in payload["edges"] if edge["relation"] == "extends")
        override = next(edge for edge in payload["edges"] if edge["relation"] == "overrides")
        clone_copy = next(
            edge for edge in payload["edges"]
            if edge["relation"] == "calls" and edge["source_qualified_name"] == "Object3D.clone"
        )
        self.assertEqual(
            (extends["source_name"], extends["target_name"], extends["target_path"]),
            ("Line", "Object3D", "Object3D.js"),
        )
        self.assertEqual(extends["resolution_confidence"], "import_resolved_inheritance")
        self.assertEqual(
            (
                override["source_qualified_name"], override["target_qualified_name"],
                override["target_path"],
            ),
            ("Line.copy", "Object3D.copy", "Object3D.js"),
        )
        self.assertEqual(
            (
                clone_copy["target_qualified_name"], clone_copy["target_path"],
                clone_copy["resolution_confidence"],
            ),
            ("Object3D.copy", "Object3D.js", "same_file"),
        )

    def test_typescript_extractor_names_nested_function_expressions(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "sample.js").write_text(
                "export default true && function httpAdapter(config) { return config; }\n"
                "const defaults = {\n"
                "  transformRequest: [function transformRequest(data) { return data; }]\n"
                "};\n"
            )
            completed = subprocess.run(
                ["node", str(Path(loop.__file__).with_name("ts_structure.mjs")), str(repo), "test"],
                check=True, text=True, capture_output=True,
            )
        payload = json.loads(completed.stdout)
        functions = {
            symbol["name"]: symbol for symbol in payload["symbols"]
            if symbol["kind"] == "function"
        }
        self.assertEqual(
            {"httpAdapter", "transformRequest"} & set(functions),
            {"httpAdapter", "transformRequest"},
        )
        self.assertEqual(functions["httpAdapter"]["start_line"], 1)
        self.assertEqual(functions["transformRequest"]["start_line"], 3)

    def test_tests_and_active_sql_expand_complete_symbols(self):
        base = fixture_snapshot()
        test_symbol = loop.SymbolRecord(
            "S2", "src/service_test.ts", "function", "availability_test",
            "availability_test", 1, 3, True, "test", {},
        )
        active_sql = loop.SymbolRecord(
            "S3", "supabase/migrations/2.sql", "function", "public.lookup",
            "public.lookup", 1, 4, True, "schema_history", {},
        )
        old_sql = loop.SymbolRecord(
            "S4", "supabase/migrations/1.sql", "function", "public.lookup",
            "public.lookup", 1, 4, False, "schema_history", {},
        )
        test_edge = loop.EdgeRecord(
            "E2", "src/service_test.ts", "availability_test", "availability_test",
            "calls", "availability", "src/service.ts", "availability", 2,
            "import_resolved", {},
        )
        snapshot = loop.FrozenSnapshot(
            base.descriptor,
            base.files | {
                "src/service_test.ts": "test\navailability()\nend",
                "supabase/migrations/2.sql": "create\nbody\nactive\nend",
                "supabase/migrations/1.sql": "create\nbody\nold\nend",
            },
            base.symbols + (test_symbol, active_sql, old_sql),
            base.edges + (test_edge,),
            base.authorities | {
                "src/service_test.ts": "test",
                "supabase/migrations/2.sql": "schema_history",
                "supabase/migrations/1.sql": "schema_history",
            },
        )
        evidence = loop.EvidenceLedger(snapshot)
        tests = evidence.resolve_request({"op": "tests", "name": "availability"})
        sql = evidence.resolve_request({"op": "sql", "name": "public.lookup"})
        self.assertTrue(any(item.path == "src/service_test.ts" and item.complete for item in tests))
        self.assertEqual({item.path for item in sql}, {"supabase/migrations/2.sql"})

    def test_index_publication_rolls_back_as_one_generation(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.executescript(
            """
            CREATE TABLE repos(
              project TEXT PRIMARY KEY,root TEXT,commit_hash TEXT,content_digest TEXT,
              extractor_version TEXT,generation_id TEXT,indexed_at TEXT
            );
            CREATE TABLE chunks(
              id INTEGER PRIMARY KEY,project TEXT,path TEXT,start_line INTEGER,
              end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT
            );
            """
        )
        structure.ensure_schema(db)
        db.execute(
            "INSERT INTO repos VALUES('sample','/old','old','old-bytes','old-ext','old-gen','now')"
        )
        db.execute(
            "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
            "VALUES('sample','old.ts',1,1,'old','old-gen','old')"
        )
        db.commit()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "new.ts").write_text("new")

            def fail_structure(*args, **kwargs):
                db.execute(
                    "INSERT INTO source_authority(project,path,authority,reason,commit_hash,generation_id) "
                    "VALUES('sample','new.ts','code','new','new','new-gen')"
                )
                raise RuntimeError("extractor failed")

            with patch.object(brain, "REPOS", {"sample": repo}), patch(
                "project_brain.require_clean_repository"
            ), patch("project_brain.git", return_value="new"), patch(
                "project_brain.repository_content_digest", return_value="new-bytes"
            ), patch("project_brain.tracked_files", return_value=[Path("new.ts")]), patch(
                "project_brain.connect", return_value=db
            ), patch("project_brain.index_structure", side_effect=fail_structure):
                with self.assertRaisesRegex(RuntimeError, "extractor failed"):
                    brain.index_project("sample", force=True)
        self.assertEqual(db.execute("SELECT content FROM chunks").fetchone()[0], "old")
        self.assertEqual(db.execute("SELECT generation_id FROM repos").fetchone()[0], "old-gen")
        self.assertEqual(db.execute("SELECT count(*) FROM source_authority").fetchone()[0], 0)

    def test_working_tree_snapshot_includes_safe_untracked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "tracked.ts").write_text("tracked")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.ts"], check=True)
            (repo / "new.ts").write_text("untracked")
            (repo / ".env").write_text("SECRET=hidden")
            (repo / ".gitignore").write_text("ignored.ts\n")
            (repo / "ignored.ts").write_text("ignored")

            paths = {str(path) for path in brain.working_tree_files(repo)}
            self.assertEqual(paths, {"tracked.ts", "new.ts"})
            first = brain.repository_content_digest(repo)
            (repo / "new.ts").write_text("changed")
            self.assertNotEqual(first, brain.repository_content_digest(repo))

    def test_structural_file_listing_includes_untracked_nonignored_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "tracked.ts").write_text("tracked")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.ts"], check=True)
            (repo / "new.ts").write_text("untracked")
            (repo / ".gitignore").write_text("ignored.ts\n")
            (repo / "ignored.ts").write_text("ignored")
            self.assertEqual(
                {str(path) for path in structure.subprocess_files(repo)},
                {"tracked.ts", "new.ts", ".gitignore"},
            )

    def test_publish_chunks_preserves_only_unchanged_embeddings(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY,project TEXT,path TEXT,"
            "start_line INTEGER,end_line INTEGER,commit_hash TEXT,generation_id TEXT,content TEXT)"
        )
        vectors.ensure_schema(db)
        db.executemany(
            "INSERT INTO chunks(project,path,start_line,end_line,commit_hash,generation_id,content) "
            "VALUES('sample',?,?,?,?,?,?)",
            [
                ("same.ts", 1, 1, "old", "old-gen", "unchanged"),
                ("changed.ts", 1, 1, "old", "old-gen", "before"),
            ],
        )
        ids = {row["path"]: row["id"] for row in db.execute("SELECT id,path FROM chunks")}
        db.executemany(
            "INSERT INTO embeddings(chunk_id,model,dimension,vector) VALUES(?,?,1,?)",
            [(ids["same.ts"], vectors.EMBED_MODEL, b"same"),
             (ids["changed.ts"], vectors.EMBED_MODEL, b"changed")],
        )

        with db:
            brain.publish_chunks(db, "sample", "new", "new-gen", [
                ("same.ts", 1, 1, "unchanged"),
                ("changed.ts", 1, 1, "after"),
            ])

        rows = {row["path"]: row for row in db.execute("SELECT * FROM chunks")}
        self.assertEqual(rows["same.ts"]["id"], ids["same.ts"])
        self.assertNotEqual(rows["changed.ts"]["id"], ids["changed.ts"])
        embedded_ids = {row[0] for row in db.execute("SELECT chunk_id FROM embeddings")}
        self.assertEqual(embedded_ids, {ids["same.ts"]})

    def test_semantic_status_recommends_deferred_incremental_maintenance(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY,project TEXT,path TEXT,content TEXT)"
        )
        vectors.ensure_schema(db)
        db.executemany(
            "INSERT INTO chunks(project,path,content) VALUES('sample',?,?)",
            [(f"file-{index}.ts", "content") for index in range(20)],
        )
        ids = [row[0] for row in db.execute("SELECT id FROM chunks ORDER BY id LIMIT 19")]
        db.executemany(
            "INSERT INTO embeddings(chunk_id,model,dimension,vector) VALUES(?,?,1,?)",
            [(chunk_id, vectors.EMBED_MODEL, b"x") for chunk_id in ids],
        )
        status = brain.semantic_status(db, "sample")
        self.assertFalse(status["semantic_current"])
        self.assertEqual(status["pending_embeddings"], 1)
        self.assertEqual(status["recommendation"], "defer")

    def test_refresh_embeddings_is_exposed_separately_from_repository_refresh(self):
        tools = {item["name"]: item for item in mcp.TOOLS}
        self.assertIn("refresh_repository", tools)
        self.assertIn("refresh_embeddings", tools)
        self.assertIn("Does not start the embedding encoder", tools["refresh_repository"]["description"])


class GeminiGatewayTests(unittest.TestCase):
    def test_count_tokens_uses_exact_generate_content_request(self):
        payload = {
            "phase": "plan",
            "response_schema": {
                "snapshot_id": "snapshot",
                "call_id": "Q1",
                "batch_id": "B0",
                "batch_digest": "digest",
            },
        }
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(payload)},
        ]
        gateway = loop.GeminiGateway("secret", max_output_tokens=321)
        response = {
            "candidates": [{
                "content": {"parts": [{"text": "{\"status\":\"ok\"}"}]},
            }],
        }
        with patch.object(
            gateway, "_post", side_effect=[{"totalTokens": 123}, response],
        ) as post:
            self.assertEqual(gateway.token_count(messages), 123)
            self.assertEqual(gateway.complete(messages), '{"status":"ok"}')
        count_request = post.call_args_list[0].args[1]["generateContentRequest"]
        generate_request = post.call_args_list[1].args[1]
        self.assertEqual(count_request, generate_request)
        self.assertEqual(
            count_request["model"], "models/gemini-3.6-flash",
        )
        self.assertEqual(
            count_request["systemInstruction"]["parts"][0]["text"], "system",
        )
        self.assertEqual(
            count_request["generationConfig"],
            {"maxOutputTokens": 321, "responseMimeType": "application/json"},
        )
        self.assertNotIn(
            "responseJsonSchema", count_request["generationConfig"],
        )

    def test_gemini_repair_uses_stable_wire_schema(self):
        gateway = loop.GeminiGateway("secret")
        messages = [
            {"role": "system", "content": "repair"},
            {"role": "user", "content": json.dumps({
                "invalid_response": "{broken", "error": "invalid JSON",
            })},
        ]
        request = gateway.request_payload(messages)
        schema = request["generationConfig"]["responseJsonSchema"]
        self.assertEqual(schema["required"], sorted(loop.RESPONSE_KEYS))
        self.assertEqual(
            schema["properties"]["status"]["enum"],
            sorted(loop.RESULT_STATUSES),
        )
        self.assertNotIn(
            "responseJsonSchema",
            gateway.request_payload([
                {"role": "user", "content": json.dumps({"question": "normal"})},
            ])["generationConfig"],
        )

    def test_transient_gemini_http_error_retries_same_payload(self):
        gateway = loop.GeminiGateway("secret")
        unavailable = loop.HTTPError(
            "https://example.invalid", 503, "busy", {},
            io.BytesIO(b'{"error":{"status":"UNAVAILABLE"}}'),
        )
        with patch(
            "exhaustive_loop.urlopen",
            side_effect=[unavailable, io.BytesIO(b'{"ok":true}')],
        ) as opened, patch("exhaustive_loop.time.sleep") as slept:
            self.assertEqual(
                gateway._post("generateContent", {"same": "payload"}, 10),
                {"ok": True},
            )
        self.assertEqual(opened.call_count, 2)
        slept.assert_called_once_with(1)

    def test_cache_is_separated_by_provider_and_model(self):
        snapshot = fixture_snapshot()
        first = ProtocolGateway()
        first.provider = "gemini"
        first.model = "gemini-a"
        first.cache_namespace = "gemini:gemini-a"
        second = ProtocolGateway()
        second.provider = "gemini"
        second.model = "gemini-b"
        second.cache_namespace = "gemini:gemini-b"
        with tempfile.TemporaryDirectory() as directory, patch(
            "exhaustive_loop.assert_snapshot_current",
        ):
            arguments = dict(
                phase="plan", payload={"question": "q"}, snapshot=snapshot,
                batch_id="B0", batch_digest="digest", allowed_edge_ids={"E0"},
            )
            loop.ExhaustiveLoop(
                None, Path("/unused"), first, cache_dir=Path(directory),
                context_window=10_000,
            )._call(**arguments)
            loop.ExhaustiveLoop(
                None, Path("/unused"), second, cache_dir=Path(directory),
                context_window=10_000,
            )._call(**arguments)
        self.assertEqual(len(first.completions), 1)
        self.assertEqual(len(second.completions), 1)

    def test_gemini_secret_file_is_read_without_shell_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "gemini.env"
            secret.write_text("GEMINI_API_KEY=value-with-shell-text-$HOME\n")
            secret.chmod(0o600)
            with patch.object(brain, "GEMINI_SECRET_FILE", secret), patch.dict(
                brain.os.environ, {"GEMINI_API_KEY": ""}, clear=False,
            ):
                self.assertEqual(
                    brain.load_gemini_api_key(), "value-with-shell-text-$HOME",
                )
                secret.chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, "group/world readable"):
                    brain.load_gemini_api_key()


if __name__ == "__main__":
    unittest.main()
