# Architecture

Project Brain is an external memory layer. It does not alter a model's weights, attention,
logits, or KV cache. It spends local compute before inference to decide which repository
evidence deserves to become model input.

## Pipeline

```text
registered Git checkout
  ├─ safe file enumeration
  ├─ text chunks + source authority ──────────────┐
  ├─ multilingual embeddings                     │
  └─ TS/JS/SQL structural extraction             │
       ├─ symbols                                 │
       ├─ imports / calls / inheritance           │
       ├─ RPC consumers                           │
       └─ SQL definitions and lineage             │
                                                  ▼
engineering question → lexical + semantic candidates → bounded graph expansion
                                                  │
                                                  ▼
                              compact anchors or deep context dossier
```

## Persistent state

One local SQLite database stores chunks, vectors, symbols, edges, source authority, and
snapshot generations. A separate JSON registry maps user-chosen project names to explicitly
authorized absolute Git roots.

The index describes a working tree, not just a commit. Its generation identity combines the
current commit, a digest of safe tracked and untracked source, and the structural extractor
fingerprint. Structural and semantic freshness are reported separately so an agent can use
a fresh graph while a bounded amount of embedding work remains.

## Retrieval

The first stage fuses SQLite FTS5 and the multilingual embedding index. Ranking also accounts
for source authority and active SQL definitions. Results are over-fetched before path
diversification so a frequently repeated test term cannot occupy every returned slot.

For complex questions, a second code-aware encoder can rerank at most 60 candidates. Its
document vectors are cached by content hash. This is a bounded reranker, not another complete
repository embedding pass.

## Structural graph

The TypeScript compiler API extracts JavaScript and TypeScript modules, declarations, calls,
imports, inheritance, and likely overrides. The SQL extractor records definitions,
dependencies, and consumers. Later same-name SQL definitions are marked active while earlier
ones remain available as lineage.

Edges express facts found by static extraction at a stated confidence. They are not proof of
which branch executes at runtime. Dynamic dispatch, reflection, generated code, runtime SQL,
and framework conventions can escape the graph.

## Context products

Compact tools return short, diverse citations and bounded adjacency hints. They are intended
to orient an agent before normal source navigation.

`load_deep_context` composes retrieval, flow expansion, contract comparison, and exact source
bodies into one declared budget. It verifies one index generation before reading and fails
closed if the checkout has changed. Its token count is a transparent character-based estimate
because the MCP process does not own the client's tokenizer.

## Trust model

- The local checkout and its tests remain the source of truth.
- Project registration is CLI-only; MCP tools cannot authorize new paths.
- Every citation includes project, path, lines, and indexed revision metadata.
- Retrieval omissions are expected and must remain visible.
- A coding agent should verify decisive files directly before changing them.

## Known limitations

- Structural coverage is strongest for TypeScript, JavaScript, and SQL.
- Static graphs cannot fully resolve reflection, runtime registration, or generated behavior.
- Embedding similarity is relevance evidence, not causality.
- Source-authority heuristics are conventions and may need adaptation for unusual repositories.
- Context reduction can omit a decisive file; normal search remains an intentional fallback.
