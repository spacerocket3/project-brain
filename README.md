# Project Brain

**Give your coding agent a map, not another giant prompt.**

[![CI](https://github.com/spacerocket3/project-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/spacerocket3/project-brain/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Project Brain is a local repository cartographer for coding agents. It combines exact
search, embeddings, source authority, and a structural graph to return the small set of
files, symbols, SQL definitions, tests, and relationships most likely to matter for a
change.

It does not replace `rg`, source reading, reasoning, or tests. It gives the agent a better
place to start and exposes relationships that do not necessarily share the same words.

```text
engineering question
        │
        ▼
 exact search + semantic retrieval
        │
        ▼
 symbols + imports + calls + SQL lineage
        │
        ▼
 source authority + active/superseded definitions
        │
        ▼
 compact, cited repository map
        │
        ▼
 coding agent → direct inspection → tests
```

## Why it exists

An unfamiliar codebase contains relationships a general model could not have learned:
an RPC redefined by a later migration, a generated type consumed by a UI query, a cache
key that must be invalidated, or an override hidden behind virtual dispatch. Agents can
reconstruct those relationships through repeated searches. Project Brain persists that
map locally and retrieves a task-specific working set on demand.

The practical contract is intentionally narrow:

- Project Brain answers **what is related even when the vocabulary differs?**
- `rg` answers **where does this exact text occur?**
- direct source inspection confirms current behavior;
- executable tests confirm the change.

## What it maps

- Git-tracked source plus safe, untracked, non-ignored source files.
- Chunked full-text search with path, line, commit, generation, and authority metadata.
- Multilingual semantic retrieval with an optional code-aware reranking pass.
- TypeScript and JavaScript modules, symbols, imports, calls, inheritance, and overrides.
- SQL objects, RPC consumers, dependency lineage, and later redefinitions.
- Tests, configuration, generated surfaces, historical documentation, and canonical docs
  as distinct evidence classes.
- Freshness of both the structural index and semantic embeddings.

Other text-based languages remain searchable. Structural extraction is currently deepest
for TypeScript, JavaScript, and SQL.

## Quick start

Requirements: Git, Python 3.11+, Node.js, and npm.

```bash
git clone https://github.com/spacerocket3/project-brain.git
cd project-brain
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
```

Register a local Git checkout explicitly:

```bash
./project-brain project add my-app /absolute/path/to/my-app \
  --description "Local checkout used by coding agents"
./project-brain index my-app
./project-brain embed my-app
./project-brain status
```

Runtime state stays local and is ignored by Git:

- repository registry: `data/projects.json`;
- SQLite index: `data/index.sqlite3`;
- model cache: `~/.cache/project-brain/models`.

Override those locations with `PROJECT_BRAIN_PROJECTS_FILE`, `PROJECT_BRAIN_DB`,
and `PROJECT_BRAIN_MODEL_CACHE`.

Registration is the filesystem authorization boundary. Only the CLI can add an absolute
Git root; MCP calls accept registered names, never arbitrary paths. Removing a project
deletes its local index and registry entry, never the checkout.

## Use it from MCP

For Codex CLI:

```bash
codex mcp add project-brain -- \
  /absolute/path/to/project-brain/.venv/bin/python \
  /absolute/path/to/project-brain/mcp_server.py
```

The same stdio server can run over SSH without exposing an HTTP port:

```bash
codex mcp add project-brain -- \
  ssh -T your-host \
  /absolute/path/to/project-brain/.venv/bin/python \
  /absolute/path/to/project-brain/mcp_server.py
```

The default MCP surface contains eight tools:

| Tool | Purpose |
| --- | --- |
| `project_brain_status` | List registered checkouts and index freshness. |
| `search_repository` | Return at most six short, diverse, cited anchors. |
| `inspect_symbol` | Find exact definitions and compact caller/callee relationships. |
| `investigate_flow` | Expand anchors into a bounded cross-layer flow map. |
| `diagnose_retrieval` | Explain missing facets, concentration, and retrieval weaknesses. |
| `load_deep_context` | Build one larger, token-budgeted dossier for a difficult task. |
| `refresh_repository` | Refresh current text and structure without editing the checkout. |
| `refresh_embeddings` | Embed only new or changed chunks. |

Every tool except status requires exactly one registered project. This prevents accidental
mixing between repositories.

### Recommended agent workflow

For routine work:

1. Call `project_brain_status`.
2. Ask `search_repository` one complete engineering question.
3. Inspect only the decisive symbols.
4. Open those files in the checkout and continue normally.

For difficult cross-layer work, replace steps 2–3 with one `load_deep_context` call. The
dossier includes exact source bodies where budget permits, but it remains navigation
evidence rather than proof.

## Retrieval modes

`search_repository`, `investigate_flow`, and `diagnose_retrieval` accept:

- `fast`: full-text plus the multilingual embedding index;
- `auto`: use the code-aware reranker for complex questions only if it is already cached;
- `code`: require the bounded code-aware reranker.

The code-aware pass reranks at most 60 candidates. It caches document vectors by content
instead of building a second full repository index.

It can also use a separate OpenAI-compatible embedding endpoint:

```bash
export PROJECT_BRAIN_CODE_EMBED_URL=http://127.0.0.1:8002/v1/embeddings
export PROJECT_BRAIN_CODE_EMBED_MODEL=qwen3-embedding-8b
```

For example, a local vLLM service can provide the embeddings:

```bash
docker run --rm --gpus all -p 127.0.0.1:8002:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:v0.20.0 Qwen/Qwen3-Embedding-8B \
  --served-model-name qwen3-embedding-8b --runner pooling --max-model-len 8192
```

An unavailable remote endpoint falls back only in `auto`; explicit `code` mode fails so an
experiment cannot silently change methods.

## Deep context without a giant blind dump

`load_deep_context` performs one composed retrieval, graph expansion, relationship check,
and contract comparison. It then fills a declared budget with verified source from a
single indexed working-tree generation.

```json
{
  "project": "my-app",
  "question": "Implement this cross-layer change, preserve compatibility, and add regression tests",
  "semantic_mode": "code",
  "max_context_tokens": 36000
}
```

The accepted estimate is 16,000–60,000 tokens and defaults to 36,000. Project Brain cannot
access the client model's tokenizer, so it clearly labels its deterministic compact-JSON
estimate and reports included and omitted evidence. It fails closed if the checkout is
newer than the index.

## Does it save tokens?

It is designed to reduce **repository context transported into the coding model**, not to
make universal cost claims. The compact tools bound snippets and fan-out; the deep tool
declares its budget. Indexing and embedding consume local compute but no model context.

Whether total task tokens fall depends on the agent, repository, question, cache behavior,
and how often direct verification is needed. This repository deliberately makes no “90%”
or quality claim without a reproducible paired evaluation. See
[Evaluation](docs/EVALUATION.md) for the protocol and measurements that count.

## Safety

- Secret files, private keys, dependency trees, build output, binaries, and oversized files
  are excluded from indexing.
- The optional HTTP server binds to `127.0.0.1` by default.
- Index generations include the current commit and working-tree digest.
- `active=0` means a SQL definition is superseded in repository order; it remains useful
  lineage, not active runtime behavior.
- A graph edge is a navigation fact, never automatic proof of runtime causality.
- Refresh operations write only Project Brain's local data directory.

Read [Architecture](docs/ARCHITECTURE.md) for the trust model and limitations.

## Optional experimental generation tools

The original research prototype also contains model-specific investigation loops. They are
hidden from MCP by default because an MCP client already supplies the reasoning model and
extra choices made agents less consistent in early trials. Set
`PROJECT_BRAIN_ENABLE_GENERATION_TOOLS=1` only if you intentionally want to experiment with
that legacy surface. It is not part of the stable cartographer contract.

## Tests

```bash
.venv/bin/python -m unittest -v
npm audit --omit=dev
```

For a meaningful agent evaluation, use clean worktrees at the same commit, identical
prompts and permissions, and allow ordinary shell tools in both arms. Compare correctness,
regressions, tests, investigation effort, wall time, and token transport—not just whether
the agent mentioned Project Brain.

## Prior art and scope

Repository maps, code graphs, semantic search, and retrieval-augmented coding all have
substantial prior art. Project Brain's contribution is the particular local, temporal,
authority-aware combination and a deliberately bounded MCP handoff. It is a practical
experiment, not a claim that repository mapping was invented here. See
[Prior art](docs/PRIOR_ART.md).

## Contributing

Issues, reproducible retrieval misses, language extractors, and controlled agent benchmarks
are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

Apache-2.0 © Nicolas Cenci.
