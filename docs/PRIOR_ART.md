# Prior art and positioning

Project Brain builds on established ideas rather than claiming the invention of repository
mapping or retrieval-augmented coding.

Relevant categories include:

- repository maps that rank symbols and dependencies under a token budget;
- codebase indexing with chunked embeddings and incremental updates;
- code graphs that connect definitions, references, callers, and callees;
- hybrid lexical and semantic retrieval for coding assistants;
- graph-native MCP servers and code-memory systems;
- retrieval-augmented code generation research such as RepoCoder and Repoformer.

Representative public systems and references:

- [Aider repository map](https://aider.chat/docs/repomap.html)
- [Sourcegraph Cody context](https://sourcegraph.com/docs/cody/core-concepts/context)
- [Codebase Memory MCP](https://github.com/DeusData/codebase-memory-mcp)
- [RepoCoder](https://arxiv.org/abs/2303.12570)
- [Repoformer](https://arxiv.org/abs/2403.10059)

Project Brain's present focus is the combination of:

- explicit local repository authorization;
- current working-tree generations rather than commit-only snapshots;
- source authority and active/superseded SQL lineage;
- semantic retrieval plus inspectable structural relations;
- compact and deep MCP products with declared bounds;
- honest diagnostics about missing facets and lexical concentration.

The project should be compared on reproducible retrieval and agent outcomes, not novelty by
analogy. Similar systems may outperform it in language coverage, scale, UX, or mature graph
queries; those are useful baselines and collaboration opportunities.
