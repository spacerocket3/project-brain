# Evaluation

Project Brain should be evaluated as an additional map, not as a replacement for engineering
tools. A benchmark that forbids `rg` in one arm measures an artificial constraint rather than
whether the map improves an agent.

## Claims that require different evidence

1. **Retrieval quality**: did the result include the decisive files and relationships?
2. **Context transport**: how many repository-derived tokens entered the model?
3. **Task quality**: was the implementation correct, complete, and regression-safe?
4. **Efficiency**: total input/output tokens, tool calls, wall time, retries, and local compute.

A compact response can have excellent transport efficiency and still omit the crucial file.
A correct answer from one run does not establish a reliable quality improvement.

## Paired agent protocol

Use at least two clean worktrees at the same commit.

| Control | Project Brain arm |
| --- | --- |
| Same model and reasoning setting | Same model and reasoning setting |
| Same task prompt | Same task prompt |
| Same shell, network, and permissions | Same shell, network, and permissions |
| Normal `rg`, reads, and tests | Normal `rg`, reads, tests, plus Project Brain |
| Same time or token ceiling | Same time or token ceiling |

Do not prescribe a detailed search sequence. A short orientation such as “Project Brain is
available as an additional repository map” is enough. Otherwise the prompt itself can dominate
agent behavior.

Run multiple seeds or fresh sessions, randomize arm order, and keep concurrent tests from
contending for fixed ports. Judge the resulting diff blind when practical.

## Record per run

- exact repository commit and task;
- model, client/runtime version, reasoning setting, and context window;
- Project Brain index generation and retrieval mode;
- files changed and decisive files missed;
- tests attempted, passed, failed, and skipped;
- introduced regressions and unrelated environment failures;
- input, cached input, output, and repository-context tokens when available;
- MCP calls, shell searches, direct reads, elapsed time, and retries;
- final diff or immutable artifact hash.

## Scoring

Prioritize executable correctness. A useful ordering is:

1. acceptance tests and behavioral invariants;
2. security and backward compatibility;
3. completeness of the requested scope;
4. unnecessary code and regressions;
5. repository-context tokens and total cost;
6. time and tool-call count.

Report raw measurements alongside any aggregate score. Do not present estimated MCP transport
tokens as exact provider billing tokens.

## Current evidence level

Development trials motivated the bounded retrieval design and exposed both successes and
misses, including decisive files later found through direct search. Those trials were not a
pre-registered, repeated agent benchmark. The public release therefore makes no universal
percentage claim. Reproducible benchmark cases and negative results are welcome.
