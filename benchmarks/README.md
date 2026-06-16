# Benchmarks

Rune benchmark prompts live in `benchmarks/prompts/*.jsonl`. Each line is one JSON object.
The frozen suite version is recorded in `benchmarks/prompts/manifest.json`.

## Prompt Schema

Required fields:

- `id`: stable string identifier unique within the file
- `family`: one of `algorithmic`, `paraphrase`, `counterfactual`, `random_memorization`, `carrier`
- `task`: task label such as `modadd`, `date_arithmetic`, `copy_induction`, or `bracket_matching`
- `input`: prompt text or compact symbolic input
- `expected`: expected symbolic answer, token, or label
- `metadata`: object with task-specific details

Optional fields:

- `pair_id`: shared identifier for counterfactual or paraphrase groups
- `tags`: list of short strings used for filtering benchmark slices

Subtemplate-decomposed prompt records use `metadata.template` and the `subtemplate` tag, for example IOI-style `ABBA` and `BABA` slices.

The prompt suite is deliberately model-agnostic. Tokenization, chat wrapping, and model-specific formatting belong in loaders, not in these JSONL files.