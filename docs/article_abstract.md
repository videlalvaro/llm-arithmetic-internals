# Article Abstract

**Arithmetic Without Numbers** asks a simple question: what kind of arithmetic can grow inside a language model that has no fingers, no abacus, no scratch paper, and no written columns, only matrices?

The article follows a month of mechanism-aware experiments on language-model arithmetic. The central story is not that a model was turned into a calculator. Ordinary tool use can already do that. The more interesting question is whether the operation and operands for a calculator route can be recovered from the model's internal activations, without parsing the prompt text at inference time.

The strongest result is deliberately scoped. On frozen Llama-3.1-8B benchmark slices, activation-derived readouts recovered enough operation and operand structure to route several arithmetic tasks to Python under a strict no-parser runtime. The same work also exposed the limits: readable internal variables were not automatically writable registers, the final route did not transfer as-is to Qwen, and longer numeric answers showed resolution pressure as digit chunks crowded the model's geometry.

The article is written for practitioners rather than interpretability specialists. It uses arithmetic as a concrete way to explain residual streams, probes, sparse autoencoders, activation patching, steering, causal tests, provenance audits, and next-token constraints. The larger lesson is scientific: the first positive signal is rarely the result. The useful result is the one that survives when prompt text, labels, shortcuts, and comforting explanations are removed.
