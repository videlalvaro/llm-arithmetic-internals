# How Would You Do Math If You Only Had Matrices? (Draft)

If you learned arithmetic the ordinary human way, you probably learned it with a body.

You counted on fingers. You grouped things into piles. You lined digits into columns. You carried a one. Maybe, later, you used an abacus, or graph paper, or a calculator. Human arithmetic is full of objects: fingers, beads, marks, columns, strokes, places.

A language model has none of that.

It has matrices.

At each layer, a huge grid of numbers transforms another huge grid of numbers. Tokens enter, activations flow, logits come out. No fingers. No abacus. No column of digits written on a page. And yet, if you ask a modern language model for the greatest common divisor of two numbers, or a multiplication, or a division with remainder, something inside that matrix-only body responds.

Sometimes it answers correctly. Often it does not.

The question that started Rune was simple to ask and hard to answer:

**How does a language model do arithmetic, if all it has are vectors?**

The first temptation is to treat this as a product problem. If the model is bad at arithmetic, call a calculator. That works. It is also not very mysterious. A parser can read the prompt `What is 84 times 37?`, translate it into `84 * 37`, send that expression to Python, and return the result.

But Rune was chasing a different question. Could we look inside the model and find the calculation it was trying to perform? Could the model's own internal states tell us the operation and operands? And, if so, could we use that information without cheating by reading the prompt text directly?

That distinction became the center of the project.

The original dream was even more ambitious. We wanted a kind of just-in-time compiler for model arithmetic. The ideal pipeline looked like this: the model reads an arithmetic prompt; we identify the internal mechanism it is using; we replace the unreliable part with an exact computation; then the model continues naturally, as if it had done the math itself.

That is a beautiful idea. It is also not what we ended up proving.

What we found was narrower, but more interesting than ordinary tool use. In a frozen Llama model, the internal activations can expose enough structure to recover the arithmetic operation and operands under a strict no-parser rule. The runtime is not allowed to use regexes, prompt text, hidden labels, command-line operands, or gold answers. It gets token IDs and activations. From those activations, it must infer: this is gcd, or lcm, or multiplication, or division with remainder; these are the two numbers. Only after that decoded tuple exists may Python compute.

The calculator is not the surprising part. The surprising part is that the arguments to the calculator can come from inside the model.

The path to that result was not linear. It was closer to debugging a machine that kept giving us plausible but misleading answers.

Early on, we found real arithmetic structure inside the model. Some of it looked almost geometric. Numbers were not stored like beads on a wire, but they were not featureless either. There were directions, rotations, periodic patterns, and value-like states.

That makes the story feel strange in exactly the way George Lakoff and Rafael Núñez's *Where Mathematics Comes From* makes human mathematics feel strange. Their book argues that human mathematics is not born in a vacuum; it is built from embodied experience, conceptual metaphor, grouping, moving, measuring, collecting, and balancing. That is not how a transformer encounters the world. A transformer has a different body. Its body is matrices, residual streams, attention maps, and learned projections.

So the question becomes: if human arithmetic can grow out of fingers, piles, and spatial metaphors, what kind of arithmetic grows out of a matrix-only body?

The answer suggested by recent mechanistic work is geometric. Kantamneni and Tegmark argued that language models can represent integers on generalized helices and use trigonometric structure for addition. Other work, such as Nikankin and colleagues' "bag of heuristics" account, warns that model arithmetic is often not a clean schoolbook algorithm but a mixture of learned features and heuristics. Rune builds on that prior line rather than originating it. The contribution here is narrower: testing activation-derived tool arguments, provenance boundaries, and resolution limits. We found helix-like and Fourier-like readouts, but also brittleness, prompt sensitivity, and failures that looked less like a perfect algorithm than a finite-resolution internal geometry.

For gcd, we found internal features that behaved like small-divisor tests. Certain directions could separate numbers divisible by 2, 3, or 4. Suppressing some of those features changed the model's gcd behavior. Similar motifs appeared across more than one model family. That was a real interpretability result: not a deployment system, but evidence that the model had learned useful arithmetic-related structure.

We also found something about how models emit numbers. Multi-digit answers are not produced all at once. They are rendered chunk by chunk. Late layers carry value-like writer states that help decide which numeric chunk gets emitted next. Those writer states could sometimes be steered. If you supplied the right internal state, the model could render a desired chunk.

This is where next-token prediction matters. A human doing subtraction on paper usually works from the least significant digit toward the left: units first, then tens, then hundreds, carrying or borrowing as needed. A language model must emit text from left to right. For the answer `15696`, it must commit to the visible prefix before the suffix exists. In tokenized form, the answer may come out as chunks like `15` then `696`. The model is not filling in a scratch sheet from right to left; it is walking forward through the string.

That left-to-right constraint creates a rendering problem separate from the arithmetic problem. The model may have partial information about the answer but still lose precision as it emits longer strings. In Rune's subtraction scaling experiment, exact greedy generation stayed at 96.7 percent for 6-digit subtraction, fell to 63.3 percent at 10 digits, reached 53.3 percent at 13 digits, and crossed below 50 percent at 14 digits. At 24 digits it was down to 6.7 percent.

The helix-resolution experiments gave a more mechanistic picture of that boundary. For 12-digit subtraction answers split into four 3-digit chunks, each chunk remained strongly phase-decodable at L31: R2 was about 0.96 for chunk 1 and about 0.92 for chunk 4. So the signal did not simply vanish. The subtler finding was crowding. Adjacent chunk readout subspaces were not cleanly orthogonal; the minimum principal angle dropped as low as 66 degrees. In a 14-digit, five-chunk follow-up, chunks 2-4 lost R2 and adjacent angles tightened further, for example c3-c4 went from 67.4 degrees to 58.8 degrees. Two of three preregistered crowding predictions passed.

That is why "the helix gets saturated" is too crude, but "there is a resolution budget" is fair. Longer answers still live in readable geometry, but the geometry becomes more crowded and less forgiving. The model can be close enough to know the prefix and still wrong enough to miss the exact digit string.

At first, that felt like a path toward the compiler dream. If we could write the right answer state into the model, perhaps the model would continue from there.

But then the evidence sharpened.

Making a model emit a known answer is not the same as making the model compute. If the experiment already knows the answer, and then uses that answer to choose a steering vector, it has measured the model's ability to render a supplied value. That is useful, but it is not arithmetic understanding, and it is not an honest deployment pipeline.

This was one of the first major lessons: **token rendering can masquerade as computation.**

So we tightened the rules.

At deployment, the prompt would be opaque. The system could not parse it. Python could not receive the operation or operands from the test harness. If the model was going to call a calculator, the model's own internal state had to supply the calculator arguments.

That rule exposed the hard part immediately. Operand extraction was often possible. Operation extraction was much more treacherous.

We trained probes that looked good until we changed the phrasing. We found classifiers that seemed to know the operation, but were really learning surface form. "What is the product of..." and "multiply..." leave different textual fingerprints. A probe can exploit those fingerprints while pretending to be an internal arithmetic detector.

That is an easy trap in this kind of work. A linear probe gives a number. The number looks scientific. But unless the controls are right, the probe may be measuring formatting, not mechanism.

So we started asking more annoying questions. Does the signal survive paraphrases? Does it fire on quoted arithmetic that should not be computed? Does it reject "do not multiply these numbers"? What happens if the same numbers appear in a wrong-operation prompt? What if the prompt is an invoice, a table, a log file, or number-heavy prose?

Many early versions failed those tests. Some routes fired on quoted expressions. Some rejected distractor-heavy target prompts. Some found the right numbers in clean symbolic forms but not in stories. Each failure clarified what the system was actually measuring.

That is the second lesson: **interpretability work improves when the controls are adversarial enough to embarrass the first explanation.**

The residual-write story also became clearer. We tried to write corrected answer information into the model's residual stream and let it continue. This was the closest version of the original JIT replacement thesis.

It did not earn its keep.

For the tested single-site writes, residual interventions had no accuracy advantage over simpler token or logit correction. Worse, they disturbed surrounding behavior more. On multi-token answers, forcing the first correct token could lead the model to complete the rest better than a crude residual write. On competent addition, where the model already knew how to continue, the residual write still provided no benefit over token-level correction and cost more in behavior preservation.

That did not prove residual replacement is impossible. It proved something more practical: **a readable variable is not necessarily a writable register.**

This is a crucial distinction. Mechanistic interpretability often celebrates reading: we decode a concept, localize a feature, find a direction. Engineering wants writing: change the state, preserve behavior, continue execution. Those are different problems. A blood test can tell you something about the body; that does not mean injecting the inverse signal cures the disease.

So the project narrowed again.

Instead of claiming that we could replace the model's internal arithmetic mechanism, we focused on the route that survived the honesty tests: activation-derived calculator arguments.

The final system used activation-only gates and operand readouts. It had to decide when an arithmetic route was safe to fire, identify the operation, identify the operand pair, and then call Python only after the decoded tuple was formed. The result was evaluated on frozen benchmark slices, with thresholds locked and provenance tracked.

On a broad frozen arithmetic and adversarial benchmark, the Llama route passed across four operations: multiplication, division with remainder, gcd, and lcm. Across 11,736 locked examples and 1,536 targets, it produced large exact-answer lifts with 0 fires on the constructed hard-negative suite used in this audit.

On a recognized DeepMind source slice, the result covered three operations: gcd, division with remainder, and lcm. Multiplication was not claimed there because the source filtering did not produce enough accepted two-integer multiplication examples for a powered result. Across 3,822 locked examples and 1,233 targets, the activation-derived route again produced strong exact-answer lifts with zero recorded false fires.

The safety and provenance work mattered as much as the benchmark lifts. We ran a full replay provenance audit over 15,558 replay bundles. The replay bundles excluded forbidden runtime fields: prompt text, regex outputs, decoded token spans, harness operands, operation labels, and gold answers. The route had to reproduce from allowed runtime artifacts. It passed with zero replay failures.

We also ran an independent hard-negative audit: quoted arithmetic, do-not-compute prompts, wrong-operation prompts with the same numbers, tables, logs, code, invoices, distractor-heavy number text, decimals, signs, and out-of-domain cases. Across 10,200 negative examples, the route did not fire.

This is where the result becomes different from "we used a calculator." A normal calculator tool route can read the user's text. Rune's route is constrained to ask: did the model's internal state carry the calculation description?

For these Llama benchmark slices, the evidence supports the narrower activation-derived argument claim.

There is causal support too, but it has to be said carefully. In causal interchange tests, selected internal operand chunks could be patched from donor examples into recipient examples, and the decoded operands and routed calculator answers followed the donor. That supports the claim that those internal chunks are causally involved in the decoded tuple. In the final DeepMind causal summary, division with remainder cleared the powered pair-count gate. Gcd and lcm had perfect rates in the available pairs, but too few pairs to clear the frozen causal gate. So the causal evidence is supportive, not fully powered across every final operation.

The cross-model story is also humbling. A real Qwen operand-localization sweep failed. The current Llama route did not transfer. That is not a footnote. It is a lesson: internal activation routes are not portable like text parsers. A parser sees the same string. A model's internal geometry may be entirely different.

So what did we learn?

First, models really do build arithmetic-shaped internal structure. Not arithmetic as humans experience it, with fingers and written columns, but arithmetic as a geometry of directions, rotations, periodic features, and value-like residual states. A matrix-only body can invent ways to represent divisibility, magnitude, chunks, and numeric emission.

Second, reading those structures is easier than turning them into a robust system. A probe can look good for the wrong reason. A steering vector can render an answer without proving computation. A residual write can change a token while damaging the model's state. The interesting science is not the first positive result; it is the sequence of controls that survives.

Third, tool use has layers. The easy layer is text-driven: parse the prompt, call the tool. The harder layer is activation-derived: prove that the tool arguments came from the model's internal state. The hardest layer is internal replacement: write the result back into the model and preserve behavior. Rune did not solve the hardest layer. It made progress on the middle one.

That middle layer may be useful beyond arithmetic. Many future AI systems will combine neural models with external tools. The engineering question will not only be whether the tool improves accuracy. It will be: where did the tool arguments come from? Were they parsed from text? Generated as a chain of thought? Decoded from internal state? Calibrated with labels but run without them? Audited by replay? Protected against false fires?

Those questions sound bureaucratic until you run the experiments. Then they become the experiment.

The final claim is deliberately modest:

A frozen Llama model's activations can supply operation and operand arguments for an exact calculator under an opaque, no-parser runtime on the current benchmark slices.

That is not native arithmetic repair. It is not a general cross-model result. It is not behavior-preserving residual JIT compilation.

But it is a real and useful boundary. It shows that the model's hidden state can contain the calculation even when its output would be wrong. It shows that mechanistic interpretability can contribute to tool systems, if the provenance rules are strict enough. And it shows how easy it is to overclaim unless the experiment is designed to catch you.

In the end, the wonder is still there.

A transformer does not have fingers. It does not line digits on paper. It does not know an abacus. It has matrices, activations, and learned geometry. And inside that geometry, under the right tests, we can find traces of arithmetic: not human arithmetic exactly, but a machine's version of it.

The practical lesson is less romantic but just as important.

If you want to build systems from those traces, do not trust the first beautiful signal. Ask what else it could be measuring. Remove the prompt text. Remove the labels. Replay without the forbidden fields. Add the annoying negatives. Compare against the boring baseline. Separate what the model represents from what you can safely write.

That is how the story gets smaller.

It is also how it becomes true.

## Notes And Sources

- George Lakoff and Rafael E. Núñez, *Where Mathematics Comes From: How the Embodied Mind Brings Mathematics into Being*, Basic Books, 2000.
- Subhash Kantamneni and Max Tegmark, "Language Models Use Trigonometry to Do Addition," arXiv:2502.00873, 2025.
- Yaniv Nikankin, Anja Reusch, Aaron Mueller, and Yonatan Belinkov, "Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics," arXiv:2410.21272, 2024.
- Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan, "A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis," arXiv:2305.15054, 2023.
- Timo Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," arXiv:2302.04761, 2023.
- Luyu Gao et al., "PAL: Program-aided Language Models," arXiv:2211.10435, 2022.
- Wenhu Chen et al., "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks," arXiv:2211.12588, 2022.
- Shunyu Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," arXiv:2210.03629, 2022.
- Neel Nanda and colleagues' activation-patching best-practice line is represented here by Zhang and Nanda, "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods," arXiv:2309.16042, 2023.
- David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli, "Analysing Mathematical Reasoning Abilities of Neural Models," ICLR 2019 / arXiv:1904.01557. This is the DeepMind Mathematics Dataset source.
- Trenton Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning," Transformer Circuits Thread, 2023; and Hoagy Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models," arXiv:2309.08600, 2023.
