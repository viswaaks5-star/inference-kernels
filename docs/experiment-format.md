# Experiment format

Every investigation in this repository uses the same format. The fields exist to enforce one habit: state what a hypothesis predicts before the hardware answers.

```markdown
### E<n>. <The question, phrased as a question>

**Hypothesis.** One claim that could turn out to be wrong.

**Prediction.** What the numbers look like if the hypothesis is true, and where
possible what they look like if it is false. Written before running. If the
experiment wasn't planned, say so instead of back-filling a prediction.

**Setup.** What changed, what was held fixed, which configuration was measured.
If the experiment changes two things at once, say so here.

**Result.** Numbers, with a link to the raw output in results/.

**Verdict.** Supported, falsified or inconclusive, judged against the
prediction and not against what would have been convenient.

**What went wrong.** Wrong assumptions, confounds, tooling mistakes.
Omitted when nothing did.

**Next.** The experiment this result makes necessary.
```

## Rules

- A hypothesis the planned experiment cannot falsify is not tested by it.
- "Supported" is not "proven". One experiment can falsify a hypothesis; no single experiment confirms one.
- When an explanation is plausible but unconfirmed, the text says "consistent with", not "because".
- Every number in the text appears in a file under `results/`.
- Dead ends stay. A falsified hypothesis, with the experiment that killed it, is often the most informative entry in a notebook.
- Each kernel's `README.md` is a front page: a summary a reader can stop after, the open question that matters most, the key results and how to reproduce them. The full log lives beside it in `EXPERIMENTS.md`.
- Experiments that validate the measurement itself are numbered M1, M2, …; experiments on the kernel, E1, E2, …. Numbers follow the order the argument needs; a timeline records the order things happened.
