# Decision Lab requirements and roadmap

## Who it is for

- **Support or operations lead.** Wants AI to triage and route work, and needs to know how often
  it is right before letting it act without review.
- **Engineer shipping an LLM feature.** Needs structured output that is validated, retried, and
  covered by a regression test in CI, without writing that plumbing again.
- **Risk or quality owner.** Needs evidence that a decision can be trusted: accuracy per field,
  whether confidence is calibrated, and a record of what went wrong.

## v0.1 (done)

- YAML decision schemas with `enum`, `int` (min/max), `bool`, and `str` fields.
- Cross-field consistency rules with exact match, `_gte`, and `_lte`.
- Strict validation that reports all problems at once; clear errors for malformed schemas.
- `Decider` with a retry loop that feeds validation errors back, confidence clamping, and
  abstention below a threshold.
- Providers: offline keyword mock, Anthropic Claude, and a small protocol for adding others.
- Evaluation on JSONL datasets: per-field accuracy, exact match, confusion counts, expected
  calibration error, mean attempts, failure list, text and JSON reports.
- CLI: `validate`, `decide`, `eval` with `--min-exact` as a CI gate.
- Example: billing support ticket triage, with schema and dataset.
- Offline test suite and CI on Python 3.10 and 3.12.

## v0.2: compare and understand

- Compare providers or models side by side on the same dataset.
- Disagreement analysis: where two providers differ, and which one is right.
- Self-consistency confidence: sample a decision several times and use agreement as confidence,
  instead of relying on the number the model reports.
- Reliability diagram to go with ECE.
- Cost and latency per decision in the report.
- Tolerance for ordinal fields, such as counting urgency within +-1 as correct.
- OpenAI-compatible provider.

## v0.3: decisions in the development workflow

- GitHub Action that comments accuracy deltas on pull requests.
- Baseline thresholds stored in the repo, per field, so a regression fails the build.
- Versioned prompts, so a report can be tied to the exact prompt that produced it.

## v0.4: decisions in production

- Audit log of every decision: input, output, confidence, attempts, prompt version.
- Human-review queue for abstained and low-confidence decisions.
- Plain-English report for non-technical stakeholders.

## Non-goals

- Not a general agent or prompt-orchestration framework.
- Not a labelling tool. Datasets are created elsewhere and brought in as JSONL.
- Not a hosted service or dashboard. It is a library and a CLI.
- No free-text quality scoring. Decision Lab measures structured decisions only.
- The mock provider will not become a real classifier. It stays a crude offline baseline.

## Open questions

- How should abstained decisions be scored: excluded, counted as wrong, or reported separately
  as coverage versus accuracy?
- Is self-reported confidence worth keeping once self-consistency confidence exists?
- How small can a dataset be before accuracy and ECE stop being meaningful, and should the
  report warn about it?
- Should rules be able to express more than `when` / `then`, such as "one of" or arithmetic
  between fields, or is that the point where users should write Python?
- Should partially labelled examples (only some expected fields) count towards exact match?
