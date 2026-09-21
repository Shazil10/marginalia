# Repository cleanup

Current implementation is `marginalia/`; original research modules remain useful
as references. Cleanup does not mean experimental code has been integrated or
that old backtest metrics have been validated.

Removed from active repository:

- Root `quantmentals.pdf`: byte-identical to retained
  `agents/strategy_extraction/quantmentals.pdf`.
- Generated `agents/strategy_extraction/outputs/` reports, scripts and returns.
- Generated `data/raw/` and `data/processed/` sentiment examples.
- `sentiment_pipeline.py` and `tavily_social_pipeline.py`: redundant wildcard
  wrappers. Direct package imports and three CLI wrappers remain supported.

Tracked removals can be recovered from parent commit. Generated outputs and
duplicate PDF also have external local recovery copy. Git history is unchanged;
cleanup reduces active-tree clutter, not historical clone size.

`Shazil-quantbros/` was a separate nested repository with unique uncommitted work.
Entire original, including its Git history, local changes, credentials and run
artifacts, was moved to external local recovery folder, not uploaded. Source,
tests, prompts, fixtures, schemas, scripts, docs and notebooks were preserved as
ordinary files under `experiments/quantbros/`. Its database, paper copies,
generated artifacts and private environment files are not part of published
snapshot. Main CI does not imply this independent experiment passes its tests.

Machine-specific `.claude/` configuration stays local and ignored. Core source,
tests, examples and portable Context Engine instructions are versioned.

Kept original extraction/report-generation scripts, unique UIs and sentiment
modules because they still provide distinct behavior. Deleting these requires
finishing their migration, rather than assuming they are unused.
