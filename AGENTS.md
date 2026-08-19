# Repository contribution rules

- Every commit pushed to GitHub must update `README.md` in the same commit so that the documentation stays synchronized with the repository state.
- Before committing, verify that the README describes the user-visible change included in that commit.

## Protected simulation data

- Treat every file under `data/simulation/` as an immutable original after it is placed there.
- Files under `data/simulation/` may be read or copied to another location, but must not be edited, overwritten, renamed, moved, or deleted.

## Trajectory archive accounting

- Date-scoped trajectory and SFT archives record only model-valid outcomes: `accepted`, `incorrect`, and `format_error`.
- Do not copy, count, summarize, or retain infrastructure failures or interrupted attempts in archive manifests, curation files, reports, or training data.
- Runners may handle those outcomes transiently for control flow. Existing source experiment state and historical archives do not need to be rewritten solely to remove legacy records.

## Documentation and reproducibility

- Every change to trajectory sources, curation, split rules, clustering, node selection, message construction, system prompts, tool protocols, loss weights, sampling, tokenizer limits, training entry points, validators, or generated SFT artifacts must update documentation in the same change.
- For date-scoped data, update the root `README.md`, the date directory `README.md`, and its reproducibility document. For 0805 this is `data/2026-08-05/REPRODUCIBILITY.md`.
- Documentation must state the immutable inputs, exact commands, parameters, expected counts, output files, known limitations, validation result, and any required environment or model/tokenizer version. A changelog entry must explain why the data changed.
- A semantic conversion change must bump the relevant manifest/selection schema version. Regenerate all derived artifacts and run the independent validator before committing.
- Generated artifacts, their manifest hashes, the converter, validator, training entry point, and reproducibility document must stay synchronized. Do not commit or push a dataset change when any of these are stale or when the documented reproduction commands do not pass.
- Target-tokenizer length and per-token loss-mask checks must be archived before formal training. Heuristic token estimates must be labeled as heuristic and must not be reported as the target-tokenizer result.
