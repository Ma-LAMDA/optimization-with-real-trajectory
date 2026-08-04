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
