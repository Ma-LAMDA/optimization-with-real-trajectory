# Repository contribution rules

- Every commit pushed to GitHub must update `README.md` in the same commit so that the documentation stays synchronized with the repository state.
- Before committing, verify that the README describes the user-visible change included in that commit.

## Protected simulation data

- Treat every file under `data/simulation/` as an immutable original after it is placed there.
- Files under `data/simulation/` may be read or copied to another location, but must not be edited, overwritten, renamed, moved, or deleted.
