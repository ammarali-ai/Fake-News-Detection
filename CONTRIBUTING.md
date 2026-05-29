# Contributing

Thanks for taking the time to contribute. This repo is small and the bar to participate is low — a clean reproduction of a bug or a focused PR is plenty.

## Filing a bug

Open a GitHub issue with:

1. **What you did.** Exact command or steps.
2. **What you expected.** One sentence.
3. **What actually happened.** Error message + stack trace, copy-pasted.
4. **Environment.** Python version, OS, whether you're running locally / Docker / HF Spaces.
5. **SavedModel source.** `./saved_model/` committed locally, or fetched via `HF_MODEL_REPO_ID`?

A minimal repro that someone else can paste into their terminal is worth its weight in triage time.

## Proposing a change

For anything beyond a typo, open an issue first describing the change and why it's needed. That avoids you sinking time into a PR that won't land.

Small unambiguous changes (typo fixes, doc improvements, obvious bugs with clear repros) can skip the issue and go straight to a PR.

## Opening a PR

1. Fork → branch off `main`.
2. Make the change. Keep the scope tight — one logical change per PR.
3. Run the pre-commit checks:
   ```bash
   python -m py_compile model_loader.py app.py api.py evaluate.py
   pytest -q tests/
   docker build -t fake-news-detector .
   ```
4. If your change touches inference or input handling, do a manual end-to-end with the model in place.
5. Update docs in the same PR — if you change `api.py`, update `docs/API.md`; if you change architecture, update `docs/ARCHITECTURE.md`.
6. Open the PR with:
   - A clear title (under 70 chars).
   - A body that says what changed and why.
   - A "test plan" section listing what you ran.

## Style

Conventions are documented in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#coding-conventions). The key invariants:

- **One source of truth for the model.** Only `model_loader.py` calls `tf.saved_model.load()` or `AutoTokenizer.from_pretrained()`.
- **Docstrings on every public function.**
- **No bare `except:` clauses.**
- **Pinned versions stay pinned.** The spec-listed packages in `requirements.txt` are pinned for a reason — don't loosen them without discussion.

## Doc-only PRs

Welcome. Spelling, grammar, broken links, clearer phrasing, better examples — all worth landing. No test plan needed if no code changed.

## What's out of scope

- **Sweeping reformats** (linter / formatter runs across the whole repo) — open an issue first so we can agree on tooling.
- **New ML approaches.** Swapping models or adding a second classifier is a significant change. Discuss in an issue first.
- **Adding heavyweight dependencies** without a clear use case. The dependency list is intentionally pinned and minimal.

## Code of conduct

Be respectful. Disagreements are normal — keep them about the work, not the person. Maintainer has the final call.

## License

By contributing, you agree your contributions are released under the same MIT license as the rest of the project. See [LICENSE](LICENSE).
