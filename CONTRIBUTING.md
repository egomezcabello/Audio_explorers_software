# Contributing to Audio Explorers

Thank you for contributing!  Please follow these guidelines so all four
team members can work in parallel without conflicts.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, tested code only — **merge via PR** |
| `feature/member1-doa` | Member 1 work |
| `feature/member2-enhance` | Member 2 work |
| `feature/member3-analysis` | Member 3 work |
| `feature/member4-fusion` | Member 4 work |

Always branch from the latest `main`:

```bash
git checkout main && git pull
git checkout -b feature/member1-doa
```

---

## Pull Request Checklist

Before opening a PR, make sure:

- [ ] My code runs without errors: `python -m src.memberX_xxx.run_all`
- [ ] All smoke tests pass: `pytest tests/ -v`
- [ ] I have not committed WAV, model weights, or large generated files.
- [ ] I updated docstrings and TODO comments where needed.
- [ ] I did not modify `src/common/` without discussing with the team first.
- [ ] I did not change `config.yaml` defaults without agreement.
- [ ] Linting passes: `make lint` (or `ruff check src/ tests/`).
- [ ] My branch is up to date with `main` (rebase or merge).

---

## Code Style

- **Python 3.10+** with type hints.
- **pathlib** for all file paths (no `os.path`).
- **Black** formatting (line length 99).
- **Ruff** linting.
- Descriptive function names, clear docstrings.
- `TODO:` comments for unimplemented sections.

---

## Shared Code Rules

The `src/common/` package is shared by everyone.  To avoid conflicts:

1. **Do not modify** `constants.py` or `json_schema.py` without a team
   discussion (these define the data contract).
2. If you need a new utility, add it to the appropriate `src/common/`
   module and open a small PR for review first.
3. Keep your member-specific code inside your own `src/memberX_*/` package.

---

## Output Conventions

All generated files follow the naming scheme defined in
`src/common/constants.py`.  Do not invent new file names without updating
the constants.

---

## Asking for Help

- Open a GitHub Issue with a clear description.
- Tag the relevant member(s).
- Include the error traceback if applicable.
