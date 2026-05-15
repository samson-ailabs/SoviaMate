# Contributing to SoviaMate

SoviaMate is an open research effort and we welcome contributions of all kinds — code, evaluation, documentation, datasets, and research discussion. This guide explains how to get involved.

## Ways to contribute

### Good first contributions
If you are looking for a concrete starting point, these areas are useful, well-scoped, and reviewer-friendly:

- **Dataset adapters** — readers/manifests for additional speech corpora (Common Voice, GigaSpeech, VCTK, expressive corpora) that plug into `soviamate.datas`.
- **Evaluation scripts & reports** — extend `scripts/eval_audio_codec.py` with objective metrics (PESQ, ViSQOL, STOI, WER, SECS) and reproducible benchmark tables against EnCodec / SoundStream / DAC.
- **Multilingual tokenizers** — train and contribute SentencePiece tokenizers for languages beyond English (see `scripts/prepare_tokenizer.py`).
- **Tests** — unit tests for `soviamate.modules` and `soviamate.layers` (the test suite is currently minimal).
- **Inference & demo tooling** — a minimal `infer.py` for encode/decode round-trips, plus a small streaming demo.
- **Documentation** — improve module docstrings, add diagrams, or write tutorials.

### Research contributions
If you want to engage at the research level — codec architectures, LLM integration, dialogue systems — please open a **GitHub Discussion** or email **[samson.ailabs@gmail.com](mailto:samson.ailabs@gmail.com)** before starting large work, so we can align on scope.

### Reporting issues
- Search existing issues first to avoid duplicates.
- Include reproduction steps, expected vs. actual behavior, environment info (OS, Python, GPU, CUDA, package versions), and a minimal example where possible.

### Feature requests
Open an issue describing the problem the feature solves, the proposed approach, and any references. For larger changes please discuss before implementing.

## Development workflow

1. **Fork** the repository on GitHub.
2. **Clone** your fork:
   ```sh
   git clone https://github.com/your-username/SoviaMate.git
   cd SoviaMate
   ```
3. **Set up** the environment as described in [`README.md`](README.md) (`uv sync --frozen`).
4. **Create a feature branch**:
   ```sh
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** following the [coding style](CODING_STYLE.md). Add or update tests in `tests/` where appropriate.
6. **Verify** locally:
   ```sh
   uv run ruff check .
   uv run mypy soviamate
   uv run pytest
   ```
7. **Commit** with clear, descriptive messages.
8. **Push** and open a pull request against `main` with a description of *what* changed and *why*.

Maintainers will review, suggest changes, and merge when ready. Please be patient — this is a research project run by a small team.

## Code of Conduct
By contributing, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

## Contact
For questions, ideas, or research collaboration: **[samson.ailabs@gmail.com](mailto:samson.ailabs@gmail.com)** or GitHub Issues / Discussions.

Thank you for helping move spoken dialogue systems forward.
