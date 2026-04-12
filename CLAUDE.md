# llm-prompt-injection-defense — CLAUDE.md

## Quality gates

```bash
cd /media/user/SSD/github-waves/llm-prompt-injection-defense
ruff check .
mypy patterns/
bandit -r patterns/ -ll
pytest --cov=patterns --cov-fail-under=60
```

## Layout

```
patterns/
  input_sanitizer.py    — regex + unicode injection detection
  output_validator.py   — leak/hijack/bypass detection in outputs
  prompt_firewall.py    — external content screening
  canary_tokens.py      — embedded token exfiltration detection
tests/                  — pytest test suite
```

## Key decisions

- Zero runtime dependencies — pure stdlib Python
- All patterns are synchronous except PromptFirewall (async classifier fn)
- Canary tokens use zero-width Unicode for invisible embedding (hard for humans to spot)
- `raise_on_detection=False` mode available for logging-only pipelines
