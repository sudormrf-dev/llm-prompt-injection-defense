# llm-prompt-injection-defense

Production patterns for defending LLM applications against prompt injection attacks. Four layered defenses covering the full attack surface.

## Attack taxonomy

| Attack type | Defense |
|---|---|
| Direct injection ("ignore previous instructions") | `InputSanitizer` |
| Indirect injection (malicious web content) | `PromptFirewall` |
| Output exfiltration detection | `OutputValidator` + `CanaryInjector` |
| Persona hijacking | `OutputValidator` |

## Patterns

### `input_sanitizer.py` — Block known injection patterns

Regex-based detection of 16+ injection signatures, Unicode direction override stripping, and structured prompt building with XML delimiters.

```python
from patterns.input_sanitizer import InputSanitizer, InjectionDetected

sanitizer = InputSanitizer()
try:
    clean = sanitizer.sanitize(user_input)
except InjectionDetected as exc:
    return "Input rejected"

# Safe variant (no raise):
result = sanitizer.sanitize_safe(user_input)
if result.warnings:
    log_suspicious(result.warnings)
```

### `output_validator.py` — Detect injection success in outputs

Checks model outputs for system prompt leakage, persona hijack, and safety bypass indicators. Format validation (JSON) and length bounds.

```python
from patterns.output_validator import OutputValidator

validator = OutputValidator(
    system_prompt_fragments=["confidential instructions"],
    expected_format="json",
)
validated = validator.validate(llm_response)
```

### `prompt_firewall.py` — Second-model content screening

Evaluate external content (web pages, documents, tool outputs) before including in the main prompt. Pluggable classifier (heuristic or LLM-based).

```python
from patterns.prompt_firewall import PromptFirewall

firewall = PromptFirewall(classifier_fn=my_llm_classifier)
decision = await firewall.check(retrieved_web_content)
if decision.is_safe:
    prompt = build_prompt(decision.content)
```

### `canary_tokens.py` — Exfiltration detection via canary strings

Embed invisible token strings in system prompts. If a token appears in any output or tool call argument, an injection succeeded.

```python
from patterns.canary_tokens import CanaryInjector

injector = CanaryInjector()
system_prompt, token = injector.inject("You are a helpful assistant.")
response = call_llm(system_prompt=system_prompt, user=user_input)
if injector.is_leaked(response, token):
    alert_security_team(token.session_id)
```

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

## Defense in depth

Use all four layers together:

1. **Sanitize** all user inputs before they reach the model
2. **Screen** all external/retrieved content through the firewall  
3. **Embed** canary tokens in your system prompt each session
4. **Validate** all model outputs before surfacing to users or tool execution

## License

MIT
