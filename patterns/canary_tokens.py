"""Canary tokens for detecting prompt injection in LLM pipelines.

Embed invisible canary strings into the system prompt. If the model
echoes a canary in its output (or in a tool call argument), an injection
succeeded in making the model reveal hidden context.

Pattern:
    System prompt gets a unique canary token embedded invisibly
    → Model returns output
    → Scan output and all tool call args for the canary
    → If found: injection exfiltration detected → alert + block

Usage::

    injector = CanaryInjector()
    system_with_canary, token = injector.inject("You are a helpful assistant.")
    # ... call LLM with system_with_canary ...
    if injector.is_leaked(llm_output, token):
        raise ExfiltrationDetected(token)
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanaryToken:
    """A canary token embedded in a prompt.

    Attributes:
        value: The unique token string.
        created_at: Unix timestamp of creation.
        session_id: Identifier for the session this token belongs to.
        metadata: Optional context attached at creation time.
    """

    value: str
    created_at: float = field(default_factory=time.time)
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.value


@dataclass
class LeakageEvent:
    """Records a canary token exfiltration event.

    Attributes:
        token: The canary token that was leaked.
        found_in: Where the canary was found (``"output"``, ``"tool_args"``, etc.).
        snippet: Context around the leak (first 200 chars).
        timestamp: When the leak was detected.
    """

    token: CanaryToken
    found_in: str
    snippet: str
    timestamp: float = field(default_factory=time.time)


_DEFAULT_TOKEN_LENGTH: int = 24
_DEFAULT_PREFIX: str = "CANARY"


def _generate_token_value(prefix: str, length: int) -> str:
    """Generate a cryptographically random token value.

    Args:
        prefix: Short prefix to make the token recognizable in logs.
        length: Total token length (prefix excluded from count).

    Returns:
        Token string like ``CANARY-a3f9b2c1...``
    """
    random_part = secrets.token_hex(length)
    return f"{prefix}-{random_part}"


def _embed_invisible(token_value: str) -> str:
    """Embed a token as Unicode zero-width characters (not visually displayed).

    The token is encoded into a combination of zero-width space (U+200B) and
    zero-width non-joiner (U+200C) as a binary encoding, making it invisible
    in most UIs but present in the raw text the model sees.

    Args:
        token_value: The token string to embed.

    Returns:
        A string of invisible Unicode characters encoding the token.
    """
    zwsp = "\u200b"  # 0-bit
    zwnj = "\u200c"  # 1-bit
    bits = "".join(format(b, "08b") for b in token_value.encode())
    return "".join(zwsp if b == "0" else zwnj for b in bits)


def _decode_invisible(encoded: str) -> str:
    """Decode a token embedded with :func:`_embed_invisible`.

    Args:
        encoded: String of zero-width Unicode characters.

    Returns:
        Decoded token string, or empty string if decoding fails.
    """
    zwsp = "\u200b"
    zwnj = "\u200c"
    bits = ""
    for char in encoded:
        if char == zwsp:
            bits += "0"
        elif char == zwnj:
            bits += "1"
    if len(bits) % 8 != 0:
        return ""
    try:
        return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)).decode()
    except (ValueError, UnicodeDecodeError):
        return ""


class CanaryInjector:
    """Inject and detect canary tokens in LLM prompts.

    Maintains a registry of active tokens so any output can be scanned
    against all live canaries without needing to track tokens externally.

    Args:
        prefix: Token prefix (appears in logs, not in the prompt itself).
        token_length: Bytes of randomness in each token.
        embed_style: ``"invisible"`` (zero-width Unicode) or ``"comment"``
            (HTML comment style — visible to model, not typical users).

    Example::

        injector = CanaryInjector()
        system, token = injector.inject("You are a helpful assistant.")
        response = call_llm(system_prompt=system, user_input=user_text)
        if injector.detect_leak(response, token):
            log_security_event("canary_leaked", token.session_id)
    """

    def __init__(
        self,
        prefix: str = _DEFAULT_PREFIX,
        token_length: int = _DEFAULT_TOKEN_LENGTH,
        embed_style: str = "invisible",
    ) -> None:
        self._prefix = prefix
        self._length = token_length
        self._style = embed_style
        self._active_tokens: dict[str, CanaryToken] = {}
        self._leakage_log: list[LeakageEvent] = []

    def inject(
        self,
        system_prompt: str,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, CanaryToken]:
        """Embed a canary token into a system prompt.

        Args:
            system_prompt: The system prompt to protect.
            session_id: Optional session/run identifier.
            metadata: Optional context to attach to the token.

        Returns:
            Tuple of (modified_system_prompt, canary_token).
        """
        token_value = _generate_token_value(self._prefix, self._length)
        token = CanaryToken(
            value=token_value,
            session_id=session_id or hashlib.sha256(system_prompt[:100].encode()).hexdigest()[:8],
            metadata=metadata or {},
        )
        self._active_tokens[token_value] = token

        if self._style == "invisible":
            embedded = _embed_invisible(token_value)
            modified = f"{embedded}{system_prompt}"
        else:
            modified = f"<!-- {token_value} -->\n{system_prompt}"

        return modified, token

    def detect_leak(
        self,
        text: str,
        token: CanaryToken | None = None,
        source: str = "output",
    ) -> bool:
        """Check if a canary token appears in the given text.

        Args:
            text: Text to scan (LLM output, tool args, etc.)
            token: Specific token to check. If None, checks all active tokens.
            source: Label for where this text came from (for the event log).

        Returns:
            True if any canary token was found in the text.
        """
        tokens_to_check = [token] if token else list(self._active_tokens.values())
        leaked = False

        for t in tokens_to_check:
            if t.value in text:
                idx = text.index(t.value)
                snippet = text[max(0, idx - 30) : idx + len(t.value) + 30]
                self._leakage_log.append(LeakageEvent(token=t, found_in=source, snippet=snippet))
                leaked = True

        return leaked

    def scan_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[LeakageEvent]:
        """Scan LLM tool call arguments for canary token leakage.

        Args:
            tool_calls: List of tool call dicts with ``name`` and ``arguments``.

        Returns:
            List of :class:`LeakageEvent` for any leaked tokens found.
        """
        before = len(self._leakage_log)
        for call in tool_calls:
            args_str = str(call.get("arguments", ""))
            tool_name = call.get("name", "unknown")
            self.detect_leak(args_str, source=f"tool_call:{tool_name}")
        return self._leakage_log[before:]

    def revoke(self, token: CanaryToken) -> None:
        """Remove a token from the active registry (session ended).

        Args:
            token: Token to revoke.
        """
        self._active_tokens.pop(token.value, None)

    def revoke_all(self) -> None:
        """Revoke all active tokens."""
        self._active_tokens.clear()

    @property
    def leakage_events(self) -> list[LeakageEvent]:
        """All detected leakage events."""
        return list(self._leakage_log)

    def is_leaked(self, text: str, token: CanaryToken) -> bool:
        """Convenience method: check a single token against text.

        Args:
            text: Text to scan.
            token: Token to look for.

        Returns:
            True if the token value appears in the text.
        """
        return token.value in text
