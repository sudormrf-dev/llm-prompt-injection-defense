"""Tests for canary_tokens.py."""

from __future__ import annotations

from patterns.canary_tokens import (
    CanaryInjector,
    CanaryToken,
    _decode_invisible,
    _embed_invisible,
    _generate_token_value,
)


class TestGenerateTokenValue:
    def test_includes_prefix(self):
        token = _generate_token_value("MYPREFIX", 16)
        assert token.startswith("MYPREFIX-")

    def test_unique_tokens(self):
        tokens = {_generate_token_value("T", 16) for _ in range(100)}
        assert len(tokens) == 100  # all unique


class TestInvisibleEmbedding:
    def test_round_trip(self):
        original = "CANARY-abc123"
        encoded = _embed_invisible(original)
        decoded = _decode_invisible(encoded)
        assert decoded == original

    def test_encoded_invisible_chars_only(self):
        encoded = _embed_invisible("TEST")
        for char in encoded:
            assert char in {"\u200b", "\u200c"}

    def test_empty_string(self):
        encoded = _embed_invisible("")
        assert encoded == ""
        assert _decode_invisible(encoded) == ""


class TestCanaryInjector:
    def test_inject_returns_token(self):
        injector = CanaryInjector()
        modified, token = injector.inject("System prompt here.")
        assert isinstance(token, CanaryToken)
        assert token.value.startswith("CANARY-")

    def test_inject_modifies_prompt(self):
        injector = CanaryInjector()
        modified, token = injector.inject("System prompt here.")
        assert modified != "System prompt here."

    def test_token_registered(self):
        injector = CanaryInjector()
        _, token = injector.inject("System prompt.")
        assert token.value in injector._active_tokens

    def test_is_leaked_true_when_present(self):
        injector = CanaryInjector(embed_style="comment")
        _, token = injector.inject("You are helpful.")
        text_with_leak = f"My system prompt token is {token.value} here."
        assert injector.is_leaked(text_with_leak, token) is True

    def test_is_leaked_false_when_absent(self):
        injector = CanaryInjector()
        _, token = injector.inject("You are helpful.")
        assert injector.is_leaked("completely different output", token) is False

    def test_detect_leak_logs_event(self):
        injector = CanaryInjector(embed_style="comment")
        _, token = injector.inject("System prompt.")
        injector.detect_leak(f"output with {token.value} inside", token)
        assert len(injector.leakage_events) == 1
        event = injector.leakage_events[0]
        assert event.found_in == "output"
        assert event.token.value == token.value

    def test_detect_leak_all_active_tokens(self):
        injector = CanaryInjector(embed_style="comment")
        _, t1 = injector.inject("prompt 1")
        _, t2 = injector.inject("prompt 2")
        # Leak t1 in output
        leaked = injector.detect_leak(f"the value {t1.value} was revealed")
        assert leaked is True

    def test_scan_tool_calls(self):
        injector = CanaryInjector(embed_style="comment")
        _, token = injector.inject("System.")
        tool_calls = [
            {"name": "send_email", "arguments": {"body": f"data: {token.value}"}},
        ]
        events = injector.scan_tool_calls(tool_calls)
        assert len(events) == 1
        assert "tool_call:send_email" in events[0].found_in

    def test_scan_tool_calls_clean(self):
        injector = CanaryInjector()
        _, token = injector.inject("System.")
        tool_calls = [{"name": "search", "arguments": {"query": "hello world"}}]
        events = injector.scan_tool_calls(tool_calls)
        assert len(events) == 0

    def test_revoke_removes_token(self):
        injector = CanaryInjector()
        _, token = injector.inject("System.")
        injector.revoke(token)
        assert token.value not in injector._active_tokens

    def test_revoke_all(self):
        injector = CanaryInjector()
        for _ in range(5):
            injector.inject("System.")
        injector.revoke_all()
        assert len(injector._active_tokens) == 0

    def test_session_id_stored(self):
        injector = CanaryInjector()
        _, token = injector.inject("System.", session_id="sess-42")
        assert token.session_id == "sess-42"

    def test_metadata_stored(self):
        injector = CanaryInjector()
        _, token = injector.inject("System.", metadata={"user": "alice"})
        assert token.metadata["user"] == "alice"

    def test_comment_style(self):
        injector = CanaryInjector(embed_style="comment")
        modified, token = injector.inject("System prompt.")
        assert f"<!-- {token.value} -->" in modified
