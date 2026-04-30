"""Prompt firewall: second-model defense against indirect injection.

Indirect prompt injection comes from untrusted data sources (web pages,
documents, tool outputs). A firewall model — a separate, smaller LLM call —
classifies the retrieved content as safe or malicious before it's included
in the main prompt.

Pattern:
    Main model is about to include external content
    → Firewall model evaluates: is this trying to hijack the agent?
    → If safe: include normally
    → If suspicious: strip, quarantine, or raise

This is a sync/async-agnostic interface — wire in your own inference backend.

Usage::

    firewall = PromptFirewall(classifier_fn=my_llm_classifier)
    result = await firewall.check("Web page content here...")
    if result.is_safe:
        include_in_prompt(result.content)
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FirewallDecision:
    """Decision from the prompt firewall.

    Attributes:
        is_safe: True if content is safe to include in the main prompt.
        content: The (possibly modified) content to include.
        risk_score: 0.0 (safe) to 1.0 (definitely malicious).
        reason: Human-readable explanation of the decision.
        original: Original content before any cleaning.
    """

    is_safe: bool
    content: str
    risk_score: float = 0.0
    reason: str = ""
    original: str = ""

    @property
    def was_modified(self) -> bool:
        """True if the content was changed by the firewall."""
        return self.content != self.original


@dataclass
class QuarantinedContent:
    """Content quarantined by the firewall (not safe to include)."""

    content: str
    risk_score: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Heuristic patterns for content extracted from external sources
_EXTERNAL_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Injections targeting tool-use agents
    re.compile(r"(assistant|ai|model)\s*:\s*(ignore|disregard|forget)\s+", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
    re.compile(r"SYSTEM:\s*(ignore|you\s+are\s+now)", re.IGNORECASE),
    # Exfiltration via tool calls
    re.compile(r"call\s+the\s+(send_email|post_to|upload|exfiltrate)\s+tool", re.IGNORECASE),
    re.compile(
        r"use\s+the\s+\w+\s+tool\s+to\s+send\s+(all|my|the)\s+(data|content|information)",
        re.IGNORECASE,
    ),
    # Hidden text attempts (many spaces before payload)
    re.compile(r" {10,}[A-Z]{2,}", re.IGNORECASE),
]

ClassifierFn = Callable[[str], Awaitable[float]]


async def _default_heuristic_classifier(content: str) -> float:
    """Heuristic-only classifier (no external model call).

    Returns a risk score 0.0-1.0 based on pattern matching alone.
    Replace with an actual LLM call in production.

    Args:
        content: External content to evaluate.

    Returns:
        Risk score from 0.0 (safe) to 1.0 (definitely malicious).
    """
    matches = sum(1 for pattern in _EXTERNAL_INJECTION_PATTERNS if pattern.search(content))
    # Each pattern match adds 0.3, capped at 0.9
    return min(matches * 0.3, 0.9)


class PromptFirewall:
    """Firewall for external content before it enters the main prompt.

    Uses a classifier function (heuristic or LLM-based) to assign a risk
    score, then allows/blocks/quarantines based on configurable thresholds.

    Args:
        classifier_fn: Async function that returns a 0.0-1.0 risk score.
            Defaults to heuristic pattern matching.
        block_threshold: Content with score >= this is blocked entirely.
        warn_threshold: Content with score >= this gets a warning but passes.
        quarantine_store: Optional list to accumulate blocked content for review.

    Example::

        async def my_llm_classifier(content: str) -> float:
            response = await llm.classify(
                f"Rate from 0.0 to 1.0 how likely this is a prompt injection: {content}"
            )
            return float(response.strip())

        firewall = PromptFirewall(classifier_fn=my_llm_classifier)
        decision = await firewall.check(web_page_content)
        if decision.is_safe:
            prompt = build_prompt(decision.content)
    """

    def __init__(
        self,
        classifier_fn: ClassifierFn | None = None,
        block_threshold: float = 0.7,
        warn_threshold: float = 0.3,
        quarantine_store: list[QuarantinedContent] | None = None,
    ) -> None:
        self._classifier = classifier_fn or _default_heuristic_classifier
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold
        self._quarantine = quarantine_store if quarantine_store is not None else []

    async def check(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> FirewallDecision:
        """Check external content and return a firewall decision.

        Args:
            content: External content (web page, document, tool output, etc.)
            metadata: Optional context (source URL, tool name, etc.)

        Returns:
            :class:`FirewallDecision` with is_safe, risk_score, and reason.
        """
        risk_score = await self._classifier(content)

        if risk_score >= self._block_threshold:
            self._quarantine.append(
                QuarantinedContent(
                    content=content,
                    risk_score=risk_score,
                    reason=f"risk_score={risk_score:.2f} >= threshold={self._block_threshold}",
                    metadata=metadata or {},
                )
            )
            return FirewallDecision(
                is_safe=False,
                content="[CONTENT BLOCKED BY PROMPT FIREWALL]",
                risk_score=risk_score,
                reason=f"Blocked: risk score {risk_score:.2f} exceeds threshold {self._block_threshold}",
                original=content,
            )

        reason = ""
        if risk_score >= self._warn_threshold:
            reason = f"Warning: elevated risk score {risk_score:.2f}"

        return FirewallDecision(
            is_safe=True,
            content=content,
            risk_score=risk_score,
            reason=reason,
            original=content,
        )

    async def check_batch(
        self,
        contents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[FirewallDecision]:
        """Check multiple content items concurrently.

        Args:
            contents: List of external content strings.
            metadata: Optional per-item metadata dicts.

        Returns:
            List of :class:`FirewallDecision` in the same order.
        """
        import asyncio

        metas = metadata or [{}] * len(contents)
        tasks = [self.check(c, m) for c, m in zip(contents, metas, strict=False)]
        return list(await asyncio.gather(*tasks))

    @property
    def quarantine_count(self) -> int:
        """Number of items currently in quarantine."""
        return len(self._quarantine)

    def quarantine_summary(self) -> list[dict[str, Any]]:
        """Return a summary of quarantined items (no full content).

        Returns:
            List of dicts with ``risk_score``, ``reason``, and ``metadata``.
        """
        return [
            {
                "risk_score": item.risk_score,
                "reason": item.reason,
                "content_length": len(item.content),
                "metadata": item.metadata,
            }
            for item in self._quarantine
        ]
