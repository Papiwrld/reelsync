"""Shared LLM machinery for the agentic intelligence layers.

Every intelligent stage (content intelligence, research, planning, QA) calls
the LLM through the same helpers so provider abstraction, bounded retries,
structured JSON extraction and per-agent observability stay consistent.

Design rules (shared with the agent graph):
- All LLM calls go through ``llm._generate_response``, which abstracts every
  provider, timeout and retry policy (provider-abstract).
- Every LLM-dependent stage has a deterministic fallback so a dead provider
  degrades gracefully instead of killing the pipeline.
- The tracker records statuses and retries per agent, never chain-of-thought.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from loguru import logger

from app.services import llm

_AGENT_LLM_RETRIES = 3


class AgenticError(RuntimeError):
    """Base error for the agentic intelligence layers."""


class AgenticScriptError(AgenticError):
    """Raised when the script itself cannot be produced (no fallback exists)."""


class AgentTracker:
    """Per-run observability: which agents used the LLM vs fell back.

    Records status (``llm`` | ``fallback`` | ``failed``), the fallback
    reason, and retry attempts per agent. Never records chain-of-thought.

    Also acts as a lightweight circuit breaker: once a provider-level failure
    is observed (quota exhausted, invalid key, network down), ``degraded``
    flips and subsequent LLM calls short-circuit to their deterministic
    fallbacks instead of firing doomed requests that only burn credits.
    """

    def __init__(self) -> None:
        self.statuses: Dict[str, str] = {}
        self.reasons: Dict[str, str] = {}
        self.retries: Dict[str, int] = {}
        self.degraded: bool = False
        self.degrade_reason: str = ""

    def set_ok(self, agent: str, attempts: int) -> None:
        self.statuses[agent] = "llm"
        self.retries[agent] = max(self.retries.get(agent, 0), attempts)

    def set_fallback(self, agent: str, reason: str, attempts: int = 0) -> None:
        self.statuses.setdefault(agent, "fallback")
        self.reasons[agent] = reason
        self.retries[agent] = max(self.retries.get(agent, 0), attempts)

    def mark_degraded(self, reason: str) -> None:
        """Trip the circuit breaker: the provider is unusable for this run."""
        if not self.degraded:
            self.degraded = True
            self.degrade_reason = reason

    def set_failed(self, agent: str, reason: str, attempts: int = 0) -> None:
        self.statuses[agent] = "failed"
        self.reasons[agent] = reason
        self.retries[agent] = max(self.retries.get(agent, 0), attempts)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            "statuses": dict(self.statuses),
            "fallback_reasons": dict(self.reasons),
            "retries": dict(self.retries),
        }


def _llm_text(prompt: str, app_config=None, tracker: Optional[AgentTracker] = None, agent: str = "") -> str:
    # 熔断：本 run 已经观察到供应商级失败（额度耗尽/Key 无效/网络断开）时，
    # 后续调用直接短路，不再发起注定失败、只会继续消耗配额的请求。
    if tracker and tracker.degraded:
        raise AgenticError(
            f"llm degraded, skipping call: {tracker.degrade_reason}"
        )
    last_error: Optional[Exception] = None
    for attempt in range(_AGENT_LLM_RETRIES):
        try:
            response = llm._generate_response(prompt=prompt, app_config=app_config)
            # 供应商失败会以 "Error: ..." 字符串返回（_generate_response 不抛
            # 异常）。必须把它识别为失败而不是内容：否则错误文案会流入脚本，
            # 并且每次失败都要重试整个循环，放大配额消耗。
            if isinstance(response, str) and response.startswith("Error: "):
                if tracker:
                    tracker.mark_degraded(response)
                raise AgenticError(response)
            if tracker and agent:
                tracker.set_ok(agent, attempt + 1)
            return response
        except AgenticError:
            raise
        except Exception as exc:  # noqa: BLE001 - providers raise heterogeneous errors
            last_error = exc
            if tracker:
                tracker.mark_degraded(f"{type(exc).__name__}: {exc}")
            logger.warning(f"agentic llm call failed (attempt {attempt + 1}): {exc}")
    if tracker and agent:
        tracker.set_failed(agent, f"llm failed after {_AGENT_LLM_RETRIES} attempts", _AGENT_LLM_RETRIES)
    # 带上底层供应商错误（如 429 额度不足、401 Key 无效、网络超时），
    # 让 UI 能向用户展示可操作的失败原因，而不是笼统的 "call failed"。
    raise AgenticError(
        f"agentic llm call failed after {_AGENT_LLM_RETRIES} attempts: {last_error}"
    ) from last_error


def _extract_json_payload(text: str) -> Any:
    """Parse a JSON object/array from an LLM response, tolerating fences and prose."""
    candidate = llm._strip_code_fence(text or "")
    candidate = candidate.strip()
    if not candidate:
        raise ValueError("empty llm response")
    try:
        return json.loads(candidate)
    except ValueError:
        pass

    start = min(
        (index for index in (candidate.find("{"), candidate.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError(f"no json payload found in response: {candidate[:120]!r}")
    opening = candidate[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    for end in range(start, len(candidate)):
        char = candidate[end]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start : end + 1])
                except ValueError as exc:
                    raise ValueError("json payload is malformed") from exc
    raise ValueError("unclosed json payload")


def _llm_json(
    prompt: str,
    fallback: Callable[[], Any],
    app_config=None,
    tracker: Optional[AgentTracker] = None,
    agent: str = "",
) -> Any:
    if tracker and tracker.degraded:
        # 熔断：供应商已确认不可用，跳过 LLM 直接使用确定性兜底，
        # 避免本次 run 中后续每个阶段都再发起一次注定失败的调用。
        if tracker and agent:
            tracker.set_fallback(agent, f"skipped (degraded): {tracker.degrade_reason}")
        return fallback()
    try:
        return _extract_json_payload(_llm_text(prompt, app_config=app_config, tracker=tracker, agent=agent))
    except Exception as exc:  # noqa: BLE001 - degrade to deterministic fallback
        logger.warning(f"agentic json step failed, using fallback: {exc}")
        if tracker and agent:
            tracker.set_fallback(agent, f"{type(exc).__name__}: {exc}")
        return fallback()


def _clamp_score(value: Any, dimension: str) -> float:
    """Validate and clamp a score to [0, 10]; invalid values raise."""
    score = float(value)
    if score != score or score in (float("inf"), float("-inf")):  # NaN/inf guard
        raise ValueError(f"score {dimension!r} is not a finite number")
    return min(10.0, max(0.0, score))
