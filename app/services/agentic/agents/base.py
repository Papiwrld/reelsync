"""Base agent class with common LLM call + fallback pattern."""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ValidationError

from app.services.agent_llm import AgentTracker, _llm_json

from .protocols import AgentInput, AgentOutput


class BaseAgent:
    """Base class implementing the standard LLM+fallback pattern.

    Subclasses must define:
      - name: str
      - output_model: type[BaseModel]
      - build_prompt(input_data: AgentInput) -> str
      - build_fallback(input_data: AgentInput) -> BaseModel
    """

    name: str = "base_agent"
    output_model: type[BaseModel]

    def build_prompt(self, input_data: AgentInput) -> str:
        raise NotImplementedError

    def build_fallback(self, input_data: AgentInput) -> BaseModel:
        raise NotImplementedError

    def run(
        self,
        state: Any,
        input_data: AgentInput,
        *,
        app_config: Optional[Any] = None,
        tracker: Optional[AgentTracker] = None,
    ) -> AgentOutput:
        prompt = self.build_prompt(input_data)

        def fallback_fn() -> BaseModel:
            return self.build_fallback(input_data)

        try:
            payload = _llm_json(
                prompt,
                fallback_fn,
                app_config=app_config,
                tracker=tracker,
                agent=self.name,
            )
            if isinstance(payload, dict):
                validated = self.output_model.model_validate(payload)
            else:
                validated = fallback_fn()
        except ValidationError as exc:
            logger.warning(f"{self.name} validation failed: {exc}")
            validated = fallback_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{self.name} failed: {exc}")
            validated = fallback_fn()

        logger.info(f"{self.name} complete")
        return validated.model_dump()