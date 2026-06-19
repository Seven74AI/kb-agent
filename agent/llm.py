"""LLM client — swappable interface for DeepSeek and Edgee.

Both clients implement the same LLMClient ABC.
DeepSeek is built first (simpler), Edgee as optional wrapper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""


class LLMClient(ABC):
    """Abstract base for LLM client implementations.

    Both DeepSeekClient and EdgeeClient implement call() with the same signature.
    """

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str,
             tools: list[dict] | None = None) -> LLMResponse:
        """Send a prompt to the LLM and return the response.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message / task description.
            tools: Optional OpenAI-compatible tool definitions.

        Returns:
            LLMResponse with content, token counts, and model identifier.
        """
        ...


class DeepSeekClient(LLMClient):
    """OpenAI-compatible client for DeepSeek API.

    Uses /v1/chat/completions with stream=False.
    """

    def __init__(self, api_key: str, model: str = "deepseek-v4-pro",
                 base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(300.0))

    def call(self, system_prompt: str, user_prompt: str,
             tools: list[dict] | None = None) -> LLMResponse:
        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            model=data.get("model", self.model),
        )


class EdgeeClient(LLMClient):
    """Client for Edgee LLM gateway (OpenAI-compatible API).

    Uses same /v1/chat/completions format, stream=False to avoid
    known Edgee streaming bugs (HTTP/2 chunk headers, StreamReset).
    """

    def __init__(self, api_key: str, model: str = "deepseek-v4-pro",
                 base_url: str = "https://api.edgee.ai/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(300.0))

    def call(self, system_prompt: str, user_prompt: str,
             tools: list[dict] | None = None) -> LLMResponse:
        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            model=data.get("model", self.model),
        )


def get_llm_client(config) -> LLMClient:
    """Factory: return the appropriate LLM client based on config.

    Args:
        config: Config dataclass from agent.config.

    Returns:
        An LLMClient instance (DeepSeekClient or EdgeeClient).

    Raises:
        ValueError: If the configured llm_provider is unknown.
    """
    import os
    provider = config.llm_provider

    if provider == "deepseek":
        api_key = config.llm_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        base_url = config.llm_base_url or "https://api.deepseek.com/v1"
        return DeepSeekClient(
            api_key=api_key,
            model=config.llm_model,
            base_url=base_url,
        )
    elif provider == "edgee":
        api_key = config.edgee_api_key or os.getenv("EDGEE_API_KEY", "")
        base_url = config.llm_base_url or "https://api.edgee.ai/v1"
        return EdgeeClient(
            api_key=api_key,
            model=config.llm_model,
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
