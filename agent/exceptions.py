"""Exception types for pipeline step error handling.

Three error buckets align with the error model:
- RETRY: transient errors (network, rate-limit) — exponential backoff
- PAUSE: operator intervention needed (cookies invalid, IP flag, infra down)
- FAIL: unrecoverable (URL deleted, 404, unsupported content)
"""

from enum import Enum


class ErrorType(str, Enum):
    """Error severity buckets for pipeline steps."""
    RETRY = "retry"    # Transient — backoff 5s→15s→60s→180s→PAUSE
    PAUSE = "pause"    # Needs operator — store reason, resume via dashboard
    FAIL = "fail"      # Unrecoverable — mark failed, move on


class StepError(Exception):
    """Pipeline step error with typed severity.

    Raised by step functions to signal the consumer how to handle the failure.

    Attributes:
        error_type: RETRY, PAUSE, or FAIL.
        step_name: Optional name of the step that raised the error.
        details: Optional additional context for the operator or logs.
    """

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.RETRY,
        step_name: str | None = None,
        details: str | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.step_name = step_name
        self.details = details

    def __repr__(self) -> str:
        return (
            f"StepError({self.error_type.value}"
            + (f", step={self.step_name}" if self.step_name else "")
            + f", msg={self.args[0]!r})"
        )
