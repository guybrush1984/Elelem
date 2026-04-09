"""
Custom exceptions for Elelem.
"""


class InfrastructureError(Exception):
    """Errors that should trigger candidate iteration (e.g., timeouts, rate limits)."""

    def __init__(self, message: str, provider: str = None, model: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class ModelError(Exception):
    """Errors that should not trigger candidate iteration (e.g., validation failures)."""

    def __init__(self, message: str, provider: str = None, model: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class TooSlowError(Exception):
    """Provider is streaming too slowly for this request's min_tps requirement.

    Triggers next candidate WITHOUT cooldown — the provider isn't broken,
    just too slow for this particular request.
    """

    def __init__(self, message: str, observed_tps: float = 0, elapsed: float = 0,
                 provider: str = None, model: str = None):
        super().__init__(message)
        self.observed_tps = observed_tps
        self.elapsed = elapsed
        self.provider = provider
        self.model = model


class JsonSchemaError(Exception):
    """JSON parsed successfully but failed schema validation. Potentially fixable by LLM."""

    def __init__(self, message: str, content: str = None):
        super().__init__(message)
        self.content = content  # The parsed but invalid JSON content
