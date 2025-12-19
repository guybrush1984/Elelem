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
