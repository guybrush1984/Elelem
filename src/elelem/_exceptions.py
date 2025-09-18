"""
Custom exceptions for Elelem.
"""


class InfrastructureError(Exception):
    """Errors that should trigger candidate iteration."""
    pass


class ModelError(Exception):
    """Errors that should not trigger candidate iteration."""
    pass