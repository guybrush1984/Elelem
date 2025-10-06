"""
Elelem - Unified API wrapper for OpenAI, GROQ, and DeepInfra
Provides cost tracking, JSON validation, and retry logic.
"""

from .core import Elelem

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["Elelem", "__version__"]