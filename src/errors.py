"""Shared errors for optional extras. Safe to import without src.baselines."""


class OptionalExtraError(RuntimeError):
    """Raised when an optional synthesizer stack is not installed.

    Callers skip the method when the optional stack is missing.
    """
