"""Shared errors for optional extras. Safe to import without src.baselines."""


class OptionalExtraError(RuntimeError):
    """Raised when an optional synthesizer stack is not installed.

    Callers must skip the method and must not write invented numeric rows.
    """
