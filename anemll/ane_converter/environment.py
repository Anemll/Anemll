"""Environment detection utilities for ANEMLL."""

from __future__ import annotations


def coreml_available() -> bool:
    """Return True if coremltools is importable."""
    try:
        import coremltools as ct  # noqa: F401

        _ = ct
        return True
    except Exception:
        return False


def require_coreml() -> None:
    """Raise EnvironmentError if coremltools is unavailable."""
    if not coreml_available():
        raise EnvironmentError(
            "Core ML framework not present; conversion cannot be executed in this environment."
        )
