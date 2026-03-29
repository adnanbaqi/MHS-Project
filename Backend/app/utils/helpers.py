"""
Utility helpers for the mental health risk system.
"""

from datetime import datetime


def get_timestamp() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().isoformat() + "Z"


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float value between min and max."""
    return max(min_val, min(max_val, value))
