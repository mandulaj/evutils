"""PyTorch integration.

Bridges between evutils and PyTorch: tensor conversion, ``Dataset`` /
``DataLoader`` wrappers, and transforms for training on event data.
"""

__all__: list[str] = []

from typing import Any
def _try_import_torch(require: bool = True) -> Any:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        if require:
            raise ImportError(
                "The torch extra is required: install with `pip install evutils[torch]`."
            ) from exc
        else:
            return False
    return torch
