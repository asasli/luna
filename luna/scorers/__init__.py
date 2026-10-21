"""OOD scorer registry and convenience imports."""
from luna.scorers.base import (
    BaseScorer,
    SCORER_REGISTRY,
    register_scorer,
    get_available_scorers,
    get_scorer_families,
)

# Import submodules to trigger @register_scorer decorators
import luna.scorers.uncertainty  # noqa: F401
import luna.scorers.distance     # noqa: F401
import luna.scorers.density      # noqa: F401
import luna.scorers.hybrid       # noqa: F401
import luna.scorers.mc_uncertainty  # noqa: F401

__all__ = [
    "BaseScorer",
    "SCORER_REGISTRY",
    "register_scorer",
    "get_available_scorers",
    "get_scorer_families",
]
