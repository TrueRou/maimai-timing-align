try:
	from .models import AlignConfig, AlignResult
except ImportError:  # pragma: no cover
	from models import AlignConfig, AlignResult

__all__ = ["AlignConfig", "AlignResult"]
