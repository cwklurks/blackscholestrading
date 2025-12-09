"""
Numba JIT compatibility layer.

Provides a graceful fallback when Numba is not available, allowing the codebase
to function (albeit more slowly) in environments without Numba installed.
"""

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        """No-op decorator when Numba is not available."""
        def decorator(func):
            return func
        return decorator

