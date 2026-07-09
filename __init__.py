# Re-export the Rust extension's public surface.
# Consumers do: from zero_copy_buffer import Transition, SimpleBuffer
# The ._core suffix is a maturin convention — callers never need to know.
from zero_copy_buffer._core import Transition, SimpleBuffer

__all__ = ["Transition", "SimpleBuffer"]