# Re-export the Rust extension's public surface.
# Consumers do: from zero_copy_buffer import Transition, RingBuffer
# The ._core suffix is a maturin convention — callers never need to know.
from zero_copy_buffer._core import Transition, RingBuffer

__all__ = ["Transition", "RingBuffer"]