from __future__ import annotations

from typing import Any, BinaryIO

from dissect.cstruct.exception import PointerBoundsError

# Reserved key under which a ReadContext is carried through the field-value context dictionaries
_READ_CONTEXT_KEY = "__read_context__"

#: Default maximum depth of a pointer dereference chain
DEFAULT_MAX_DEPTH = 32


class ReadContext:
    """Optional reading context that constrains pointer dereferences to a region of the input stream.

    When a ``ReadContext`` is supplied to ``read`` or ``reads``, all pointers read from that point on
    (including nested pointers and string pointers, which inherit the same context) may only be
    dereferenced within the absolute stream offsets ``[start, end)``. Dereferencing a pointer that
    targets a location outside of this range raises a :class:`PointerBoundsError`. Reads of
    dynamically sized values, such as null-terminated strings, are clamped to the range and fail
    closed with an ``EOFError`` if they would run past it.

    Nested dereferences form a chain with an explicit depth policy: following a chain of pointers
    (e.g. a cyclic linked list) for more than ``max_depth`` hops raises a :class:`PointerDepthError`.
    Aliased pointers that legitimately share data are unaffected, since the depth is tracked per
    dereference chain and not per address.

    Every dereference records the interval it accessed in ``accessed`` as ``(start, end)`` absolute
    stream offsets, including the partially read interval when a dereference fails halfway.

    Args:
        start: First absolute stream offset (inclusive) that pointers may target.
        end: Last absolute stream offset (exclusive) that pointers may target.
             ``None`` means no upper bound.
        max_depth: Maximum length of a chained pointer dereference before raising
                   :class:`PointerDepthError`.
    """

    __slots__ = ("start", "end", "max_depth", "accessed", "depth")

    def __init__(self, start: int = 0, end: int | None = None, *, max_depth: int = DEFAULT_MAX_DEPTH):
        if start < 0:
            raise ValueError("start must be a positive offset")
        if end is not None and end < start:
            raise ValueError("end must not be smaller than start")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        self.start = start
        self.end = end
        self.max_depth = max_depth
        self.accessed: list[tuple[int, int]] = []
        self.depth = 0

    def __repr__(self) -> str:
        end = "None" if self.end is None else hex(self.end)
        return f"<ReadContext [{self.start:#x}, {end}) depth={self.depth}/{self.max_depth}>"

    def derive(self) -> ReadContext:
        """Return a derived context for a nested dereference, sharing the bounds and access log."""
        context = ReadContext.__new__(ReadContext)
        context.start = self.start
        context.end = self.end
        context.max_depth = self.max_depth
        context.accessed = self.accessed
        context.depth = self.depth + 1
        return context

    def check(self, offset: int, size: int = 0) -> None:
        """Validate that the range ``[offset, offset + size)`` lies within the bounds of this context."""
        if offset < self.start or (self.end is not None and offset + size > self.end):
            end = "EOF" if self.end is None else hex(self.end)
            raise PointerBoundsError(
                f"Pointer dereference of {size} bytes at {offset:#x} is outside the allowed range "
                f"[{self.start:#x}, {end})"
            )

    def record(self, start: int, end: int) -> None:
        """Record an accessed interval of absolute stream offsets."""
        self.accessed.append((start, end))

    def bounded_stream(self, stream: BinaryIO) -> _BoundedStream:
        """Return a proxy of the given stream that clamps reads to the bounds of this context."""
        return _BoundedStream(stream, self)


class _BoundedStream:
    """A stream proxy that clamps reads to the end of a :class:`ReadContext`.

    Seeks are passed through to the underlying stream; reads at or past the end of the allowed range
    behave like EOF, so readers of dynamically sized values fail closed.
    """

    def __init__(self, stream: BinaryIO, context: ReadContext):
        self._stream = stream
        self._context = context

    def read(self, size: int = -1) -> bytes:
        if (end := self._context.end) is not None:
            remaining = max(0, end - self._stream.tell())
            size = remaining if size is None or size < 0 else min(size, remaining)
        return self._stream.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()

    def seekable(self) -> bool:
        try:
            return self._stream.seekable()
        except Exception:
            return False


def _context_dict(read_context: ReadContext | None) -> dict[str, Any] | None:
    """Return a field-value context dictionary carrying the given read context, if any."""
    return {_READ_CONTEXT_KEY: read_context} if read_context is not None else None


def _inherit_read_context(values: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
    """Carry the read context of a parent context dictionary over into a new field-value dictionary."""
    if context and (read_context := context.get(_READ_CONTEXT_KEY)) is not None:
        values[_READ_CONTEXT_KEY] = read_context
    return values


def _strip_read_context(values: dict[str, Any]) -> dict[str, Any]:
    """Return the field-value dictionary without a carried read context, for use as init arguments."""
    if _READ_CONTEXT_KEY in values:
        return {key: value for key, value in values.items() if key != _READ_CONTEXT_KEY}
    return values
