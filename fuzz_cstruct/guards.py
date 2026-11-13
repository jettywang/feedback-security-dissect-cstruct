"""Resource guards that keep every fuzzing trial bounded.

Three independent termination strategies map to the three dangerous constructs the
fuzzer targets:

* undefined names / expression blow-ups -> :class:`StreamBudgetExceeded` plus the
  library's own fast ``OverflowError``;
* cyclic type dependencies -> a lowered recursion ceiling that turns infinite
  nesting into a quick :class:`RecursionError` instead of a real stack crash;
* extreme shifts / oversized counts -> a wall-clock :class:`signal.SIGALRM` budget
  and a read cap on :class:`CappedStream`.

The guards never patch the library under test; they only constrain the process and
the input stream handed to it.
"""

from __future__ import annotations

import io
import signal
import sys
from contextlib import contextmanager
from types import FrameType


class DoSBudgetExceeded(TimeoutError):
    """Raised when a trial uses more wall-clock time than its budget."""


class StreamBudgetExceeded(RuntimeError):
    """Raised when the library asks the stream for more bytes than the cap allows."""


class RecursionCeilingExceeded(RecursionError):
    """Marker for recursion exhaustion under the fuzzer's lowered ceiling."""


@contextmanager
def time_budget(seconds: float):
    """Interrupt the current trial after ``seconds`` of wall-clock time.

    ``SIGALRM`` only works in the main thread, which is where the fuzzer runs.
    """

    def _handler(signum: int, frame: FrameType | None) -> None:
        raise DoSBudgetExceeded(f"trial exceeded {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, _handler)
    # setitimer supports fractional seconds, unlike signal.alarm
    previous_itimer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
        if previous_itimer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_itimer[0], previous_itimer[1])


@contextmanager
def recursion_ceiling(limit: int):
    """Temporarily lower Python's recursion limit for the duration of a trial."""
    original = sys.getrecursionlimit()
    current = min(original, limit)
    sys.setrecursionlimit(current)
    try:
        yield
    finally:
        sys.setrecursionlimit(original)


class CappedStream(io.BytesIO):
    """A BytesIO that fails loudly on oversized reads instead of over-allocating.

    A normal ``BytesIO.read(n)`` happily returns the short buffer for an astronomic
    ``n``, which hides the fact that the library *attempted* a giant allocation right
    afterwards (list comprehensions, ``struct`` formats, ...). We treat any request
    beyond ``read_cap`` bytes as a denial-of-service signal, while bounded reads
    behave exactly like a regular stream, including short reads at EOF.
    """

    def __init__(self, data: bytes, read_cap: int):
        super().__init__(data)
        self.read_cap = read_cap
        self.max_requested = 0
        self.request_count = 0

    def read(self, size: int | None = -1) -> bytes:
        self.request_count += 1
        if size is None or size < 0:
            return super().read()

        self.max_requested = max(self.max_requested, size)
        if size > self.read_cap:
            raise StreamBudgetExceeded(
                f"library requested {size} bytes, cap is {self.read_cap}"
            )
        return super().read(size)


# Exception classes that prove an attempt to consume unbounded resources.
DOS_EXCEPTIONS = (
    DoSBudgetExceeded,
    StreamBudgetExceeded,
    MemoryError,
    RecursionCeilingExceeded,
)
