from __future__ import annotations


class Error(Exception):
    pass


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class ResolveError(Error):
    pass


class NullPointerDereference(Error):
    pass


class ArraySizeError(Error):
    pass


class LayoutPreservationError(Error):
    """Raised when a fidelity edit cannot be applied without moving bytes in the original layout.

    A fidelity (in-place) edit must keep every byte at its original offset. This is impossible when a
    structure contains a dynamically sized field, when a fixed array is replaced by a different number
    of entries, or when any modification would change the encoded size of a field and shift the bytes
    that follow it.
    """


class ExpressionParserError(Error):
    pass


class UnknownIdentifierError(ExpressionParserError):
    pass
