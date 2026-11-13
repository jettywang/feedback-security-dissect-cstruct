from __future__ import annotations

import contextlib
import functools
import warnings
from io import SEEK_CUR, BytesIO, UnsupportedOperation
from types import MethodType
from typing import TYPE_CHECKING, Any, BinaryIO, ClassVar, TypeVar

from dissect.cstruct.exception import ArraySizeError
from dissect.cstruct.expression import Expression
from dissect.cstruct.util import classproperty, normalize_endianness

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from dissect.cstruct.cstruct import AllowedEndianness, Endianness, cstruct


EOF = -0xE0F  # Negative counts are illegal anyway, so abuse that for our EOF sentinel

_IMMUTABLE_TYPES = (int, float, bytes, str)


class MetaType(type):
    """Base metaclass for cstruct type classes."""

    __cs__: cstruct
    """The cstruct instance this type class belongs to."""
    __size__: int | None
    """The size of the type in bytes. Can be ``None`` for dynamic sized types."""
    __dynamic__: bool
    """Whether or not the type is dynamically sized."""
    __alignment__: int | None
    """The alignment of the type in bytes. A value of ``None`` will be treated as 1-byte aligned."""

    # This must be the actual type, but since Array is a subclass of BaseType, we correct this at the bottom of the file
    __ArrayType__: type[BaseArray] = "Array"
    """The array type for this type class."""

    def __call__(cls, *args, **kwargs) -> Self:  # type: ignore
        """Adds support for ``TypeClass(bytes | file-like object)`` parsing syntax."""
        if len(args) == 1 and not kwargs:
            # Fast path for the most common case without keyword arguments
            obj = args[0]
            if type(obj) in _BUFFER_TYPES:
                if len(obj) == cls.__size__ and isinstance(obj, bytes) and issubclass(cls, bytes):
                    # Shortcut for char/bytes type
                    return type.__call__(cls, obj)
                return cls._read(BytesIO(obj), endian=cls.__cs__.endian)

            if hasattr(obj, "read") and not isinstance(obj, cls):
                return cls._read(obj, endian=cls.__cs__.endian)

            if not _is_buffer_type(obj):
                # E.g. initializing with a value
                return type.__call__(cls, obj)

        if len(args) == 1 and not isinstance(obj := args[0], cls):
            if type(obj) not in _BUFFER_TYPES and hasattr(obj, "read"):
                endian = (
                    normalize_endianness(endian) if (endian := kwargs.get("endian")) is not None else cls.__cs__.endian
                )
                return cls._read(obj, endian=endian)

            if isinstance(obj, bytes) and len(obj) == cls.__size__ and issubclass(cls, bytes):
                # Shortcut for char/bytes type
                return type.__call__(cls, *args, **kwargs)

            if _is_buffer_type(obj):
                return cls.reads(obj, endian=kwargs.get("endian"))

        return type.__call__(cls, *args, **kwargs)

    def __getitem__(cls, num_entries: int | str | Expression | None) -> type[BaseArray]:
        """Create a new array with the given number of entries."""
        if isinstance(num_entries, str):
            num_entries = Expression(num_entries)
            with contextlib.suppress(Exception):
                num_entries = num_entries.evaluate(cls.__cs__)

        return cls.__cs__._make_array(cls, num_entries)

    def __bool__(cls) -> bool:
        """Type class is always truthy."""
        return True

    def __len__(cls) -> int:
        """Return the byte size of the type."""
        if cls.__size__ is None:
            raise TypeError("Dynamic size")

        return cls.__size__

    def __default__(cls) -> Self:  # type: ignore
        """Return the default value of this type."""
        return cls()

    def reads(cls, data: bytes | memoryview | bytearray, *, endian: AllowedEndianness | None = None) -> Self:  # type: ignore
        """Parse the given data from a bytes-like object.

        Args:
            data: Bytes-like object to parse.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
                    When omitted, the cstruct instance's current endianness is used.

        Returns:
            The parsed value of this type.
        """
        endian = normalize_endianness(endian) if endian is not None else cls.__cs__.endian
        return cls._read(BytesIO(data), endian=endian)

    def read(cls, obj: BinaryIO | bytes | memoryview | bytearray, *, endian: AllowedEndianness | None = None) -> Self:  # type: ignore
        """Parse the given data.

        Args:
            obj: Data to parse. Can be a bytes-like object or a file-like object.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
                    When omitted, the cstruct instance's current endianness is used.

        Returns:
            The parsed value of this type.
        """
        if _is_buffer_type(obj):
            return cls.reads(obj, endian=endian)

        if not hasattr(obj, "read"):
            raise TypeError("Invalid object type")

        endian = normalize_endianness(endian) if endian is not None else cls.__cs__.endian
        return cls._read(obj, endian=endian)

    def write(cls, stream: BinaryIO, value: Any, *, endian: AllowedEndianness | None = None) -> int:
        """Write a value to a writable file-like object.

        Args:
            stream: File-like objects that supports writing.
            value: Value to write.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
                    When omitted, the cstruct instance's current endianness is used.

        Returns:
            The amount of bytes written.
        """
        endian = normalize_endianness(endian) if endian is not None else cls.__cs__.endian
        return cls._write(stream, value, endian=endian)

    def dumps(cls, value: Any, *, endian: AllowedEndianness | None = None) -> bytes:
        """Dump a value to a byte string.

        Args:
            value: Value to dump.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
                    When omitted, the cstruct instance's current endianness is used.

        Returns:
            The raw bytes of this type.
        """
        endian = normalize_endianness(endian) if endian is not None else cls.__cs__.endian

        out = BytesIO()
        cls._write(out, value, endian=endian)
        return out.getvalue()

    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:  # type: ignore
        """Internal function for reading value.

        Must be implemented per type.

        Args:
            stream: The stream to read from.
            context: Optional reading context.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
        """
        raise NotImplementedError

    def _read_array(
        cls, stream: BinaryIO, count: int, *, context: dict[str, Any] | None = None, endian: Endianness
    ) -> list[Self]:  # type: ignore
        """Internal function for reading array values.

        Allows type implementations to do optimized reading for their type.

        Args:
            stream: The stream to read from.
            count: The amount of values to read.
            context: Optional reading context.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
        """
        result = []
        if count == EOF:
            if _is_seekable(stream):
                while not _is_eof(stream):
                    result.append(cls._read(stream, context=context, endian=endian))
                return result

            # A non-seekable stream can't be probed and rewound, so rely on the exact reads of each element
            # to raise an EOFError once the stream is truly exhausted
            while True:
                try:
                    result.append(cls._read(stream, context=context, endian=endian))
                except EOFError:
                    return result

        # This is purposefully a simple loop
        # PyPy will pre-allocate the list for a list comprehension ahead of time, which will cause huge allocations in
        # case wrong count is read, instead of causing an EOFError when trying to read the type.
        for _ in range(count):
            result.append(cls._read(stream, context=context, endian=endian))  # noqa: PERF401
        return result

    def _read_0(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> list[Self]:
        """Internal function for reading null-terminated data.

        "Null" is type specific, so must be implemented per type.

        Args:
            stream: The stream to read from.
            context: Optional reading context.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
        """
        raise NotImplementedError

    def _write(cls, stream: BinaryIO, data: Any, *, endian: Endianness) -> int:
        raise NotImplementedError

    def _write_array(cls, stream: BinaryIO, array: list[Self], *, endian: Endianness) -> int:  # type: ignore
        """Internal function for writing arrays.

        Allows type implementations to do optimized writing for their type.

        Args:
            stream: The stream to read from.
            array: The array to write.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
        """
        write = cls._write
        return sum(write(stream, entry, endian=endian) for entry in array)

    def _write_0(cls, stream: BinaryIO, array: list[Self], *, endian: Endianness) -> int:  # type: ignore
        """Internal function for writing null-terminated arrays.

        Allows type implementations to do optimized writing for their type.

        Args:
            stream: The stream to read from.
            array: The array to write.
            endian: The endianness to use when parsing (little, big, network, <, >, !, @ or =).
        """
        return cls._write_array(stream, [*array, cls.__default__()], endian=endian)

    @property
    def cs(cls) -> cstruct:
        """Access the cstruct object for this type."""
        warnings.warn(
            "The 'cs' property is deprecated, use '__cs__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__cs__

    @property
    def size(cls) -> int | None:
        """Access the size of this type."""
        warnings.warn(
            "The 'size' property is deprecated, use '__size__', len(cls) or sizeof(cls) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__size__

    @property
    def dynamic(cls) -> bool:
        """Access whether this type is dynamically sized."""
        warnings.warn(
            "The 'dynamic' property is deprecated, use '__dynamic__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__dynamic__

    @property
    def alignment(cls) -> int | None:
        """Access the alignment of this type."""
        warnings.warn(
            "The 'alignment' property is deprecated, use '__alignment__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__alignment__

    @property
    def ArrayType(cls) -> type[BaseArray]:
        """Access the array type of this type."""
        warnings.warn(
            "The 'ArrayType' property is deprecated, use '__ArrayType__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__ArrayType__


class _overload:
    """Descriptor to use on the ``write`` and ``dumps`` methods on cstruct types.

    Allows for calling these methods on both the type and instance.

    Example:
        >>> int32.dumps(123)
        b'\\x7b\\x00\\x00\\x00'
        >>> int32(123).dumps()
        b'\\x7b\\x00\\x00\\x00'
    """

    def __init__(self, func: Callable[..., Any], method: Callable[..., Any]) -> None:
        self.func = func
        self.method = method

    def __get__(self, instance: BaseType | None, owner: type[BaseType]) -> Callable[..., Any]:
        if instance is None:
            return functools.partial(self.func, owner)
        return MethodType(self.method, instance)


def _instance_dumps(self: BaseType, *, endian: AllowedEndianness | None = None) -> bytes:
    return MetaType.dumps(self.__class__, self, endian=endian)


def _instance_write(self: BaseType, stream: BinaryIO, *, endian: AllowedEndianness | None = None) -> int:
    return MetaType.write(self.__class__, stream, self, endian=endian)


class BaseType(metaclass=MetaType):
    """Base class for cstruct type classes."""

    dumps = _overload(MetaType.dumps, _instance_dumps)
    write = _overload(MetaType.write, _instance_write)

    def __len__(self) -> int:
        """Return the byte size of the type."""
        if self.__class__.__size__ is None:
            raise TypeError("Dynamic size")

        return self.__class__.__size__


T = TypeVar("T", bound=BaseType)


class BaseArray(BaseType):
    """Implements a fixed or dynamically sized array type.

    Example:
        When using the default C-style parser, the following syntax is supported:

            x[3] -> 3 -> static length.
            x[] -> None -> null-terminated.
            x[expr] -> expr -> dynamic length.
    """

    __type__: ClassVar[type[BaseType]]
    __num_entries__: ClassVar[int | Expression | None]
    __null_terminated__: ClassVar[bool]

    @classmethod
    def __default__(cls) -> BaseType:
        count = cls.__num_entries__ if isinstance(cls.__num_entries__, int) else 0
        if isinstance(value := cls.__type__.__default__(), _IMMUTABLE_TYPES):
            return type.__call__(cls, [value] * count)
        return type.__call__(cls, [cls.__type__.__default__() for _ in range(count)])

    @classmethod
    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> list[BaseType]:
        if cls.__null_terminated__:
            return cls.__type__._read_0(stream, context=context, endian=endian)

        return cls.__type__._read_array(stream, cls._num_entries(context), context=context, endian=endian)

    @classmethod
    def _num_entries(cls, context: dict[str, Any] | None) -> int:
        """Return the number of entries to read for this (non null-terminated) array type."""
        if isinstance(cls.__num_entries__, int):
            return max(0, cls.__num_entries__)

        if cls.__num_entries__ is None:
            return EOF

        try:
            return max(0, cls.__num_entries__.evaluate(cls.__cs__, context))
        except Exception:
            if cls.__num_entries__.expression != "EOF":
                raise
            return EOF

    @classmethod
    def _write(cls, stream: BinaryIO, data: list[Any], *, endian: Endianness) -> int:
        if cls.__null_terminated__:
            return cls.__type__._write_0(stream, data, endian=endian)

        if not cls.__dynamic__:
            data = cls._pad(data)

        return cls.__type__._write_array(stream, data, endian=endian)

    @classmethod
    def _pad(cls, data: list[Any]) -> list[Any]:
        """Internal function for padding the entries of a static array to its size.

        Args:
            data: The entries of the array.
        """
        if remaining := cls.__num_entries__ - (actual_size := len(data)):
            if remaining < 0:
                raise ArraySizeError(f"Expected static array size {cls.__num_entries__}, got {actual_size} instead.")
            data = data + [cls.__type__.__default__()] * remaining

        return data

    @classproperty
    def type(cls) -> type[BaseType]:
        """Access the array element type."""
        warnings.warn(
            "The 'type' property is deprecated, use '__type__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__type__

    @classproperty
    def num_entries(cls) -> int | Expression | None:
        """Access the number of entries for this array type."""
        warnings.warn(
            "The 'num_entries' property is deprecated, use '__num_entries__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__num_entries__

    @classproperty
    def null_terminated(cls) -> bool:
        """Access whether this array type is null-terminated."""
        warnings.warn(
            "The 'null_terminated' property is deprecated, use '__null_terminated__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__null_terminated__


class Array(list[T], BaseArray):
    @classmethod
    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> list[T]:
        return type.__call__(cls, super()._read(stream, context=context, endian=endian))


_BUFFER_TYPES = (bytes, memoryview, bytearray)


def _is_buffer_type(value: object) -> bool:
    return isinstance(value, _BUFFER_TYPES)


def _is_seekable(stream: BinaryIO) -> bool:
    try:
        return stream.seekable()
    except Exception:
        return False


def _require_seekable(stream: BinaryIO, feature: str) -> None:
    """Raise a clear error when a feature needs seeking but the stream isn't seekable.

    This keeps non-seekable features from silently dropping or repositioning bytes.
    """
    if not _is_seekable(stream):
        raise UnsupportedOperation(f"{feature} requires a seekable stream")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes from ``stream``, retrying on short reads.

    A stream may return fewer bytes than requested while more data is still in flight (e.g. a rate-limited
    socket), so a short read is retried instead of being treated as EOF. Only a read that returns no bytes
    signals a true end of stream.

    Args:
        stream: The stream to read from.
        size: The exact amount of bytes to read.

    Raises:
        EOFError: If the stream ends before ``size`` bytes could be read.
    """
    if size <= 0:
        return b""

    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            # An empty read is the only reliable signal of a true EOF
            read = size - remaining
            raise EOFError(f"Read {read} bytes, but expected {size}")

        chunks.append(chunk)
        remaining -= len(chunk)

    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _read_remaining(stream: BinaryIO) -> bytes:
    """Read every byte left in a stream, tolerating short reads."""
    chunks = []
    while True:
        chunk = stream.read()
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_partial(stream: BinaryIO, size: int) -> bytes:
    """Read up to ``size`` bytes, retrying short reads but stopping at a true EOF.

    Unlike :func:`_read_exact`, an empty read is a normal end of stream here, so a shorter result than
    ``size`` is returned instead of raising.
    """
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break

        chunks.append(chunk)
        remaining -= len(chunk)

    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _read_null_terminated(stream: BinaryIO, width: int) -> bytes:
    """Read data up to a null terminator of the given width from a seekable stream, in chunks.

    Leaves the stream just after the terminator.
    """
    _require_seekable(stream, "Reading a null-terminated array in chunks")

    terminator = b"\x00" * width
    buf = []
    while True:
        chunk = _read_partial(stream, 128 * width)
        if not isinstance(chunk, (bytes, bytearray)):
            # E.g. a memoryview, which doesn't support find()
            chunk = bytes(chunk)

        # Find a terminator that is aligned on an item
        idx = chunk.find(terminator)
        while idx != -1 and idx % width:
            idx = chunk.find(terminator, idx + 1)

        if idx != -1:
            buf.append(chunk[:idx])
            stream.seek(idx + width - len(chunk), SEEK_CUR)
            return b"".join(buf)

        if not chunk or len(chunk) % width:
            # The end of the stream, possibly with a partial item
            raise EOFError(f"Read {len(chunk) % width} bytes, but expected {width}")

        buf.append(chunk)


def _is_eof(stream: BinaryIO) -> bool:
    """Check if the stream has reached EOF.

    Seekable streams are probed with a single byte that is put back. A non-seekable stream can't put the
    probe byte back, so reading it until EOF is handled by the type implementations instead.
    """
    if not _is_seekable(stream):
        raise UnsupportedOperation("Detecting EOF requires a seekable stream")

    pos = stream.tell()
    if not stream.read(1):
        return True

    stream.seek(pos)
    return False


# As mentioned in the BaseType class, we correctly set the type here
MetaType.__ArrayType__ = Array
