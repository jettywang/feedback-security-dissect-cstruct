from __future__ import annotations

from typing import TYPE_CHECKING, Any, BinaryIO

from dissect.cstruct.types.base import EOF, BaseType, MetaType, _read_exact
from dissect.cstruct.util import ENDIANNESS_TO_BYTEORDER_MAP

if TYPE_CHECKING:
    from typing_extensions import Self

    from dissect.cstruct.cstruct import Endianness


class Int(int, BaseType):
    """Integer type that can span an arbitrary amount of bytes."""

    signed: bool

    @classmethod
    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:
        data = _read_exact(stream, cls.__size__)
        return type.__call__(cls, int.from_bytes(data, ENDIANNESS_TO_BYTEORDER_MAP[endian], signed=cls.signed))

    @classmethod
    def _read_array(
        cls, stream: BinaryIO, count: int, *, context: dict[str, Any] | None = None, endian: Endianness
    ) -> list[Self]:
        if count == EOF:
            return MetaType._read_array(cls, stream, count, context=context, endian=endian)

        size = cls.__size__
        data = _read_exact(stream, size * count)

        byteorder = ENDIANNESS_TO_BYTEORDER_MAP[endian]
        return [
            type.__call__(cls, int.from_bytes(data[i : i + size], byteorder, signed=cls.signed))
            for i in range(0, size * count, size)
        ]

    @classmethod
    def _read_0(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:
        result = []

        while True:
            if (value := cls._read(stream, context=context, endian=endian)) == 0:
                break

            result.append(value)

        return result

    @classmethod
    def _write(cls, stream: BinaryIO, data: int, *, endian: Endianness) -> int:
        return stream.write(data.to_bytes(cls.__size__, ENDIANNESS_TO_BYTEORDER_MAP[endian], signed=cls.signed))
