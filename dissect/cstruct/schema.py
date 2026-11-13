from __future__ import annotations

import hashlib
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from dissect.cstruct import compiler
from dissect.cstruct.exception import ResolveError
from dissect.cstruct.expression import Expression
from dissect.cstruct.types.base import BaseArray, BaseType, MetaType
from dissect.cstruct.types.enum import EnumMetaType
from dissect.cstruct.types.pointer import Pointer
from dissect.cstruct.types.structure import (
    Field,
    StructureMetaType,
    UnionMetaType,
    UnionProxy,
    _generate_structure__init__,
    _generate_union__init__,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dissect.cstruct.cstruct import Endianness, cstruct


def definition_digest(cs: cstruct) -> str:
    """Return a stable digest of the definitions currently registered in ``cs``.

    Only definition state that can affect a parse is included: endianness, pointer width, constants, field names,
    types, bit fields and explicit offsets. Runtime compile state and instance values are not part of the digest.
    """
    seen: dict[type[BaseType], str] = {}
    parts = [f"endian={cs.endian}", f"pointer={_signature(cs.pointer, seen)}"]
    parts.extend(f"const:{name}={value!r}" for name, value in sorted(cs.consts.items()))

    # A type is included once under its canonical (first sorted) name; aliases don't change a parse.
    canonical: dict[type[BaseType], str] = {}
    for name, type_ in sorted(cs.types.items()):
        canonical.setdefault(type_, name)
    for type_, name in sorted(canonical.items(), key=lambda item: item[1]):
        parts.append(f"type:{name}={_signature(type_, seen)}")

    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _signature(type_: type[BaseType], seen: dict[type[BaseType], str]) -> str:
    if not isinstance(type_, type):
        return repr(type_)

    if type_ in seen:
        return seen[type_]

    if isinstance(type_, EnumMetaType):
        seen[type_] = f"enum({type_.__name__})"
        members = ",".join(f"{name}={member.value!r}" for name, member in type_.__members__.items())
        return f"enum({type_.__name__}:{_signature(type_.__type__, seen)}{{{members}}})"

    if issubclass(type_, BaseArray):
        seen[type_] = "array(recursive)"
        entries = type_.__num_entries__
        count = entries.expression if isinstance(entries, Expression) else repr(entries)
        return f"array({_signature(type_.__type__, seen)}:{count})"

    if issubclass(type_, Pointer):
        seen[type_] = "pointer(recursive)"
        return f"pointer({_signature(type_.__type__, seen)})"

    if isinstance(type_, StructureMetaType):
        seen[type_] = f"{_kind(type_)}(recursive:{type_.__name__})"
        fields = ",".join(
            f"{field.name!r}:{_signature(field.type, seen)}:{field.bits!r}:{field.offset!r}:{field.alignment!r}"
            for field in type_.__fields__
        )
        return f"{_kind(type_)}({type_.__name__}:{type_.__align__}:{type_.__anonymous__}:{fields})"

    return f"{type_.__module__}.{type_.__qualname__}:{type_.__size__!r}:{type_.__alignment__!r}"


def _kind(type_: type[BaseType]) -> str:
    return "union" if isinstance(type_, UnionMetaType) else "struct"


def _readonly(*args: Any, **kwargs: Any) -> None:
    raise TypeError("Schema definitions are immutable; take a new snapshot to use changed definitions")


def _lazy_init(cls: type[BaseType]) -> Any:
    """Return an ``__init__`` that is generated with this snapshot's field types on first use."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        generator = _generate_union__init__ if isinstance(cls, UnionMetaType) else _generate_structure__init__
        init = generator(cls.__fields__)
        # Replace the proxy on the class so subsequent constructions use the generated initializer directly.
        cls.__init__ = init
        init(self, *args, **kwargs)

    return __init__


# Instances are keyed by id with a weak reference, so keeping them alive doesn't keep stale definitions alive.
# We don't attach the schema to __dict__, because structure instances intentionally have a list of field values there.
_instances: dict[int, tuple[weakref.ref[Any], Schema]] = {}


# Arrays are unhashable and not weak-referenceable, so they use a strong reference entry. It is removed when the
# owning value is no longer reachable through the structure that contains it; callers that keep a top-level array
# keep its version with it.
_arrays: dict[int, tuple[Any, Schema]] = {}


def _bind_array(value: Any, schema: Schema) -> None:
    """Bind an array value and its structure elements to the parse version."""
    if not isinstance(value, list):
        return
    if len(_arrays) > 1024:
        # Keep the side table bounded if callers parse many short-lived arrays.
        _arrays.clear()
    _arrays[id(value)] = (value, schema)
    for item in value:
        if isinstance(item, list):
            _bind_array(item, schema)
        elif hasattr(item, "__dict__"):
            _bind(item, schema)
            _bind_nested(item, schema)


def _bind_nested(obj: Any, schema: Schema, seen: set[int] | None = None) -> None:
    """Bind nested values recursively so old dumps can't mix layouts."""
    seen = set() if seen is None else seen

    if isinstance(obj, list):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for value in obj:
            _bind_nested(value, schema, seen)
        return

    if isinstance(obj, UnionProxy):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        _bind(obj, schema)
        _bind_nested(obj.__target__, schema, seen)
        return

    if not isinstance(type(obj), StructureMetaType) or id(obj) in seen:
        return
    seen.add(id(obj))
    _bind(obj, schema)
    for value in obj.__dict__.values():
        if isinstance(value, list):
            _bind_array(value, schema)
            _bind_nested(value, schema, seen)
        elif isinstance(type(value), StructureMetaType):
            _bind_nested(value, schema, seen)


def _bind(obj: Any, schema: Schema) -> None:
    """Bind an object to a schema until it's garbage collected."""
    if isinstance(obj, list):
        # Lists can't be weak-referenced; entries are removed when their elements are collected.
        _arrays[id(obj)] = (obj, schema)
        return
    if isinstance(obj, (StructureMetaType, Schema)):
        return
    if id(obj) in _instances:
        return
    if not hasattr(obj, "__dict__"):
        return
    try:
        key = id(obj)
        ref = weakref.ref(obj, lambda _ref: _instances.pop(key, None))
    except TypeError:
        return
    _instances[id(obj)] = (ref, schema)


def active_schema(cs: cstruct) -> Schema | None:
    """Return the schema currently used by a cstruct instance for parses, if any."""
    return getattr(cs, "_active_schema", None) or getattr(cs, "_pinned_schema", None)


def instance_schema(obj: Any) -> Schema | None:
    """Return the schema an object was parsed with, if any."""
    if isinstance(obj, list):
        entry = _arrays.get(id(obj))
        return entry[1] if entry is not None and entry[0] is obj else None
    entry = _instances.get(id(obj))
    if entry is not None and entry[0]() is obj:
        return entry[1]
    schema = obj.__dict__.get("__schema__") if hasattr(obj, "__dict__") else None
    return schema if isinstance(schema, Schema) else None


def schema_class(obj: Any) -> type[BaseType]:
    """Return the definition class an object should use for serialization."""
    schema = instance_schema(obj)
    if schema is not None:
        cls = schema.serialization_class(type(obj))
        if cls is not None:
            return cls
    return type(obj)


def _reinstate(obj: Any, schema: Schema) -> None:
    """Return a parsed value to its snapshot classes after its live definitions changed."""
    if isinstance(obj, list):
        for value in obj:
            _reinstate(value, schema)
        return

    if isinstance(obj, UnionProxy):
        _reinstate(obj.__target__, schema)
        return

    if isinstance(obj, StructureMetaType) or isinstance(type(obj), StructureMetaType):
        cls = type(obj) if not isinstance(obj, StructureMetaType) else obj
        wherever = schema.serialization_class(cls) or schema._cache.get(cls)
        if wherever is not None and wherever is not cls:
            try:
                object.__setattr__(obj, "__class__", wherever)
            except TypeError:
                pass
        for value in getattr(obj, "__dict__", {}).values():
            _reinstate(value, schema)


def on_definition_change(cs: cstruct) -> None:
    """Handle values parsed with an outdated definition version."""
    for ref, schema in list(_instances.values()):
        obj = ref()
        if obj is None or schema.source is not cs:
            continue

        if isinstance(type(obj), StructureMetaType):
            snapshot = schema.serialization_class(type(obj)) or schema._cache.get(type(obj))
            live = type(obj)
            if snapshot is not None and live is not snapshot and not schema.structurally_equal(snapshot, live):
                _reinstate(obj, schema)


def _adopt(obj: Any, schema: Schema) -> None:
    """Expose a snapshot object graph as the live classes of its source instance.

    The definitions of a snapshot and the live instance are structurally identical at parse time. Reassigning the
    class preserves the original public type identity while the schema binding preserves the definition version for
    later serialization.
    """
    if isinstance(obj, list):
        for value in obj:
            _adopt(value, schema)
        return

    if isinstance(obj, UnionProxy):
        _bind(obj, schema)
        _adopt(obj.__target__, schema)
        return

    if isinstance(type(obj), StructureMetaType):
        snapshot = type(obj)
        live = schema.live_class(snapshot)
        if live is not None and live is not snapshot and schema.structurally_equal(snapshot, live):
            try:
                object.__setattr__(obj, "__class__", live)
                _instances.setdefault(id(obj), (weakref.ref(obj), schema))
            except TypeError:
                pass
        _bind(obj, schema)
        # Nested values were already created with the snapshot classes; returning them through the live class
        # preserves type identity without re-interpreting their fields.
        for value in obj.__dict__.values():
            _adopt(value, schema)
    elif isinstance(obj, BaseType):
        _bind(obj, schema)


class Schema:
    """An immutable snapshot of the definitions of a :class:`~dissect.cstruct.cstruct` instance.

    The snapshot owns copies of every registered structure, union, array and pointer type, plus a copy of the
    constants. Reading with a schema therefore never observes later definition changes. Objects parsed by a schema
    remember it, so their ``dumps`` output keeps using the same fields and sizes.

    Create a schema with :meth:`cstruct.snapshot`, make it the default for new parses with :meth:`cstruct.pin`, and
    force compilation with :meth:`recompile`.
    """

    def __init__(self, source: cstruct):
        self._source_ref = weakref.ref(source)
        self.revision: int = source.revision
        self.endian: Endianness = source.endian
        self.includes: list[str] = list(source.includes)
        self.consts: dict[str, int | str | bytes] = dict(source.consts)
        self.types: dict[str, type[BaseType]] = {}
        self._cache: dict[type[BaseType], type[BaseType]] = {}
        self._live: dict[type[BaseType], type[BaseType]] = {}
        self._snapshot_classes: dict[type[BaseType], type[BaseType]] = {}
        self._digest: str = definition_digest(source)

        # A snapshot owns a frozen copy of every registered type. This is the version the caller chose.
        self.pointer = self._type_for(source.pointer)
        for name, type_ in source.types.items():
            self.types[name] = self._type_for(type_)

    @property
    def digest(self) -> str:
        """A stable digest of the definitions in this snapshot."""
        return self._digest

    @property
    def source(self) -> cstruct | None:
        """The cstruct instance this snapshot was taken from, if it's still alive."""
        return self._source_ref()

    def __getitem__(self, type_: type[BaseType]) -> type[BaseType]:
        """Return the snapshot version of a live type."""
        return self._type_for(type_)

    @contextmanager
    def _parse_context(self) -> Iterator[Schema]:
        """Make this snapshot the active definition version for a parse."""
        yield self

    def add_type(self, name: str, type_: Any, replace: bool = False) -> None:
        """Ignore implicit re-registration while a declarative class is being cloned."""
        return

    def add_const(self, name: str, value: Any) -> None:
        """Registration hook used by the declarative machinery; snapshot constants are otherwise frozen."""
        self.consts[name] = value

    def del_const(self, name: str) -> None:
        """Registration hook used by the declarative machinery."""
        self.consts.pop(name, None)

    def __getattr__(self, attr: str) -> Any:
        try:
            return self.consts[attr]
        except KeyError:
            pass
        try:
            return self.types[attr]
        except KeyError:
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {attr!r}")

    def __repr__(self) -> str:
        return f"<Schema revision={self.revision} digest={self._digest[:12]}>"

    def __hash__(self) -> int:
        return hash((self._digest, self.revision))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Schema)
            and self._digest == other._digest
            and self.revision == other.revision
            and self._source_ref() is other._source_ref()
        )

    def resolve(self, name: type[BaseType] | str) -> type[BaseType]:
        """Resolve a type name in this snapshot."""
        if not isinstance(name, str):
            return name
        try:
            return self.types[name]
        except KeyError:
            raise ResolveError(f"Unknown type {name}") from None

    def read(self, name: str, stream: Any, *, endian: Endianness | None = None) -> Any:
        """Read a value using the definitions in this snapshot."""
        return self.resolve(name).read(stream, endian=endian or self.endian)

    def recompile(self) -> Schema:
        """Explicitly compile every structure and union in this snapshot.

        Compilation is otherwise lazy (on first read), like on a regular cstruct instance.
        """
        for type_ in self._cache.values():
            if isinstance(type_, StructureMetaType):
                compiler.compile_structure(type_)
        return self

    def live_class(self, snapshot: type[BaseType]) -> type[BaseType] | None:
        """Return the live definition class for a snapshot class."""
        return self._snapshot_classes.get(snapshot)

    def structurally_equal(self, snapshot: type[BaseType], live: type[BaseType]) -> bool:
        """Return whether two type classes describe the same definitions, recursively.

        Used to decide whether a parsed value can keep the live class without changing its meaning. Any structural
        edit of the live definition makes it a different version, even if the class object is the same.
        """
        return self._equivalent(snapshot, live, {}, {})

    def _equivalent(
        self,
        snapshot: type[BaseType],
        live: type[BaseType],
        left: dict[int, int],
        right: dict[int, int],
    ) -> bool:
        if snapshot is live:
            return self._snapshot_classes.get(snapshot, live) is live

        if id(snapshot) in left or id(live) in right:
            return left.get(id(snapshot)) == id(live)

        if not isinstance(snapshot, type) or not isinstance(live, type):
            return snapshot == live

        if isinstance(snapshot, EnumMetaType) or isinstance(live, EnumMetaType):
            return snapshot is live

        if issubclass(snapshot, BaseArray) or issubclass(live, BaseArray):
            if not (issubclass(snapshot, BaseArray) and issubclass(live, BaseArray)):
                return False
            left[id(snapshot)] = id(live)
            right[id(live)] = id(snapshot)
            return (
                snapshot.__num_entries__ == live.__num_entries__
                and snapshot.__null_terminated__ == live.__null_terminated__
                and self._equivalent(snapshot.__type__, live.__type__, left, right)
            )

        if issubclass(snapshot, Pointer) or issubclass(live, Pointer):
            if not (issubclass(snapshot, Pointer) and issubclass(live, Pointer)):
                return False
            left[id(snapshot)] = id(live)
            right[id(live)] = id(snapshot)
            return self._equivalent(snapshot.__type__, live.__type__, left, right)

        if isinstance(snapshot, StructureMetaType) or isinstance(live, StructureMetaType):
            if not (isinstance(snapshot, StructureMetaType) and isinstance(live, StructureMetaType)):
                return False
            if snapshot.__align__ != live.__align__ or snapshot.__anonymous__ != live.__anonymous__:
                return False
            if len(snapshot.__fields__) != len(live.__fields__):
                return False
            left[id(snapshot)] = id(live)
            right[id(live)] = id(snapshot)
            for snapshot_field, live_field in zip(snapshot.__fields__, live.__fields__, strict=False):
                if (
                    snapshot_field.name != live_field.name
                    or snapshot_field.bits != live_field.bits
                    or snapshot_field.offset != live_field.offset
                    or not self._equivalent(snapshot_field.type, live_field.type, left, right)
                ):
                    return False
            return True

        return snapshot.__name__ == live.__name__ and snapshot.__module__ == live.__module__

    def serialization_class(self, cls: type[BaseType]) -> type[BaseType] | None:
        """Return the snapshot class to use for serializing an instance of ``cls``.

        Adopted instances have the live class, but must serialize with the snapshot definition. Snapshot
        instances already have their snapshot class. Types that aren't structures (or arrays of structures) are
        returned as is.
        """
        if cls in self._cache:
            return self._cache[cls]
        if cls in self._snapshot_classes:
            return cls
        # An adopted instance has a live class, or a user subclass of it. Walk the MRO to the definition origin.
        # Prefer the snapshot class for a live base; if the instance is already a snapshot class, use its base.
        for base in cls.__mro__:
            if (snapshot := self._cache.get(base)) is not None:
                return snapshot
            if base in self._snapshot_classes:
                return base
        return None

    def _type_for(self, type_: type[BaseType]) -> type[BaseType]:
        """Return the snapshot equivalent of a type, creating it on demand."""
        if type_ in self._cache:
            return self._cache[type_]

        if isinstance(type_, EnumMetaType):
            clone = self._clone_enum(type_)
        elif isinstance(type_, StructureMetaType):
            clone = self._clone_structure(type_)
        elif issubclass(type_, BaseArray):
            clone = self._clone_array(type_)
        elif issubclass(type_, Pointer):
            clone = self._clone_pointer(type_)
        else:
            # Scalars are value classes shared with the live definitions. Cloning them would break
            # ``type(value) is cs.uint8`` compatibility without changing parse semantics.
            clone = type_
            self._cache[type_] = clone
            self._live[clone] = type_
            self._snapshot_classes[clone] = type_

        self._cache[type_] = clone
        self._live[clone] = type_
        self._snapshot_classes[clone] = type_
        return clone

    def _clone_enum(self, type_: type[BaseType]) -> type[BaseType]:
        """Reuse the enum class.

        Enum equality includes the class identity, so value compatibility requires reusing the same enum members.
        Enum values are immutable; a redefinition creates a new enum class at the cstruct instance.
        """
        self._cache[type_] = type_
        self._live[type_] = type_
        self._snapshot_classes[type_] = type_
        return type_

    def _clone_structure(self, type_: type[BaseType]) -> type[BaseType]:
        """Clone a structure or union as a lean, immutable definition snapshot.

        The clone copies field metadata, not generated initializers or default values. That keeps taking a snapshot
        cheap and side-effect free, which matters because a default parse captures one at entry.
        """
        attrs = {
            "__module__": type_.__module__,
            "__qualname__": type_.__qualname__,
            "__doc__": type_.__doc__,
            "__align__": type_.__align__,
            "__anonymous__": type_.__anonymous__,
            "__cs__": self,
            "__compiled__": False,
            "__snapshot__": True,
            "__snapshot_clone__": True,
        }
        if hasattr(type_, "__init_subclass__"):
            attrs["__init_subclass__"] = classmethod(lambda *args, **kwargs: None)
        # Bypass __init_subclass__ hooks (e.g. the declarative registration) while creating the snapshot class.
        # A declarative class inherits those through its MRO, so temporarily remove the hook from its bases.
        clone = type.__new__(type(type_), type_.__name__, type_.__bases__, attrs)
        clone.__cs__ = self
        # Register before cloning fields, so recursive structures resolve to this class.
        self._cache[type_] = clone
        self._live[clone] = type_
        self._snapshot_classes[clone] = type_

        clone.__fields__ = [
            Field(field.name, self._type_for(field.type), bits=field.bits, offset=field.offset)
            for field in type_.__fields__
        ]

        # Metadata can be copied from the source definition: field names, sizes and layout were fixed at capture.
        for name in (
            "__size__",
            "__alignment__",
            "__dynamic__",
            "__static_sizes__",
            "__bool__",
            "__eq__",
            "__hash__",
        ):
            setattr(clone, name, getattr(type_, name))

        # Default construction must use this version's nested types, but generating initializers eagerly would
        # allocate default values during a normal parse. Generate it on first use instead.
        clone.__init__ = _lazy_init(clone)

        # Rebuild the raw lookup and folded member mapping so dumpstruct and anonymous members see the snapshot
        # field objects, not the source definition's fields.
        clone.__lookup__ = {field._name: field for field in clone.__fields__}
        lookup = {}
        for field in clone.__fields__:
            if isinstance(field.type, StructureMetaType) and field.name is None:
                for anon_field in field.type.__members__.values():
                    lookup[anon_field._name] = anon_field
            else:
                lookup[field._name] = field
        clone.__members__ = lookup

        # A snapshot is a boundary; structural edits would create an unversioned third definition.
        clone.add_field = classmethod(_readonly)
        clone.commit = classmethod(_readonly)

        base = UnionMetaType if isinstance(type_, UnionMetaType) else StructureMetaType
        clone._read = classmethod(base._read)
        clone._write = classmethod(base._write)
        return clone

    def _clone_array(self, type_: type[BaseType]) -> type[BaseType]:
        attrs = {
            "__module__": type_.__module__,
            "__qualname__": type_.__qualname__,
            "__type__": None,
            "__num_entries__": type_.__num_entries__,
            "__null_terminated__": type_.__null_terminated__,
            "__cs__": self,
        }
        clone = type.__new__(type(type_), type_.__name__, type_.__bases__, attrs)
        clone.__cs__ = self
        self._cache[type_] = clone
        self._live[clone] = type_
        self._snapshot_classes[clone] = type_

        element = self._type_for(type_.__type__)
        clone.__type__ = element
        clone.__num_entries__ = type_.__num_entries__
        clone.__null_terminated__ = type_.__null_terminated__
        # Use the source array's exact size: this preserves nested arrays whose element size depends on an
        # implicit dimension, instead of recomputing a different layout.
        clone.__size__ = type_.__size__
        clone.__dynamic__ = type_.__dynamic__
        clone.__alignment__ = type_.__alignment__
        return clone

    def _clone_pointer(self, type_: type[BaseType]) -> type[BaseType]:
        attrs = {
            "__module__": type_.__module__,
            "__qualname__": type_.__qualname__,
            "__type__": None,
            "__cs__": self,
        }
        clone = type.__new__(type(type_), type_.__name__, type_.__bases__, attrs)
        clone.__cs__ = self
        self._cache[type_] = clone
        self._live[clone] = type_
        self._snapshot_classes[clone] = type_
        clone.__type__ = self._type_for(type_.__type__)
        clone.__size__ = type_.__size__
        clone.__dynamic__ = type_.__dynamic__
        clone.__alignment__ = type_.__alignment__
        return clone

    def _make_array(self, type_: type[BaseType], num_entries: int | Expression | None) -> type[BaseType]:
        """Create an array type in the context of this snapshot."""
        attrs = {"__type__": type_, "__num_entries__": num_entries, "__null_terminated__": num_entries is None}
        if num_entries is None or isinstance(num_entries, Expression) or type_.__dynamic__:
            attrs.update({"__size__": None, "__dynamic__": True, "__alignment__": type_.__alignment__})
        else:
            attrs.update(
                {
                    "__size__": num_entries * type_.__size__,
                    "__dynamic__": False,
                    "__alignment__": type_.__alignment__,
                }
            )
        name = f"{type_.__name__}[]" if num_entries is None else f"{type_.__name__}[{num_entries}]"
        array_type = MetaType.__ArrayType__
        clone = type.__new__(type(array_type), name, (array_type,), attrs)
        clone.__cs__ = self
        return clone

    def _make_pointer(self, target: type[BaseType]) -> type[BaseType]:
        """Create a pointer type in the context of this snapshot."""
        clone = type.__new__(
            Pointer,
            f"{target.__name__}*",
            (Pointer,),
            {
                "__type__": target,
                "__cs__": self,
                "__size__": self.pointer.__size__,
                "__dynamic__": self.pointer.__dynamic__,
                "__alignment__": self.pointer.__alignment__,
            },
        )
        clone.__cs__ = self
        return clone


def snapshot(cs: cstruct) -> Schema:
    """Create a snapshot of the definitions in ``cs``."""
    return Schema(cs)
