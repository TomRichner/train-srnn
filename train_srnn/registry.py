"""Minimal name -> class registry used for tasks and cells."""
from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj
        return deco

    def __getitem__(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} {name!r}. "
                           f"Available: {sorted(self._items)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def names(self) -> list[str]:
        return sorted(self._items)
