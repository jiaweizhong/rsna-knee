from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise KeyError(f"Duplicate {self.kind} registration: {name}")
            self._items[name] = factory
            return factory

        return decorator

    def build(self, spec: Mapping[str, Any], **injected: Any) -> T:
        if "name" not in spec:
            raise KeyError(f"{self.kind} config requires a 'name'")
        name = str(spec["name"])
        if name not in self._items:
            choices = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.kind} {name!r}. Available: {choices}")
        params = dict(spec.get("params", {}))
        params.update(injected)
        return self._items[name](**params)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

