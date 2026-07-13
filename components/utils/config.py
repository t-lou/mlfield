from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from typing import Type, TypeVar

import yaml

from components.utils.logger import logger

T = TypeVar("T")


def _enum_to_value(d):
    r = {}
    for k, v in d.items():
        if isinstance(v, Enum):
            r[k] = v.value
        else:
            r[k] = v
    return r


def dump_yaml(obj: T, path: Path) -> None:
    raw_dict = asdict(obj)

    raw_dict = _enum_to_value(raw_dict)

    ordered_dict = {f.name: raw_dict[f.name] for f in fields(obj)}

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ordered_dict, f, sort_keys=False)


def create_default(cls: Type[T], path: Path) -> None:
    obj = cls(**{f.name: f.default if f.default is not None else None for f in fields(cls)})
    dump_yaml(obj, path)


def load_yaml(path: Path, cls: Type[T]) -> T:
    if not path.exists():
        logger.info(f"Cannot find config {path}, with create with default values.")
        create_default(cls, path)

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}

    return cls(**filtered)
