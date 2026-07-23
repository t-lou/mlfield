from dataclasses import MISSING, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Type, TypeVar, get_type_hints

import yaml

from components.utils.logger import logger

T = TypeVar("T")


def _enum_to_value(d):
    """Convert Enum values in a dictionary to their underlying values."""
    r = {}
    for k, v in d.items():
        if isinstance(v, Enum):
            r[k] = v.value
        else:
            r[k] = v
    return r


def dump_yaml(obj: T, path: Path) -> None:
    """Dump a dataclass object to a YAML file, preserving field order and converting Enums to their values."""
    raw_dict = asdict(obj)

    raw_dict = _enum_to_value(raw_dict)

    ordered_dict = {f.name: raw_dict[f.name] for f in fields(obj)}

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ordered_dict, f, sort_keys=False)


def create_default(cls: Type[T], path: Path) -> None:
    """Create a default instance of a dataclass and dump it to a YAML file."""
    defaults = {}
    for f in fields(cls):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:
            defaults[f.name] = f.default_factory()
        else:
            defaults[f.name] = None

    obj = cls(**defaults)
    dump_yaml(obj, path)


def _get_field_types(cls: Type[T]) -> dict[str, object]:
    try:
        return get_type_hints(cls)
    except Exception:
        return {f.name: f.type for f in fields(cls)}


def load_yaml(path: Path, cls: Type[T]) -> T:
    """Load a YAML file into a dataclass object, ignoring unknown fields."""
    if not path.exists():
        logger.info(f"Cannot find config {path}, with create with default values.")
        create_default(cls, path)

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    allowed = _get_field_types(cls)
    filtered = {}
    for k, v in data.items():
        if k in allowed:
            field_type = allowed[k]
            # Check if the field type is a dataclass and v is a dict
            if v is not None and hasattr(field_type, "__dataclass_fields__") and isinstance(v, dict):
                filtered[k] = load_yaml_recursive(v, field_type)
            else:
                filtered[k] = v

    return cls(**filtered)


def load_yaml_recursive(data: dict, cls: Type[T]) -> T:
    """Recursively convert a dict to a dataclass, handling nested dataclasses."""
    allowed = _get_field_types(cls)
    filtered = {}
    for k, v in data.items():
        if k in allowed:
            field_type = allowed[k]
            # Check if the field type is a dataclass and v is a dict
            if v is not None and hasattr(field_type, "__dataclass_fields__") and isinstance(v, dict):
                filtered[k] = load_yaml_recursive(v, field_type)
            else:
                filtered[k] = v

    return cls(**filtered)
