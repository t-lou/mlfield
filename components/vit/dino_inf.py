from pathlib import Path

import torch

from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession

from enum import Enum


class DINOPartID(Enum):
    TEACHER = 0
    STUDENT = 1


def _to_part_id(value: str) -> DINOPartID:
    try:
        return DINOPartID[value.upper()]
    except KeyError:
        raise ValueError(f"{value!r} is not a valid DINOPartID")


def load_from_checkpoint(config: DINOConfig, ckpt_path: Path, device: torch.device, part_name: str) -> torch.nn.Module:
    """Load either student or teacher via DINOSession checkpoint as requested."""
    part_id = _to_part_id(part_name)

    session = DINOSession(config, device=device)
    session.load(ckpt_path)

    if part_id == DINOPartID.TEACHER:
        part = session.teacher.to(device)
    elif part_id == DINOPartID.STUDENT:
        part = session.student.to(device)
    else:
        raise ValueError(f"Unsupported DINO part {part_name}")
    part.eval()
    return part
