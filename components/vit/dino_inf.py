from enum import Enum
from pathlib import Path

import torch

from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession


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

    if part_id == DINOPartID.TEACHER:
        part = DINOSession.create_teacher(config, device)
    elif part_id == DINOPartID.STUDENT:
        part = DINOSession.create_student(config, device)
    else:
        raise ValueError(f"Unsupported DINO part {part_name}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    part.load_state_dict(ckpt[part_name.lower()])
    part.eval()
    return part
