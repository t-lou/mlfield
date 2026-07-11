from pathlib import Path

import torch

from components.vit.dino_defs import DINOConfig
from components.vit.dino_session import DINOSession


def load_student_from_checkpoint(config: DINOConfig, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load student via DINOSession checkpoint as requested."""
    session = DINOSession(config, device=device)
    session.load(ckpt_path)
    student = session.student.to(device)
    student.eval()
    return student
