import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from components.utils.device import get_device
from components.vit.dino_defs import DINOConfig
from components.vit.dino_model import DINOModel, ViTBackbone


class DINOLoss(nn.Module):
    """DINO loss function for self-supervised learning, comparing student and teacher outputs across multiple crops."""

    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum):
        """Initialize the DINO loss with temperature parameters and center momentum.

        Args:
            out_dim: Output dimension of the DINO head.
            teacher_temp: Temperature for the teacher outputs.
            student_temp: Temperature for the student outputs.
            center_momentum: Momentum for updating the center of teacher outputs.
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs, global_crop_indices=None):
        """Compute the DINO loss between student and teacher outputs.

        Args:
            student_outputs: List of tensors from the student model for each crop.
            teacher_outputs: List of tensors from the teacher model for each global crop.
            global_crop_indices: Indices of the global crops in the student outputs
                (default: the first ``len(teacher_outputs)`` student crops).
        """
        if global_crop_indices is None:
            global_crop_indices = list(range(len(teacher_outputs)))
        if len(global_crop_indices) != len(teacher_outputs):
            raise ValueError("global_crop_indices must have one entry per teacher output")

        num_teachers, num_students = len(teacher_outputs), len(student_outputs)

        # Every (teacher crop, student crop) pair counts, except a teacher crop paired with the
        # student view of the very same global crop.
        pair_mask = torch.ones(num_teachers, num_students)
        for t_idx, s_idx in enumerate(global_crop_indices):
            pair_mask[t_idx, s_idx] = 0.0
        num_terms = int(pair_mask.sum().item())  # CPU tensor: no device sync

        # The loss needs fp32 probabilities (teacher probs are tiny); keep it out of autocast.
        with torch.autocast(device_type=student_outputs[0].device.type, enabled=False):
            student_logp = torch.stack([F.log_softmax(s.float() / self.student_temp, dim=-1) for s in student_outputs])
            teacher_prob = torch.stack(
                [F.softmax((t.float() - self.center.float()) / self.teacher_temp, dim=-1) for t in teacher_outputs]
            )

            # Cross-entropy of all teacher/student crop pairs at once, mean over the batch: (T, S)
            cross_entropy = -torch.einsum("tbd,sbd->ts", teacher_prob, student_logp) / student_logp.shape[1]
            total_loss = (cross_entropy * pair_mask.to(cross_entropy.device)).sum() / num_terms

            # Update the center of the teacher outputs using momentum
            batch_center = torch.cat([t.float() for t in teacher_outputs]).mean(dim=0, keepdim=True)
            # Update the center with momentum to stabilize training
            self.center.mul_(self.center_momentum).add_(
                batch_center.to(self.center.dtype), alpha=1 - self.center_momentum
            )

        return total_loss


class DINOSession(nn.Module):
    def __init__(
        self,
        config: DINOConfig,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
        device=None,
    ):
        """Initialize a DINO training session with student and teacher models, loss function, and parameters."""
        super().__init__()
        self.device = device if device is not None else get_device()
        self.loss_fn = DINOLoss(
            out_dim=config.out_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum,
        ).to(self.device)

        # The teacher is an exact copy of the student (same encoder configuration and positional
        # embedding buffer, so EMA updates stay compatible); the crop size differs only at runtime.
        self.student = self.create_student(config, self.device)
        self.teacher = copy.deepcopy(self.student)

        # Freeze teacher parameters.
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    @staticmethod
    def create_student(config: DINOConfig, device: torch.device) -> DINOModel:
        return DINOModel(config).to(device)

    @staticmethod
    def create_teacher(config: DINOConfig, device: torch.device) -> DINOModel:
        return DINOModel(config).to(device)

    def forward(self, student_inputs, teacher_inputs):
        """Compute the DINO loss for a batch of student and teacher inputs."""
        # Compute student and teacher outputs (crops of equal size share one forward pass)
        student_outputs = self.student.forward_crops(list(student_inputs))
        teacher_outputs = self.teacher.forward_crops(list(teacher_inputs))
        # Compute the DINO loss
        loss = self.loss_fn(student_outputs, teacher_outputs)
        return loss

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        # fused multi-tensor EMA update using detached student parameters
        teacher_params = list(self.teacher.parameters())
        student_params = [p.detach() for p in self.student.parameters()]
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)

    @staticmethod
    def build_encoder(config: DINOConfig, path_ckpt: Path, device: torch.device | None = None) -> ViTBackbone:
        device = device if device is not None else get_device()
        teacher_part = DINOModel(config).to(device)
        ckpt = torch.load(str(path_ckpt), map_location=device)
        teacher_part.load_state_dict(ckpt["teacher"])
        teacher_part.eval()
        return teacher_part.backbone

    def save(self, path_ckpt: Path, extra: dict | None = None):
        """Save student, teacher and loss state; ``extra`` entries (e.g. optimizer) are stored alongside."""
        state = {
            "student": {k: v.cpu() for k, v in self.student.state_dict().items()},
            "teacher": {k: v.cpu() for k, v in self.teacher.state_dict().items()},
            "loss_fn": {k: v.cpu() for k, v in self.loss_fn.state_dict().items()},
        }
        if extra:
            state.update(extra)
        torch.save(state, str(path_ckpt))

    def load(self, path_ckpt: Path) -> dict:
        """Load student, teacher and loss state; returns the checkpoint dict (for extra entries)."""
        ckpt = torch.load(str(path_ckpt), map_location=self.device)

        self.student.load_state_dict(ckpt["student"])
        self.teacher.load_state_dict(ckpt["teacher"])
        self.loss_fn.load_state_dict(ckpt["loss_fn"])
        return ckpt
