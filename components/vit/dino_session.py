from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from components.utils.device import get_device
from components.vit.dino_defs import MODEL_BASE_RES
from components.vit.dino_model import DINOModel


class DINOLoss(nn.Module):
    """DINO loss function for self-supervised learning, comparing student and teacher outputs across multiple crops."""

    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
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

    def forward(self, student_outputs, teacher_outputs, global_crop_indices=[0, 1]):
        """Compute the DINO loss between student and teacher outputs.

        Args:
            student_outputs: List of tensors from the student model for each crop.
            teacher_outputs: List of tensors from the teacher model for each global crop.
            global_crop_indices: Indices of the global crops in the student outputs.
        """
        student_out = [s.float() / self.student_temp for s in student_outputs]
        teacher_out = [(t.float() - self.center.float()) / self.teacher_temp for t in teacher_outputs]

        student_probs = [F.log_softmax(s, dim=-1) for s in student_out]
        teacher_probs = [F.softmax(t, dim=-1) for t in teacher_out]

        total_loss = 0
        n_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            # Get the corresponding global crop index for the teacher output
            t_view = global_crop_indices[t_idx]

            # Compute the cross-entropy loss between the teacher and student outputs for all crops except the
            # corresponding global crop
            for s_idx, s_prob in enumerate(student_probs):
                if s_idx == t_view:
                    # Skip the student output corresponding to the same global crop as the teacher output
                    continue

                # Compute the cross-entropy loss between the teacher and student outputs
                total_loss += torch.sum(-t_prob * s_prob, dim=-1).mean()
                n_terms += 1

        total_loss /= n_terms

        # Update the center of the teacher outputs using momentum
        batch_center = torch.cat([t.float() for t in teacher_outputs]).mean(dim=0, keepdim=True)
        # Update the center with momentum to stabilize training
        self.center.mul_(self.center_momentum).add_(batch_center.to(self.center.dtype), alpha=1 - self.center_momentum)

        return total_loss


class DINOSession(nn.Module):
    def __init__(
        self,
        out_dim=65536,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
        device=None,
    ):
        """Initialize a DINO training session with student and teacher models, loss function, and parameters."""
        super().__init__()
        self.device = device if device is not None else get_device()
        self.loss_fn = DINOLoss(
            out_dim=out_dim, teacher_temp=teacher_temp, student_temp=student_temp, center_momentum=center_momentum
        ).to(self.device)

        # Create student and teacher models with the same encoder configuration.
        # The crop size differs at runtime via the transforms, but the positional
        # embedding buffer must stay identical so teacher initialization can copy
        # the student weights and EMA updates remain compatible.
        self.student = DINOModel(base_res=MODEL_BASE_RES, out_dim=out_dim).to(self.device)
        self.teacher = DINOModel(base_res=MODEL_BASE_RES, out_dim=out_dim).to(self.device)

        # Initialize teacher with student weights and freeze teacher parameters.
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def forward(self, student_inputs, teacher_inputs):
        """Compute the DINO loss for a batch of student and teacher inputs."""
        # Compute student and teacher outputs
        student_outputs = [self.student(x) for x in student_inputs]
        teacher_outputs = [self.teacher(x) for x in teacher_inputs]
        # Compute the DINO loss
        loss = self.loss_fn(student_outputs, teacher_outputs)
        return loss

    def update_teacher(self, momentum=0.996):
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                # Update teacher parameters with momentum using detached student parameters.
                pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)

    def save(self, path_ckpt: Path):
        torch.save(
            {
                "student": {k: v.cpu() for k, v in self.student.state_dict().items()},
                "teacher": {k: v.cpu() for k, v in self.teacher.state_dict().items()},
                "loss_fn": {k: v.cpu() for k, v in self.loss_fn.state_dict().items()},
            },
            str(path_ckpt),
        )

    def load(self, path_ckpt: Path):
        ckpt = torch.load(str(path_ckpt), map_location=self.device)

        self.student.load_state_dict(ckpt["student"])
        self.teacher.load_state_dict(ckpt["teacher"])
        self.loss_fn.load_state_dict(ckpt["loss_fn"])
