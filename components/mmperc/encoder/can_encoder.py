import torch
from torch import Tensor, nn


class CANEncoder(nn.Module):
    """
    Ego-state encoder for CAN signals. Deliberately excludes:
      - lat/long (absolute position → route/location memorization risk)
      - action-adjacent signals (throttle/brake/steer, if present) — see
        earlier discussion; kept out of the encoder entirely for now.

    Only measured ego-motion state goes in: this is a fourth modality
    alongside lidar/camera, not a privileged action-conditioning signal.
    """

    DEFAULT_KEYS = (
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "angular_velocity_omega_x",
        "angular_velocity_omega_y",
        "angular_velocity_omega_z",
        "pitch_angle",
        "roll_angle",
        "vehicle_speed",
        # distance_pulse_* intentionally omitted until confirmed cumulative-vs-rate
    )

    def __init__(self, dim, input_keys=DEFAULT_KEYS, field_mean=None, field_std=None):
        super().__init__()
        self.input_keys = list(input_keys)
        n = len(self.input_keys)

        # Per-field normalization stats — static data, computed once from the
        # training set, cached as buffers (not learned, but must move with .to(device)).
        mean = torch.zeros(n) if field_mean is None else torch.as_tensor(field_mean, dtype=torch.float32)
        std = torch.ones(n) if field_std is None else torch.as_tensor(field_std, dtype=torch.float32)
        self.register_buffer("field_mean", mean)
        self.register_buffer("field_std", std)

        self.mlp = nn.Sequential(
            nn.Linear(n, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, can_dict: dict[str, Tensor]) -> Tensor:
        missing = [k for k in self.input_keys if k not in can_dict]
        if missing:
            raise ValueError(f"CANEncoder missing required fields: {missing}")

        x = torch.stack([can_dict[k] for k in self.input_keys], dim=1)  # (B, N)
        x = (x - self.field_mean) / self.field_std.clamp_min(1e-6)

        token = self.mlp(x).unsqueeze(1)  # (B, 1, dim)
        return token
