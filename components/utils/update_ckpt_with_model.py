"""Update a checkpoint with a new model architecture."""

import json
import os

import torch
import torch.nn.functional as F


def interpolate_tensor(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Interpolate src tensor into target_shape.

    Supports Linear, Conv2d, and generic tensors.
    """
    if src.shape == target_shape:
        return src

    # Fully-connected layers: (out, in)
    if len(src.shape) == 2 and len(target_shape) == 2:
        return (
            F.interpolate(src.unsqueeze(0).unsqueeze(0), size=target_shape, mode="bilinear", align_corners=False)
            .squeeze(0)
            .squeeze(0)
        )

    # Conv or patch embedding: (C_out, C_in, H, W)
    if len(src.shape) == 4 and len(target_shape) == 4:
        return F.interpolate(src, size=target_shape[2:], mode="bicubic", align_corners=False)

    # Generic fallback: reshape with nearest interpolation
    return F.interpolate(src.unsqueeze(0), size=target_shape, mode="nearest").squeeze(0)


def initialize_model_from_checkpoint(model: torch.nn.Module, ckpt_path: str, output_path: str) -> None:
    """Initialize a model from a checkpoint, updating the model architecture as needed."""

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state = ckpt["model"] if "model" in ckpt else ckpt

    model_state = model.state_dict()

    report = {
        "loaded_direct": [],
        "interpolated": [],
        "random_initialized": [],
        "unused_checkpoint_keys": [],
        "warnings": [],
    }

    # Track checkpoint keys
    used_ckpt_keys = set()

    new_state = {}

    for key, target_tensor in model_state.items():
        if key in ckpt_state:
            src_tensor = ckpt_state[key]
            used_ckpt_keys.add(key)

            if src_tensor.shape == target_tensor.shape:
                # Case 1: direct load
                new_state[key] = src_tensor
                report["loaded_direct"].append(key)

            else:
                # Case 2: interpolate
                try:
                    new_state[key] = interpolate_tensor(src_tensor, target_tensor.shape)
                    report["interpolated"].append(
                        {"key": key, "src_shape": list(src_tensor.shape), "target_shape": list(target_tensor.shape)}
                    )
                except Exception as e:
                    report["warnings"].append(f"Interpolation failed for {key}: {str(e)}")
                    new_state[key] = torch.randn_like(target_tensor)
                    report["random_initialized"].append(key)

        else:
            # Case 3: missing in checkpoint → random init
            new_state[key] = torch.randn_like(target_tensor)
            report["random_initialized"].append(key)

    # Case 4: checkpoint keys not used
    for key in ckpt_state.keys():
        if key not in used_ckpt_keys:
            report["unused_checkpoint_keys"].append(key)
            report["warnings"].append(f"Checkpoint key unused: {key}")

    # Load new state
    model.load_state_dict(new_state)

    # Save CPU checkpoint
    torch.save({"model": model.state_dict()}, output_path)

    # Save JSON report
    json_path = os.path.splitext(output_path)[0] + "_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)


########################################################################
## Example usage in model:                                            ##
########################################################################
# def partial_initialization(self, ckpt_path: str, output_path: str):
#     initialize_model_from_checkpoint(self, ckpt_path, output_path)
#


if __name__ == "__main__":
    import argparse

    from model import MyModel  # Replace with your actual model import

    parser = argparse.ArgumentParser(description="Update a checkpoint with a new model architecture.")
    parser.add_argument("--ckpt_in_path", type=str, required=True, help="Path to the input checkpoint.")
    parser.add_argument("--ckpt_out_path", type=str, required=True, help="Path to save the updated checkpoint.")
    args = parser.parse_args()

    # Initialize your model (replace MyModel with your actual model class)
    model = MyModel()

    assert args.ckpt_in_path != args.ckpt_out_path

    report = initialize_model_from_checkpoint(model, args.ckpt_in_path, args.ckpt_out_path)
    print("Update completed. Report saved.")
