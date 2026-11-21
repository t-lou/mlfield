from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    path: str | Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
) -> Path:
    """Export tensor inputs/outputs with a dynamic batch axis unless axes are specified."""
    if not sample_inputs or len(sample_inputs) != len(input_names):
        raise ValueError("Provide one input name for every sample input tensor")
    if not output_names:
        raise ValueError("At least one ONNX output name is required")

    try:
        import onnx
    except ImportError as err:
        raise ImportError("ONNX export requires the 'onnx' package; install it in the active environment") from err

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    export_dynamic_axes = (
        {name: {0: "batch"} for name in (*input_names, *output_names)} if dynamic_axes is None else dict(dynamic_axes)
    )
    module_modes = tuple((module, module.training) for module in model.modules())
    mha_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                sample_inputs,
                str(temporary_path),
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=export_dynamic_axes or None,
                opset_version=17,
                dynamo=False,
                external_data=False,
            )
        onnx.checker.check_model(onnx.load(str(temporary_path)))
        temporary_path.replace(output_path)
    finally:
        torch.backends.mha.set_fastpath_enabled(mha_fastpath_enabled)
        for module, was_training in module_modes:
            module.training = was_training
        temporary_path.unlink(missing_ok=True)
    return output_path
