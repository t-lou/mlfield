from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from components.mmperc.encoder.point_transformer_v3 import PointTransformerV3Config
from experiments.ssl.export_sonata import _load_checkpoint_model
from experiments.ssl.sonata import (
    Sonata,
    SonataConfig,
    SonataTrainConfig,
    SonataTrainer,
    export_sonata_encoder_pt,
    export_sonata_full_pt,
)


def test_training_checkpoint_exports_encoder_and_full_teacher_weights():
    config = SonataConfig(
        backbone=PointTransformerV3Config(
            in_channels=2,
            order=("z",),
            enc_depths=(1, 1),
            enc_channels=(4, 8),
            enc_num_head=(1, 2),
            enc_patch_size=(8, 8),
            stride=(2,),
            drop_path=0.0,
            serial_depth=8,
            stem_kernel=3,
            enc_mode=True,
        ),
        input_grid=None,
        up_cast_level=1,
        num_prototypes=4,
        head_hidden=8,
        head_bottleneck=4,
    )
    model = Sonata(config)

    with TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "step_0000001.pt"
        trainer = SonataTrainer(model, [], SonataTrainConfig(max_steps=1), device="cpu")
        trainer.save(str(checkpoint_path))

        restored, channels = _load_checkpoint_model(
            checkpoint_path,
            Path(temp_dir) / "unused-config.yml",
            torch.device("cpu"),
        )
        encoder_path = Path(temp_dir) / "encoder.pt"
        full_path = Path(temp_dir) / "full.pt"
        export_sonata_encoder_pt(restored, encoder_path)
        export_sonata_full_pt(restored, full_path)

        encoder = torch.load(encoder_path, map_location="cpu", weights_only=True)
        full = torch.load(full_path, map_location="cpu", weights_only=True)

    assert channels == 2
    assert "mask_head.prototypes" in full
    assert "unmask_head.prototypes" in full
    for key, value in encoder.items():
        torch.testing.assert_close(full[f"backbone.{key}"], value)
