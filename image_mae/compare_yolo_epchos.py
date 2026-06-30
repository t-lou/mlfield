"""
For the same model:

    python -m image_mae.yolo_inference -i input.jpg -o output.png --conf-threshold 0.4 --iou-threshold 0.5

we need to compare the results of different epochs, so we can check the results of different epochs.
"""

import argparse
import os

from yolo_inference import inference_and_draw

base_ckpt_paths = [
    "yolo_checkpoints/nodistill/",
    "yolo_checkpoints/mae_distilled/",
]

epoches = [10, 20, 30, 40, 50]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-s Inference")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input image path")
    parser.add_argument("-o", "--output-dir", required=True, type=str, help="Output image path")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for base_ckpt_path in base_ckpt_paths:
        for epoch in epoches:
            ckpt_path = f"{base_ckpt_path}/epoch_{epoch}.pth"

            if not os.path.exists(ckpt_path):
                print(f"Checkpoint {ckpt_path} does not exist. Skipping.")
                continue

            inference_and_draw(
                checkpoint_path=ckpt_path,
                image_path=args.input,
                output_path=f"{args.output_dir}/output_{base_ckpt_path.split('/')[-2]}_epoch_{epoch}.png",
                conf_threshold=0.4,
                iou_threshold=0.5,
            )
