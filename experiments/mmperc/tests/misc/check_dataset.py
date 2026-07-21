from pathlib import Path

from components.dataset.a2d2_dataset import A2D2Dataset, Split
from components.definitions.mmperc import MmpercParams

path_tar = Path("/repo/data/camera_lidar_semantic_bboxes.tar")

if __name__ == "__main__":
    split = Split.FULL
    params = MmpercParams()
    dataset = A2D2Dataset(path_tar, params, split)
    print(f"The size of the dataset is: {len(dataset)}")

    num_demo_samples = max(20, len(dataset))
    for i in range(num_demo_samples):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Points shape: {sample['points'].shape}")
        print(f"  Camera shape: {sample['camera'].shape}")
        print(f"  Semantics shape: {sample['semantics'].shape}")
        print(f"  GT boxes shape: {sample['gt_boxes'].shape}")
