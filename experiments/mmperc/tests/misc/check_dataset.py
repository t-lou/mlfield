from pathlib import Path

from components.dataset.a2d2_dataset import A2D2Dataset

path_tar = Path("/repo/data/camera_lidar_semantic_bboxes.tar")

if __name__ == "__main__":
    dataset = A2D2Dataset(path_tar)
    print(f"The size of the dataset is: {len(dataset)}")

    num_demo_samples = 20
    for i in range(num_demo_samples):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Points shape: {sample['points'].shape}")
        print(f"  Camera shape: {sample['camera'].shape}")
        print(f"  Semantics shape: {sample['semantics'].shape}")
        print(f"  GT boxes shape: {sample['gt_boxes'].shape}")
