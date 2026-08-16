import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from components.dataset.a2d2_dataset_unlabeled import A2D2DatasetUnlabeled
from components.definitions.mmperc_params import MmpercParams
from components.utils.calibration import load_sensor_calibration
from components.utils.config import load_yaml
from components.utils.logger import configure_logger
from PIL import Image

if __name__ == "__main__":
    configure_logger("mmperc_check")

    parser = argparse.ArgumentParser(description="MMPERC unlabeled dataset checking")
    parser.add_argument(
        "--path-config",
        type=Path,
        default="./experiments/mmperc/mmperc_config.yaml",
        help="Path to MMPERC config YAML",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of datato show.",
    )

    args = parser.parse_args()

    cfg = load_yaml(Path(args.path_config), MmpercParams)
    dataset = A2D2DatasetUnlabeled(cfg, recording_time="20190401145936")

    data = dataset.get_with_index(args.index)
    data_per_sensor = {}
    for (sensor_type, sensor_position), fileobj in data.items():
        if sensor_type not in data_per_sensor:
            data_per_sensor[sensor_type] = {}
        data_per_sensor[sensor_type][sensor_position] = fileobj

    plt.figure(1)
    line_offset = 0
    if "lidar" in data_per_sensor:
        for i, (sensor_position, fileobj) in enumerate(data_per_sensor["lidar"].items()):
            with fileobj:
                lidar_data = np.load(fileobj)
                key = "points" if "points" in lidar_data else "pcloud_points"
                points = lidar_data[key]
                plt.subplot(2, len(data_per_sensor["lidar"]), i + 1)
                plt.plot(points[:, 0], points[:, 1], ".")
                plt.title(sensor_position)

                plt.figure(2)
                vehicle_T = load_sensor_calibration(
                    Path(cfg.path_calibration),
                    sensor_name=sensor_position,
                    sensor_type="camera",
                ).pose.vehicle_from_sensor
                xyz_vehicle = points @ vehicle_T[:3, :3].T
                xyz_vehicle += vehicle_T[:3, -1]
                plt.plot(xyz_vehicle[:, 0], xyz_vehicle[:, 1], ".", alpha=0.3, label=sensor_position)
                plt.figure(1)  # switch back

        line_offset += 1

    if "camera" in data_per_sensor:
        for i, (sensor_position, fileobj) in enumerate(data_per_sensor["camera"].items()):
            with fileobj:
                img = Image.open(fileobj).convert("RGB")
                plt.subplot(2, len(data_per_sensor["camera"]), i + 1 + line_offset * len(data_per_sensor["camera"]))
                plt.imshow(img)
                plt.title(sensor_position)

plt.tight_layout()

plt.figure(2)
plt.tight_layout()
plt.legend()

plt.show()
