import tarfile
from pathlib import Path

from components.definitions.mmperc_params import MmpercParams
from components.utils.logger import logger
from torch.utils.data import Dataset


class A2D2DatasetUnlabeled(Dataset):
    """ """

    def __init__(self, params: MmpercParams, recording_time: str = "20190401145936"):
        self.params = params
        self.path_data = Path(params.path_data)
        assert self.path_data.is_dir()
        self.path_calib = Path(params.path_calibration)
        assert self.path_calib.exists()
        self._tars = {}
        self._calibs = {}  # Currently in encoder, here more for debugging.
        self.recording_time = recording_time

        sensor_positions = [
            "front_center",
            "front_right",
            "front_left",
            "side_right",
            "side_left",
            "rear_center",
        ]
        sensor_types = ["camera", "lidar"]

        for sensor_type in sensor_types:
            for sensor_position in sensor_positions:
                pos_in_name = sensor_position.replace("_", "")
                filename = f"camera_lidar-{self.recording_time}_{sensor_type}_{pos_in_name}.tar"
                path = self.path_data / filename
                if path.exists():
                    logger.info(f"Path {path} found")
                    self._tars[(sensor_type, sensor_position)] = tarfile.open(path, mode="r")
                else:
                    logger.warning(f"Path {path} not found")

        assert self._tars

        # If any tar is found, extract all the sequence id of the files
        self.sequence_ids = []
        first_tar = next(iter(self._tars.values()))
        for member in first_tar.getmembers():
            if member.isfile():
                # Extract the sequence id from the filename
                sequence_id = Path(member.name).stem.split("_")[-1]
                self.sequence_ids.append(sequence_id)
        self.sequence_ids.sort()
        logger.info(f"Found {len(self.sequence_ids)} files.")

    def __del__(self) -> None:
        """Close all opened tars."""
        for tar in self._tars.values():
            try:
                if tar is not None:
                    tar.close()
            except Exception:
                pass

    def get_with_index(self, index: int) -> dict:
        assert 0 <= index < len(self.sequence_ids)
        sequence_id = self.sequence_ids[index]

        recording_date = self.recording_time[:8]
        recording_clock = self.recording_time[8:]

        result = {}

        for (sensor_type, sensor_position), tar in self._tars.items():
            path_dir = f"./camera_lidar/{recording_date}_{recording_clock}/{sensor_type}/cam_{sensor_position}"
            pos_in_name = sensor_position.replace("_", "")
            ext = "npz" if sensor_type == "lidar" else "png"
            path_name = f"{self.recording_time}_{sensor_type}_{pos_in_name}_{sequence_id}.{ext}"
            logger.info(f"loading {path_dir}/{path_name} for {sensor_type} {sensor_position}")
            fileobj = tar.extractfile(f"{path_dir}/{path_name}")
            result[(sensor_type, sensor_position)] = fileobj

        return result
