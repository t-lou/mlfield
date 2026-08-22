import json
import tarfile
from enum import Enum
from pathlib import Path

from components.definitions.mmperc_params import MmpercParams
from components.utils.logger import logger
from torch.utils.data import Dataset


class Mode(Enum):
    TRAIN = "train"
    REFINE = "refine"


def _reformat_recording_time(recording_time: str) -> str:
    if "_" in recording_time:
        assert len(recording_time) == 15
        return recording_time.replace("_", "")
    else:
        assert len(recording_time) == 14

        recording_date = recording_time[:8]
        recording_clock = recording_time[8:]

        return f"{recording_date}_{recording_clock}"


class A2D2DatasetUnlabeled(Dataset):
    """
    The Dataset for A2D2 without labeling (BBox and Semantics).

    This dataset will load all available sensor frames, and output the frames which are already divided.
    The lidars should be synchronized with the corresponding cameras, but the cameras are not synchronized.
    The timestamp difference is ignored for now.

    In traning mode the CAN data containing the vehicle status are read and interpolated with any camera frame,
    and in refinement mode the actuator status is also included. TODO
    """

    def __init__(self, params: MmpercParams, recording_time: str = "20190401145936", mode: Mode = Mode.TRAIN):
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
                # Extract the sequence id from the filename with extension npz or png
                path_member = Path(member.name)
                if path_member.suffix in (".npz", ".png"):
                    sequence_id = path_member.stem.split("_")[-1]
                    self.sequence_ids.append(sequence_id)
        self.sequence_ids.sort()
        logger.info(f"Found {len(self.sequence_ids)} files.")

        # Read CAN and interpolate
        self.signal_in_names = [
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "angular_velocity_omega_x",
            "angular_velocity_omega_y",
            "angular_velocity_omega_z",
            "distance_pulse_front_left",
            "distance_pulse_front_right",
            "distance_pulse_rear_left",
            "distance_pulse_rear_right",
            "latitude_degree",
            "latitude_direction",
            "longitude_degree",
            "longitude_direction",
            "pitch_angle",
            "roll_angle",
            "vehicle_speed",
        ]
        self.signal_out_names = [
            "accelerator_pedal",
            "accelerator_pedal_gradient_sign",
            "brake_pressure",
            "steering_angle_calculated",
            "steering_angle_calculated_sign",
        ]
        path_can = self.path_data / f"camera_lidar-{self.recording_time}_bus_signals.tar"
        assert path_can.exists(), f"Path {path_can} not found"
        with tarfile.open(path_can, mode="r") as can_file:
            # Open and parse CAN content
            path_json = (
                f"camera_lidar/{_reformat_recording_time(self.recording_time)}/"
                f"bus/{self.recording_time}_bus_signals.json"
            )
            with can_file.extractfile(path_json) as f:
                self.can_data = json.load(f)

                # Check the data
                missing_in_signals = [n for n in self.signal_in_names if n not in self.can_data]
                missing_out_signals = [n for n in self.signal_out_names if n not in self.can_data]
                assert not missing_in_signals, f"Missing CAN signals: {missing_in_signals}"
                assert not missing_out_signals, f"Missing CAN signals: {missing_out_signals}"

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

        result = {}

        for (sensor_type, sensor_position), tar in self._tars.items():
            path_dir = (
                f"./camera_lidar/{_reformat_recording_time(self.recording_time)}/{sensor_type}/cam_{sensor_position}"
            )
            pos_in_name = sensor_position.replace("_", "")
            ext = "npz" if sensor_type == "lidar" else "png"
            path_name = f"{self.recording_time}_{sensor_type}_{pos_in_name}_{sequence_id}.{ext}"
            logger.info(f"loading {path_dir}/{path_name} for {sensor_type} {sensor_position}")
            fileobj = tar.extractfile(f"{path_dir}/{path_name}")
            result[(sensor_type, sensor_position)] = fileobj

        return result
