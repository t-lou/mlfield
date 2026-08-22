import bisect
import io
import json
import os
import random
import tarfile
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Optional

from components.definitions.mmperc_params import MmpercParams
from components.utils.logger import logger
from torch.utils.data import Dataset, get_worker_info

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_PIL = False

import numpy as np


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

    In training mode the CAN data containing the vehicle status are read and interpolated with any camera frame,
    and in refinement mode the actuator status is also included.

    Multi-worker / performance notes
    ---------------------------------
    - Tar files are NOT kept open across process boundaries. Each process (the main process, or each
      DataLoader worker) lazily opens and indexes its own tar file handles on first access. This avoids
      a subtle but serious correctness bug: forked worker processes would otherwise inherit file
      descriptors that share the same underlying OS file offset as the parent, so concurrent reads from
      different workers can silently corrupt each other's data. It also makes the dataset safe to use
      with the ``spawn`` multiprocessing start method, since open tar handles are dropped before pickling
      (see ``__getstate__``).
    - Tar members are indexed once per tar (name -> TarInfo) so per-item extraction is O(1) instead of
      the O(n) linear scan that ``tarfile.extractfile(name: str)`` performs.
    - ``__getitem__`` decodes images/point clouds to numpy arrays (or returns raw bytes if
      ``decode_images=False``) instead of returning live file objects, since file objects tied to a tar
      are not picklable and cannot cross the process boundary used by DataLoader workers.
    - CAN signal interpolation uses binary search instead of a linear scan.

    Recommended DataLoader usage: ``DataLoader(dataset, num_workers=N, persistent_workers=True,
    prefetch_factor=2-4, pin_memory=True)``. Optionally pass ``worker_init_fn=worker_init_fn`` (defined in
    this module) to eagerly open/index tars when each worker starts, rather than paying that cost on the
    first ``__getitem__`` call.
    """

    _SENSOR_POSITIONS = [
        "front_center",
        "front_right",
        "front_left",
        "side_right",
        "side_left",
        "rear_center",
    ]
    _SENSOR_TYPES = ["camera", "lidar"]

    def __init__(
        self,
        params: MmpercParams,
        recording_time: str = "20190401145936",
        mode: Mode = Mode.TRAIN,
        shuffle: bool = False,
        seed: int = 42,
        decode_images: bool = True,
    ):
        """
        Args:
            params: Dataset parameters (data/calibration paths).
            recording_time: Recording identifier used to locate the tar files.
            mode: TRAIN or REFINE (REFINE also yields CAN "out"/actuator signals).
            shuffle: If False (default), items are returned in chronological (sequential) order -
                useful for temporal modeling. If True, a fixed random permutation (seeded by ``seed``)
                is used instead, so you can A/B test whether shuffling helps a non-temporal model.
            seed: Seed for the shuffle permutation, only used when ``shuffle=True``.
            decode_images: If True (default), camera frames are decoded to HxWx3 uint8 numpy arrays and
                lidar frames to a dict of numpy arrays. If False, raw bytes are returned instead (still
                picklable, unlike the raw tar file objects the previous implementation returned).
        """
        self.params = params
        self.path_data = Path(params.path_data)
        self.mode = mode
        assert self.path_data.is_dir()
        self.path_calib = Path(params.path_calibration)
        assert self.path_calib.exists()

        self.recording_time = recording_time
        self.decode_images = decode_images
        self.shuffle = shuffle
        self.seed = seed

        # Lazily-populated, per-process tar state. Never assume these survive a fork/spawn boundary -
        # always go through _ensure_tars_open().
        self._tars: dict = {}
        self._tar_index: dict = {}
        self._owner_pid = -1

        self._ensure_tars_open()
        assert self._tars
        assert any(sensor_type == "camera" for sensor_type, _ in self._tars.keys())

        # Open any camera file, collect the sequence ids and load json for timestamps.
        camera_key = next(key for key in self._tars if key[0] == "camera")
        camera_tar = self._tars[camera_key]
        self.sequence_ids: list = []
        self.timestamps: list = []
        for name, member in self._tar_index[camera_key].items():
            if not member.isfile():
                continue
            path_member = Path(name)
            if path_member.suffix != ".json":
                continue
            sequence_id = path_member.stem.split("_")[-1]
            fileobj = camera_tar.extractfile(member)
            if fileobj is None:
                continue
            with fileobj as f:
                data = json.load(f)
            self.sequence_ids.append(sequence_id)
            self.timestamps.append(data["cam_tstamp"])

        assert len(self.sequence_ids) == len(self.timestamps)
        # NOTE: the previous implementation sorted `sequence_ids` and `timestamps` independently, which
        # silently desynchronized the pairing between a frame's id and its timestamp. Sort them together,
        # by timestamp, so index order is also chronological (needed for the sequential mode below).
        if self.sequence_ids:
            paired = sorted(zip(self.timestamps, self.sequence_ids))
            self.timestamps, self.sequence_ids = (list(values) for values in zip(*paired))
        num_items = len(self.sequence_ids)
        assert num_items > 0, "No sequence ids found."
        logger.info(f"Found {num_items} files.")

        if self.shuffle:
            rng = random.Random(self.seed)
            self._order = rng.sample(range(num_items), num_items)
        else:
            self._order = list(range(num_items))

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
        missing_in_signals = [name for name in self.signal_in_names if name not in self.can_data]
        missing_out_signals = [name for name in self.signal_out_names if name not in self.can_data]
        assert not missing_in_signals, f"Missing CAN signals: {missing_in_signals}"
        assert not missing_out_signals, f"Missing CAN signals: {missing_out_signals}"

        # Pre-sort each signal's (timestamp, value) pairs once and cache just the timestamps, so lookups
        # can use binary search (O(log n)) instead of a linear scan (O(n)) per call.
        self._can_ts_cache: dict = {}
        for name in self.signal_in_names + self.signal_out_names:
            entries = sorted(self.can_data[name]["values"], key=lambda pair: pair[0])
            self.can_data[name]["values"] = entries
            self._can_ts_cache[name] = [ts for ts, _ in entries]

    # -- process-safe tar lifecycle -------------------------------------------------------------

    def _ensure_tars_open(self) -> None:
        """(Re)open and index tar files for the current process, if not already done.

        Safe to call unconditionally before any tar access. Tar handles are never shared across a
        fork/spawn boundary: this checks the current pid and transparently reopens fresh handles for
        each new process (main process, or each DataLoader worker).
        """
        pid = os.getpid()
        if self._tars and self._owner_pid == pid:
            return

        self._close_tars()
        for sensor_type in self._SENSOR_TYPES:
            for sensor_position in self._SENSOR_POSITIONS:
                pos_in_name = sensor_position.replace("_", "")
                filename = f"camera_lidar-{self.recording_time}_{sensor_type}_{pos_in_name}.tar"
                path = self.path_data / filename
                if not path.exists():
                    logger.warning(f"Path {path} not found")
                    continue
                try:
                    tar = tarfile.open(path, mode="r")
                    members = {member.name: member for member in tar.getmembers()}
                except Exception as exc:  # noqa: BLE001 - tolerate a single bad tar, don't crash the run
                    logger.warning(f"Failed to open/index {path}: {exc}")
                    continue
                logger.info(f"Path {path} found")
                self._tars[(sensor_type, sensor_position)] = tar
                self._tar_index[(sensor_type, sensor_position)] = members
        self._owner_pid = pid

    def _close_tars(self) -> None:
        for tar in self._tars.values():
            try:
                tar.close()
            except Exception:  # noqa: BLE001
                pass
        self._tars = {}
        self._tar_index = {}

    def close(self) -> None:
        """Explicitly release all open tar handles."""
        self._close_tars()

    def __enter__(self) -> "A2D2DatasetUnlabeled":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        """Close all opened tars."""
        try:
            self._close_tars()
        except Exception:  # noqa: BLE001
            pass

    def __getstate__(self) -> dict:
        # Never pickle open tar handles across a process boundary (e.g. the `spawn` start method) -
        # they aren't picklable, and even if they were, sharing them would be unsafe (see class
        # docstring). Dropping them here plus resetting _owner_pid forces a fresh, safe reopen in
        # whichever process unpickles this dataset.
        state = self.__dict__.copy()
        state["_tars"] = {}
        state["_tar_index"] = {}
        state["_owner_pid"] = -1
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # -- decoding ---------------------------------------------------------------------------------

    @staticmethod
    def _decode_camera(data: bytes):
        if not _HAS_PIL:
            logger.warning("PIL not available, returning raw camera bytes instead of a decoded array.")
            return data
        with Image.open(io.BytesIO(data)) as img:
            return np.array(img.convert("RGB"))

    @staticmethod
    def _decode_lidar(data: bytes):
        with np.load(io.BytesIO(data)) as npz:
            return {key: npz[key] for key in npz.files}

    # -- CAN signal interpolation -------------------------------------------------------------------

    def _find_nearest_last_can_signal(self, signal_name: str, timestamp: int) -> Optional[float]:
        """Find the last nearest CAN signal value for a given timestamp (binary search)."""
        ts_list = self._can_ts_cache[signal_name]
        idx = bisect.bisect_right(ts_list, timestamp) - 1
        if idx < 0:
            return None
        return self.can_data[signal_name]["values"][idx][1]

    # -- public API -----------------------------------------------------------------------------

    def iter_sensor(self) -> Iterator[tuple[str, str, tarfile.TarFile]]:
        """Iterate over all sensors and return the sensor type, position, and tarfile."""
        self._ensure_tars_open()
        for (sensor_type, sensor_position), tar in self._tars.items():
            yield sensor_type, sensor_position, tar

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        real_index = self._order[index]
        return self.get_with_index(real_index)

    def get_with_index(self, index: int) -> dict:
        assert 0 <= index < len(self.sequence_ids)
        self._ensure_tars_open()
        sequence_id = self.sequence_ids[index]

        result: dict = {}

        for (sensor_type, sensor_position), tar in self._tars.items():
            path_dir = (
                f"./camera_lidar/{_reformat_recording_time(self.recording_time)}/{sensor_type}/cam_{sensor_position}"
            )
            pos_in_name = sensor_position.replace("_", "")
            ext = "npz" if sensor_type == "lidar" else "png"
            path_name = f"{self.recording_time}_{sensor_type}_{pos_in_name}_{sequence_id}.{ext}"
            member_name = f"{path_dir}/{path_name}"
            logger.info(f"loading {member_name} for {sensor_type} {sensor_position}")

            tarinfo = self._tar_index.get((sensor_type, sensor_position), {}).get(member_name)
            if tarinfo is None:
                logger.warning(f"Member {member_name} not found for {sensor_type} {sensor_position}")
                result[(sensor_type, sensor_position)] = None
                continue

            fileobj = tar.extractfile(tarinfo)  # O(1): seeks directly using the cached TarInfo offset
            if fileobj is None:
                result[(sensor_type, sensor_position)] = None
                continue
            with fileobj as f:
                raw = f.read()

            if not self.decode_images:
                result[(sensor_type, sensor_position)] = raw
            elif sensor_type == "camera":
                result[(sensor_type, sensor_position)] = self._decode_camera(raw)
            else:
                result[(sensor_type, sensor_position)] = self._decode_lidar(raw)

        # Find the last nearest CAN in signals, if mode is REFINE, also add CAN out signals
        timestamp = self.timestamps[index]
        result["can_in"] = {
            signal_name: self._find_nearest_last_can_signal(signal_name, timestamp)
            for signal_name in self.signal_in_names
        }
        if self.mode == Mode.REFINE:
            result["can_out"] = {
                signal_name: self._find_nearest_last_can_signal(signal_name, timestamp)
                for signal_name in self.signal_out_names
            }

        result["sequence_id"] = sequence_id
        result["timestamp"] = timestamp

        return result


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001 - required signature for DataLoader
    """Optional DataLoader ``worker_init_fn``.

    Eagerly opens and indexes tar files as soon as each worker process starts, instead of paying that
    cost lazily on the first ``__getitem__`` call. Not required for correctness (the dataset opens tars
    lazily and safely on its own per-process), but avoids a first-batch latency spike.

    Usage:
        DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn, persistent_workers=True)
    """
    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if isinstance(dataset, A2D2DatasetUnlabeled):
        dataset._ensure_tars_open()
