from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SensorPose:
    name: str
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    vehicle_from_sensor: np.ndarray
    sensor_from_vehicle: np.ndarray


@dataclass
class SensorCalibration:
    name: str
    pose: SensorPose
    intrinsic: np.ndarray | None = None
    distortion: np.ndarray | None = None
    resolution: tuple[int, int] | None = None


@dataclass
class CamsLidarsCalibration:
    camera: SensorCalibration
    lidar: SensorCalibration
    lidar_to_camera: np.ndarray
    camera_to_lidar: np.ndarray


def _build_pose_from_view(name: str, view: dict[str, Any]) -> SensorPose:
    origin = np.asarray(view["origin"], dtype=np.float64)
    x_axis = np.asarray(view["x-axis"], dtype=np.float64)
    y_axis = np.asarray(view["y-axis"], dtype=np.float64)

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    vehicle_from_sensor = np.eye(4, dtype=np.float64)
    vehicle_from_sensor[:3, :3] = rotation
    vehicle_from_sensor[:3, 3] = origin
    sensor_from_vehicle = np.linalg.inv(vehicle_from_sensor)

    return SensorPose(
        name=name,
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        vehicle_from_sensor=vehicle_from_sensor,
        sensor_from_vehicle=sensor_from_vehicle,
    )


def _load_camera_info(name: str, info: dict[str, Any]) -> SensorCalibration:
    view = info.get("view") or info.get("view_")
    if view is None:
        raise ValueError(f"Camera calibration for '{name}' missing view information.")

    pose = _build_pose_from_view(name=name, view=view)
    intrinsic = np.asarray(info["CamMatrix"], dtype=np.float32)
    distortion = np.asarray(info.get("Distortion", []), dtype=np.float32).reshape(-1)
    resolution_raw = info.get("Resolution")
    if resolution_raw is None or len(resolution_raw) != 2:
        raise ValueError(f"Camera calibration for '{name}' missing Resolution.")

    resolution = (int(resolution_raw[1]), int(resolution_raw[0]))
    return SensorCalibration(
        name=name,
        pose=pose,
        intrinsic=intrinsic,
        distortion=distortion,
        resolution=resolution,
    )


def _load_lidar_info(name: str, info: dict[str, Any]) -> SensorCalibration:
    view = info.get("view")
    if view is None:
        raise ValueError(f"LIDAR calibration for '{name}' missing view information.")

    pose = _build_pose_from_view(name=name, view=view)
    return SensorCalibration(name=name, pose=pose)


def load_cams_lidars_calibration(
    path: Path,
    camera_name: str = "front_center",
    lidar_name: str = "front_center",
) -> CamsLidarsCalibration:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cameras = data.get("cameras")
    lidars = data.get("lidars")
    if cameras is None or lidars is None:
        raise ValueError("Calibration JSON must contain both 'cameras' and 'lidars' sections.")

    camera_info = cameras.get(camera_name)
    lidar_info = lidars.get(lidar_name)
    if camera_info is None:
        raise KeyError(f"Camera '{camera_name}' not found in calibration file.")
    if lidar_info is None:
        raise KeyError(f"LIDAR '{lidar_name}' not found in calibration file.")

    camera_calib = _load_camera_info(camera_name, camera_info)
    lidar_calib = _load_lidar_info(lidar_name, lidar_info)

    lidar_to_camera = camera_calib.pose.sensor_from_vehicle @ lidar_calib.pose.vehicle_from_sensor
    camera_to_lidar = np.linalg.inv(lidar_to_camera)

    return CamsLidarsCalibration(
        camera=camera_calib,
        lidar=lidar_calib,
        lidar_to_camera=lidar_to_camera.astype(np.float32),
        camera_to_lidar=camera_to_lidar.astype(np.float32),
    )
