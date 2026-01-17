import sys
import tarfile
from pathlib import PurePosixPath


class A2D2TarDatasetConverter:
    def __init__(self, tar_path: str):
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")

        self._name = "cam_front_center"
        self._root_parsing = "camera"
        self._dataset_type = "camera_lidar_semantic_bboxes"
        self._group_size = 200

        # internal caches
        self._frames = {}

    def list_timestamps(self):
        """
        Find all timestamps under:
            camera_lidar_semantic_bboxes/<timestamp>/
        """
        timestamps = set()

        for member in self.tar.getmembers():
            p = PurePosixPath(member.name)

            # Expect: camera_lidar_semantic_bboxes/<timestamp>/...
            if len(p.parts) >= 2 and p.parts[0] == self._dataset_type:
                timestamps.add(p.parts[1])

        return sorted(timestamps)

    def find_frames(self):
        self._frames = {}

        for member in self.tar.getmembers():
            p = PurePosixPath(member.name)

            if len(p.parts) >= 4:
                ts, sensor_type, cam_name = p.parts[1], p.parts[2], p.parts[3]

                if sensor_type == self._root_parsing and cam_name == self._name:
                    frame_id = p.parts[-1].split(".")[0].split("_")[-1]
                    full_frame_id = (ts, frame_id)
                    if full_frame_id not in self._frames:
                        self._frames[full_frame_id] = {}
                    self._frames[full_frame_id][sensor_type] = member.name

    def shuffle_and_group_pngs(self):
        """
        Shuffle and group the front center PNG files into groups of given size.
        """
        self.find_frames()
        shuffled = self._frames.values()
        for i in shuffled:
            print(len(i), i)

        # grouped = [
        #     shuffled[i : i + self._group_size]
        #     for i in range(0, len(shuffled), self._group_size)
        # ]
        # if len(grouped[-1]) < self._group_size:
        #     grouped = grouped[:-1]  # drop last incomplete group
        grouped = []
        return grouped


if __name__ == "__main__":
    tar_path = sys.argv[1]
    converter = A2D2TarDatasetConverter(tar_path)

    groups = converter.shuffle_and_group_pngs()
    # print(len(groups))
    # print(groups[0][0])
