import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpochLoss:
    epoch_id: int
    batch_id: int
    loss_hm: float
    loss_reg: float
    loss_sem: float
    loss_total: float

    def to_dict(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "batch_id": self.batch_id,
            "loss_hm": self.loss_hm,
            "loss_reg": self.loss_reg,
            "loss_sem": self.loss_sem,
            "loss_total": self.loss_total,
        }


class JSONLossLLogger:
    def __init__(self, path: str, cache_size: int = 10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_size = cache_size
        self._cache = []

    def append(self, record: EpochLoss):
        self._cache.append(record)

        if len(self._cache) >= self._cache_size:
            self.flush()

    def flush(self):
        """Write cached records to disk."""
        if not self._cache:
            return

        with self.path.open("a") as f:
            for rec in self._cache:
                f.write(json.dumps(asdict(rec)) + "\n")

        self._cache = []

    def close(self):
        """Flush remaining records."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.flush()
