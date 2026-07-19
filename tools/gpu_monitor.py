import datetime
import subprocess
import time


def get_gpu_stats():
    """Return raw nvidia-smi output."""
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        return result.decode("utf-8").strip().split("\n")
    except Exception as e:
        return [f"Error: {e}"]


def monitor_gpu(interval=2):
    """Print GPU usage every `interval` seconds."""
    print("Starting GPU monitor…")
    while True:
        print("\n" + "=" * 60)
        print("GPU Status @", datetime.datetime.now().strftime("%H:%M:%S"))
        print("=" * 60)

        stats = get_gpu_stats()
        for line in stats:
            timestamp, idx, name, util, mem_used, mem_total = line.split(", ")
            print(f"GPU {idx} | {name}")
            print(f"  Utilization: {util}%")
            print(f"  Memory: {mem_used} MB / {mem_total} MB")
        print("=" * 60)

        time.sleep(interval)


if __name__ == "__main__":
    monitor_gpu(interval=1)
