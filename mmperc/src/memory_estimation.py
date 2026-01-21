import torch


def measure_memory(model, inputs, batch_size, training=True, device="cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    if training:
        model.train()
        out = model(**inputs)

        if isinstance(out, dict):
            loss = sum(v.sum() for v in out.values() if torch.is_tensor(v))
        else:
            loss = out.sum()

        loss.backward()
    else:
        model.eval()
        with torch.no_grad():
            out = model(**inputs)

    peak = torch.cuda.max_memory_allocated(device)
    return peak


def find_max_batch_size(model, inputs, training=True, max_bs=1024, device="cuda"):
    low, high = 1, max_bs
    best = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            _ = measure_memory(model, inputs, mid, training=training, device=device)
            best = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e
    return best


def get_parameter_size(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


if __name__ == "__main__":
    import random

    from common.device import get_best_device
    from datasets.a2d2_dataset import A2D2Dataset
    from model.simple_model import SimpleModel

    device = get_best_device()
    print(f"Using device: {device}")

    dataset = A2D2Dataset("/workspace/mmperc/data/a2d2")
    sample = dataset[random.randint(0, len(dataset) - 1)]

    model = SimpleModel().to(device)
    param_mem = get_parameter_size(model) / 1024**2
    print(f"Parameter memory: {param_mem:.2f} MB")

    points: torch.Tensor = sample["points"].unsqueeze(0).to(device)
    images: torch.Tensor = sample["camera"].unsqueeze(0).to(device)

    print(f"Input sizes are {points.shape} {images.shape}")

    inputs = {"points": points, "images": images}

    max_bs = find_max_batch_size(model, inputs, device=device)
    print(f"Max batch size that fits: {max_bs}")
