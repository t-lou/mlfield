import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    torch.save(ckpt, args.dst)
