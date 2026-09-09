"""Ordinary PyTorch U-Net training; also used by numerical and hardware checks."""

import argparse
import json
import time

import torch
from torch import nn
from torch.nn import functional as F


def block(source, target):
    return nn.Sequential(
        nn.Conv2d(source, target, 3, padding=1),
        nn.BatchNorm2d(target),
        nn.ReLU(),
        nn.Conv2d(target, target, 3, padding=1),
        nn.BatchNorm2d(target),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, width=8, classes=3):
        super().__init__()
        self.encoder1 = block(3, width)
        self.encoder2 = block(width, width * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(width * 2, width * 4)
        self.dropout = nn.Dropout2d(0.2)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.decoder2 = block(width * 4, width * 2)
        self.decoder1 = block(width * 3, width)
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, images):
        skip1 = self.encoder1(images)
        skip2 = self.encoder2(self.pool(skip1))
        deep = self.dropout(self.bottleneck(self.pool(skip2)))
        decoded = self.decoder2(torch.cat((self.up2(deep), skip2), dim=1))
        decoded = F.interpolate(decoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder1(torch.cat((decoded, skip1), dim=1)))


def train(model_factory):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-device", default=None, help="CPU simulation harness only")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="AdamW")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(321)
    model = model_factory(args.width).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=1e-3, foreach=False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    device = args.input_device or args.device
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    phase = "inputs"
    try:
        for _ in range(args.steps):
            started = time.perf_counter()
            images = torch.randn(args.batch, 3, args.size, args.size, device=device)
            masks = torch.randint(3, (args.batch, args.size, args.size), device=device)
            optimizer.zero_grad()
            phase = "forward"
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            loss = F.cross_entropy(outputs, masks)
            losses.append(loss.item())
            phase = "backward"
            loss.backward()
            phase = "optimizer.step"
            optimizer.step()
            times.append(time.perf_counter() - started)
    except torch.OutOfMemoryError:
        print(
            json.dumps(
                {
                    "loss": losses,
                    "step_seconds": times,
                    "failed_phase": phase,
                    "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved": torch.cuda.max_memory_reserved(),
                    "gpu": torch.cuda.get_device_name(),
                    "gpu_vram": torch.cuda.get_device_properties(0).total_memory,
                }
            )
        )
        raise
    if args.checkpoint:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, args.checkpoint)
    print(
        json.dumps(
            {
                "loss": losses,
                "step_seconds": times,
                "peak_cuda_allocated": torch.cuda.max_memory_allocated()
                if device.startswith("cuda")
                else None,
                "peak_cuda_reserved": torch.cuda.max_memory_reserved() if device.startswith("cuda") else None,
                "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                "gpu_vram": torch.cuda.get_device_properties(0).total_memory
                if device.startswith("cuda")
                else None,
            }
        )
    )


if __name__ == "__main__":
    train(UNet)
