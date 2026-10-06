#!/usr/bin/env python3
"""Train a tiny BESS fault classifier and export to ONNX for Arm deployment.

Usage:
    python train_model.py --output ../models/bess_fault.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BessFaultMlp(nn.Module):
    """8 → 16 → 5 MLP matching the Rust kernel architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def synthetic_batch(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate labeled feature vectors matching Rust rule thresholds."""
    rng = np.random.default_rng(42)
    x = np.zeros((n, 8), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    per_class = n // 5
    for label in range(5):
        start = label * per_class
        end = start + per_class
        for i in range(start, end):
            x[i, 0] = rng.uniform(3.4, 3.7)  # mean_v
            x[i, 1] = rng.uniform(25, 30)  # mean_t
            x[i, 2] = rng.uniform(1.0, 1.5)  # mean_z
            x[i, 3] = rng.uniform(26, 32)  # max_t
            x[i, 4] = rng.uniform(3.4, 3.7)  # min_v
            x[i, 5] = rng.uniform(-0.01, 0.01)  # dv_dt
            x[i, 6] = rng.uniform(-0.2, 0.2)  # dt_dt
            x[i, 7] = rng.uniform(0.01, 0.08)  # v_spread
            y[i] = label

            if label == 1:  # thermal_runaway
                x[i, 3] = rng.uniform(42, 48)
                x[i, 6] = rng.uniform(0.9, 1.3)
                x[i, 5] = rng.uniform(-0.12, -0.05)
            elif label == 2:  # cell_imbalance
                x[i, 7] = rng.uniform(0.18, 0.28)
                x[i, 0] = rng.uniform(3.4, 3.5)
                x[i, 5] = rng.uniform(-0.015, -0.005)
            elif label == 3:  # impedance_fault
                x[i, 2] = rng.uniform(2.4, 3.5)
            elif label == 4:  # voltage_sag
                x[i, 5] = rng.uniform(-0.04, -0.025)
                x[i, 4] = rng.uniform(3.2, 3.34)

    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BESS fault MLP and export ONNX")
    parser.add_argument("--output", type=Path, default=Path("../models/bess_fault.onnx"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    x, y = synthetic_batch()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)

    model = BessFaultMlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = model(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            acc = (logits.argmax(1) == yt).float().mean().item()
            print(f"epoch {epoch + 1}: loss={loss.item():.4f} acc={acc:.3f}")

    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 8)
    # torch>=2.9 defaults to the dynamo exporter, which needs onnxscript.
    # Keep the TorchScript exporter this script already uses.
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
