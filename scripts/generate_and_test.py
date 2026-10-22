#!/usr/bin/env python3
"""
EdgeSense Nano: pipeline self-test on synthetic data with a KNOWN control law
Version: 2.1.0

Generates synthetic data from a known PD-type law plus noise, runs the same
symbolic-regression step as train_on_real_data.py, and reports whether the
known law was recovered. This tests the pipeline only; it says nothing about
real vessels.

Usage:
    python scripts/generate_and_test.py
    python scripts/generate_and_test.py --noise 0.05 --rows 2000 --seed 1
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# Ground-truth law used to generate the data
KP, KD, BIAS = -0.28, 0.27, 0.36   # thrust = KP*error + KD*gyro_y + BIAS + noise


def main():
    p = argparse.ArgumentParser(description="EdgeSense synthetic self-test")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.02, help="std of additive noise on thrust")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iterations", type=int, default=40)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    t = np.linspace(0.0, args.rows * 0.1, args.rows)          # 10 Hz log
    error = 0.8 * np.sin(0.3 * t) + 0.3 * np.sin(1.1 * t) + rng.normal(0, 0.05, args.rows)
    gyro_y = 1.5 * np.cos(0.9 * t) + rng.normal(0, 0.1, args.rows)
    thrust = KP * error + KD * gyro_y + BIAS + rng.normal(0, args.noise, args.rows)

    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "synthetic_pd.csv")
    pd.DataFrame({"error": error, "gyro_y": gyro_y, "thrust": thrust}).to_csv(csv_path, index=False)
    print(f"Synthetic data written: {csv_path}")
    print(f"Ground truth: thrust = {KP}*error + {KD}*gyro_y + {BIAS}  (noise std {args.noise})\n")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_on_real_data.py")
    cmd = [sys.executable, script,
           "--csv", csv_path,
           "--inputs", "error", "gyro_y",
           "--target", "thrust",
           "--iterations", str(args.iterations),
           "--seed", str(args.seed),
           "--out-dir", os.path.join("outputs", "synthetic_selftest"),
           "--model-name", "synthetic_model",
           "--data-label", "SYNTHETIC (known PD law)"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print("\nCompare the selected equation with the ground truth above.")
    print("Coefficients should match within roughly the noise level; "
          "held-out R^2 should be close to 1 for small noise.")


if __name__ == "__main__":
    main()
