"""Workload 1: matrix multiplication benchmark.
Allocates three n x n float32 matrices that together fill the declared memory (minus CUDA overhead),
multiplies them repeatedly until the deadline, and reports the achieved GFLOPS.
Output: one info line, then "RESULT {...json...}" with n, iterations and gflops, which the agent shows as the
finished note. GFLOPS = billions of floating point operations per second (higher = faster GPU).
Uses PyTorch (CUDA). With --device cpu and no PyTorch it falls back to NumPy (development only)."""
import argparse
import json
import math
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-gb", type=float, required=True)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--overhead-gb", type=float, default=0.4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args()

    # Memory we may use = declared memory minus CUDA's own overhead, converted from GB to bytes.
    usable = max(0.05, a.memory_gb - a.overhead_gb) * 1024 ** 3
    # Matrix side n: three n*n float32 matrices need 3*4*n^2 bytes = 12*n^2, so n = sqrt(usable/12).
    n = int(math.sqrt(usable / 12.0))                  # 3 matrices * 4 bytes
    # Clamp: at least 64, at most 8192 on GPU (1024 on CPU) so a CPU test finishes quickly.
    n = max(64, min(n, 8192 if a.device == "cuda" else 1024))

    try:
        import torch
    except ImportError:
        # PyTorch missing: use the NumPy fallback below.
        torch = None

    if torch is not None:
        dev = "cuda" if a.device == "cuda" else "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            print("ERROR: CUDA is not available to PyTorch", flush=True)
            # Exit code 3 = CUDA unavailable (the agent shows 'failed (exit code 3)' with the ERROR line).
            sys.exit(3)
        try:
            # A, B: random inputs; C: preallocated output so repeated multiplies do not allocate new
            # memory.
            A = torch.rand(n, n, device=dev)
            B = torch.rand(n, n, device=dev)
            C = torch.empty(n, n, device=dev)
            # GPU calls are asynchronous; synchronize waits until they finish so the timing is honest.
            sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
            print(f"matmul n={n} on {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})", flush=True)
            iters, t0 = 0, time.time()
            while time.time() < a.deadline_epoch:
                # Do 5 multiplications per round, then synchronize once. Loop runs until the declared end
                # time.
                for _ in range(5):
                    torch.matmul(A, B, out=C)
                sync()
                iters += 5
            elapsed = max(1e-9, time.time() - t0)
        except Exception as e:                         # includes CUDA out-of-memory
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
            sys.exit(4)
    else:
        import numpy as np
        A = np.random.rand(n, n).astype("float32")
        B = np.random.rand(n, n).astype("float32")
        print(f"matmul n={n} on cpu (NumPy fallback, development only)", flush=True)
        iters, t0 = 0, time.time()
        while time.time() < a.deadline_epoch:
            A @ B
            iters += 1
        elapsed = max(1e-9, time.time() - t0)

    # One n*n matmul = 2*n^3 operations (n multiplies + n adds per output cell).
    gflops = 2.0 * n ** 3 * iters / elapsed / 1e9
    print("RESULT " + json.dumps({"workload": "matmul", "n": n, "iterations": iters, "gflops": round(gflops, 1)}), flush=True)


if __name__ == "__main__":
    main()
