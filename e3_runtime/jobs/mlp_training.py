"""Workload 2: small neural network training on synthetic data (no dataset download).
A 3-layer MLP learns to classify 10 Gaussian clusters. A reserve tensor is allocated so the job really
occupies the declared memory. Trains until the deadline and reports loss and accuracy.
Output: an info line, then "RESULT {...json...}" with steps (mini-batches done), final_loss (last batch's
cross-entropy, lower = better) and accuracy (fraction of the 20000 points classified correctly).
Uses PyTorch (CUDA). With --device cpu and no PyTorch it falls back to a tiny NumPy network (development only)."""
import argparse
import json
import sys
import time


def numpy_train(deadline):
    """Development fallback: a 2-layer network trained by hand-written gradient descent in NumPy. Returns (steps, loss, accuracy)."""
    import numpy as np
    # Fixed seed 0 so every run makes the same data.
    rng = np.random.default_rng(0)
    # 10 random cluster centres in 32 dimensions; each sample is its centre plus noise.
    centers = rng.normal(0, 3, (10, 32))
    y = rng.integers(0, 10, 4000)
    X = centers[y] + rng.normal(0, 1, (4000, 32))
    W1 = rng.normal(0, 0.1, (32, 64)); b1 = np.zeros(64)
    W2 = rng.normal(0, 0.1, (64, 10)); b2 = np.zeros(10)
    steps, loss = 0, 0.0
    while time.time() < deadline:
        idx = rng.integers(0, 4000, 256)
        xb, yb = X[idx], y[idx]
        # Forward pass: hidden = ReLU(x W1 + b1), then scores z = h W2 + b2.
        h = np.maximum(0, xb @ W1 + b1)
        z = h @ W2 + b2
        # Softmax (subtract the max for numerical stability) turns scores into probabilities p.
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        # Cross-entropy loss: -log of the probability given to the true class.
        loss = float(-np.log(p[np.arange(256), yb] + 1e-9).mean())
        # Backward pass: gradient of the loss w.r.t. scores is (p - one_hot(y)) / batch size, then the
        # chain rule.
        g = p; g[np.arange(256), yb] -= 1; g /= 256
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (h > 0)
        gW1 = xb.T @ gh; gb1 = gh.sum(0)
        # Gradient descent step with learning rate 0.1.
        for P, G in ((W1, gW1), (b1, gb1), (W2, gW2), (b2, gb2)):
            P -= 0.1 * G
        steps += 1
    # Accuracy: share of points whose highest score is the true class.
    pred = (np.maximum(0, X @ W1 + b1) @ W2 + b2).argmax(1)
    return steps, loss, float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-gb", type=float, required=True)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--overhead-gb", type=float, default=0.4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args()

    try:
        import torch
    except ImportError:
        torch = None

    if torch is None:
        steps, loss, acc = numpy_train(a.deadline_epoch)
        print("train on cpu (NumPy fallback, development only)", flush=True)
    else:
        dev = "cuda" if a.device == "cuda" else "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            print("ERROR: CUDA is not available to PyTorch", flush=True)
            sys.exit(3)
        try:
            # Fixed seed so results are repeatable.
            torch.manual_seed(0)
            # n samples, d features, k classes (clusters).
            n, d, k = 20000, 64, 10
            centers = torch.randn(k, d, device=dev) * 3
            y = torch.randint(0, k, (n,), device=dev)
            X = centers[y] + torch.randn(n, d, device=dev)
            # 3-layer MLP: 64 -> 256 -> 256 -> 10 with ReLU in between.
            model = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(),
                                        torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                        torch.nn.Linear(256, k)).to(dev)
            # AdamW optimizer, learning rate 0.001; CrossEntropyLoss compares scores with the true labels.
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            lossf = torch.nn.CrossEntropyLoss()
            if dev == "cuda":                          # hold the declared memory
                # Allocate (and zero) a big tensor so the job truly occupies its declared GPU memory (200
                # MB left for the model).
                reserve = max(0, int((a.memory_gb - a.overhead_gb) * 1024 ** 3) - 200 * 1024 ** 2)
                hold = torch.empty(reserve // 4, dtype=torch.float32, device=dev)
                hold.zero_()
            print(f"train on {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})", flush=True)
            steps, loss = 0, 0.0
            while time.time() < a.deadline_epoch:
                # Each epoch: shuffle the data, then walk through it in mini-batches of 512.
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, 512):
                    if time.time() >= a.deadline_epoch:
                        break
                    idx = perm[i:i + 512]
                    # Standard training step: clear old gradients, forward, loss, backward, update weights.
                    opt.zero_grad()
                    out = lossf(model(X[idx]), y[idx])
                    out.backward()
                    opt.step()
                    steps += 1
                    loss = float(out)
            # Evaluation only: no gradients needed. Accuracy = fraction of correct predictions on the data.
            with torch.no_grad():
                acc = float((model(X).argmax(1) == y).float().mean())
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
            sys.exit(4)

    print("RESULT " + json.dumps({"workload": "train", "steps": steps, "final_loss": round(loss, 4),
                                  "accuracy": round(acc, 3)}), flush=True)


if __name__ == "__main__":
    main()
