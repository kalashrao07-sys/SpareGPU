"""Runs a user's script inside the Docker sandbox. Before the script starts it caps this process's PyTorch GPU
memory to the amount the user declared, so the declared memory (HC1) is actually enforced for PyTorch code."""
import os
import runpy
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: sandbox_wrapper.py <script.py>")
    script = sys.argv[1]
    # The agent passes the user's declared GPU memory (GB) through this variable.
    declared = float(os.environ.get("GPUSCHED_MEMORY_GB", "0") or 0)
    try:
        import torch
        if declared > 0 and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            # Hard cap: PyTorch raises out-of-memory beyond this fraction, so one job cannot take more than
            # declared.
            torch.cuda.set_per_process_memory_fraction(min(1.0, declared / total), 0)
            print(f"[sandbox] GPU memory capped to {declared:.2f} of {total:.2f} GB", flush=True)
    except Exception as e:                    # no PyTorch (e.g. TensorFlow user): the declared value stays advisory
        print(f"[sandbox] memory cap not applied ({e.__class__.__name__})", flush=True)
    # Make the script see a normal command line (argv[0] = itself).
    sys.argv = [script]
    # Let the script import its sibling files from the project folder.
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    # Run the user's file as if started with 'python script.py' (so __name__ == '__main__' works).
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
