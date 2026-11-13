"""Start everything:
    python run.py                      college portal  (KLE branding, port 5000)
    python run.py --portal exchange    SpareGPU exchange (own data folder, port 5100)

--portal chooses which of the two websites runs (same code, different config). It is copied into the
GPUSCHED_PORTAL environment variable, which config.py reads to pick branding, data folder and port.
"""
import os
import sys

# This must run BEFORE anything imports config, because config reads the variable when it is imported.
if "--portal" in sys.argv:
    i = sys.argv.index("--portal")
    os.environ["GPUSCHED_PORTAL"] = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""   # before config is imported; children inherit it

# Imported only now (after the environment is set); the supervisor's children inherit the variable too.
from e3_runtime.supervisor import main

if __name__ == "__main__":
    main()
