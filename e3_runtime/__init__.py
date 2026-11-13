"""E3 Process Isolation & Recovery Engine: supervisor, provider monitor, provider agent.

How the pieces fit:
  * supervisor.py       - run.py's parent process; starts web / scheduler / monitor as separate OS processes so
                          one crash does not kill the others, and restarts whichever dies.
  * provider_monitor.py - server side: handles each provider heartbeat; the watchdog re-queues jobs of providers
                          that were silent for more than 20 s.
  * agent.py            - runs on the lender's laptop; polls the server every 2 s with a Bearer token, runs jobs
                          (user code only inside Docker) and kills leftovers after its own restart.
  * recovery.py         - kill_tree helper and the startup consistency check.
  * gpu_probe.py        - reads GPU info with nvidia-smi.
  * jobs/               - the prepared workloads and the sandbox wrapper that run on the provider.
Stop requests carry a reason: "user" (the submitter) or "owner" (the GPU owner).
"""
