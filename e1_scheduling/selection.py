"""E1: hard-constraint filters (HC1, HC2, HC4-HC7) and the ranked preference order (SP2, SP3, tie-break).

Pure functions on plain dicts: no database, no clock, no I/O. That makes them easy to test.
A GPU dict needs: gpu_id, gpu_type, memory_gb, current_job_id, free_since
                  (SpareGPU extras: provider_id, owner_user_id, price_per_hour, avail_from, avail_to, private_only).
A job dict needs:  memory_gb and (optionally) pref_gpu_type, kind, duration_min, user_id,
                  trusted_providers, max_price_per_hour, available_credits.

How a decision is made (select_gpu):
  1. HARD constraints remove GPUs that can NEVER be used for this job (HC1, HC4, HC7, HC5, HC6) - a violation here is a
     "hold" with a human-readable reason, so the user sees WHY the job waits.
  2. HC2 then removes GPUs that are busy right now (waiting for one to free up is a different reason).
  3. Among the survivors the SOFT preferences pick one winner, in a fixed order (price, SP2, SP3, gpu_id).
Hard constraints are never traded off against preferences: a GPU either qualifies or it does not.
"""
from dataclasses import dataclass
from typing import Optional

import config
import rules

REASON_NO_MEMORY = "no GPU with sufficient memory capacity"                    # FR11
REASON_NONE_FREE = "a suitable GPU exists but none is currently free"           # FR12
REASON_NO_PROVIDER = "no provider GPU is online right now"                      # extension
REASON_NO_SCRIPTS = "no online provider accepts your own scripts"               # HC4
REASON_PRIVATE = "the GPUs that fit are private to their owners"                # HC7
REASON_HOURS = "the lenders' available hours do not cover your whole booking"   # HC5
REASON_PRICE = "every available GPU costs more than your maximum price"         # HC6
REASON_CREDITS = "not enough credits for the booked time"                       # HC6


# ---- Hold reasons shown to the user. Each is tied to the constraint that caused it (FR numbers from the SRS).
def hc1_ok(gpu, job):
    """HC1: the GPU's total memory must cover the job's memory requirement (FR5)."""
    return gpu["memory_gb"] >= job["memory_gb"]


def hc2_ok(gpu):
    """HC2: a GPU runs at most one job at a time, so it must be free now (FR6).
    Only the GPU's current commitment is looked at, never past jobs (NFR7)."""
    return gpu["current_job_id"] is None


def hc4_ok(gpu, job):
    """HC4: a script or session job (user code) may only run where the provider has switched sandbox mode on.
    Prepared-workload jobs run anywhere."""
    return job.get("kind", "template") == "template" or bool(gpu.get("accepts_scripts"))


def hc7_ok(gpu, job):
    """HC7: a private GPU may only be used by its owner or the owner's trusted users."""
    return rules.may_use(gpu, job)


def hc5_ok(gpu, job, now):
    """HC5: the whole booking [now, now+duration] must sit inside the lender's available hours."""
    return rules.fits_window(gpu.get("avail_from"), gpu.get("avail_to"), now, now + job.get("duration_min", 0) * 60)


def rate_of(gpu, job):
    """Price per hour THIS renter would pay on THIS GPU (0 if it is theirs or they are trusted)."""
    return rules.effective_rate(gpu, job)


def hc6_price_ok(gpu, job):
    """HC6 (part 1): the GPU must not cost more per hour than the most the renter said they would pay.
    max_price_per_hour = None means "no limit". The 1e-9 absorbs floating-point rounding (e.g. 0.1+0.2)."""
    mx = job.get("max_price_per_hour")
    return mx is None or rate_of(gpu, job) <= mx + 1e-9


def hc6_credits_ok(gpu, job):
    """HC6 (part 2): the renter's spendable credits must cover rate x booked time.
    Skipped entirely when the wallet feature is off (config.CREDITS_ENABLED = False)."""
    if not config.CREDITS_ENABLED:
        return True
    cost = rules.estimate_cost(rate_of(gpu, job), job.get("duration_min", 0))
    return cost <= job.get("available_credits", float("inf")) + 1e-9


def filter_gpus(job, gpus):
    """Evaluate EVERY GPU (FR7). Returns (passed_hc1, passed_hc1_and_hc2)."""
    passed_hc1 = [g for g in gpus if hc1_ok(g, job)]
    candidates = [g for g in passed_hc1 if hc2_ok(g)]
    return passed_hc1, candidates


def rank_key(gpu, job, now):
    """Lower tuple wins. The order is lexicographic, never blended (FR8):
    price: cheapest GPU first (SpareGPU; every college GPU costs 0, so this changes nothing there),
    SP2: smaller idle gap first (idle time this assignment adds to the GPU's history),
    SP3: GPU type equal to the type this submitter's previous job used,
    then the lowest gpu_id as the final deterministic tie-break."""
    # idle gap = how long the GPU has been sitting free. Python compares tuples left to right, so a later element
    # only matters when every earlier element is equal - that is what "lexicographic" means.
    gap = max(0.0, now - gpu["free_since"])
    # SP3: 0 (better) when this GPU's type equals the type the user's previous job ran on, otherwise 1.
    same_type = 0 if (job.get("pref_gpu_type") and gpu["gpu_type"] == job["pref_gpu_type"]) else 1
    return (rate_of(gpu, job), gap, same_type, gpu["gpu_id"])


def deciding_rule(best, candidates, job, now):
    """Which rule in the SP order separated the winner from the rest (FR10)."""
    if len(candidates) == 1:
        return "only valid GPU"          # nothing to rank: the hard constraints already left one choice
    others = [c for c in candidates if c is not best]
    k_best = rank_key(best, job, now)
    # Walk the ranking positions in order; the FIRST position where some other GPU differs from the winner is the rule
    # that actually decided. Names "SP2 idle time" / "SP3 GPU-type continuity" are shown in the UI and tests.
    for pos, name in ((0, "lowest price"), (1, "SP2 idle time"), (2, "SP3 GPU-type continuity")):
        if any(rank_key(o, job, now)[pos] != k_best[pos] for o in others):
            return name
    return "tie-break (lowest GPU id)"   # all three preferences were equal; gpu_id made the choice


@dataclass
class Decision:
    """Result of select_gpu: either assign to gpu_id (rule says why this one) or hold (reason says why not yet)."""
    kind: str                       # "assign" or "hold"
    gpu_id: Optional[int] = None
    rule: Optional[str] = None
    reason: Optional[str] = None


def select_gpu(job, gpus, now):
    """Pick a GPU for one job. gpus = the online, approved, sharing GPUs right now. Returns a Decision."""
    if not gpus:
        return Decision("hold", reason=REASON_NO_PROVIDER)
    passed_hc1, candidates = filter_gpus(job, gpus)
    if not passed_hc1:
        return Decision("hold", reason=REASON_NO_MEMORY)
    # HC4: keep only GPUs whose provider allows user code (script/session jobs). Prepared workloads pass everywhere.
    passed_hc1 = [g for g in passed_hc1 if hc4_ok(g, job)]
    if not passed_hc1:
        return Decision("hold", reason=REASON_NO_SCRIPTS)
    # Apply HC7, HC5, HC6 (price), HC6 (credits) one after another. The first filter that empties the list decides
    # which hold reason the user sees, so each message names the exact rule that blocked the job.
    for ok, reason in ((lambda g: hc7_ok(g, job), REASON_PRIVATE), (lambda g: hc5_ok(g, job, now), REASON_HOURS),
                       (lambda g: hc6_price_ok(g, job), REASON_PRICE), (lambda g: hc6_credits_ok(g, job), REASON_CREDITS)):
        passed_hc1 = [g for g in passed_hc1 if ok(g)]
        if not passed_hc1:
            return Decision("hold", reason=reason)
    # HC2: of the GPUs that are allowed in principle, only those free right now can take the job.
    candidates = [g for g in passed_hc1 if hc2_ok(g)]
    if not candidates:
        return Decision("hold", reason=REASON_NONE_FREE)
    # min() with a tuple key = pick the winner by the preference order in rank_key.
    best = min(candidates, key=lambda g: rank_key(g, job, now))
    return Decision("assign", gpu_id=best["gpu_id"], rule=deciding_rule(best, candidates, job, now))
