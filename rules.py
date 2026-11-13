"""Pure rules shared by E1 (scheduler) and E2 (the commit re-check). No database, no clock of its own.

Exchange rules: a lender's GPU may be open only during certain hours (HC5), may cost credits per hour, with a
renter's maximum price and credit balance as hard limits (HC6), and may be private (HC7).
College portal: every provider has no hours, price 0 and is not private, so these rules never reject anything.
"""
import time


def minutes_of_day(ts):
    """Minutes after midnight (0..1439) of a Unix timestamp, in the server's local time zone (see GPUSCHED_TZ)."""
    t = time.localtime(ts)
    return t.tm_hour * 60 + t.tm_min


def fits_window(avail_from, avail_to, start_ts, end_ts):
    """True if [start, end] lies inside the daily window avail_from..avail_to (minutes after midnight, server local
    time). None/None = always open. A window may cross midnight (e.g. 22:00-07:00). The whole booking must fit in ONE
    opening of the window, so a 10-hour job cannot start in a 9-hour window."""
    if avail_from is None or avail_to is None or avail_from == avail_to:
        return True
    start_day = time.localtime(start_ts)
    # Unix time of 00:00 on the day the booking starts (mktime turns a local date/time tuple into a timestamp).
    midnight = time.mktime((start_day.tm_year, start_day.tm_mon, start_day.tm_mday, 0, 0, 0, 0, 0, -1))
    open_h = avail_from * 60                     # window opening, in seconds after midnight (minutes x 60)
    close_h = avail_to * 60                      # window closing, in seconds after midnight
    # Each candidate is one concrete (open, close) interval on the timeline that could contain the booking.
    candidates = []
    if avail_from < avail_to:                    # same-day window
        candidates.append((midnight + open_h, midnight + close_h))
    else:                                        # crosses midnight: yesterday-evening..today-morning and today-evening..tomorrow
        candidates.append((midnight - 86400 + open_h, midnight + close_h))
        candidates.append((midnight + open_h, midnight + 86400 + close_h))
    # The booking fits if it starts after some opening AND ends before that same opening's close.
    return any(o <= start_ts and end_ts <= c for o, c in candidates)


def effective_rate(gpu, job):
    """Credits per hour this renter pays on this GPU: 0 on the owner's own laptop or if the owner trusts them."""
    pid = gpu.get("provider_id")
    if pid is None:
        return 0.0                               # a local/college GPU has no lender, so it is never charged
    if job.get("user_id") is not None and job.get("user_id") == gpu.get("owner_user_id"):
        return 0.0
    if pid in (job.get("trusted_providers") or ()):
        return 0.0
    return float(gpu.get("price_per_hour") or 0.0)


def may_use(gpu, job):
    """HC7: a private laptop is for its owner and the users the owner trusts."""
    if not gpu.get("private_only"):
        return True
    pid = gpu.get("provider_id")
    return (job.get("user_id") is not None and job.get("user_id") == gpu.get("owner_user_id")) \
        or pid in (job.get("trusted_providers") or ())


def estimate_cost(rate, duration_min):
    """Worst-case cost of a booking: rate per hour x booked minutes / 60, rounded to 4 decimals."""
    return round(float(rate) * float(duration_min) / 60.0, 4)
