"""E1: priority queue. Order = highest priority first (SP1), earlier submission breaks ties (FR4)."""
import heapq


class JobQueue:
    """Binary min-heap keyed on (-priority, submitted_at, job_id).
    heapify is O(n); push / pop are O(log n). No full re-sort is ever done."""

    def __init__(self, jobs=()):
        self._h = [self._key(j) for j in jobs]       # the heap stores only small sort keys
        heapq.heapify(self._h)                       # rearrange the list into heap order in O(n)
        self._jobs = {j["job_id"]: j for j in jobs}

    @staticmethod
    def _key(job):
        # Python's heap is a MIN-heap, so priority is negated: priority 5 becomes -5 and comes out before priority 1.
        # submitted_at then prefers the older job; job_id makes every key unique so ties never compare dicts.
        return (-job["priority"], job["submitted_at"], job["job_id"])

    def push(self, job):
        self._jobs[job["job_id"]] = job
        heapq.heappush(self._h, self._key(job))

    def pop(self):
        """Remove and return the best job (smallest key)."""
        _, _, job_id = heapq.heappop(self._h)
        return self._jobs.pop(job_id)

    def __len__(self):
        return len(self._h)
