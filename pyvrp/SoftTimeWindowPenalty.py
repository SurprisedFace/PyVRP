from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvrp._pyvrp import ProblemData, Solution


@dataclass
class SoftTimeWindowParams:
    """
    Parameters for soft time window penalties near the end of a client's
    time window, following Blauth et al. (2024) [1]_.

    Parameters
    ----------
    bands
        Default penalty bands, given as a list of ``(duration, penalty)``
        tuples, used for every client not listed in ``client_bands``. A
        client whose service starts within ``duration`` time units before its
        ``tw_late`` deadline (but not after it, which is instead handled by
        the regular time warp penalty) incurs the associated ``penalty``.
        When several bands apply, the one with the smallest ``duration``
        (i.e. closest to the deadline) is used. Durations must be distinct
        and positive; penalties must be non-negative. Defaults to the bands
        used by Blauth et al.: $1 within 15 min, $2 within 10 min, and $4
        within 5 min of the deadline.
    client_bands
        Optional per-client override, keyed by client location index: a
        client present here uses its own band list instead of ``bands``.
        Each band list is validated the same way as ``bands``. Empty by
        default, so every client falls back to the shared ``bands`` -- the
        original, single-band-list behaviour.

    References
    ----------
    .. [1] Blauth et al. (2024). Vehicle routing with time-dependent travel times: Theory, practice,
and benchmarks.
    """

    bands: list[tuple[int, int]] = field(
        default_factory=lambda: [(15, 1), (10, 2), (5, 4)]
    )
    client_bands: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def __post_init__(self):
        self.bands = self._validate(self.bands)
        self.client_bands = {
            location: self._validate(bands)
            for location, bands in self.client_bands.items()
        }

    @staticmethod
    def _validate(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
        durations = [duration for duration, _ in bands]

        if any(duration <= 0 for duration in durations):
            raise ValueError("Band durations must be positive.")

        if any(penalty < 0 for _, penalty in bands):
            raise ValueError("Band penalties must be non-negative.")

        if len(set(durations)) != len(durations):
            raise ValueError("Band durations must be distinct.")

        # Sorted ascending, so the tightest (smallest duration) band is
        # matched first in penalty_for().
        return sorted(bands)

    def penalty_for(self, location: int, slack: int) -> int:
        """
        Returns the penalty for a client at the given location index whose
        service starts ``slack`` time units before its deadline, using that
        client's own bands if it has one in ``client_bands``, else the
        shared ``bands``. ``slack`` must be non-negative: arriving after the
        deadline is a hard constraint violation (time warp), handled
        elsewhere.
        """
        for duration, penalty in self.client_bands.get(location, self.bands):
            if slack < duration:
                return penalty

        return 0


def soft_time_window_penalty(
    solution: Solution, data: ProblemData, params: SoftTimeWindowParams
) -> int:
    """
    Computes the total soft time window penalty of the given solution: the
    sum, over all client visits, of the penalty incurred for starting
    service close to (but not after) that client's time window deadline.
    Visits that already incur (hard) time warp are skipped, since those are
    penalised separately by the regular time warp penalty.
    """
    total = 0
    num_depots = data.num_depots

    for route in solution.routes():
        for visit in route.schedule():
            if visit.location < num_depots or visit.time_warp > 0:
                continue  # depot visit, or already time-warped

            client = data.location(visit.location)
            slack = client.tw_late - visit.start_service
            total += params.penalty_for(visit.location, slack)

    return total
