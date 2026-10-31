from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.Result import Result
from pyvrp.RingBuffer import RingBuffer
from pyvrp.SoftTimeWindowPenalty import (
    SoftTimeWindowParams,
    soft_time_window_penalty,
)
from pyvrp.Statistics import Statistics

if TYPE_CHECKING:
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp._pyvrp import ProblemData, RandomNumberGenerator, Solution
    from pyvrp.search.SearchMethod import SearchMethod
    from pyvrp.stop.StoppingCriterion import StoppingCriterion


@dataclass
class IteratedLocalSearchParams:
    """
    Parameters for the iterated local search algorithm.

    Parameters
    ----------
    num_iters_no_improvement
        Number of iterations without improvement before a restart occurs.
    history_length
        History length for the late acceptance hill-climbing stopping criterion
        used by the algorithm. Must be positive.
    exhaustive_on_best
        Whether to perform a more expensive, exhaustive search for newly found
        best solutions.
    soft_tw
        Optional soft time window parameters. When set, an additional
        penalty is added (on top of the regular cost) for candidate, current,
        and best solutions whenever comparing them during the search, to
        discourage schedules with arrivals close to a client's time window
        deadline. This does not affect the neighbourhood search itself, only
        which of its outcomes this algorithm prefers. Disabled by default.
    """

    num_iters_no_improvement: int = 150_000
    history_length: int = 300
    exhaustive_on_best: bool = True
    soft_tw: SoftTimeWindowParams | None = None

    def __post_init__(self):
        if self.num_iters_no_improvement < 0:
            raise ValueError("num_iters_no_improvement < 0 not understood.")

        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")


class IteratedLocalSearch:
    """
    Creates an IteratedLocalSearch instance.

    Parameters
    ----------
    data
        The problem data instance.
    penalty_manager
        Penalty manager to use.
    rng
        Random number generator.
    search_method
        Search method to use.
    initial_solution
        Initial solution to start the search with.
    params
        Iterated local search parameters to use. If not provided, a default
        will be used.
    """

    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: RandomNumberGenerator,
        search_method: SearchMethod,
        initial_solution: Solution,
        params: IteratedLocalSearchParams = IteratedLocalSearchParams(),
    ):
        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._search = search_method
        self._init = initial_solution
        self._params = params

    def run(
        self,
        stop: StoppingCriterion,
        collect_stats: bool = True,
        display: bool = False,
        display_interval: float = 5.0,
    ) -> Result:
        """
        Runs the iterated local search algorithm with the provided stopping
        criterion. The algorithm uses late acceptance hill-climbing as the
        acceptance criterion; see [1]_ for details.

        Parameters
        ----------
        stop
            Stopping criterion to use. The algorithm runs until the first time
            the stopping criterion returns ``True``.
        collect_stats
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display
            Whether to display information about the solver progress. Default
            ``False``. Progress information is only available when
            ``collect_stats`` is also set.
        display_interval
            Time (in seconds) between iteration logs. Defaults to 5s.

        Returns
        -------
        Result
            A Result object, containing statistics (if collected) and the best
            found solution.

        References
        ----------
        .. [1] Burke, E.K., and Y. Bykov (2017). The Late Acceptance
               Hill-Climbing Heuristic. *European Journal of Operational
               Research*, 258(1): 70 - 78.
               https://doi.org/10.1016/j.ejor.2016.07.012.
        """
        print_progress = ProgressPrinter(display, display_interval)
        print_progress.start(self._data)

        soft_tw = self._params.soft_tw

        # The soft time window penalty is computed in Python over every visit
        # and was evaluated about six times per iteration, although only the
        # candidate is new: best, curr and the late-acceptance reference are
        # immutable solutions whose penalty cannot change. On 2000 clients
        # this made a solve up to 27 times slower. The penalty is therefore
        # cached per solution object. The cache holds a reference to each
        # solution, so its id cannot be reused while it is cached, and a miss
        # just recomputes the penalty, so results do not depend on the cache.
        penalty_cache: OrderedDict[int, tuple[Solution, int]] = OrderedDict()
        cache_size = 2 * self._params.history_length + 16

        def soft_penalty(sol: Solution) -> int:
            key = id(sol)
            hit = penalty_cache.get(key)
            if hit is not None and hit[0] is sol:
                penalty_cache.move_to_end(key)
                return hit[1]

            penalty = soft_time_window_penalty(sol, self._data, soft_tw)
            penalty_cache[key] = (sol, penalty)
            if len(penalty_cache) > cache_size:
                penalty_cache.popitem(last=False)
            return penalty

        def augmented_cost(base_cost: int, sol: Solution) -> int:
            # Adds the soft time window penalty (if configured) on top of a
            # cost value already computed by the cost evaluator. This lets
            # the outer search prefer solutions with fewer arrivals close to
            # a deadline, without changing how the neighbourhood search
            # itself evaluates moves.
            if soft_tw is None:
                return base_cost

            return base_cost + soft_penalty(sol)

        history: RingBuffer[Solution] = RingBuffer(self._params.history_length)
        stats = Statistics(collect_stats=collect_stats)

        start = time.perf_counter()
        iters = iters_no_improvement = 0
        best = curr = self._init

        cost_eval = self._pm.cost_evaluator()
        while not stop(cost_eval.cost(best)):
            iters += 1

            if iters_no_improvement == self._params.num_iters_no_improvement:
                print_progress.restart()
                history.clear()

                curr = best
                iters_no_improvement = 0

            cost_eval = self._pm.cost_evaluator()
            cand = self._search(curr, cost_eval, exhaustive=False)
            self._pm.register(cand)

            iters_no_improvement += 1
            cand_best_cost = augmented_cost(cost_eval.cost(cand), cand)
            best_cost = augmented_cost(cost_eval.cost(best), best)
            if cand_best_cost < best_cost:
                best = cand
                iters_no_improvement = 0

                if self._params.exhaustive_on_best:
                    # Candidate is already a new (global) best, but let's see
                    # if we can improve it via an exhaustive search. That new
                    # candidate solution might be infeasible, so we need to
                    # check before updating best.
                    cand = self._search(cand, cost_eval, exhaustive=True)
                    if cand.is_feasible():
                        best = cand

            cand_cost = augmented_cost(cost_eval.penalised_cost(cand), cand)
            curr_cost = augmented_cost(cost_eval.penalised_cost(curr), curr)

            # We use either the initial cost or the current cost value from
            # some iterations ago to determine whether to accept the candidate
            # solution, if available.
            late_cost = augmented_cost(
                cost_eval.penalised_cost(self._init), self._init
            )
            if (late := history.peek()) is not None:
                late_cost = augmented_cost(
                    cost_eval.penalised_cost(late), late
                )

            # Late-acceptance hill climbing of Burke and Bykov (2017). We use
            # both enhancements of section 4.2:
            # 1. We accept also when the candidate improves over the current
            #    solution;
            if cand_cost < late_cost or cand_cost < curr_cost:
                curr = cand
                curr_cost = cand_cost

            # 2. We update the history only when the current solution is better
            #    than the one already in the history.
            if curr_cost < late_cost or late is None:
                history.append(curr)
            else:
                history.skip()

            stats.collect(curr, cand, best, cost_eval)
            print_progress.iteration(stats)

        runtime = time.perf_counter() - start
        res = Result(best, stats, iters, runtime)

        print_progress.end(res)

        return res
