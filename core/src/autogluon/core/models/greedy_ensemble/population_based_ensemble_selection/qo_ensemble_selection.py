import logging
import time
from functools import partial
from typing import List

import numpy as np

# FIXME: remove the ribs==0.4.0 dependency (requires re-implementing archive logic and emitter logic...)
from ribs.archives._archive_base import ArchiveBase
from ribs.optimizers import Optimizer

from ....constants import PROBLEM_TYPES
from ..ensemble_selection import AbstractWeightedEnsemble
from .emitters import DiscreteWeightSpaceEmitter

logger = logging.getLogger(__name__)


class QualityArchive(ArchiveBase):
    """A baseline archive that is not a typical QDO archive.

    It functions like a pure quality population. Thus, allowing us to compare diversity inspired populations of elites
    to simple quality archives. It is a collection of N items w.r.t. some simple characteristics.
    Usually, it is the top N  performing solutions.

    We assume that we want to maximize the objective following the default QDO implementation.

    Parameters
    ----------
    archive_size: int
        Size of the beam.
    seed: int
        Seed for the random number generator of archive base. (Beam is not random)
    """

    def __init__(self, archive_size, seed):
        self._dims = [archive_size]
        self.archive_size = archive_size
        self.stored_values_count = 0

        ArchiveBase.__init__(self, storage_dims=tuple(self._dims), behavior_dim=0, seed=seed)

    def get_index(self, behavior_values):
        """Return the index of the position in the archive that should be replaced by the behavior values"""

        # Fill empty bin
        if len(self._occupied_indices) < self.archive_size:
            return tuple(np.unravel_index(np.argmin(self._occupied), self._occupied.shape))

        # Find bin with the smallest value and return (local competition will resolve the if it is inserted or not)
        return tuple(np.unravel_index(np.argmin(self._objective_values), self._objective_values.shape))

    def add(self, solution, objective_value, behavior_values, metadata=None):
        status, value = ArchiveBase.add(self, solution, objective_value, behavior_values, metadata)
        self.stored_values_count += 1
        return status, value


class QOEnsembleSelection(AbstractWeightedEnsemble):
    """Using QO-ES to find a good weight vector for post hoc ensembling.

    Parameters
    ----------
    n_iterations: int
        The number of iterations determines the number of evaluations. By default, number of evaluations
        = n_iterations * n_base_models. The true number of iterations is also determined by the batch size.
        If n_base_models == batch_size, then used n_iterations is the value that is passed. Otherwise, we need to adapt
        the number of iterations.
    population_size: int, default=49
        Defines the size of the population for the evolutionary search.
    batch_size: int, default=20
        Defines the batch size for ask/tell loop.
    emitter_initialization_method: str in
                {"AllL1", "RandomL2Combinations", "L2ofSingleBest"}, default="AllL1"
        Defines the initialization method for the first batch of solutions. That is, the set of initial weight
        vectors.
        *   "AllL1": The first batch of solutions are all base models. That is, the weight vector is a one-hot vector.
        *   "RandomL2Combinations": The first batch of solutions contains random combination of base models.
            As a result, it will start with ensembles of size 2, i.e., Layer 2 (L2). The combinations are done
            exhaustively such that we will produce n_base_models/2 many solutions.
        *   "L2ofSingleBest": Extend the single best model to an L2 ensemble for all possible combinations. Produces
            n_base_models-1 many initial solutions.
    """

    def __init__(
        self,
        n_iterations: int,
        problem_type: str,
        metric,
        random_state: np.random.RandomState = None,
        batch_size: int = 20,
        population_size: int = 50,
        emitter_initialization_method: str = "L2ofSingleBest",
        **kwargs,
    ):
        self.n_iterations = int(n_iterations)
        self.problem_type = problem_type
        self.metric = metric
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=0)

        self.quantile_levels = kwargs.get("quantile_levels", None)

        if self.n_iterations < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if not self.problem_type in PROBLEM_TYPES:
            raise ValueError("Unknown problem type %s." % self.problem_type)

        # QO Vars
        self.batch_size = batch_size
        self.population_size = population_size
        self.emitter_initialization_method = emitter_initialization_method
        self._n_init_evals = 0
        self.init_iteration_results = None  # type: Optional[List[float, np.ndarray]]

    def fit(self, predictions, labels, time_limit=None, sample_weight=None):
        self.n_base_models = len(predictions)
        self.sample_weight = sample_weight
        self.n_evaluations = self.n_iterations * self.n_base_models
        self.internal_n_iterations, self.n_rest_evaluations = _compute_n_iterations(self.n_evaluations, self.batch_size)

        # -- Init QDO Elements
        self.archive = QualityArchive(
            archive_size=self.population_size,
            seed=self.random_state.randint(1, 100000),
        )

        self._init_emitters(predictions, labels)

        # -- Call optimizer
        self._optimize(predictions, labels, time_limit)

        # -- Get final weights
        self._compute_weights(predictions, labels)

        logger.log(15, "Ensemble weights: ")
        logger.log(15, self.weights_)
        return self

    def _init_start_weight_vectors(self, predictions: List[np.ndarray], y_true: np.ndarray):
        """Initialize the weight vectors for the first batch of solutions to be evaluated.

        We must set self.init_iteration_results if the start weight vectors do not include the Single Best.
        """

        # -- Evaluate Base Models and to determine start weight vectors
        base_weight_vector = np.zeros((self.n_base_models,))
        _start_weight_vectors = []

        # -- Get L1 Weight Vectors
        for i in range(self.n_base_models):
            tmp_w = base_weight_vector.copy()
            tmp_w[i] = 1
            _start_weight_vectors.append((tmp_w, 1))

        # -- Switch case for different initialization methods
        if self.emitter_initialization_method != "AllL1":

            # -- Get Single Best to guarantee performance improvement based on validation data
            objs, _ = self._evaluate_batch_of_solutions(np.array([w for w, _ in _start_weight_vectors]), predictions, y_true)
            sb_weight_vector = _start_weight_vectors[np.argmax(objs)]
            self._update_n_iterations(self.n_base_models)

            # -- Switch case
            if self.emitter_initialization_method == "RandomL2Combinations":
                self.init_iteration_results = (np.max(objs), sb_weight_vector[0])

                l2_combinations = self.random_state.choice(self.n_base_models, (self.n_base_models // 2, 2), replace=False)
                # Fill combinations
                _start_weight_vectors = []
                for i, j in l2_combinations:
                    tmp_w = base_weight_vector.copy()
                    tmp_w[i] = 0.5
                    tmp_w[j] = 0.5
                    _start_weight_vectors.append((tmp_w, 2))

            elif self.emitter_initialization_method == "L2ofSingleBest":
                self.init_iteration_results = (np.max(objs), sb_weight_vector[0])
                index_single_best = np.argmax(objs)

                # Fill combinations
                _start_weight_vectors = []
                for i in range(self.n_base_models):
                    if i == index_single_best:
                        continue
                    tmp_w = base_weight_vector.copy()
                    tmp_w[i] = 0.5
                    tmp_w[index_single_best] = 0.5
                    _start_weight_vectors.append((tmp_w, 2))

        # If it is "AllL1" we do need to anything and just return all of them
        return _start_weight_vectors

    def _update_n_iterations(self, n_evals_outside_of_loop: int):
        # Set parameter for logging to see how many values were evaluated outside the loop
        self._n_init_evals += n_evals_outside_of_loop
        # Set number of iterations to reflect this and to guarantee that we do not exceed the allowed
        # number of evaluations
        self.internal_n_iterations, self.n_rest_evaluations = _compute_n_iterations(self.n_evaluations - self._n_init_evals, self.batch_size)

    def _init_emitters(self, predictions, y_true):
        _start_weight_vectors = self._init_start_weight_vectors(predictions, y_true)

        emitters = [
            DiscreteWeightSpaceEmitter(
                self.archive,
                self.n_base_models,
                _start_weight_vectors,
                batch_size=self.batch_size,
                seed=self.random_state.randint(1, 100000),
                # As this is an evolutionary search, there are many possible parameters here.
                # I selected some I think are reasonable for the use case at hand.
                # If I take another look at the results from the QDO paper, we might find the best possible preset.
                elite_selection_method="combined_dynamic",  # questionable if this is best
                crossover="two_point_crossover",  # questionable if this is best
                crossover_probability=0.5,
                mutation_probability_after_crossover=0.5,
                crossover_probability_dynamic=True,
                mutation_probability_after_crossover_dynamic=True,
            )
        ]
        self.emitters = emitters

    def _optimize(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None) -> None:
        time_start = time.time()
        time_left = 1

        # -- Build Optimizer
        opt = Optimizer(self.archive, self.emitters)

        # -- Set up Results collector
        optimize_stats = {"Archive Size": [], "Max Objective": []}
        if self.init_iteration_results is not None:
            optimize_stats["Archive Size"].append(0)
            optimize_stats["Max Objective"].append(self.init_iteration_results[0])

        for itr in range(1, self.internal_n_iterations + 1):
            # Get solutions
            sols = opt.ask()

            # Evaluate solutions
            objs, bcs = self._evaluate_batch_of_solutions(sols, predictions, labels)

            # Report back and restart
            opt.tell(objs, bcs)

            # Log stats
            optimize_stats["Archive Size"].append(len(opt.archive))
            optimize_stats["Max Objective"].append(float(opt.archive.stats.obj_max))

            if time_limit is not None:
                time_elapsed = time.time() - time_start
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.warning(
                        "Warning: Ensemble Selection ran out of time, early stopping at iteration %s. This may mean that the time_limit specified is very small for this problem."
                        % itr
                    )
                    break

        # -- Rest Iteration, required if wanted number of evaluations can not be evenly
        #    distributed across batches.
        if self.n_rest_evaluations and (time_left > 0):
            sols = opt.ask()
            org_length = len(sols)
            sols = sols[: self.n_rest_evaluations, :]

            objs, bcs = self._evaluate_batch_of_solutions(sols, predictions, labels)

            # Get some existing behavior values and a worse objective value
            dummy_obj = objs[0]
            dummy_bc = bcs[0, :]

            # fill rest of the solutions with dummy values
            objs = np.hstack([objs, [dummy_obj] * (org_length - len(sols))])
            bcs = np.vstack([bcs, [dummy_bc] * (org_length - len(sols))])

            opt.tell(objs, bcs)
            optimize_stats["Archive Size"].append(len(opt.archive))
            optimize_stats["Max Objective"].append(float(opt.archive.stats.obj_max))

        self.optimize_stats_ = optimize_stats

    def _evaluate_batch_of_solutions(self, solutions: np.ndarray, predictions, y_true):
        """Return objective value

        Parameters
        ----------
        solutions: np.ndarray (batch_size, n_base_models)
            A batch of weight vectors.

        Returns
        -------
            objs (np.ndarray): (batch_size,) array with objective values.
            bcs (np.ndarray): (batch_size,) array with a BC in each row.
        """

        # Create static function arguments list
        func_args = [predictions, y_true]

        res = np.apply_along_axis(partial(self.evaluate_single_solution, *func_args), axis=1, arr=solutions)

        return res[:, 0], res[:, 1:]

    def evaluate_single_solution(self, predictions, y_true, weight_vector):
        # Get Score
        y_pred_ensemble = self.weight_pred_probas(predictions, weight_vector)

        # Negative loss because we want to maximize
        s_m = -self._calculate_regret(y_true, y_pred_ensemble, self.metric, sample_weight=self.sample_weight)

        return s_m, *[]

    def _compute_weights(self, predictions, labels):
        """Code to compute the final weight vector (among other things)

        Code does the following:
            1. First it tests if merging all found solutions is better than the single
                best solution. If yes, the merged solution is used.
            2. Second if verifies (if needed) if the best found solution improved
                over the single best base model and returns the single best
                base model's results.
            3. Lastly, it fills metadat about the training process such that we can analysis them later.
        """
        # -- Get Best Weight Vector
        elites = [elite for elite in self.archive]
        performances = [e.obj for e in elites]

        # - Get merged weights
        disc_weights = np.array([elite.sol * elite.meta for elite in elites])
        merged_weights = np.sum(disc_weights, axis=0) / np.sum(np.sum(disc_weights, axis=1))

        # Get performance for merged weights
        merge_obj = self._evaluate_batch_of_solutions(np.array([merged_weights]), predictions, labels)[0][0]

        # max/Argmax because we made this a maximization problem to work with ribs
        if merge_obj > np.max(performances):
            self.optimize_stats_["merged_weights"] = True
            self.weights_ = merged_weights
            self.validation_loss_ = -float(merge_obj)
        else:
            self.optimize_stats_["merged_weights"] = False
            self.weights_ = elites[np.argmax(performances)].sol
            self.validation_loss_ = -float(np.max(performances))

        # -- Verify that the optimization method improved over the single best
        #   Only done if the method does/can not do this by itself
        if self.init_iteration_results is not None:
            init_score, init_weights = self.init_iteration_results

            # >= because ini iteration has most likely a smaller ensemble size
            if self.validation_loss_ >= -float(init_score):
                self.optimize_stats_["merged_weights"] = False
                self.weights_ = init_weights
                self.validation_loss_ = -float(init_score)

        # -- Set to save metadata
        self.iteration_batch_size_ = self.batch_size
        self.val_loss_over_iterations_ = [-i for i in self.optimize_stats_["Max Objective"]]

        # - set additional metadata
        self.model_specific_metadata_ = dict(
            evaluation_types=dict(
                total=int(self.n_base_models * self.n_iterations),
                explore=sum(int(em.explore) for em in self.emitters),
                exploit=sum(int(em.exploit) for em in self.emitters),
                init=int(self._n_init_evals),
                rejects=sum(int(em._total_rejects) for em in self.emitters),
                crossover_rejects=sum(int(em._total_crossover_rejects) for em in self.emitters),
                n_mutate=sum(int(em.n_mutate) for em in self.emitters),
                n_crossover=sum(int(em.n_crossover) for em in self.emitters),
            ),
            internal_n_iterations=int(self.internal_n_iterations) + int(self.n_rest_evaluations > 0),
            archive_size=[int(i) for i in self.optimize_stats_["Archive Size"]],
        )


def _compute_n_iterations(n_eval, batch_size):
    internal_n_iterations = n_eval // batch_size

    if n_eval % batch_size == 0:
        n_rest_evaluations = 0
    else:
        n_rest_evaluations = n_eval % batch_size

    return internal_n_iterations, n_rest_evaluations
