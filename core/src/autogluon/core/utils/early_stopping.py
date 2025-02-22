from __future__ import annotations

import copy

import numpy as np
from dataclasses import dataclass
from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION, BINARY
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from autogluon.core.utils.utils import CVSplitter


class AbstractES:
    """
    Abstract early stopping class
    """

    def update(self, cur_round, is_best: bool = False) -> bool:
        raise NotImplementedError

    def early_stop(self, cur_round, is_best: bool = False) -> bool:
        raise NotImplementedError


class NoES(AbstractES):
    """
    Dummy early stopping method that never triggers early stopping
    """

    def update(self, cur_round: int, is_best: bool = False) -> bool:
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round: int, is_best: bool = False) -> bool:
        return False


class SimpleES(AbstractES):
    """
    Implements early stopping with fixed patience

    Parameters
    ----------
    patience : int, default 10
        If no improvement occurs in `patience` rounds or greater, self.early_stop will return True.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_round = 0

    def update(self, cur_round: int, is_best: bool = False) -> bool:
        if is_best:
            self.best_round = cur_round
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round: int, is_best: bool = False) -> bool:
        if is_best:
            return False
        return cur_round - self.best_round >= self.patience


# TODO: Add time component
#  if given a large amount of time and training is fast, should check more rounds before early stopping
# TODO: Incorporate score, rolling window
class AdaptiveES(AbstractES):
    """
    Implements early stopping with adaptive patience

    Patience follows the formula `patience = ax + b`, where `a = adaptive_rate`, `x = round` and `b = adaptive_offset`.
    Patience is only updated when a new `best_round` is observed.

    Patience is adaptively adjusted across training instead of being a fixed value.
    This generally outperforms fixed patience strategies. Examples below:
    1. If the current best_round is 10000, it is reasonable to assume that it could take more than 100 rounds before finding a new best.
    2. If the current best_round is 3, it is unlikely that there will be 100 rounds before finding a new best at round 103.
    In the above examples, a fixed patience of 100 would be too little for round 10000, but too large for round 3.
    However, with `adaptive_rate=0.2`, `adaptive_offset=10`, round 3 would have a patience of ~10, while round 10000 would have a patience of ~2000.

    Parameters
    ----------
    adaptive_rate : float, default 0.3
        The rate of increase in patience.
        Set to 0 to disable, or negative to shrink patience during training.
    adaptive_offset : int, default 10
        The initial patience when cur_round is 0.
    min_patience : int | None, default None
        The minimum value of patience. Ignored if None.
    max_patience : int | None, default None
        The maximum value of patience. Ignored if None.

    Attributes
    ----------
    best_round : int
        The most recent round passed to self.update with `is_best=True`.
        Dictates patience and is used to determine if self.early_stop() returns True.
    patience : int
        If no improvement occurs in `patience` rounds or greater, self.early_stop will return True.
        patience is dictated by the following formula:
        patience = min(self.max_patience, (max(self.min_patience, round(self.best_round * self.adaptive_rate + self.adaptive_offset))))
        Effectively, patience = self.best_round * self.adaptive_rate + self.adaptive_offset, bound by min_patience and max_patience
    """

    def __init__(self, adaptive_rate: float = 0.3, adaptive_offset: int = 10, min_patience: int | None = None, max_patience: int | None = None):
        self.adaptive_rate = adaptive_rate
        self.adaptive_offset = adaptive_offset
        self.min_patience = min_patience
        self.max_patience = max_patience
        self.best_round = 0
        self.patience = self._update_patience(self.best_round)

    def update(self, cur_round: int, is_best: bool = False) -> bool:
        """
        Updates the state of the object. Identical to calling self.early_stop, but if `is_best=True`, it will set `self.best_round=cur_round`.
        If cur_round achieved a new best score, set `is_best=True`.
        Ideally, this should be called every round during training, with the output used to determine if the model should stop training.
        """
        if is_best:
            self.best_round = cur_round
            self.patience = self._update_patience(self.best_round)
        return self.early_stop(cur_round, is_best=is_best)

    def early_stop(self, cur_round: int, is_best: bool = False) -> bool:
        """
        Returns True if (cur_round - self.best_round) equals or exceeds self.patience, otherwise returns False.
        This can be used to indicate if training should stop.
        """
        if is_best:
            return False
        return cur_round - self.best_round >= self.patience

    def _update_patience(self, best_round: int) -> int:
        patience = round(self.adaptive_rate * best_round + self.adaptive_offset)  # ax + b
        if self.min_patience is not None:
            patience = max(self.min_patience, patience)
        if self.max_patience is not None:
            patience = min(self.max_patience, patience)
        return patience


ES_CLASS_MAP = {
    "simple": SimpleES,
    "adaptive": AdaptiveES,
}


@dataclass
class ESOutput:
    early_stop: bool
    is_best: bool
    is_best_or_tie: bool
    score: float


@dataclass
class ESOOFOutput:
    early_stop: bool


# TODO: Should be able to make a really nice unit test of this class
# TODO: Use error instead of score? Probably better.
class ESWrapper:
    def __init__(
        self,
        es: AbstractES,
        score_func: callable,
        best_is_later_if_tie: bool = True,
    ):
        """

        Parameters
        ----------
        es: AbstractES
        score_func: callable
        best_is_later_if_tie : bool, default True
            If True, ties for best will consider the earlier round as best for early stopping, but the later round as best for the value of `self.round_to_use`.
            If False, ties for best will use the earlier round as best for early stopping and for `self.round_to_use`.
        """
        self.es = es
        self.score_func = score_func
        self.best_is_later_if_tie = best_is_later_if_tie

        self.best_score = None
        self.round_to_use = None  # round to use at test time

    def update(self, y: np.ndarray, y_score: np.ndarray, cur_round: int) -> ESOutput:
        score = self.score_func(y, y_score)
        is_best, is_best_or_tie = self._check_is_best(score=score)
        if is_best_or_tie:
            self.round_to_use = cur_round
        if is_best:
            self.best_score = score
        early_stop = self.es.update(cur_round=cur_round, is_best=is_best)
        es_output = ESOutput(
            early_stop=early_stop,
            is_best=is_best,
            is_best_or_tie=is_best_or_tie,
            score=score,
        )
        return es_output

    def _check_is_best(self, score: float) -> tuple[bool, bool]:
        if self.best_score is None:
            is_best = True
            is_best_or_tie = True
        elif score > self.best_score:
            is_best = True
            is_best_or_tie = True
        elif score == self.best_score:
            is_best = False
            is_best_or_tie = self.best_is_later_if_tie
        else:
            is_best = False
            is_best_or_tie = False
        return is_best, is_best_or_tie


# TODO: Can crash during LOO scoring if roc_auc and all same class
# TODO: This isn't really OOF, it is LOO.
# TODO: Should be able to make a really nice unit test of this class
class ESWrapperOOF:
    def __init__(
        self,
        es: AbstractES,
        score_func: callable,
        problem_type,
        best_is_later_if_tie: bool = True,
        use_ts: str = "None",
    ):
        self.es = es
        self.score_func=score_func
        self.best_is_later_if_tie=best_is_later_if_tie
        self.score_func = score_func
        self.y_pred_proba_val_best_oof = None
        self.len_val = None
        self.best_val_metric_oof = None
        self.early_stop_oof = None
        self.early_stopping_wrapper_val_lst = None
        self.early_stop_oof_score_over_time = None
        self.early_stop_oof_score_over_time_avg = None
        self.problem_type = problem_type
        self.use_ts = use_ts
        self.debug = False

    def _init_wrappers(self, y: np.ndarray, y_pred_proba: np.ndarray):
        self._es_template = ESWrapper(es=self.es, score_func=self.score_func, best_is_later_if_tie=self.best_is_later_if_tie)
        self.y_pred_proba_shape = y_pred_proba.shape
        self.len_val = len(y)
        self.early_stop_oof_score_over_time = []
        self.early_stop_oof_score_over_time_avg = []
        self.early_stop_custom_score_over_time = []

        self.early_stop_score_over_time = []




        match self.use_ts:
            case "None":
                self.n_repeats = 5
                self.n_folds = 10

                if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                    _, counts = np.unique(y, return_counts=True)
                    if min(counts) < self.n_folds:
                        if min(counts) == 1:
                            print("Cannot perform 1-fold cross-validation, as there is only 1 sample in a class.")
                            exit(-1)  # try to error out inside of AutoGluon, this should work (?)
                        self.n_folds = min(counts)

                self.spliter = CVSplitter(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=0,
                                          stratify=self.problem_type in PROBLEM_TYPES_CLASSIFICATION)
                self.splits = self.spliter.split(list(range(self.len_val)), y=y)
                self.y_pred_proba_val_best_oof_list = [copy.deepcopy(y_pred_proba) for i in range(self.n_repeats)]

            case "25r2f":
                self.n_repeats = 25
                self.n_folds = 2

                if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                    _, counts = np.unique(y, return_counts=True)
                    if min(counts) == 1:
                        print("Cannot perform 1-fold cross-validation, as there is only 1 sample in a class.")
                        exit(-1)  # try to error out inside of AutoGluon, this should work (?)

                self.spliter = CVSplitter(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=0,
                                          stratify=self.problem_type in PROBLEM_TYPES_CLASSIFICATION)
                self.splits = self.spliter.split(list(range(self.len_val)), y=y)
                self.y_pred_proba_val_best_oof_list = [copy.deepcopy(y_pred_proba) for i in range(self.n_repeats)]

            case "50sT0.67":
                self.n_repeats = 50
                self.n_folds = 1  # higher nuber means more overlap of writing to OOF
                # n_folds * n_repeats = n_splits given to spliter below
                n_splits = 50

                if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                    self.spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.67, random_state=0)
                else:
                    self.spliter = ShuffleSplit(n_splits=n_splits, test_size=0.67, random_state=0)
                self.splits = list(self.spliter.split(list(range(self.len_val)), y=y))
                self.y_pred_proba_val_best_oof_list = [np.full_like(y_pred_proba, np.nan) for _ in range(self.n_repeats)]

            case "50sT0.67Mix":
                self.n_repeats = 25
                self.n_folds = 2
                n_splits = 50

                if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
                    self.spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.67, random_state=0)
                else:
                    self.spliter = ShuffleSplit(n_splits=n_splits, test_size=0.67, random_state=0)
                self.splits = list(self.spliter.split(list(range(self.len_val)), y=y))

                self.y_pred_proba_val_best_oof_list = [copy.deepcopy(y_pred_proba) for _ in range(self.n_repeats)]

            case _:
                raise ValueError(f"Unknown use_ts: {self.use_ts}")


        self.y_pred_proba_val_best_oof = copy.deepcopy(y_pred_proba)
        self.y_pred_proba_val_best_oof_fallback = copy.deepcopy(y_pred_proba)
        # self.y_pred_proba_val_best_oof = self.y_pred_proba_val_best_oof.astype(np.float64) # TODO this might be needed
        self.n_splits = len(self.splits)
        self.early_stopping_wrapper_val_lst: list[ESWrapper] = [ESWrapper(es=self.es, score_func=self.score_func, best_is_later_if_tie=self.best_is_later_if_tie) for _ in range(self.n_splits)]
        self.best_val_metric_oof = [[] for i in range(self.n_splits)]  # higher = better
        self.early_stop_oof = np.zeros(self.n_splits, dtype=np.bool_)


    @property
    def round_to_use(self) -> np.ndarray:
        return np.array([es.round_to_use for es in self.early_stopping_wrapper_val_lst])

    # FIXME: docstring
    # FIXME: y_pred_proba isn't the right name. But what should it be? It is the correct prediction format for stacker inputs.
    def update(self, y: np.ndarray, y_score: np.ndarray, cur_round: int, y_pred_proba: np.ndarray) -> ESOOFOutput:
        if self.y_pred_proba_val_best_oof is None:
            self._init_wrappers(y=y, y_pred_proba=y_pred_proba)

        early_stop = True
        for i, (val_idx, oof_idx) in enumerate(self.splits):
            if not self.early_stop_oof[i]:
                y_i = y[val_idx]
                y_score_i = y_score[val_idx]

                es_output = self.early_stopping_wrapper_val_lst[i].update(
                    y=y_i,
                    y_score=y_score_i,
                    cur_round=cur_round,
                )
                self.early_stop_oof[i] = es_output.early_stop
                if not es_output.early_stop:
                    early_stop = False
                # update best validation
                if es_output.is_best_or_tie:
                    self.best_val_metric_oof[i].append(self.early_stopping_wrapper_val_lst[i].best_score)
                    self.y_pred_proba_val_best_oof_list[int((i - (i % self.n_folds)) / self.n_folds)][oof_idx] = y_pred_proba[oof_idx]

        if len(self.y_pred_proba_val_best_oof_list) == 1:
            self.y_pred_proba_val_best_oof = self.y_pred_proba_val_best_oof_list[0]
        else:
            self.y_pred_proba_val_best_oof = np.nanmean(self.y_pred_proba_val_best_oof_list, axis=0)

            if np.isnan(self.y_pred_proba_val_best_oof).any():
                self.y_pred_proba_val_best_oof  = np.where(np.isnan(self.y_pred_proba_val_best_oof), self.y_pred_proba_val_best_oof_fallback, self.y_pred_proba_val_best_oof )

        # if early_stop:
        #     # Final corrections
        #     es_oof_score = self.score_func(y, self.y_pred_proba_val_best_oof)
        #     # es_oof_score =  float(np.mean(self.best_val_metric_oof))
        #     self.early_stop_oof_score_over_time.append(float(es_oof_score))
        #
        if self.debug:
            es_oof_score = self._es_template.score_func(y, self.y_pred_proba_val_best_oof)
            self.early_stop_oof_score_over_time.append(es_oof_score)
            self.early_stop_oof_score_over_time_avg.append(np.mean([i[-1] for i in self.best_val_metric_oof]))
        # print(f"round: {cur_round}, es_oof_score: {es_oof_score}, no-update% {no_updated_count/self.len_val}, {np.mean(self.early_stop_oof)}")

        # # Custom new version of LOO ES
        # from scipy.stats import mannwhitneyu
        # y_pred_proba_val_best_oof_custom = copy.deepcopy(y_pred_proba)
        # true_history = np.array(val_metric_over_time)
        # true_history = true_history[:np.argmax(true_history) + 1]
        # for i in range(self.len_val):
        #     loo_history = self.best_val_metric_oof[i]
        #     _, pvalue = mannwhitneyu(loo_history, true_history)
        #
        #     if pvalue >= 0.5: # 50 % confidence
        #         # no significant difference
        #         y_pred_proba_val_best_oof_custom[i] = best_y_pred_proba_val[i]
        #     else:
        #         # significant difference
        #         y_pred_proba_val_best_oof_custom[i] = self.y_pred_proba_val_best_oof[i]
        # self.early_stop_custom_score_over_time.append(self._es_template.score_func(y, y_pred_proba_val_best_oof_custom))

        # if (self.problem_type in PROBLEM_TYPES_CLASSIFICATION) and ((self.use_ts and early_stop) or self.debug):
        #     from probmetrics.calibrators import get_calibrator
        #     is_binary = self.problem_type == BINARY
        #
        #     if is_binary:
        #         from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
        #         y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(self.y_pred_proba_val_best_oof) # self.y_pred_proba_val_best_oof
        #     calib = get_calibrator('temp-scaling', calibrate_with_mixture=False)
        #     calib.fit(y_pred_proba, y)
        #     y_pred_proba_val_best_oof_custom = calib.predict_proba(y_pred_proba)
        #     if is_binary:
        #         y_pred_proba_val_best_oof_custom = y_pred_proba_val_best_oof_custom[:, 1]
        #     self.early_stop_custom_score_over_time.append(self._es_template.score_func(y, y_pred_proba_val_best_oof_custom))
        #
        #     if self.use_ts:
        #         self.y_pred_proba_val_best_oof = y_pred_proba_val_best_oof_custom

        if self.problem_type in PROBLEM_TYPES_CLASSIFICATION:
            # Fix precision errors
            if early_stop:

                is_binary = self.problem_type == BINARY

                if is_binary:
                    from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
                    y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(self.y_pred_proba_val_best_oof) # self.y_pred_proba_val_best_oof
                else:
                    y_pred_proba = self.y_pred_proba_val_best_oof

                if (not np.allclose(y_pred_proba.sum(axis=1), 1, rtol=np.sqrt(np.finfo(y_pred_proba.dtype).eps))):
                    y_pred_proba /= np.sum(y_pred_proba, axis=1)[:, np.newaxis]

                if is_binary:
                    y_pred_proba = y_pred_proba[:, 1]

                self.y_pred_proba_val_best_oof = y_pred_proba

        return ESOOFOutput(early_stop=early_stop)
