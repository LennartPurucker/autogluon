import gc
import logging
import os
import random
import re
import time
import warnings

import numpy as np
from pandas import DataFrame, Series

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_lightgbm
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds

from . import lgb_utils
from .hyperparameters.parameters import DEFAULT_NUM_BOOST_ROUND, get_lgb_objective, get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .lgb_utils import construct_dataset, train_lgb_model

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
logger = logging.getLogger(__name__)


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class LGBModel(AbstractModel):
    """
    LightGBM model: https://lightgbm.readthedocs.io/en/latest/

    Hyperparameter options: https://lightgbm.readthedocs.io/en/latest/Parameters.html

    Extra hyperparameter options:
        ag.early_stop : int, specifies the early stopping rounds. Defaults to an adaptive strategy. Recommended to keep default.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._features_internal_map = None
        self._features_internal_list = None
        self._requires_remap = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type)

    # Use specialized LightGBM metric if available (fast), otherwise use custom func generator
    def _get_stopping_metric_internal(self):
        stopping_metric = lgb_utils.convert_ag_metric_to_lgbm(ag_metric_name=self.stopping_metric.name,
                                                              problem_type=self.problem_type)
        if stopping_metric is None:
            stopping_metric = lgb_utils.func_generator(
                metric=self.stopping_metric, is_higher_better=True,
                needs_pred_proba=not self.stopping_metric.needs_pred, problem_type=self.problem_type
            )
            stopping_metric_name = self.stopping_metric.name
        else:
            stopping_metric_name = stopping_metric
        return stopping_metric, stopping_metric_name

    def _estimate_memory_usage(self, X, **kwargs):
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem
        data_mem_usage = get_approximate_df_mem_usage(X).sum()
        approx_mem_size_req = data_mem_usage * 7 + data_mem_usage / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved
        return approx_mem_size_req

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=0, sample_weight=None,
             sample_weight_val=None, verbosity=2, **kwargs):


        try_import_lightgbm()  # raise helpful error message if LightGBM isn't installed
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        self._stacking_dropout = params.pop("stacking_dropout", False)
        self._stacking_dropout_per = params.pop("stacking_dropout_per", 0.5)
        self._threshold_norm = params.pop("threshold_norm", False)


        if self._threshold_norm:
            def _find_optimal_threshold(proba):
                from sklearn.metrics import balanced_accuracy_score

                threshold_pos = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

                def proba_to_binary(t):
                    return np.where(proba >= t, 1, 0)

                tf = threshold_pos[np.argmax([balanced_accuracy_score(y, proba_to_binary(ti)) for ti in threshold_pos])]
                return tf

            stack_f = self.feature_metadata.get_features(required_special_types=['stack'])

            for f in stack_f:
                opt_tf = _find_optimal_threshold(X[f])

                print()


        if self._stacking_dropout:
            import math

            import pandas as pd
            rng = np.random.RandomState(int(self.name.replace("F", "").replace("S", "")))

            _f_sel = self.feature_metadata.get_features(required_special_types=['stack'])
            # no_stack_f = self.feature_metadata.get_features(invalid_special_types=['stack'])
            # _f_sel = list(rng.choice(stack_f, math.floor(len(stack_f)*0.95), replace=False))
            # _f_sel += list(rng.choice(no_stack_f, math.floor(len(no_stack_f)*0.5), replace=False))
            # _f_sel = X.columns

            X = X.copy()
            more_X = X.copy()
            for f in _f_sel:
                i_sel = rng.choice(X.index, math.floor(len(X.index) * 0.67), replace=False)
                X.loc[i_sel, f] = -1
                i_sel = rng.choice(X.index, math.floor(len(X.index) * 0.67), replace=False)
                more_X.loc[i_sel, f] = -2
            X = pd.concat([X, more_X], axis=0).reset_index(drop=True)
            y = pd.concat([y, y]).reset_index(drop=True)
            shuffle_i = rng.choice(X.index, len(X.index), replace=False)
            X = X.loc[shuffle_i].reset_index(drop=True)
            y = y.loc[shuffle_i].reset_index(drop=True)

        if verbosity <= 1:
            log_period = False
        elif verbosity == 2:
            log_period = 1000
        elif verbosity == 3:
            log_period = 50
        else:
            log_period = 1

        stopping_metric, stopping_metric_name = self._get_stopping_metric_internal()

        num_boost_round = params.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND)
        dart_retrain = params.pop("dart_retrain",
                                  False)  # Whether to retrain the model to get optimal iteration if model is trained in 'dart' mode.
        if num_gpus != 0:
            if "device" not in params:
                # TODO: lightgbm must have a special install to support GPU: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version
                #  Before enabling GPU, we should add code to detect that GPU-enabled version is installed and that a valid GPU exists.
                #  GPU training heavily alters accuracy, often in a negative manner. We will have to be careful about when to use GPU.
                params["device"] = "gpu"
                logger.log(20,
                           f"\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.")
        logger.log(15, f"\tFitting {num_boost_round} rounds... Hyperparameters: {params}")

        if "num_threads" not in params:
            params["num_threads"] = num_cpus
        if "objective" not in params:
            params["objective"] = get_lgb_objective(problem_type=self.problem_type)
        if self.problem_type in [MULTICLASS, SOFTCLASS] and "num_classes" not in params:
            params["num_classes"] = self.num_classes
        if "verbose" not in params:
            params["verbose"] = -1

        self._scale = params.pop("scale_tests", False)
        self._label_flip_protection = params.pop("label_flip_constraints", False)
        self._test = params.pop("test", False)

        if self._label_flip_protection or self._test:
            stack_f = self.feature_metadata.get_features(required_special_types=['stack'])

            from scripts.leakage_benchmark.src.other.post_hoc_ensembling import (
                caruana_weighted,
                roc_auc_binary_loss_proba,
            )
            self._l1_ges_weights, _ = caruana_weighted([np.array(x) for x in X[stack_f].values.T.tolist()],
                                                       y, 42, 50, roc_auc_binary_loss_proba)
            self._stack_f = stack_f

        # ----------- DUPLICATE TESTING START -----------
        if params.pop("drop_duplicates", False):
            # Duplicate Code
            label = "class"
            l2_train_data = X.copy()
            l2_train_data[label] = y
            l2_train_data.reset_index(drop=True, inplace=True)
            oof_col_names = self.feature_metadata.get_features(required_special_types=['stack'])

            ignore_feature_duplicates = oof_col_names + [label]
            # ignore_feature_label_duplicates = oof_col_names
            ignore_cols = ignore_feature_duplicates
            mask = l2_train_data.drop(columns=ignore_cols).duplicated()
            # print("n+duplicates:", sum(mask) / len(mask))

            # # Compute sample weight raw
            # rel_cols = [x for x in l2_train_data.columns if x not in ignore_cols]
            # idx_sample_weight = []
            # for group_idx_list in l2_train_data.groupby(rel_cols).groups.values():
            #     group_idx_list = list(group_idx_list)
            #     n_dup = len(group_idx_list)
            #     for idx in group_idx_list:
            #         idx_sample_weight.append((idx, n_dup))
            #
            # sample_weight = np.array([s_c for _, s_c in sorted(idx_sample_weight, key=lambda x: x[0])])
            # sample_weight = sample_weight[~mask]

            # # Equalize code
            # rel_cols = [x for x in l2_train_data.columns if x not in ignore_cols]
            # keep_weight_list = []
            # org_len = len(l2_train_data)
            # l, counts = np.unique(y, return_counts=True)
            # major_label = l[np.argmax(counts)]
            # rng = np.random.RandomState(42)
            # for group_idx_list in l2_train_data.groupby(rel_cols).groups.values():
            #
            #     # --- Sample weight and clever drop code
            #     group_idx_list = list(group_idx_list)
            #     n_dup = len(group_idx_list)
            #     if n_dup == 1:
            #         keep_index = group_idx_list[0]
            #         sample_count = 1
            #     else:
            #         sample_count = n_dup
            #         subset = l2_train_data.loc[group_idx_list, label]
            #         counts = subset.value_counts()
            #
            #         # Random dropping
            #         # sel_label = rng.choice(counts.index)
            #
            #         # Clever dropping
            #         if len(counts) == 1:
            #             # No disparity  (could merge this but meh for rng)
            #             sel_label = counts.index[0]
            #         elif all(counts == counts.iloc[0]):
            #             # Tie breaker random
            #             sel_label = rng.choice(counts.index)
            #             # Tie breaker major class
            #             # sel_label = major_label
            #         else:
            #             # Popularity/majority vote
            #             sel_label = counts.index[0]
            #
            #         keep_index = subset[subset == sel_label].index[0]
            #     keep_weight_list.append((keep_index, sample_count))
            #
            #     # # - Equalize code
            #     # if len(group_idx_list) == 1:
            #     #     continue
            #     # for oof_col in oof_col_names:
            #     #     curr_vals = l2_train_data.loc[group_idx_list, oof_col]
            #     #     # could do majority or avg here... not sure what is better, stick to avg for now
            #     #     l2_train_data.loc[group_idx_list, oof_col] = curr_vals.max()
            #
            # # clever drop
            # keep_idx_list = [idx for idx, _ in keep_weight_list]
            # l2_train_data = l2_train_data.loc[keep_idx_list]
            #
            # # FIXME Would need to add more clever logic to merge existing sample weights
            # sample_weight = np.array([s_c for _, s_c in keep_weight_list]) # /org_len

            # Drop code
            l2_train_data = l2_train_data[~mask]
            X = l2_train_data.drop(columns=[label])
            y = l2_train_data[label]
        # ----------- DUPLICATE TESTING END ----------

        # ----------- NOISE TESTING START -----------
        if params.pop("random_noise_for_stack", False):
            rng = np.random.RandomState(42)
            stack_f = self.feature_metadata.get_features(required_special_types=['stack'])
            for col in stack_f:
                X.loc[:, col] = rng.randn(*X.loc[:, col].values.shape)
        # ----------- NOISE TESTING START -----------

        # ----------- OCI TESTING START -----------
        if params.pop("only_correct_instances", False):

            if self.num_classes != 2:
                raise NotImplementedError(
                    "only_correct_instances is only implemented for binary classification for now.")

            stack_f = self.feature_metadata.get_features(required_special_types=['stack'])
            tmp_X = X[stack_f].copy()
            classes_ = np.unique(y)

            threshold = 0.5  # technically would need to find the correct threshold per column and tune it...
            tmp_X = tmp_X.mask(tmp_X <= threshold, classes_[0])
            tmp_X = tmp_X.mask(tmp_X > threshold, classes_[1])
            s_tmp = tmp_X.sum(axis=1)

            no_diversity_rows = (s_tmp == 0) | (s_tmp == len(stack_f))
            s_tmp = s_tmp[no_diversity_rows]
            s_tmp[s_tmp == len(stack_f)] = 1
            always_wrong_row_indices = s_tmp.index[s_tmp != y[no_diversity_rows]]

            X = X.drop(index=always_wrong_row_indices)
            y = y.drop(index=always_wrong_row_indices)
        # ----------- OCI TESTING START -----------

        num_rows_train = len(X)
        dataset_train, dataset_val = self.generate_datasets(
            X=X, y=y, params=params, X_val=X_val, y_val=y_val, sample_weight=sample_weight,
            sample_weight_val=sample_weight_val
        )
        gc.collect()

        callbacks = []
        valid_names = []
        valid_sets = []
        if dataset_val is not None:
            from .callbacks import early_stopping_custom

            # TODO: Better solution: Track trend to early stop when score is far worse than best score, or score is trending worse over time
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train,
                                                                        strategy=early_stopping_rounds)
            if early_stopping_rounds is None:
                early_stopping_rounds = 999999
            reporter = kwargs.get("reporter", None)
            train_loss_name = self._get_train_loss_name() if reporter is not None else None
            if train_loss_name is not None:
                if "metric" not in params or params["metric"] == "":
                    params["metric"] = train_loss_name
                elif train_loss_name not in params["metric"]:
                    params["metric"] = f'{params["metric"]},{train_loss_name}'
            # early stopping callback will be added later by QuantileBooster if problem_type==QUANTILE
            early_stopping_callback_kwargs = dict(
                stopping_rounds=early_stopping_rounds,
                metrics_to_use=[("valid_set", stopping_metric_name)],
                max_diff=None,
                start_time=start_time,
                time_limit=time_limit,
                ignore_dart_warning=True,
                verbose=False,
                manual_stop_file=False,
                reporter=reporter,
                train_loss_name=train_loss_name,
            )
            callbacks += [
                # Note: Don't use self.params_aux['max_memory_usage_ratio'] here as LightGBM handles memory per iteration optimally.  # TODO: Consider using when ratio < 1.
                early_stopping_custom(**early_stopping_callback_kwargs)
            ]
            valid_names = ["valid_set"] + valid_names
            valid_sets = [dataset_val] + valid_sets
        else:
            early_stopping_callback_kwargs = None
        from lightgbm.callback import log_evaluation

        if log_period is not None:
            callbacks.append(log_evaluation(period=log_period))

        seed_val = params.pop("seed_value", 0 if params.pop('static_seed', True) else self._get_fold_seed())
        train_params = {
            "params": params,
            "train_set": dataset_train,
            "num_boost_round": num_boost_round,
            "valid_sets": valid_sets,
            "valid_names": valid_names,
            "callbacks": callbacks,
        }
        if not isinstance(stopping_metric, str):
            train_params["feval"] = stopping_metric
        else:
            if "metric" not in train_params["params"] or train_params["params"]["metric"] == "":
                train_params["params"]["metric"] = stopping_metric
            elif stopping_metric not in train_params["params"]["metric"]:
                train_params["params"]["metric"] = f'{train_params["params"]["metric"]},{stopping_metric}'
        if self.problem_type == SOFTCLASS:
            train_params["fobj"] = lgb_utils.softclass_lgbobj
        elif self.problem_type == QUANTILE:
            train_params["params"]["quantile_levels"] = self.quantile_levels
        if seed_val is not None:
            train_params["params"]["seed"] = seed_val
            random.seed(seed_val)
            np.random.seed(seed_val)

            # Sanity
            train_params["params"]["bagging_seed"] = seed_val + 1
            train_params["params"]["feature_fraction_seed"] = seed_val + 2
            train_params["params"]["extra_seed"] = seed_val + 3
            train_params["params"]["drop_seed"] = seed_val + 4
            train_params["params"]["data_random_seed"] = seed_val + 5
            train_params["params"]["objective_seed"] = seed_val + 6

        # Add monotone constraints for all stack features
        if train_params["params"].pop("monotone_constraints_for_stack_features", False):
            stack_features = self.feature_metadata.get_features(required_special_types=['stack'])
            train_params["params"]["monotone_constraints"] = \
                [1 if col in stack_features else 0 for col in dataset_train.data.columns]

            # Our defaults for now
            train_params["params"]["monotone_penalty"] = 0
            train_params["params"]["monotone_constraints_method"] = 'advanced'

        if train_params["params"].pop("stack_feature_interactions_map", False):
            # Interaction constraints test
            stack_features = self.feature_metadata.get_features(required_special_types=['stack'])
            X_f = [i for i, col in enumerate(dataset_train.data.columns) if col not in stack_features]
            oof_f = [i for i, col in enumerate(dataset_train.data.columns) if col in stack_features]
            train_params["params"]["interaction_constraints"] = [X_f + [f] for f in oof_f]

        # Train LightGBM model:
        from lightgbm.basic import LightGBMError

        with warnings.catch_warnings():
            # Filter harmless warnings introduced in lightgbm 3.0, future versions plan to remove: https://github.com/microsoft/LightGBM/issues/3379
            warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset.")
            warnings.filterwarnings("ignore", message="categorical_column in param dict is overridden.")
            try:
                self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs,
                                             **train_params)
            except LightGBMError:
                if train_params["params"].get("device", "cpu") != "gpu":
                    raise
                else:
                    logger.warning(
                        "Warning: GPU mode might not be installed for LightGBM, GPU training raised an exception. Falling back to CPU training..."
                        "Refer to LightGBM GPU documentation: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version"
                        "One possible method is:"
                        "\tpip uninstall lightgbm -y"
                        "\tpip install lightgbm --install-option=--gpu"
                    )
                    train_params["params"]["device"] = "cpu"
                    self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs,
                                                 **train_params)
            retrain = False
            if train_params["params"].get("boosting_type", "") == "dart":
                if dataset_val is not None and dart_retrain and (self.model.best_iteration != num_boost_round):
                    retrain = True
                    if time_limit is not None:
                        time_left = time_limit + start_time - time.time()
                        if time_left < 0.5 * time_limit:
                            retrain = False
                    if retrain:
                        logger.log(15, f"Retraining LGB model to optimal iterations ('dart' mode).")
                        train_params.pop("callbacks", None)
                        train_params.pop("valid_sets", None)
                        train_params.pop("valid_names", None)
                        train_params["num_boost_round"] = self.model.best_iteration
                        self.model = train_lgb_model(**train_params)
                    else:
                        logger.log(15, f"Not enough time to retrain LGB model ('dart' mode)...")

        if dataset_val is not None and not retrain:
            self.params_trained["num_boost_round"] = self.model.best_iteration
        else:
            self.params_trained["num_boost_round"] = self.model.current_iteration()

    def _predict_proba(self, X, num_cpus=0, val_label=None, **kwargs):
        X = self.preprocess(X, **kwargs)

        y_pred_proba = self.model.predict(X, num_threads=num_cpus)

        if self._label_flip_protection:
            if self.problem_type != BINARY:
                raise ValueError("Label flip protection is only supported for binary classification so far.")
            from scripts.leakage_benchmark.src.other.exp_method_store import current_best
            current_best(self, X, y_pred_proba, val_label)

        if self._test:
            # org_y_pred_proba = y_pred_proba.copy()


            pass


            # -- Count flips
            # print(f'{sum(org_y_pred_proba != y_pred_proba) / len(y_pred_proba):.3f}', f'Test: {val_label is None}')

        if self.problem_type == REGRESSION:
            return y_pred_proba
        elif self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == MULTICLASS:
            return y_pred_proba
        elif self.problem_type == SOFTCLASS:  # apply softmax
            y_pred_proba = np.exp(y_pred_proba)
            y_pred_proba = np.multiply(y_pred_proba, 1 / np.sum(y_pred_proba, axis=1)[:, np.newaxis])
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def _preprocess_nonadaptive(self, X, is_train=False, **kwargs):
        X = super()._preprocess_nonadaptive(X=X, **kwargs)

        if is_train:
            # Dropout stuff
            self._requires_select = False\

            self._requires_remap = False
            for column in self._features_internal:
                if isinstance(column, str):
                    new_column = re.sub(r'[",:{}[\]]', "", column)
                    if new_column != column:
                        self._features_internal_map = {feature: i for i, feature in enumerate(list(X.columns))}
                        self._requires_remap = True
                        break
            if self._requires_remap:
                self._features_internal_list = np.array(
                    [self._features_internal_map[feature] for feature in list(X.columns)])
            else:
                self._features_internal_list = self._features_internal



        #     if self._scale:
        #         from sklearn.preprocessing import StandardScaler
        #         self._oof_col_names = self.feature_metadata.get_features(required_special_types=['stack'])
        #         self._sc = StandardScaler()
        #         self._sc.fit(X[self._oof_col_names])
        #
        # if self._scale:
        #     X_new = X.copy()
        #     X_new[self._oof_col_names] = self._sc.transform(X_new[self._oof_col_names])
        #     X = X_new

        if self._requires_remap:
            X_new = X.copy(deep=False)
            X_new.columns = self._features_internal_list
            return X_new
        else:
            return X

    def generate_datasets(self, X: DataFrame, y: Series, params, X_val=None, y_val=None, sample_weight=None,
                          sample_weight_val=None, save=False):
        lgb_dataset_params_keys = ["two_round"]  # Keys that are specific to lightGBM Dataset object construction.
        data_params = {key: params[key] for key in lgb_dataset_params_keys if key in params}.copy()

        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike

        y_og = None
        y_val_og = None
        if self.problem_type == SOFTCLASS:
            y_og = np.array(y)
            y = None
            if X_val is not None:
                y_val_og = np.array(y_val)
                y_val = None

        # X, W_train = self.convert_to_weight(X=X)
        dataset_train = construct_dataset(
            x=X, y=y, location=os.path.join("self.path", "datasets", "train"), params=data_params, save=save,
            weight=sample_weight
        )
        # dataset_train = construct_dataset_lowest_memory(X=X, y=y, location=self.path + 'datasets/train', params=data_params)
        if X_val is not None:
            # X_val, W_val = self.convert_to_weight(X=X_val)
            dataset_val = construct_dataset(
                x=X_val,
                y=y_val,
                location=os.path.join(self.path, "datasets", "val"),
                reference=dataset_train,
                params=data_params,
                save=save,
                weight=sample_weight_val,
            )
            # dataset_val = construct_dataset_lowest_memory(X=X_val, y=y_val, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        else:
            dataset_val = None
        if self.problem_type == SOFTCLASS:
            if y_og is not None:
                dataset_train.softlabels = y_og
            if y_val_og is not None:
                dataset_val.softlabels = y_val_og
        return dataset_train, dataset_val

    def _get_train_loss_name(self):
        if self.problem_type == BINARY:
            train_loss_name = "binary_logloss"
        elif self.problem_type == MULTICLASS:
            train_loss_name = "multi_logloss"
        elif self.problem_type == REGRESSION:
            train_loss_name = "l2"
        else:
            raise ValueError(f"unknown problem_type for LGBModel: {self.problem_type}")
        return train_loss_name

    def _get_early_stopping_rounds(self, num_rows_train, strategy="auto"):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _is_gpu_lgbm_installed(self):
        # Taken from https://github.com/microsoft/LightGBM/issues/3939
        try_import_lightgbm()
        import lightgbm

        try:
            data = np.random.rand(50, 2)
            label = np.random.randint(2, size=50)
            train_data = lightgbm.Dataset(data, label=label)
            params = {"device": "gpu"}
            gbm = lightgbm.train(params, train_set=train_data, verbose=-1)
            return True
        except Exception as e:
            return False

    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 1,
        }
        if is_gpu_available and self._is_gpu_lgbm_installed():
            minimum_resources["num_gpus"] = 0.5
        return minimum_resources

    def _get_default_resources(self):
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    @property
    def _features(self):
        return self._features_internal_list

    def _ag_params(self) -> set:
        return {"early_stop"}

    def _more_tags(self):
        # `can_refit_full=True` because num_boost_round is communicated at end of `_fit`
        return {"can_refit_full": True}
