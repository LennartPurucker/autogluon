from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration

from cir_model import CenteredIsotonicRegression

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from functools import partial
from sklearn.pipeline import Pipeline


# Clean Code
def clean_oof_predictions(
        X: pd.DataFrame,
        y: pd.Series,
        X_val,
        y_val,
        stack_cols: List[str],
        problem_type: str,
        fold_model=None,
        sample_weight: Optional[np.ndarray] = None,
) -> dict:
    if not stack_cols:
        raise ValueError(f"stack_cols are empty!")
    ir_map = {}

    if problem_type != BINARY:
        raise ValueError(f"Unsupported Problem Type for cleaning oof predictions! Got: {problem_type}")

    # from autogluon.tabular.models import NNFastAiTabularModel, TabularNeuralNetTorchModel, LGBModel, \
    #     CatBoostModel  # future LinearModel, TabPFNModel
    # is_smooth_fiter = (fold_model is None) or isinstance(fold_model, (NNFastAiTabularModel, TabularNeuralNetTorchModel))

    if sample_weight is None:
        curr_sample_weight = np.ones((len(X),), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()

    for f_idx, f in enumerate(sorted(stack_cols)):

        proba = X[f].values.astype(np.float64)
        y_fit = y
        fit_curr_sample_weight = curr_sample_weight
        y_min = 0
        y_max = 1

        apply_at_train, apply_at_val_train = True, True
        apply_at_val_predict, apply_at_test = False, True

        # FIXME move this information to feature metadata somehow
        if any(x in f for x in ["NeuralNetTorch", "NeuralNetFastAI"]):
            # for future: "LinearModel", "Transformer", "FTTransformer", "TabPFN"
            reg = _SigmoidCalibration().fit(proba, y_fit, sample_weight=fit_curr_sample_weight)
            # reg = ts_calibration(proba, y_fit, y_min=y_min, y_max=y_max, sample_weight=curr_sample_weight, problem_type=problem_type)
        elif any(x in f for x in ["KNeighbors", "RandomForest", "ExtraTrees"]):  # "CatBoost", "LightGBMXT",
            reg = CenteredIsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip", increasing=True,
                                             non_centered_points=[y_min, y_max])
            reg.fit(proba, y_fit, sample_weight=fit_curr_sample_weight)
        else:
            reg = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip", increasing=True)
            reg.fit(proba, y_fit, sample_weight=fit_curr_sample_weight)

        ir_map[f] = [apply_at_train, apply_at_val_train, apply_at_val_predict, apply_at_test, reg]

    # plot_insights(X, tmp_y, stack_cols, ir_map, "Train", reverse_offset)
    # plot_insights(X_val, y_val, stack_cols, ir_map, "Val", y_offset)

    return ir_map


def clean_leaking_predictions(
        proba_to_clean,
        reasonable_proba,
        y
        # sample_weight: Optional[np.ndarray] = None,
) -> dict:
    # Idea: Proba calibration at l2 informed from l1
    #   - either L1 as target, or L1 as reg fit model
    #   - motivation: ensure motonoc constraitns between L1 and L2+

    # Problems / Variations:
    #   - maybe it only works good if L1 model same model type as L2 model for reasonable proba
    #       - could make that work and fall back if needed.
    #   - maybe need to do it per repeat

    # Test:
    #  - different y
    #  - different reg model fit on
    reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip") # , increasing=False
    # reg = CenteredIsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True, non_centered_points=[0, 1])
    # reg = _SigmoidCalibration()

    # reg.fit(proba_b, y)
    new_proba = proba_to_clean.copy()
    new_proba[y==1] = reg.fit_transform(proba_to_clean[y==1], reasonable_proba[y==1])
    plot_thresholds_proba(proba_to_clean[y==1], reasonable_proba[y==1],  reg)

    new_proba[y==0] = reg.fit_transform(proba_to_clean[y==0], reasonable_proba[y==0])
    plot_thresholds_proba(proba_to_clean[y==0], reasonable_proba[y==0],  reg)

    reg.fit(proba_to_clean, reasonable_proba)
    # reg.fit(proba_to_clean, y)

    # reg.fit(np.hstack([proba_to_clean, proba_to_clean]), np.hstack([proba_b, y]),)


    # new_proba = reg.predict(proba_to_clean)

    return new_proba, reg

def plot_thresholds_proba(proba_a, prob_b, reg):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(proba_a, prob_b, "C0.", markersize=12)
    # ax.plot(proba_a, reg.predict(proba_a), "C2.", markersize=12)
    # ax.plot(reg.predict(proba_a), prob_b, "C3.", markersize=12)
    x_th = None
    # if hasattr(reg, "X_thresholds_"):
    #     x_th, y_th = reg.X_thresholds_, reg.y_thresholds_

    if x_th is not None:
        ax.plot(x_th, y_th, "C1.-", markersize=12, alpha=0.5)

    ax.plot([0, 1], [0, 1], "C8.-", markersize=12, alpha=0.5)
    ax.set_xlim(-0.1, 1.1)

    fig.supxlabel("Proba Clan")
    fig.supylabel("Proba Ref")
    plt.show()

# def inject_offset(X, offset):
#     return (X + offset).astype(np.float64)
#
#
# def reverse_offset(X, offset):
#     return (X - offset).astype(np.float32)
# def const_maker(X):
#     return np.full_like(X, fill_value=1)
#
#
# is_smooth_fiter_disc = isinstance(fold_model, LGBModel) and fold_model._user_params.get('extra_trees', False)
# is_smooth_fiter_disc = is_smooth_fiter_disc or isinstance(fold_model, CatBoostModel)
# from ._custom_function_transformer import FunctionTransformer
#
# from .temperature_scaling import tune_temperature_scaling, apply_temperature_scaling
# from ..data.label_cleaner import LabelCleanerMulticlassToBinary
# def ts_calibration(proba,  y, y_min, y_max, sample_weight, problem_type):
#     # reg = ts_calibration(proba, y, y_min=y_min, y_max=y_max, sample_weight=curr_sample_weight, problem_type=problem_type)
#
#     eps = np.finfo(np.float32).eps
#     tmp_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(proba)
#     tmp_proba = np.clip(tmp_proba, a_min=y_min + eps, a_max=y_max - eps)
#     temperature_scale = tune_temperature_scaling(tmp_proba, y.values)
#
#     reg = Pipeline(
#         [
#             ("temperature_scaling", FunctionTransformer(
#                 func=partial(apply_temperature_scaling, temperature_scalar=temperature_scale,
#                              problem_type=problem_type))),
#         ]
#     )
#     return reg
#
# def logit_func(Xf):
#     return np.log(np.clip(Xf, a_min=np.finfo(np.float32).eps, a_max=1))
#
# def dtype_func(X, dtype):
#     return X.astype(dtype)
#
# def id_func(X, offset=0, offset_mult=1):
#     return (X/offset_mult) - offset   # np.full_like(X, fill_value=1)
#

# def passthrough(X):
#     return X
#
# def plot_calibration(X, y, stack_cols, ir_map, plt_title):
#     import matplotlib.pyplot as plt
#     from sklearn.calibration import CalibrationDisplay
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#     colors = plt.get_cmap("tab20")
#     colors.colors = colors.colors + plt.get_cmap("tab20b").colors
#     colors.colors = colors.colors + plt.get_cmap("tab20c").colors
#     colors.colors = colors.colors + plt.get_cmap("Paired").colors
#     colors.N = len(colors.colors)
#
#     for i, f in enumerate(stack_cols):
#         display = CalibrationDisplay.from_predictions(y, X[f], n_bins=20, name="Before-" + f, ax=ax, color=colors((i) + (2 * i)), strategy="quantile")
#         display = CalibrationDisplay.from_predictions(
#             y, ir_map[f][-1].predict(X[f]), n_bins=20, name="After-" + f, ax=ax, color=colors((i + 1) + (2 * i)), strategy="quantile"
#         )
#     plt.title(plt_title)
#     plt.show()
#
#
# def plot_thresholds(X, y, stack_cols, ir_map, plt_title, reverse_offset):
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots(ncols=len(stack_cols), figsize=(20, 6))
#
#     for f_idx, f in enumerate(stack_cols):
#         ax[f_idx].plot(X[f], y, "C0.", markersize=12)
#         ax[f_idx].plot(X[f], ir_map[f][-1].predict(X[f]), "C2.", markersize=12)
#
#         x_th = None
#         if hasattr(ir_map[f][-1], "X_thresholds_"):
#             x_th, y_th = ir_map[f][-1].X_thresholds_, ir_map[f][-1].y_thresholds_
#         elif hasattr(ir_map[f][-1], "named_steps") and hasattr(ir_map[f][-1][-1], "X_thresholds_"):
#             x_th, y_th = ir_map[f][-1][-1].X_thresholds_, ir_map[f][-1][-1].y_thresholds_
#         elif hasattr(ir_map[f][-1], "named_steps") and hasattr(ir_map[f][-1][0], "X_thresholds_"):
#             x_th, y_th = ir_map[f][-1][0].X_thresholds_, ir_map[f][-1][0].y_thresholds_
#         elif hasattr(ir_map[f][-1], "named_steps") and hasattr(ir_map[f][-1][1], "X_thresholds_"):
#             x_th, y_th = ir_map[f][-1][1].X_thresholds_, ir_map[f][-1][1].y_thresholds_
#
#         if x_th is not None:
#             y_th = reverse_offset(y_th)
#             ax[f_idx].plot(x_th, y_th, "C1.-", markersize=12, alpha=0.5)
#             print(f, y_th)
#
#         ax[f_idx].plot([0, 1], [0, 1], "C8.-", markersize=12, alpha=0.5)
#
#         ax[f_idx].set_title(f)
#         ax[f_idx].set_xlim(-0.1, 1.1)
#
#     fig.supxlabel("Proba L1")
#     fig.supylabel("Label / Adjusted Proba")
#     fig.suptitle(plt_title + f" | Offset: {reverse_offset}")
#     plt.show()
#
#
# def plot_insights(X, y, stack_cols, ir_map, plt_title, reverse_offset):
#     # plot_calibration(X, y, stack_cols, ir_map, plt_title)
#     plot_thresholds(X, y, stack_cols, ir_map, plt_title, reverse_offset)


# -- Potentially usable agnostic and more efficient solution for multi or binary class from sklearns's calibration class
# https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV

# # Calibrate
# Y = label_binarize(y, classes=classes)
# label_encoder = LabelEncoder().fit(classes)
# pos_class_indices = label_encoder.transform(clf.classes_)
# for class_idx, this_pred in zip(pos_class_indices, predictions.T):
#     calibrator = IsotonicRegression(out_of_bounds="clip")
#     X[f] = calibrator.fit_transform(this_pred, Y[:, class_idx], sample_weight)
#
#     elif problem_type == MULTICLASS:
#         raise NotImplementedError(
#             f"{MULTICLASS} not yet supported!"
#         )  # multiclass requires entirely different implementation than the one working below to function well (future work)
#         # FIXME: get this from outer scope somehow (could not find it)
#         classes_in_X = set([f.rsplit("_")[-1] for f in stack_cols])
#         base_models = set([f.rsplit("_", 1)[0] for f in stack_cols])
#         bm_proba_map = {bm: [f for f in stack_cols if f.startswith(bm)] for bm in base_models}
#
#         # - Checks
#         # assert set(np.unique(y.astype(str))) != classes_in_X)  # does not necessarily need to hold true for this to work
#         class_overflow = classes_in_X - set(np.unique(y.astype(str)))  # TODO: verify required assumption
#         if class_overflow:  # needs to hold
#             raise ValueError(
#                 f"The OOF predictions contain classes that are not in the label! The follow missing classes are missing in the label: {class_overflow}"
#             )
#
#         for bm, bm_stack_cols in bm_proba_map.items():
#             c_ordered = np.unique(list(classes_in_X))
#             f_ordered = np.unique(bm_stack_cols)
#             assert len(c_ordered) == len(f_ordered), "Something is wrong with proba and classes for OOF!"  # TODO: verify required assumption
#
#             for current_class, current_f in zip(c_ordered, f_ordered):
#                 y_ovr = y.copy().astype(str)
#                 pos_class_mask = y_ovr == str(current_class)
#                 y_ovr[pos_class_mask] = 1
#                 y_ovr[~pos_class_mask] = 0
#                 y_ovr = y_ovr.astype(int)
#
#                 reg = IsotonicRegression(y_max=1, y_min=0, out_of_bounds="clip", increasing=True)
#                 X[current_f] = reg.fit_transform(X[current_f], y_ovr)
#
#             # Normalize the probabilities for multiclass following sklearn's implementation
#             full_proba = X[f_ordered].values
#             denominator = np.sum(full_proba, axis=1)[:, np.newaxis]
#             uniform_proba = np.full_like(full_proba, 1 / len(c_ordered))
#             full_proba = np.divide(full_proba, denominator, out=uniform_proba, where=denominator != 0)
#             full_proba[(1.0 < full_proba) & (full_proba <= 1.0 + 1e-5)] = 1.0
#             X[f_ordered] = full_proba
#         # TODO: add this during predict time of model: refactor everything
#     elif problem_type == REGRESSION:
#         raise NotImplementedError(f"{REGRESSION} not yet supported!")
# -- add o/1 samples code rollback
#     # TODO: verify assumption that proba are always positive class proba (think this is true)
#     if sample_weight is None:
#         curr_sample_weight = np.ones((len(X),), dtype=np.float64)
#     else:
#         curr_sample_weight = sample_weight.copy()
#
#     _y = np.concatenate([y.values, np.array([0, 1])])
#     _curr_sample_weight = np.concatenate([curr_sample_weight, np.array([1, 1])])
#
#     for f in stack_cols:
#         reg = IsotonicRegression(y_min=None, y_max=None, out_of_bounds="clip", increasing=True)
#         vals = np.concatenate([X[f].values, np.array([0, 1])])
#         X[f] = reg.fit_transform(vals, _y, sample_weight=_curr_sample_weight)[:-2]
#         ir_map[f] = reg
#
#     return X, ir_map
# --- Test for before preprocess
# # if not kwargs.get('fit', False):
#         #     stack_cols = self.feature_metadata.get_features(required_special_types=["stack"])
#         #     if stack_cols and hasattr(self, "_re_ir") and self._re_ir:
#         #         from autogluon.core.calibrate.stacked_overfitting_mitigation import re_clean_oof_predictions
#         #         X = re_clean_oof_predictions(X, self._ir_map, stack_cols, self.problem_type)
#
#         # labels = np.unique(y)
#         # y_prob = X[f]
#         # n_bins= 20
#         # y_true = y == 1
#         #
#         # quantiles = np.linspace(0, 1, n_bins + 1)
#         # bins = np.percentile(y_prob, quantiles * 100)
#         # #bins = np.linspace(0.0, 1.0, n_bins + 1)
#         # binids = np.searchsorted(bins[1:-1], y_prob)
#         #
#         # bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
#         # bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
#         # bin_total = np.bincount(binids, minlength=len(bins))
#         #
#         # nonzero = bin_total != 0
#         # prob_true = bin_true[nonzero] / bin_total[nonzero]
#         # prob_pred = bin_sums[nonzero] / bin_total[nonzero]
#         # weight = abs(prob_pred - prob_true)
#         # h_mask = prob_pred > prob_true
#         #
#         # # weight[~h_mask] = 1
#         # # weight[h_mask] = 10
#         # weight += 1
#         #
#         # map_weights = {k: v for k, v in zip(np.arange(len(bins))[nonzero], weight)}
#         # curr_sample_weight *= np.array(list(map(lambda x: map_weights[x], binids)))
#
#         #   # _SigmoidCalibration()
#         #
#         # X[f] = reg.predict(X[f]) - 1
#
#         # FIXME: round thresholds? - minimal numerical differences in threshold can make larger differnce in proba
#         #       -> round input to IR! looks promising... .round(decimals=2)
#         #       -> add noise
#         #       -> use pseudo labels? majority vote, current f, GES l1
#         #   -> higher precision output?
#         #   -> higher precision input to IR? (offset)
#
#         # tmp_y = y
#
#         # preds = X[stack_cols].copy()
#         # presd = 1
#         # for _f, opt_t in opt_threshold_map.items():
#         #     preds.loc[X[_f] < opt_t, _f] = 0
#         #
#         # tmp_y = np.mean(preds, axis=1)
#         # certain_mask = (tmp_y == 1)  | (tmp_y == 0)
#         # tmp_y[certain_mask] = y[certain_mask]
#
#         # preds = preds.sum(axis=1)
#         # votes = np.full_like(preds, 0)
#         # votes[preds > np.ceil(len(stack_cols) / 2)] = 1
#         # tmp_y = votes
#
#         # tmp_y = np.full_like(y, fill_value=1)
#         # tmp_y[X[f] < opt_threshold_map[f]] = 0
#         # choice(np.arange(10)
#
#         # Random sample weights is like noise btw.
#         # curr_sample_weight = np.random.RandomState(42 + f_idx).poisson(5, size=len(tmp_y)) # np.clip(np.random.RandomState(42).normal(5, 0.5, size=len(tmp_y)), a_min=0.01, a_max=None) #
#
#         # reverse_offset = partial(id_func, offset=y_offset,offset_mult=y_offset_mult)
#         # _SigmoidCalibration() if is_smooth_fiter else CenteredIsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip", increasing=True)
#         # reg = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip", increasing=True)
#         # apply_at_train, apply_at_val_train, apply_at_val_predict,  apply_at_test = True, False, False, False
#         #
#         # reg.fit(X[f].values.astype(np.float64), (tmp_y + y_offset)*y_offset_mult, sample_weight=curr_sample_weight)
#         # reg = Pipeline(
#         #     [
#         #         ("inject_dtype", FunctionTransformer(func=partial(dtype_func, dtype=np.float32))),
#         #         ("reg", reg),
#         #         ("repair_offset", FunctionTransformer(func=reverse_offset)),
#         #         ("inject_dtype", FunctionTransformer(func=partial(dtype_func, dtype=np.float32))),
#         #     ]
#         # )
#
#         # reg = Pipeline(
#         #     [
#         #         # ("logit_maker", FunctionTransformer(func=logit_func)),
#         #         # ("passthrough", FunctionTransformer(func=passthrough)),
#         #         # ("const_maker", FunctionTransformer(func=const_maker)),
#         #     ]
#         # )
#
#         # IR (v2)
#         # reg = IsotonicRegression()
#         # reg.fit(X[f].values.astype(np.float64), y, sample_weight=curr_sample_weight)
#         # apply_at_train, apply_at_val_train, apply_at_val_predict, apply_at_test = True, False, False, False
#
#
#         # Plat scaling
#         # reg = _SigmoidCalibration()
#         # reg.fit(X[f].values.astype(np.float64), y, sample_weight=curr_sample_weight)
#         # apply_at_train, apply_at_val_train, apply_at_val_predict, apply_at_test = True, False, False, False
#
#         # Temperature scaling
#         # tmp_f = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(X[f].values.astype(np.float64))
#         # tmp_f = np.clip(tmp_f, a_min=np.finfo(np.float32).eps, a_max=1-np.finfo(np.float32).eps)
#         # temperature_scale = tune_temperature_scaling(tmp_f, y.values, init_val=1, max_iter=1000, lr=0.01)
#         # if temperature_scale is None:
#         #     raise ValueError("fix edge case pls")
#         #
#         # reg = Pipeline(
#         #     [
#         #         ("temperature_scaling", FunctionTransformer(func=partial(apply_temperature_scaling, temperature_scalar=temperature_scale, problem_type=problem_type))),
#         #     ]
#         # )
#         #
#         # apply_at_train, apply_at_val_train  , apply_at_val_predict, apply_at_test = True, True, False, False
#
#
#         # tune_temperature_scaling()
#
#         # print(reg.predict(X[f]).dtype)

#     # Make sure we have iloc index no matter the index of dataframes
#     best_oof , opt_threshold, opt_threshold_map = best_oof
#     all_index = list(X.index) + list(X_val.index)
#     iloc_index = np.where(np.arange(len(all_index))[np.argsort(all_index)] < len(X.index))[0]
#     val_best_oof = best_oof[iloc_index]
#     y_offset = 0
#     y_offset_mult = 1
#     y_min = (0 + y_offset) * y_offset_mult
#     y_max = (1 + y_offset) * y_offset_mult
# core_kwargs['best_oof'] = None # [self.get_model_oof(self.model_best), self.calibrate_decision_threshold(metric='balanced_accuracy'), {k: self.calibrate_decision_threshold(metric='balanced_accuracy', model=k) for k in  base_model_names}]
