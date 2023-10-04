from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION


def clean_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    stack_cols: List[str],
    problem_type: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, dict]:
    if not stack_cols:
        raise ValueError(f"stack_cols are empty!")
    X = X.copy()
    ir_map = {}

    if problem_type != BINARY:
        raise ValueError(f"Unsupported Problem Type for cleaning oof predictions! Got: {problem_type}")

    # TODO: verify assumption that proba are always positive class proba (think this is true)
    if sample_weight is None:
        curr_sample_weight = np.ones((len(X),), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()
    for f in stack_cols:
        reg = CenteredIsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True)
        # reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True)
        X[f] = reg.fit_transform(X[f], y, sample_weight=curr_sample_weight)
        ir_map[f] = reg

    return X, ir_map


def re_clean_oof_predictions(X, ir_map, stack_cols, problem_type):
    X = X.copy()

    if problem_type != BINARY:
        raise ValueError(f"Unsupported Problem Type for cleaning oof predictions! Got: {problem_type}")

    for f in stack_cols:
        X[f] = ir_map[f].transform(X[f])

    return X


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
