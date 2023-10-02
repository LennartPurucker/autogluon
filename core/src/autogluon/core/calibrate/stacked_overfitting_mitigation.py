from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION


def clean_oof_predictions(
    X: pd.DataFrame, y: pd.Series, feature_metadata: FeatureMetadata, problem_type: str, sample_weight: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    stack_cols = feature_metadata.get_features(required_special_types=["stack"])
    if not stack_cols:
        return X, y
    X = X.copy()

    if problem_type == BINARY:
        # TODO: verify assumption that proba are always positive class proba (think this is true)
        for f in stack_cols:
            reg = IsotonicRegression(y_max=1, y_min=0, out_of_bounds="clip", increasing=True)
            X[f] = reg.fit_transform(X[f], y, sample_weight=sample_weight)

    elif problem_type == MULTICLASS:
        raise NotImplementedError(
            f"{MULTICLASS} not yet supported!"
        )  # multiclass requires entirely different implementation than the one working below to function well (future work)
        # FIXME: get this from outer scope somehow (could not find it)
        classes_in_X = set([f.rsplit("_")[-1] for f in stack_cols])
        base_models = set([f.rsplit("_", 1)[0] for f in stack_cols])
        bm_proba_map = {bm: [f for f in stack_cols if f.startswith(bm)] for bm in base_models}

        # - Checks
        # assert set(np.unique(y.astype(str))) != classes_in_X)  # does not necessarily need to hold true for this to work
        class_overflow = classes_in_X - set(np.unique(y.astype(str)))  # TODO: verify required assumption
        if class_overflow:  # needs to hold
            raise ValueError(
                f"The OOF predictions contain classes that are not in the label! The follow missing classes are missing in the label: {class_overflow}"
            )

        for bm, bm_stack_cols in bm_proba_map.items():
            c_ordered = np.unique(list(classes_in_X))
            f_ordered = np.unique(bm_stack_cols)
            assert len(c_ordered) == len(f_ordered), "Something is wrong with proba and classes for OOF!"  # TODO: verify required assumption

            for current_class, current_f in zip(c_ordered, f_ordered):
                y_ovr = y.copy().astype(str)
                pos_class_mask = y_ovr == str(current_class)
                y_ovr[pos_class_mask] = 1
                y_ovr[~pos_class_mask] = 0
                y_ovr = y_ovr.astype(int)

                reg = IsotonicRegression(y_max=1, y_min=0, out_of_bounds="clip", increasing=True)
                X[current_f] = reg.fit_transform(X[current_f], y_ovr)

            # Normalize the probabilities for multiclass following sklearn's implementation
            full_proba = X[f_ordered].values
            denominator = np.sum(full_proba, axis=1)[:, np.newaxis]
            uniform_proba = np.full_like(full_proba, 1 / len(c_ordered))
            full_proba = np.divide(full_proba, denominator, out=uniform_proba, where=denominator != 0)
            full_proba[(1.0 < full_proba) & (full_proba <= 1.0 + 1e-5)] = 1.0
            X[f_ordered] = full_proba
        # TODO: add this during predict time of model: refactor everything
    elif problem_type == REGRESSION:
        raise NotImplementedError(f"{REGRESSION} not yet supported!")
    else:
        raise ValueError(f"Unsupported Problem Type for cleaning oof predictions! Got: {problem_type}")

    return X, y


# -- Potentially usable agnostic and more efficient solution for multi or binary class from sklearns's calibration class
# https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV

# # Calibrate
# Y = label_binarize(y, classes=classes)
# label_encoder = LabelEncoder().fit(classes)
# pos_class_indices = label_encoder.transform(clf.classes_)
# for class_idx, this_pred in zip(pos_class_indices, predictions.T):
#     calibrator = IsotonicRegression(out_of_bounds="clip")
#     X[f] = calibrator.fit_transform(this_pred, Y[:, class_idx], sample_weight)
