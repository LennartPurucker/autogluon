from sklearn.isotonic import IsotonicRegression
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION


def clean_oof_predictions(X, y, feature_metadata, problem_type):
    stack_cols = feature_metadata.get_features(required_special_types=['stack'])
    if not stack_cols:
        return X, y
    X = X.copy()

    if problem_type == BINARY:
        for f in stack_cols:
            reg = IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)
            X[f] = reg.fit_transform(X[f], y)
    elif problem_type == MULTICLASS:
        raise NotImplementedError()
    elif problem_type == REGRESSION:
        raise NotImplementedError()
    else:
        raise ValueError(f"Unsupported Problem Type for cleaning oof predictions! Got: {problem_type}")

    return X, y
