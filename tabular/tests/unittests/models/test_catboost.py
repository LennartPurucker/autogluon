import sys

import pytest

from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"iterations": 10}


@pytest.mark.skipif(
    sys.version_info >= (3, 11) and sys.platform == "darwin", reason="catboost has no wheel for py311 darwin"
)
def test_catboost():
    model_cls = CatBoostModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


@pytest.mark.parametrize("eval_metric", ["r2", "mean_absolute_error"])
def test_catboost_can_train_with_nondefault_regression_eval_metrics(eval_metric):
    model_cls = CatBoostModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        init_args={"eval_metric": eval_metric},
        problem_types=["regression"],
    )


def test_catboost_predict_uses_fit_thread_count(tmp_path, monkeypatch):
    """Predict passes `thread_count` equal to the CPU budget recorded at fit (CatBoost's own default is
    every core on the machine), falls back to that default when no budget is recorded, and the thread
    count leaves the predictions unchanged.
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y)

    model = CatBoostModel(
        path=str(tmp_path),
        name="CatBoost",
        problem_type="binary",
        eval_metric="log_loss",
        hyperparameters=dict(toy_model_params),
    )
    model.fit(X=X, y=y, num_cpus=1, num_gpus=0)
    assert model.fit_num_cpus == 1

    thread_counts = []
    predict_proba = model.model.predict_proba

    def spy(data, **kwargs):
        thread_counts.append(kwargs.get("thread_count"))
        return predict_proba(data, **kwargs)

    monkeypatch.setattr(model.model, "predict_proba", spy)

    y_pred_proba_1 = model.predict_proba(X)
    model._fit_metadata["num_cpus"] = 2
    y_pred_proba_2 = model.predict_proba(X)
    model._fit_metadata["num_cpus"] = None
    y_pred_proba_default = model.predict_proba(X)

    assert thread_counts == [1, 2, None]
    np.testing.assert_array_equal(y_pred_proba_1, y_pred_proba_2)
    np.testing.assert_array_equal(y_pred_proba_1, y_pred_proba_default)
