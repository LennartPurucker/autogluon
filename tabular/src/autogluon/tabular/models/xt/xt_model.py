from autogluon.core.constants import MULTICLASS, QUANTILE, REGRESSION

from ..rf.rf_model import RFModel


class XTModel(RFModel):
    """
    Extra Trees model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    """

    def _get_model_type(self, clean_oof_predictions=False):
        if clean_oof_predictions and (self.problem_type in [REGRESSION, QUANTILE, MULTICLASS]):
            raise ValueError(f"{self.problem_type} not supported yet for clean_oof_predictions!")

        if self.problem_type == REGRESSION:
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor
        elif self.problem_type == QUANTILE:
            from ..rf.rf_quantile import ExtraTreesQuantileRegressor

            return ExtraTreesQuantileRegressor
        else:
            from ..rf.custom_forest import ExtraTreesClassifier

            return ExtraTreesClassifier
