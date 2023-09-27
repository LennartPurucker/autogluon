from ..behavior_space import BehaviorFunction
from .diversity_metrics import LossCorrelation, LossCorrelationRegression

# -- Make Behavior Functions
# - Diversity Metrics
LossCorrelationMeasureClassification = BehaviorFunction(
    LossCorrelation, ["y_true", "Y_pred_base_models"], (0, 1), "proba", name=LossCorrelation.name + "(Lower is more Diverse)"
)
LossCorrelationMeasureRegression = BehaviorFunction(
    LossCorrelationRegression, ["y_true", "Y_pred_base_models"], (0, 1), "proba", name=LossCorrelation.name + "(Lower is more Diverse)"
)

