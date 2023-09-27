"""Selection of Pre-defined behavior spaces"""


def bs_configspace_similarity_and_loss_correlation():
    # "bs_configspace_similarity_and_loss_correlation"
    # - Used in original paper but not used here as ConfigSpaceGowerSimilarity would require config space
    #   to be defined. But AutoGluon has not real overlapping config space.
    from .behavior_space import BehaviorSpace
    from .behavior_functions.basic import LossCorrelationMeasure
    from .behavior_functions.implicit_diversity_metrics import ConfigSpaceGowerSimilarity

    bs = BehaviorSpace([ConfigSpaceGowerSimilarity, LossCorrelationMeasure])

    return bs


def bs_loss_correlation(classification_problem):
    # 1D behavior space that can be used for now
    from .behavior_space import BehaviorSpace
    from .behavior_functions.basic import LossCorrelationMeasureClassification, LossCorrelationMeasureRegression

    if classification_problem:
        msr = LossCorrelationMeasureClassification
    else:
        msr = LossCorrelationMeasureRegression

    bs = BehaviorSpace([msr])

    return bs
