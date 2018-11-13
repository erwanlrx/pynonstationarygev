from extreme_estimator.R_model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.R_model.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractFullEstimator(AbstractEstimator):
    pass


class SmoothMarginalsThenUnitaryMsp(AbstractFullEstimator):

    def __init__(self, dataset: AbstractDataset, margin_model: AbstractMarginModel,
                 max_stable_model: AbstractMaxStableModel):
        super().__init__(dataset)
        # Instantiate the two associated estimators
        self.margin_estimator = SmoothMarginEstimator(dataset=dataset, margin_model=margin_model)
        self.max_stable_estimator = MaxStableEstimator(dataset=dataset, max_stable_model=max_stable_model)

    def _fit(self):
        # Estimate the margin parameters
        self.margin_estimator.fit()
        # Compute the maxima_frech
        maxima_frech = self.margin_estimator.margin_model.gev2frech(maxima_gev=self.dataset.maxima_gev,
                                                                    df_gev_params=self.margin_estimator.df_gev_params)
        # Update maxima frech field through the dataset object
        self.dataset.maxima_frech = maxima_frech
        # Estimate the max stable parameters
        self.max_stable_estimator.fit()


class FullEstimatorInASingleStep(AbstractFullEstimator):
    pass


class FullEstimatorInASingleStepWithSmoothMarginals(AbstractFullEstimator):
    """The method of Gaume, check if its method is in a single step or not"""
    pass


class PointwiseAndThenUnitaryMsp(AbstractFullEstimator):
    pass


class StochasticExpectationMaximization(AbstractFullEstimator):
    pass


class INLAgoesExtremes(AbstractFullEstimator):
    pass
