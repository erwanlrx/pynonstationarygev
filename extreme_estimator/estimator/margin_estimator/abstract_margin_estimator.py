from abc import ABC

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearMarginModel, \
    LinearAllParametersAllDimsMarginModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMarginEstimator(AbstractEstimator, ABC):

    def __init__(self, dataset: AbstractDataset):
        super().__init__(dataset)
        assert self.dataset.maxima_gev() is not None
        self._margin_function_fitted = None


class LinearMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def _error(self, true_max_stable_params: dict):
        pass

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel):
        super().__init__(dataset)
        assert isinstance(margin_model, LinearMarginModel)
        self.margin_model = margin_model

    def _fit(self):
        maxima_gev_specialized = self.dataset.maxima_gev_for_spatial_extremes_package(self.train_split)
        df_coordinates_spat = self.dataset.coordinates.df_spatial_coordinates(self.train_split)
        df_coordinates_temp = self.dataset.coordinates.df_temporal_coordinates(self.train_split)
        self._result_from_fit = self.margin_model.fitmargin_from_maxima_gev(data=maxima_gev_specialized,
                                                                            df_coordinates_spat=df_coordinates_spat,
                                                                            df_coordinates_temp=df_coordinates_temp)

    def extract_function_fitted(self):
        return self.extract_function_fitted_from_function_to_fit(self.margin_model.margin_function_start_fit)

