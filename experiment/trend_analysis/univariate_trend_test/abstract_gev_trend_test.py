import pandas as pd
from scipy.stats import chi2
import numpy as np
from sklearn.preprocessing import normalize
from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import AbstractTrendTest
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.temporal_linear_margin_model import StationaryStationModel, \
    NonStationaryLocationStationModel
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractGevTrendTest(AbstractTrendTest):

    def __init__(self, years_after_change_point, maxima_after_change_point, non_stationary_model_class):
        super().__init__(years_after_change_point, maxima_after_change_point)
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: years_after_change_point})
        df_maxima_gev = pd.DataFrame(maxima_after_change_point, index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
        self.coordinates = AbstractTemporalCoordinates.from_df(df, transformation_class=BetweenZeroAndOneNormalization)
        # self.coordinates = AbstractTemporalCoordinates.from_df(df, transformation_class=CenteredScaledNormalization)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)

        # Fit stationary model
        self.stationary_estimator = LinearMarginEstimator(self.dataset, StationaryStationModel(self.coordinates))
        self.stationary_estimator.fit()

        # Fit non stationary model
        self.non_stationary_estimator = LinearMarginEstimator(self.dataset,
                                                              non_stationary_model_class(self.coordinates))
        self.non_stationary_estimator.fit()

    @property
    def likelihood_ratio(self):
        return 2 * (self.non_stationary_estimator.result_from_fit.deviance -
                    self.stationary_estimator.result_from_fit.deviance)

    @property
    def is_significant(self) -> bool:
        return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=1)


class GevLocationTrendTest(AbstractGevTrendTest):

    def __init__(self, years_after_change_point, maxima_after_change_point):
        super().__init__(years_after_change_point, maxima_after_change_point, NonStationaryLocationStationModel)

    @property
    def test_sign(self) -> int:
        return np.sign(self.non_stationary_estimator.margin_function_fitted.mu1_temporal_trend)
