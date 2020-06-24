import unittest

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import \
    NonStationaryQuadraticLocationModel, \
    NonStationaryQuadraticScaleModel, NonStationaryQuadraticLocationGumbelModel, NonStationaryQuadraticScaleGumbelModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    NonStationaryLocationSpatioTemporalLinearityModel, NonStationaryLocationSpatioTemporalLinearityModel2
from extreme_fit.model.margin_model.polynomial_margin_model.utils import ALTITUDINAL_MODELS
from extreme_trend.abstract_gev_trend_test import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.utils import r, set_seed_r
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.two_fold_datasets_generator import TwoFoldDatasetsGenerator
from projects.altitude_spatial_model.altitudes_fit.two_fold_fit import TwoFoldFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.slicer.split import Split
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from test.test_projects.test_contrasting.test_two_fold_fit import load_two_fold_fit


class TestGevTemporalQuadraticExtremesMle(unittest.TestCase):

    def get_estimator_fitted(self, model_class):
        altitudes = [900, 1200]
        study_class = SafranSnowfall1Day
        studies = AltitudesStudies(study_class, altitudes, year_max=2019)
        two_fold_datasets_generator = TwoFoldDatasetsGenerator(studies, nb_samples=1, massif_names=['Vercors'])
        model_family_name_to_model_class = {'Non stationary': [model_class]}
        two_fold_fit = TwoFoldFit(two_fold_datasets_generator=two_fold_datasets_generator,
                                  model_family_name_to_model_classes=model_family_name_to_model_class,
                                  fit_method=MarginFitMethod.extremes_fevd_mle)
        massif_fit = two_fold_fit.massif_name_to_massif_fit['Vercors']
        sample_fit = massif_fit.sample_id_to_sample_fit[0]
        model_fit = sample_fit.model_class_to_model_fit[model_class]  # type: TwoFoldModelFit
        estimator = model_fit.estimator_fold_1
        return estimator

    def common_test(self, model_class):
        estimator = self.get_estimator_fitted(model_class)
        # Assert that indicators are correctly computed
        self.assertAlmostEqual(estimator.result_from_model_fit.nllh, estimator.nllh(split=estimator.train_split))
        self.assertAlmostEqual(estimator.result_from_model_fit.aic, estimator.aic(split=estimator.train_split))
        self.assertAlmostEqual(estimator.result_from_model_fit.bic, estimator.bic(split=estimator.train_split))

    def test_location_spatio_temporal_models(self):
        for model_class in [NonStationaryLocationSpatioTemporalLinearityModel,
                            NonStationaryLocationSpatioTemporalLinearityModel2]:
            self.common_test(model_class)

    def test_altitudinal_models(self):
        for model_class in ALTITUDINAL_MODELS:
            self.common_test(model_class)


if __name__ == '__main__':
    unittest.main()
