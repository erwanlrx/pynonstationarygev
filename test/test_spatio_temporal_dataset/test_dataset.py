import unittest
from itertools import product

import numpy as np
from rpy2.rinterface import RRuntimeError

from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearNonStationaryLocationMarginModel
from extreme_estimator.extreme_models.utils import set_seed_for_test
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset, MarginDataset
from test.test_utils import load_test_max_stable_models, load_test_3D_spatial_coordinates, \
    load_test_1D_and_2D_spatial_coordinates, load_test_spatiotemporal_coordinates


class TestDataset(unittest.TestCase):
    nb_obs = 2
    nb_points = 2

    def test_max_stable_dataset_R1_and_R2(self):
        max_stable_models = load_test_max_stable_models()[:]
        coordinates = load_test_1D_and_2D_spatial_coordinates(self.nb_points)
        for coordinates, max_stable_model in product(coordinates, max_stable_models):
            dataset = MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                                     max_stable_model=max_stable_model,
                                                     coordinates=coordinates)
            assert len(dataset.df_dataset.columns) == self.nb_obs + dataset.coordinates.nb_coordinates
        self.assertTrue(True)

    def test_max_stable_dataset_crash_R3(self):
        """Test to warn me when spatialExtremes handles R3"""
        with self.assertRaises(RRuntimeError):
            smith_process = load_test_max_stable_models()[0]
            coordinates = load_test_3D_spatial_coordinates(nb_points=self.nb_points)[0]
            MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                           max_stable_model=smith_process,
                                           coordinates=coordinates)


class TestSpatioTemporalDataset(unittest.TestCase):
    nb_obs = 2
    nb_points = 3
    nb_steps = 2

    def setUp(self) -> None:
        set_seed_for_test(seed=42)
        self.coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[1]

    def load_dataset(self, nb_obs):
        smooth_margin_model = LinearNonStationaryLocationMarginModel(coordinates=self.coordinates,
                                                                     starting_point=1)
        self.dataset = MarginDataset.from_sampling(nb_obs=nb_obs,
                                                   margin_model=smooth_margin_model,
                                                   coordinates=self.coordinates)

    def test_spatio_temporal_array(self):
        self.load_dataset(nb_obs=1)

        # Load observation for time 0
        ind_time_0 = self.dataset.coordinates.ind_of_df_all_coordinates(coordinate_name=AbstractCoordinates.COORDINATE_T,
                                                                        value=0)
        observation_at_time_0_v1 = self.dataset.observations.df_maxima_gev.loc[ind_time_0].values.flatten()

        # Load observation correspond to time 0
        maxima_gev = self.dataset.maxima_gev_for_spatial_extremes_package()
        maxima_gev = np.transpose(maxima_gev)
        self.assertEqual(maxima_gev.shape, (3, 2))
        observation_at_time_0_v2 = maxima_gev[:, 0]
        equality = np.equal(observation_at_time_0_v1, observation_at_time_0_v2).all()
        self.assertTrue(equality, msg='v1={} is different from v2={}'.format(observation_at_time_0_v1,
                                                                             observation_at_time_0_v2))

    def test_spatio_temporal_case_to_resolve(self):
        self.load_dataset(nb_obs=2)
        print(self.dataset.maxima_gev_for_spatial_extremes_package())


if __name__ == '__main__':
    unittest.main()
