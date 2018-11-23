from rpy2.rinterface import RRuntimeError
import unittest
from itertools import product

from extreme_estimator.extreme_models.max_stable_model.utils import load_max_stable_models
from spatio_temporal_dataset.coordinates.utils import load_coordinates
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset


class TestDataset(unittest.TestCase):
    nb_obs = 10
    nb_points = 10

    def test_max_stable_dataset_R1_and_R2(self):
        max_stable_models = load_max_stable_models()[:]
        coordinatess = load_coordinates(self.nb_points)[:-1]
        for coordinates, max_stable_model in product(coordinatess, max_stable_models):
            MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                           max_stable_model=max_stable_model,
                                           coordinates=coordinates)
        self.assertTrue(True)

    def test_max_stable_dataset_crash_R3(self):
        # test to warn me when spatialExtremes handles R3
        with self.assertRaises(RRuntimeError):
            smith_process = load_max_stable_models()[0]
            coordinates_R3 = load_coordinates(self.nb_points)[-1]
            MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                           max_stable_model=smith_process,
                                           coordinates=coordinates_R3)


if __name__ == '__main__':
    unittest.main()
