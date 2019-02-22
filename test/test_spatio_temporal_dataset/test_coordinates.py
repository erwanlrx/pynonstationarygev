import unittest
import pandas as pd
from collections import Counter, OrderedDict

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    UniformSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import \
    CircleSpatialCoordinates
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class TestSpatialCoordinates(unittest.TestCase):
    DISPLAY = False

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.coord = None  # type:  AbstractCoordinates

    def tearDown(self):
        if self.DISPLAY:
            self.coord.visualize()
        self.assertTrue(True)

    def test_unif(self):
        self.coord = UniformSpatialCoordinates.from_nb_points(nb_points=10)

    def test_circle(self):
        self.coord = CircleSpatialCoordinates.from_nb_points(nb_points=500)

    def test_normalization(self):
        self.coord = AlpsStation2DCoordinatesBetweenZeroAndOne.from_csv()

    def test_anisotropy(self):
        self.coord = AlpsStation3DCoordinatesWithAnisotropy.from_csv()


class SpatioTemporalCoordinates(unittest.TestCase):
    nb_points = 4
    nb_steps = 2

    def test_temporal_circle(self):
        self.coordinates = UniformSpatioTemporalCoordinates.from_nb_points_and_nb_steps(nb_points=self.nb_points,
                                                                                        nb_steps=self.nb_steps,
                                                                                        train_split_ratio=0.5)
        c = Counter([len(self.coordinates.df_coordinates(split)) for split in SpatioTemporalSlicer.SPLITS])
        good_count = c == Counter([2, 2, 2, 2]) or c == Counter([0, 0, 4, 4])
        self.assertTrue(good_count)

    def test_ordered_coordinates(self):
        # Order coordinates, to ensure that the first dimension/the second dimension and so on..
        # Always are in the same order to a given type (e.g. spatio_temporal= of coordinates
        # Check space coordinates
        d = OrderedDict()
        d[AbstractCoordinates.COORDINATE_Z] = [1]
        d[AbstractCoordinates.COORDINATE_X] = [1]
        d[AbstractCoordinates.COORDINATE_Y] = [1]
        df = pd.DataFrame.from_dict(d)
        for df2 in [df, df.loc[:, ::-1]][-1:]:
            coordinates = AbstractCoordinates(df=df2, slicer_class=SpatioTemporalSlicer)
            self.assertEqual(list(coordinates.df_all_coordinates.columns),
                             [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y,
                              AbstractCoordinates.COORDINATE_Z])
        # Check space/time ordering
        d = OrderedDict()
        d[AbstractCoordinates.COORDINATE_T] = [1]
        d[AbstractCoordinates.COORDINATE_X] = [1]
        df = pd.DataFrame.from_dict(d)
        for df2 in [df, df.loc[:, ::-1]][-1:]:
            coordinates = AbstractCoordinates(df=df2, slicer_class=SpatioTemporalSlicer)
            self.assertEqual(list(coordinates.df_all_coordinates.columns),
                             [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_T])


if __name__ == '__main__':
    unittest.main()
