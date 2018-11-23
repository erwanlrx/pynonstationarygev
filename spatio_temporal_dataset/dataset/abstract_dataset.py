import os
import numpy as np
import os.path as op
import pandas as pd
from spatio_temporal_dataset.temporal_observations.abstract_temporal_observations import AbstractTemporalObservations
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractDataset(object):

    def __init__(self, temporal_observations: AbstractTemporalObservations, coordinates: AbstractCoordinates):
        # is_same_index = temporal_observations.index == coordinates.index  # type: pd.Series
        # assert is_same_index.all()
        self.temporal_observations = temporal_observations
        self.coordinates = coordinates

    @classmethod
    def from_csv(cls, csv_path: str):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        temporal_maxima = AbstractTemporalObservations.from_df(df)
        coordinates = AbstractCoordinates.from_df(df)
        return cls(temporal_maxima, coordinates)

    def to_csv(self, csv_path: str):
        dirname = op.dirname(csv_path)
        if not op.exists(dirname):
            os.makedirs(dirname)
        self.df_dataset.to_csv(csv_path)

    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        return self.temporal_observations.df_maxima_gev.join(self.coordinates.df_coordinates)

    @property
    def df_coordinates(self):
        return self.coordinates.df_coordinates

    @property
    def coordinates_values(self):
        return self.coordinates.coordinates_values

    @property
    def maxima_gev(self) -> np.ndarray:
        return self.temporal_observations.maxima_gev

    @property
    def maxima_frech(self):
        return self.temporal_observations.maxima_frech

    @maxima_frech.setter
    def maxima_frech(self, maxima_frech_to_set):
        self.temporal_observations.maxima_frech = maxima_frech_to_set

