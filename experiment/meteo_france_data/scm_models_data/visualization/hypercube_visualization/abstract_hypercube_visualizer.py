import os
import os.path as op
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from utils import cached_property, VERSION_TIME, get_display_name_from_object_type


class AbstractHypercubeVisualizer(object):
    """
    A study visualizer contain some massifs and years. This forms the base DataFrame of the hypercube
    Additional index will come from the tuple.
    Tuple could contain altitudes, type of snow quantity
    """

    def __init__(self, tuple_to_study_visualizer: Dict[Tuple, StudyVisualizer],
                 trend_test_class,
                 nb_data_reduced_for_speed=False,
                 save_to_file=False,
                 first_starting_year=None,
                 last_starting_year=None,
                 exact_starting_year=None,
                 verbose=True):
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.trend_test_class = trend_test_class
        self.tuple_to_study_visualizer = tuple_to_study_visualizer  # type: Dict[Tuple, StudyVisualizer]

        if exact_starting_year is not None:
            assert first_starting_year is None
            assert last_starting_year is None
            self.first_starting_year, self.first_starting_year = exact_starting_year, exact_starting_year
        else:
            default_first_starting_year, *_, default_last_starting_year = self.all_potential_starting_years
            self.first_starting_year = first_starting_year if first_starting_year is not None else default_first_starting_year
            self.last_starting_year = last_starting_year if last_starting_year is not None else default_last_starting_year
        if isinstance(nb_data_reduced_for_speed, bool):
            self.nb_data_for_fast_mode = 7 if nb_data_reduced_for_speed else None
        else:
            assert isinstance(nb_data_reduced_for_speed, int)
            self.nb_data_for_fast_mode = nb_data_reduced_for_speed

        if self.verbose:
            print('Hypercube with parameters:')
            print('Starting years:', self.starting_years)
            print('Trend test class:', get_display_name_from_object_type(self.trend_test_class))

    # Main attributes defining the hypercube

    @property
    def trend_test_name(self):
        return get_display_name_from_object_type(self.trend_test_class)

    @property
    def all_potential_starting_years(self):
        return self.study_visualizer.starting_years

    @cached_property
    def starting_years(self):
        starting_years = [year for year in self.all_potential_starting_years if self.first_starting_year <= year <= self.last_starting_year]
        if self.nb_data_for_fast_mode is not None:
            starting_years = starting_years[:self.nb_data_for_fast_mode]
        return starting_years

    def tuple_values(self, idx):
        return sorted(set([t[idx] if isinstance(t, tuple) else t for t in self.tuple_to_study_visualizer.keys()]))

    @cached_property
    def df_trends_spatio_temporal(self):
        return [study_visualizer.df_trend_spatio_temporal(self.trend_test_class, self.starting_years,
                                                          self.nb_data_for_fast_mode)
                for study_visualizer in self.tuple_to_study_visualizer.values()]

    def _df_hypercube_trend_meta(self, idx) -> pd.DataFrame:
        df_spatio_temporal_trend_strength = [e[idx] for e in self.df_trends_spatio_temporal]
        return pd.concat(df_spatio_temporal_trend_strength, keys=list(self.tuple_to_study_visualizer.keys()), axis=0)


    @cached_property
    def df_hypercube_trend_type(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=0
                                             )
    @cached_property
    def df_hypercube_trend_strength(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=1)

    @cached_property
    def df_hypercube_trend_nllh(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=2)

    # Some properties

    @property
    def study_title(self):
        return self.study.title

    def show_or_save_to_file(self, specific_title=''):
        if self.save_to_file:
            main_title, *_ = '_'.join(self.study_title.split()).split('/')
            filename = "{}/{}/".format(VERSION_TIME, main_title)
            filename += specific_title
            filepath = op.join(self.study.result_full_path, filename + '.png')
            dirname = op.dirname(filepath)
            if not op.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(filepath)
        else:
            plt.show()
        plt.close()

    @property
    def study_visualizer(self) -> StudyVisualizer:
        return list(self.tuple_to_study_visualizer.values())[0]

    @property
    def study(self):
        return self.study_visualizer.study
