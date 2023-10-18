import datetime
import io
import os
import os.path as op
from collections import OrderedDict
from contextlib import redirect_stdout
from itertools import chain
from multiprocessing.pool import Pool
from typing import List, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from netCDF4 import Dataset

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.utils import ALTITUDES, ZS_INT_23, ZS_INT_MASK, LONGITUDES, \
    LATITUDES, ORIENTATIONS, SLOPES, ORDERED_ALLSLOPES_ALTITUDES, ORDERED_ALLSLOPES_ORIENTATIONS, \
    ORDERED_ALLSLOPES_SLOPES, ORDERED_ALLSLOPES_MASSIFNUM, date_to_str, Season, \
    first_day_and_last_day, FrenchRegion, ZS_INT_MASK_PYRENNES, ZS_INT_MASK_PYRENNES_LIST, \
    season_to_str
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import create_colorbase_axis, \
    get_shifted_map, get_colors
from extreme_data.meteo_france_data.scm_models_data.visualization.utils import get_km_formatter
from root_utils import get_full_path, cached_property, NB_CORES, classproperty
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima

f = io.StringIO()
with redirect_stdout(f):
    from simpledbf import Dbf5

filled_marker_legend_list = ['Filled marker =', 'Selected model is significant', 'w.r.t $\mathcal{M}_0$']
filled_marker_legend_list2 = ['Filled marker = Selected', 'model is significant', 'w.r.t $\mathcal{M}_0$']

YEAR_MIN = 1959
YEAR_MAX = 2019


class AbstractStudy(object):
    """
    A Study is defined by:
        - a variable class that correspond to the meteorogical quantity of interest
        - an altitude of interest
        - a start and a end year

    Les fichiers netcdf de SAFRAN et CROCUS sont autodocumentés (on peut les comprendre avec ncdump -h notamment).

    The year 2017 represents the nc file that correspond to the winter between the year 2017 and 2018.
    """
    # REANALYSIS_FLAT_FOLDER = 'SAFRAN_montagne-CROCUS_2019/alp_flat/reanalysis'
    REANALYSIS_ALPS_FLAT_FOLDER = 'alps_flat/'
    REANALYSIS_PYRENEES_FLAT_FOLDER = 'pyrennes_flat/'
    REANALYSIS_ALPS_ALLSLOPES_FOLDER = 'alps_allslopes'

    YEAR_MIN = 1959
    YEAR_MAX = 2019

    # REANALYSIS_FOLDER = 'SAFRAN_montagne-CROCUS_2019/postes/reanalysis'

    def __init__(self, variable_class: type, altitude: int = 1800,
                 year_min=YEAR_MIN, year_max=YEAR_MAX,
                 multiprocessing=True, orientation=None, slope=20.0,
                 season=Season.annual,
                 french_region=FrenchRegion.alps,
                 split_years=None):
        assert isinstance(altitude, int), type(altitude)
        assert altitude in ALTITUDES, altitude
        self.french_region = french_region
        self.altitude = altitude
        self.model_name = None
        self.variable_class = variable_class
        assert self.YEAR_MIN <= year_min <= year_max <= self.YEAR_MAX
        self.year_min = year_min
        self.year_max = year_max
        self.multiprocessing = multiprocessing
        self.season = season
        if split_years is None:
            split_years = list(range(year_min, year_max + 1))
        self.split_years = set(split_years)
        # Add some attributes, for the "allslopes" reanalysis
        assert orientation is None or orientation in ORIENTATIONS
        assert slope in SLOPES
        self.orientation = orientation
        self.slope = slope
        # Add some cache for computation
        self._cache_for_pointwise_fit = {}
        self._massif_names_for_cache = None

    """ Time """

    @cached_property
    def year_to_first_index_and_last_index(self):
        year_to_first_index_and_last_index = OrderedDict()
        first_day, last_day = first_day_and_last_day(self.season)
        for year, all_days in self.year_to_all_days.items():
            year_first_index = year - 1 if self.season is not Season.spring else year
            year_last_index = year - 1 if self.season is Season.autumn else year
            first_index = all_days.index('{}-{}'.format(year_first_index, first_day))
            last_index = all_days.index('{}-{}'.format(year_last_index, last_day))
            year_to_first_index_and_last_index[year] = (first_index, last_index)
        return year_to_first_index_and_last_index

    @cached_property
    def year_to_days(self) -> OrderedDict:
        year_to_days = OrderedDict()
        for year, (start_index, last_index) in self.year_to_first_index_and_last_index.items():
            year_to_days[year] = self.year_to_all_days[year][start_index:last_index + 1]
        return year_to_days

    @cached_property
    def year_to_all_days(self) -> OrderedDict:
        # Map each year to the 'days since year-08-01 06:00:00'
        year_to_days = OrderedDict()
        for year in self.ordered_years:
            # Load days for the full year
            date = datetime.datetime(year=year - 1, month=8, day=1, hour=6, minute=0, second=0)
            days = []
            for i in range(366):
                day = date_to_str(date)
                days.append(day)
                date += datetime.timedelta(days=1)
                if date.month == 8 and date.day == 1:
                    break
            year_to_days[year] = days
        return year_to_days

    @cached_property
    def massif_name_to_df_ordered_by_maxima(self):
        df_annual_maxima = pd.DataFrame(self.year_to_annual_maxima)
        df_wps = pd.DataFrame(self.year_to_wp_for_annual_maxima)
        massif_name_to_df_ordered_by_maxima = {}
        for massif_id, s_annual_maxima in df_annual_maxima.iterrows():
            massif_name = self.study_massif_names[massif_id]
            s_annual_maxima.sort_values(inplace=True, ascending=False)
            d = {
                'Year': s_annual_maxima.index,
                'Maxima': s_annual_maxima.values,
                'WP': df_wps.loc[massif_id, s_annual_maxima.index],
            }
            df = pd.DataFrame(d)
            df.set_index('Year', inplace=True)
            assert len(df) == self.nb_years
            massif_name_to_df_ordered_by_maxima[massif_name] = df
        assert set(self.study_massif_names) == set(massif_name_to_df_ordered_by_maxima.keys())
        return massif_name_to_df_ordered_by_maxima

    @cached_property
    def year_to_wp_for_annual_maxima(self):
        year_to_wp_for_annual_maxima = OrderedDict()
        for year, idx in self.year_to_annual_maxima_index.items():
            wps_for_annual_maxima = self.year_to_wps[year][idx]
            year_to_wp_for_annual_maxima[year] = wps_for_annual_maxima
        return year_to_wp_for_annual_maxima

    @property
    def all_days(self):
        return list(chain(*list(self.year_to_days.values())))

    @property
    def all_daily_series(self) -> np.ndarray:
        """Return an array of approximate shape (total_number_of_days, 23) x """
        all_daily_series = np.concatenate([time_serie_array
                                           for time_serie_array in self.year_to_daily_time_serie_array.values()])
        assert len(all_daily_series) == len(self.all_days)
        return all_daily_series

    """ Annual maxima """

    @property
    def observations_annual_maxima(self) -> AnnualMaxima:
        return AnnualMaxima(df_maxima_gev=pd.DataFrame(self.year_to_annual_maxima, index=self.study_massif_names))

    @cached_property
    def observations_annual_mean(self) -> pd.DataFrame:
        return pd.DataFrame(self.year_to_annual_mean, index=self.study_massif_names)

    def annual_maxima_and_years(self, massif_name) -> Tuple[np.ndarray, np.ndarray]:
        df = self.observations_annual_maxima.df_maxima_gev
        return df.loc[massif_name].values, np.array(df.columns)

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_maxima = OrderedDict()
        for year, time_serie in self._year_to_max_daily_time_serie.items():
            year_to_annual_maxima[year] = time_serie.max(axis=0)
        return year_to_annual_maxima

    @cached_property
    def year_to_annual_mean(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_mean = OrderedDict()
        for year, time_serie in self._year_to_max_daily_time_serie.items():
            year_to_annual_mean[year] = time_serie.mean(axis=0)
        return year_to_annual_mean

    @cached_property
    def year_to_annual_maxima_index(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_maxima = OrderedDict()
        for year, time_serie in self._year_to_max_daily_time_serie.items():
            year_to_annual_maxima[year] = time_serie.argmax(axis=0)
        return year_to_annual_maxima

    @cached_property
    def year_to_annual_maxima_tuple_indices_for_daily_time_series(self):
        year_to_annual_maxima_indices_for_daily_time_series = OrderedDict()
        for year in self.ordered_years:
            l = [(idx, i) for i, idx in enumerate(self.year_to_annual_maxima_index[year])]
            year_to_annual_maxima_indices_for_daily_time_series[year] = l
        return year_to_annual_maxima_indices_for_daily_time_series

    @cached_property
    def massif_name_to_annual_maxima_ordered_index(self):
        massif_name_to_annual_maxima_ordered_index = OrderedDict()
        for i, (massif_name, years) in enumerate(self.massif_name_to_annual_maxima_ordered_years.items()):
            ordered_index = [self.year_to_annual_maxima_index[year][i] for year in years]
            massif_name_to_annual_maxima_ordered_index[massif_name] = ordered_index
        return massif_name_to_annual_maxima_ordered_index

    @cached_property
    def massif_name_to_annual_maxima_index(self):
        massif_name_to_annual_maxima_index = OrderedDict()
        for i, massif_name in enumerate(self.study_massif_names):
            index = [self.year_to_annual_maxima_index[year][i] for year in self.ordered_years]
            massif_name_to_annual_maxima_index[massif_name] = index
        return massif_name_to_annual_maxima_index

    @cached_property
    def massif_name_to_annual_maxima_angle(self):
        normalization_denominator = [366 if year % 4 == 0 else 365 for year in self.ordered_years]
        massif_name_to_annual_maxima_angle = OrderedDict()
        for massif_name, annual_maxima_index in self.massif_name_to_annual_maxima_index.items():
            angle = 2 * np.pi * np.array(annual_maxima_index) / np.array(normalization_denominator)
            massif_name_to_annual_maxima_angle[massif_name] = angle
        return massif_name_to_annual_maxima_angle

    @cached_property
    def massif_name_to_annual_maxima(self):
        massif_name_to_annual_maxima = OrderedDict()
        for i, massif_name in enumerate(self.study_massif_names):
            maxima = np.array([self.year_to_annual_maxima[year][i] for year in self.ordered_years])
            massif_name_to_annual_maxima[massif_name] = maxima
        return massif_name_to_annual_maxima

    def year_to_annual_maxima_for_a_massif(self, massif_name):
        maxima = self.massif_name_to_annual_maxima[massif_name]
        assert len(maxima) == len(self.ordered_years)
        return dict(zip(self.ordered_years, maxima))


    @cached_property
    def massif_name_to_annual_mean(self):
        massif_name_to_annual_mean = OrderedDict()
        for i, massif_name in enumerate(self.study_massif_names):
            maxima = np.array([self.year_to_annual_mean[year][i] for year in self.ordered_years])
            massif_name_to_annual_mean[massif_name] = maxima
        return massif_name_to_annual_mean

    @cached_property
    def massif_name_to_daily_time_series(self):
        massif_name_to_daily_time_series = OrderedDict()
        for i, massif_name in enumerate(self.study_massif_names):
            a = [self.year_to_daily_time_serie_array[year][:, i] for year in self.ordered_years]
            daily_time_series = np.array(list(chain.from_iterable(a)))
            massif_name_to_daily_time_series[massif_name] = daily_time_series
        return massif_name_to_daily_time_series

    @cached_property
    def massif_name_to_annual_maxima_ordered_years(self):
        massif_name_to_annual_maxima_ordered_years = OrderedDict()
        for massif_name in self.study_massif_names:
            maxima = self.massif_name_to_annual_maxima[massif_name]
            annual_maxima_ordered_index = np.argsort(maxima)
            annual_maxima_ordered_years = [self.ordered_years[idx] for idx in annual_maxima_ordered_index]
            massif_name_to_annual_maxima_ordered_years[massif_name] = annual_maxima_ordered_years
        return massif_name_to_annual_maxima_ordered_years

    """ Annual total """

    @property
    def df_annual_total(self) -> pd.DataFrame:
        return pd.DataFrame(self.year_to_annual_total, index=self.study_massif_names).transpose()

    def annual_aggregation_function(self, *args, **kwargs):
        raise NotImplementedError()

    @cached_property
    def year_to_annual_total(self) -> OrderedDict:
        # Map each year to an array of size nb_massif
        year_to_annual_mean = OrderedDict()
        for year, time_serie in self._year_to_daily_time_serie_array.items():
            year_to_annual_mean[year] = self.apply_annual_aggregation(time_serie)
        return year_to_annual_mean

    @cached_property
    def massif_name_to_annual_total(self):
        # Map each massif to an array of size nb_years
        massif_name_to_annual_total = OrderedDict()
        for i, massif_name in enumerate(self.study_massif_names):
            maxima = np.array([self.year_to_annual_total[year][i] for year in self.ordered_years])
            massif_name_to_annual_total[massif_name] = maxima
        return massif_name_to_annual_total

    def apply_annual_aggregation(self, time_serie):
        return self.annual_aggregation_function(time_serie, axis=0)

    """ Load daily observations """

    @cached_property
    def year_to_daily_time_serie_array(self) -> OrderedDict:
        return self._year_to_daily_time_serie_array

    @property
    def _year_to_max_daily_time_serie(self) -> OrderedDict:
        return self._year_to_daily_time_serie_array

    @property
    def _year_to_daily_time_serie_array(self) -> OrderedDict:
        # Map each year to a matrix of size 365-nb_days_consecutive+1 x nb_massifs
        year_to_daily_time_serie_array = OrderedDict()
        for year in self.ordered_years:
            # Check daily data
            daily_time_serie = self.daily_time_series(year)
            # Filter only the data corresponding:
            # 1: to treturnhe start_index and last_index of the season
            # 2: to the massifs for the altitude of interest
            assert daily_time_serie.shape == (len(self.year_to_days[year]), len(self.study_massif_names))
            year_to_daily_time_serie_array[year] = daily_time_serie
        return year_to_daily_time_serie_array

    def daily_time_series(self, year):
        daily_time_serie = self.year_to_variable_object[year].daily_time_serie_array
        nb_days = daily_time_serie.shape[0]
        assert nb_days == 365 or (nb_days == 366 and year % 4 == 0)
        assert daily_time_serie.shape[1] == len(self.column_mask)
        first_index, last_index = self.year_to_first_index_and_last_index[year]
        daily_time_serie = daily_time_serie[first_index:last_index + 1, self.column_mask]
        return daily_time_serie

    """ Load Variables and Datasets """

    @cached_property
    def year_to_variable_object(self) -> OrderedDict:
        # Map each year to the variable array
        path_files, ordered_years = self.ordered_years_and_path_files
        return self.efficient_variable_loading(ordered_years, path_files, multiprocessing=self.multiprocessing)

    def efficient_variable_loading(self, ordered_years, arguments, multiprocessing):
        if multiprocessing:
            with Pool(NB_CORES) as p:
                variables = p.map(self.load_variable_object, arguments)
        else:
            variables = [self.load_variable_object(argument) for argument in arguments]
        return OrderedDict(zip(ordered_years, variables))

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        return self.variable_class(variable_array)

    def load_variable_array(self, dataset):
        return np.array(dataset.variables[self.load_keyword()])

    def load_variable_object(self, path_file):
        dataset = Dataset(path_file)
        variable_array = self.load_variable_array(dataset)
        return self.instantiate_variable_object(variable_array)

    def load_keyword(self):
        return self.variable_class.keyword()

    @property
    def year_to_dataset_ordered_dict(self) -> OrderedDict:
        print('This code is quite long... '
              'You should consider year_to_variable which is way faster when multiprocessing=True')
        # Map each year to the correspond netCDF4 Dataset
        path_files, ordered_years = self.ordered_years_and_path_files
        datasets = [Dataset(path_file) for path_file in path_files]
        return OrderedDict(zip(ordered_years, datasets))

    @cached_property
    def ordered_years_and_path_files(self):
        nc_files = [(int(f.split('_')[-2][:4]) + 1, f) for f in os.listdir(self.study_full_path) if f.endswith('.nc')]
        ordered_years, path_files = zip(*[(year, op.join(self.study_full_path, nc_file))
                                          for year, nc_file in sorted(nc_files, key=lambda t: t[0])
                                          if (self.year_min <= year <= self.year_max)
                                          and (year in self.split_years)])
        return path_files, ordered_years

    """ Temporal properties """

    @property
    def nb_years(self):
        return len(self.ordered_years)

    @property
    def ordered_years(self):
        return self.ordered_years_and_path_files[1]

    @property
    def start_year_and_stop_year(self) -> Tuple[int, int]:
        ordered_years = self.ordered_years
        return ordered_years[0], ordered_years[-1]

    """ Spatial properties """

    @property
    def study_massif_names(self) -> List[str]:
        # Massif names that are present in the current study (i.e. for the current altitude)
        return self.altitude_to_massif_names[self.altitude]

    @property
    def nb_study_massif_names(self) -> int:
        return len(self.study_massif_names)

    @property
    def df_massifs_longitude_and_latitude(self) -> pd.DataFrame:
        # DataFrame object that represents the massif coordinates in degrees extracted from the SCM data
        # Another way of getting the latitudes and longitudes could have been the following:
        # any_ordered_dict = list(self.year_to_dataset_ordered_dict.values())[0]
        # longitude = np.array(any_ordered_dict.variables['longitude'])
        # latitude = np.array(any_ordered_dict.variables['latitude'])
        longitude = np.array(LONGITUDES)
        latitude = np.array(LATITUDES)
        columns = [AbstractSpatialCoordinates.COORDINATE_X, AbstractSpatialCoordinates.COORDINATE_Y]
        data = dict(zip(columns, [longitude[self.flat_mask], latitude[self.flat_mask]]))
        return pd.DataFrame(data=data, index=self.study_massif_names, columns=columns)

    @property
    def df_latitude_longitude(self):
        any_ordered_dict = list(self.year_to_dataset_ordered_dict.values())[0]
        print(any_ordered_dict.variables.keys())
        longitude = np.array(any_ordered_dict.variables['LON'])[self.flat_mask]
        latitude = np.array(any_ordered_dict.variables['LAT'])[self.flat_mask]
        data = [longitude, latitude]
        df = pd.DataFrame(data=data, index=['Longitude', 'Latitude'], columns=self.study_massif_names).transpose()
        return df

    @property
    def missing_massif_name(self):
        return set(self.all_massif_names(self.reanalysis_path, self.dbf_filename)) - set(
            self.altitude_to_massif_names[self.altitude])

    @property
    def column_mask(self):
        return self.allslopes_mask if self.has_orientation else self.flat_mask

    @property
    def allslopes_mask(self):
        altitude_mask = np.array(ORDERED_ALLSLOPES_ALTITUDES) == self.altitude
        orientation_mask = np.array(ORDERED_ALLSLOPES_ORIENTATIONS) == self.orientation
        slope_mask = np.array(ORDERED_ALLSLOPES_SLOPES) == self.slope
        allslopes_mask = altitude_mask & orientation_mask & slope_mask
        # Exclude all the data corresponding to the 24th massif
        massif_24_mask = np.array(ORDERED_ALLSLOPES_MASSIFNUM) == 30
        return allslopes_mask & ~massif_24_mask

    @cached_property
    def flat_mask(self):
        if self.french_region is FrenchRegion.alps:
            altitude_mask = ZS_INT_MASK == self.altitude
        elif self.french_region is FrenchRegion.pyrenees:
            altitude_mask = ZS_INT_MASK_PYRENNES == self.altitude
        else:
            raise ValueError('{}'.format(self.french_region))
        assert np.sum(altitude_mask) == len(self.altitude_to_massif_names[self.altitude])
        return altitude_mask

    """ Path properties """

    @property
    def title(self):
        return "{}/at {}m ({} massifs)".format(self.variable_name, self.altitude,
                                                                len(self.study_massif_names))

    @property
    def variable_name(self):
        return self.variable_class.NAME + ' ({})'.format(self.variable_unit)

    @property
    def variable_unit(self):
        return self.variable_class.UNIT

    """ Visualization methods """

    @classmethod
    def massifs_coordinates_for_display(cls, massif_names) -> AbstractSpatialCoordinates:
        # Coordinate object that represents the massif coordinates in Lambert extended
        # extracted for a csv file, and used only for display purposes
        df = cls.load_df_centroid()
        # Lower a bit the Mercantour massif
        df.loc['Mercantour', 'coord_x'] += 14000  # shift to the right
        df.loc['Mercantour', 'coord_y'] -= 7000  # shift down
        # Lower a bit the Maurienne massif
        # df.loc['Mercantour', 'coord_x'] += 14000 # shift to the right
        df.loc['Maurienne', 'coord_y'] -= 6000  # shift down
        df.loc['Maurienne', 'coord_y'] -= 5000  # shift down
        df.loc['Maurienne', 'coord_x'] += 3000  # shift down
        df.loc['Vanoise', 'coord_y'] -= 4000  # shift down
        df.loc['Ubaye', 'coord_y'] -= 4000  # shift down
        # Filter, keep massifs present at the altitude of interest
        df = df.loc[massif_names, :]

        # Build coordinate object from df_centroid
        return AbstractSpatialCoordinates.from_df(df)

    @classmethod
    def visualize_study(cls, ax=None, massif_name_to_value: Union[None, Dict[str, float]] = None,
                        show=True, fill=True,
                        replace_blue_by_white=False,
                        label=None, add_text=False, cmap=None, add_colorbar=False, vmax=100, vmin=0,
                        default_color_for_missing_massif='gainsboro',
                        default_color_for_nan_values='w',
                        massif_name_to_color=None,
                        show_label=True,
                        scaled=True,
                        fontsize=8,
                        axis_off=False,
                        massif_name_to_hatch_boolean_list=None,
                        norm=None,
                        massif_name_to_marker_style=None,
                        marker_style_to_label_name=None,
                        ticks_values_and_labels=None,
                        massif_name_to_text=None,
                        fontsize_label=15,
                        add_legend=True,
                        massif_names_with_white_dot=None
                        ):
        if ax is None:
            ax = plt.gca()

        if massif_name_to_value is not None:
            massif_names, values = list(zip(*massif_name_to_value.items()))
        else:
            massif_names, values = None, None

        if massif_name_to_color is None:
            # Load the colors
            if cmap is None:
                cmap = get_shifted_map(vmin, vmax)
            norm = Normalize(vmin, vmax)
            colors = get_colors(values, cmap, vmin, vmax, replace_blue_by_white)
            massif_name_to_color = dict(zip(massif_names, colors))
        massif_name_to_fill_kwargs = {massif_name: {'color': color} for massif_name, color in
                                      massif_name_to_color.items()}
        massif_names = list(massif_name_to_fill_kwargs.keys())
        masssif_coordinate_for_display = cls.massifs_coordinates_for_display(massif_names)

        for coordinate_id, coords_list in cls.idx_to_coords_list.items():
            # Retrieve the list of coords (x,y) that define the contour of the massif of id coordinate_id
            # Plot the contour of the massif
            coords_list = list(zip(*coords_list))
            ax.plot(*coords_list, color='black')

            # Potentially, fill the inside of the polygon with some color
            if fill and coordinate_id in cls.coordinate_id_to_massif_name:
                massif_name = cls.coordinate_id_to_massif_name[coordinate_id]
                if massif_name_to_marker_style is not None and massif_name in massif_name_to_marker_style:
                    massif_coordinate = masssif_coordinate_for_display.df_all_coordinates.loc[massif_name, :].values
                    ax.plot(massif_coordinate[0],
                            massif_coordinate[1], **massif_name_to_marker_style[massif_name])

                if massif_name_to_fill_kwargs is not None and massif_name in massif_name_to_fill_kwargs:
                    fill_kwargs = massif_name_to_fill_kwargs[massif_name]
                    ax.fill(*coords_list, **fill_kwargs)
                    if massif_names_with_white_dot is not None and massif_name in massif_names_with_white_dot:
                        fill_kwargs = {"facecolor":"none", "hatch":"o"*2, "edgecolor":"w"}
                        ax.fill(*coords_list, **fill_kwargs)
                else:
                    # For the missing massifs
                    ax.fill(*coords_list, **{'color': default_color_for_missing_massif})

                # Add a hatch for the missing massifs
                hatch_list = ['//', '\\\\']
                if massif_name_to_hatch_boolean_list is not None:
                    if massif_name in massif_name_to_hatch_boolean_list:
                        a = np.array(coords_list).transpose()
                        hatch_boolean_list = massif_name_to_hatch_boolean_list[massif_name]
                        for hatch, is_hatch in zip(hatch_list, hatch_boolean_list):
                            if is_hatch:
                                ax.add_patch(Polygon(xy=a, fill=False, hatch=hatch))

        if show_label:
            # Improve some explanation on the X axis and on the Y axis
            ax.set_xlabel('Longitude (km)')
            ax.xaxis.set_major_formatter(get_km_formatter())
            ax.set_ylabel('Latitude (km)')
            ax.yaxis.set_major_formatter(get_km_formatter())
        else:
            # Remove the ticks
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # Display the name or value of the massif
        if add_text:
            for _, row in masssif_coordinate_for_display.df_all_coordinates.iterrows():
                x, y = list(row)
                massif_name = row.name
                if massif_name_to_text is None:
                    value = massif_name_to_value[massif_name]
                    str_value = str(value)
                else:
                    str_value = massif_name_to_text[massif_name]
                ax.text(x, y, str_value, horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
                        weight="bold", fontweight="bold")

        if scaled:
            plt.axis('scaled')

        # create the colorbar only at the end
        if add_colorbar:
            if len(set(values)) > 1:
                create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels=ticks_values_and_labels,
                                      fontsize=fontsize_label)
        if axis_off:
            plt.axis('off')

        # Add legend for the marker
        if add_legend and massif_name_to_marker_style is not None:
            legend_elements = cls.get_legend_for_model_symbol(marker_style_to_label_name, markersize=7)
            ax.legend(handles=legend_elements[:], loc='upper right', prop={'size': 9})
            # ax.legend(handles=legend_elements[4:], bbox_to_anchor=(0.01, 0.03), loc='lower left')
            # ax.annotate(' '.join(filled_marker_legend_list),
            #             xy=(0.05, 0.015), xycoords='axes fraction', fontsize=7)

        if show:
            plt.show()

        return ax

    @classmethod
    def get_legend_for_model_symbol(cls, marker_style_to_label_name, markersize):
        legend_elements = [
            Line2D([0], [0], marker=marker, color='w', label='${}$'.format(label),
                   markerfacecolor='w', markeredgecolor='k', markersize=markersize)
            for marker, label in marker_style_to_label_name.items()
        ]
        return legend_elements

    """ 
    CLASS ATTRIBUTES COMMON TO ALL OBJECTS 
    (written as object attributes/methods for simplicity)
    """

    """ Path properties """

    @classproperty
    def relative_path(self) -> str:
        return r'data'

    @classproperty
    def full_path(self) -> str:
        return get_full_path(relative_path=self.relative_path)

    @classproperty
    def map_full_path(self) -> str:
        return op.join(self.full_path, 'map')

    @classproperty
    def result_full_path(cls) -> str:
        return op.join(op.dirname(cls.full_path), 'results')

    @property
    def study_full_path(self) -> str:
        assert self.model_name in ['Safran', 'Crocus']
        study_folder = 'meteo' if self.model_name is 'Safran' else 'pro'
        return op.join(self.reanalysis_path, study_folder)

    @property
    def reanalysis_path(self):
        if self.french_region is FrenchRegion.alps:
            if self.has_orientation:
                reanalysis_folder = self.REANALYSIS_ALPS_ALLSLOPES_FOLDER
            else:
                reanalysis_folder = self.REANALYSIS_ALPS_FLAT_FOLDER
        elif self.french_region is FrenchRegion.pyrenees and not self.has_orientation:
            reanalysis_folder = self.REANALYSIS_PYRENEES_FLAT_FOLDER
        else:
            raise ValueError(
                'french_region = {}, has_orientation = {}'.format(self.french_region, self.has_orientation))

        return op.join(self.full_path, reanalysis_folder)

    @property
    def dbf_filename(self) -> str:
        if self.french_region is FrenchRegion.alps:
            return 'massifs_alpes'
        elif self.french_region is FrenchRegion.pyrenees:
            return 'massifs_pyrenees'
        else:
            raise ValueError('{}'.format(self.french_region))

    @property
    def has_orientation(self):
        return self.orientation is not None

    @property
    def season_name(self):
        return season_to_str(self.season)

    """  Spatial properties """

    @cached_property
    def massif_name_to_massif_id(self):
        return {name: i for i, name in enumerate(self.study_massif_names)}

    @classproperty
    def original_safran_massif_id_to_massif_name(cls) -> Dict[int, str]:
        return {massif_id: massif_name for massif_id, massif_name in enumerate(cls.all_massif_names)}

    @classmethod
    def all_massif_names(cls, reanalysis_path=None, dbf_filename='massifs_alpes') -> List[str]:
        """
        Pour l'identification des massifs, le numéro de la variable massif_num correspond à celui de l'attribut num_opp
        """
        if reanalysis_path is None:
            # Default case if the french alps
            reanalysis_path = op.join(cls.full_path, cls.REANALYSIS_ALPS_FLAT_FOLDER)
        if cls.REANALYSIS_ALPS_FLAT_FOLDER in reanalysis_path or cls.REANALYSIS_ALPS_ALLSLOPES_FOLDER in reanalysis_path:
            french_region = FrenchRegion.alps
            key = 'num_opp'
        else:
            french_region = FrenchRegion.pyrenees
            key = 'massif_num'

        metadata_path = op.join(cls.full_path, 'metadata')
        dbf = Dbf5(op.join(metadata_path, '{}.dbf'.format(dbf_filename)))
        df = dbf.to_dataframe().copy()  # type: pd.DataFrame
        dbf.f.close()
        # Important part (for the alps & pyrenees all data is order from the smaller massif number to the bigger)
        df.sort_values(by=key, inplace=True)
        all_massif_names = list(df['nom'])
        # Correct a massif name
        if french_region is FrenchRegion.alps:
            all_massif_names[all_massif_names.index('Beaufortin')] = 'Beaufortain'
        return all_massif_names

    @classmethod
    def load_df_centroid(cls) -> pd.DataFrame:
        # Load df_centroid containing all the massif names
        df_centroid = pd.read_csv(op.join(cls.map_full_path, 'coordonnees_massifs_alpes.csv'))
        df_centroid.set_index('NOM', inplace=True)
        # Check that the names of massifs are the same
        symmetric_difference = set(df_centroid.index).symmetric_difference(cls.all_massif_names())
        assert len(symmetric_difference) == 0, symmetric_difference
        # Sort the column in the order of the SAFRAN dataset
        df_centroid = df_centroid.reindex(cls.all_massif_names(), axis=0)
        for coord_column in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            df_centroid.loc[:, coord_column] = df_centroid[coord_column].str.replace(',', '.').astype(float)
        return df_centroid

    @cached_property
    def massif_name_to_altitudes(self) -> Dict[str, List[int]]:
        zs = ZS_INT_23 if self.french_region is FrenchRegion.alps else ZS_INT_MASK_PYRENNES_LIST
        s = zs + [0]
        zs_list = []
        zs_all_list = []
        for a, b in zip(s[:-1], s[1:]):
            zs_list.append(a)
            if a > b:
                zs_all_list.append(zs_list)
                zs_list = []
        all_massif_names = self.all_massif_names(self.reanalysis_path, self.dbf_filename)
        return OrderedDict(zip(all_massif_names, zs_all_list))

    @cached_property
    def altitude_to_massif_names(self) -> Dict[int, List[str]]:
        altitude_to_massif_names = {altitude: [] for altitude in ALTITUDES}
        for massif_name in self.massif_name_to_altitudes.keys():
            for altitude in self.massif_name_to_altitudes[massif_name]:
                altitude_to_massif_names[altitude].append(massif_name)
        # massif_names are ordered in the same way as all_massif_names
        return altitude_to_massif_names

    """ Visualization methods """

    @classproperty
    def coordinate_id_to_massif_name(cls) -> Dict[int, str]:
        df_centroid = cls.load_df_centroid()
        return dict(zip(df_centroid['id'], df_centroid.index))

    @classproperty
    def idx_to_coords_list(self):
        df_massif = pd.read_csv(op.join(self.map_full_path, 'massifsalpes.csv'))
        coord_tuples = [(row_massif['idx'], row_massif[AbstractCoordinates.COORDINATE_X],
                         row_massif[AbstractCoordinates.COORDINATE_Y])
                        for _, row_massif in df_massif.iterrows()]
        all_idxs = set([t[0] for t in coord_tuples])
        return {idx: [coords for idx_loop, *coords in coord_tuples if idx == idx_loop] for idx in all_idxs}








