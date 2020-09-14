import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.snowfall_plot import \
    fit_linear_regression
from projects.exceeding_snow_loads.utils import paper_altitudes


class PointwiseGevStudyVisualizer(AltitudesStudies):

    def __init__(self, study_class, altitudes, spatial_transformation_class=None, temporal_transformation_class=None,
                 **kwargs_study):
        super().__init__(study_class, altitudes, spatial_transformation_class, temporal_transformation_class,
                         **kwargs_study)
        # self.altitudes_for_temporal_hypothesis = [min(self.altitudes), 2100, max(self.altitudes)]
        self.altitudes_for_temporal_hypothesis = [900, 2100, 3000]

    def plot_gev_params_against_altitude(self):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_against_altitude_one_massif(ax, massif_name, param_name)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Altitude')
            ax.set_ylabel(param_name)
            plot_name = '{} change /with altitude'.format(param_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_against_altitude_one_massif(self, ax, massif_name, param_name):
        altitudes = []
        params = []
        confidence_intervals = []
        for altitude, study in self.altitude_to_study.items():
            if massif_name in study.study_massif_names:
                altitudes.append(altitude)
                gev_params = study.massif_name_to_stationary_gev_params[massif_name]
                params.append(gev_params.to_dict()[param_name])
                confidence_intervals.append(gev_params.param_name_to_confidence_interval[param_name])
        massif_id = self.study.all_massif_names().index(massif_name)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, params, fill=False)
        # plot_against_altitude(altitudes, ax, massif_id, massif_name, confidence_intervals, fill=True)

    # Plot against the time

    @property
    def year_min_and_max_list(self):
        l = []
        year_min, year_max = 1959, 1989
        for shift in range(0, 7):
            l.append((year_min + 5 * shift, year_max + 5 * shift))
        return l

    @property
    def min_years_for_plot_x_axis(self):
        return [c[0] for c in self.year_min_and_max_list]

    def plot_gev_params_against_time_for_all_altitudes(self):
        for altitude in self.altitudes_for_temporal_hypothesis:
            self._plot_gev_params_against_time_for_one_altitude(altitude)

    def _plot_gev_params_against_time_for_one_altitude(self, altitude):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_against_time_for_one_altitude_and_one_massif(ax, massif_name, param_name,
                                                                                   altitude,
                                                                                   massif_name_as_labels=True)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Year')
            ax.set_ylabel(param_name + ' for altitude={}'.format(altitude))
            xlabels = ['-'.join([str(e) for e in t]) for t in self.year_min_and_max_list]
            ax.set_xticks(self.min_years_for_plot_x_axis)
            ax.set_xticklabels(xlabels)
            # ax.tick_params(labelsize=5)
            plot_name = '{} change /all with years /for altitude={}'.format(param_name, altitude)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_against_time_for_one_altitude_and_one_massif(self, ax, massif_name, param_name, altitude,
                                                                      massif_name_as_labels):
        study = self.altitude_to_study[altitude]
        if massif_name in study.study_massif_names:
            gev_params_list = study.massif_name_to_gev_param_list(self.year_min_and_max_list)[massif_name]
            params = [gev_params.to_dict()[param_name] for gev_params in gev_params_list]
            # params = np.array(params)
            # param_normalized = params / np.sqrt(np.sum(np.power(params, 2)))
            # confidence_intervals = [gev_params.param_name_to_confidence_interval[param_name] for gev_params in
            #                         gev_params_list]
            massif_id = self.study.all_massif_names().index(massif_name)
            plot_against_altitude(self.min_years_for_plot_x_axis, ax, massif_id, massif_name, params,
                                  altitude, False,
                                  massif_name_as_labels)
            # plot_against_altitude(self.years, ax, massif_id, massif_name, confidence_intervals, True)

    # plot for each massif against the time

    def plot_gev_params_against_time_for_all_massifs(self):
        for massif_name in self.study.all_massif_names():
            self._plot_gev_params_against_time_for_one_massif(massif_name)

    def _plot_gev_params_against_time_for_one_massif(self, massif_name):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for altitude in self.altitudes_for_temporal_hypothesis:
                self._plot_gev_params_against_time_for_one_altitude_and_one_massif(ax, massif_name, param_name,
                                                                                   altitude,
                                                                                   massif_name_as_labels=False)
            ax.legend()
            ax.set_xlabel('Year')
            ax.set_ylabel(param_name + ' for {}'.format(massif_name))
            xlabels = ['-'.join([str(e) for e in t]) for t in self.year_min_and_max_list]
            ax.set_xticks(self.min_years_for_plot_x_axis)
            ax.set_xticklabels(xlabels)
            plot_name = '{} change /with years /for {}'.format(param_name, massif_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    # PLot for each massif the derivative against the time for each altitude

    def plot_time_derivative_against_time(self):
        for param_name in GevParams.PARAM_NAMES[:]:
            ax = plt.gca()
            for massif_name in self.study.all_massif_names()[:]:
                self._plot_gev_params_time_derivative_against_altitude_one_massif(ax, massif_name, param_name)
            ax.legend(prop={'size': 7}, ncol=3)
            ax.set_xlabel('Altitude')
            ax.set_ylabel(param_name)
            plot_name = '{} change /time derivative with altitude'.format(param_name)
            self.show_or_save_to_file(plot_name, no_title=True, tight_layout=True, show=False)
            ax.clear()
            plt.close()

    def _plot_gev_params_time_derivative_against_altitude_one_massif(self, ax, massif_name, param_name):
        altitudes = []
        time_derivatives = []
        for altitude, study in self.altitude_to_study.items():
            if (massif_name in study.study_massif_names) and ("Mercan" not in massif_name):
                gev_params_list = study.massif_name_to_gev_param_list(self.year_min_and_max_list)[massif_name]
                params = [gev_params.to_dict()[param_name] for gev_params in gev_params_list]
                x = list(range(len(params)))
                y = params
                a = self.get_robust_slope(x, y)
                time_derivatives.append(a)
                altitudes.append(altitude)
        massif_id = self.study.all_massif_names().index(massif_name)
        plot_against_altitude(altitudes, ax, massif_id, massif_name, time_derivatives, fill=False)

    def get_robust_slope(self, x, y):
        a, *_ = fit_linear_regression(x=x, y=y)
        a_list = [a]
        for i in range(len(x)):
            x_copy, y_copy = x[:], y[:]
            x_copy.pop(i)
            y_copy.pop(i)
            a, *_ = fit_linear_regression(x=x_copy, y=y_copy)
            a_list.append(a)
        return np.mean(np.array(a_list))


if __name__ == '__main__':
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]
    # altitudes = paper_altitudes
    # altitudes = [1800, 2100]
    visualizer = PointwiseGevStudyVisualizer(SafranSnowfall1Day, altitudes=altitudes)
    visualizer.plot_gev_params_against_altitude()
    visualizer.plot_gev_params_against_time_for_all_altitudes()
    visualizer.plot_gev_params_against_time_for_all_massifs()
    # visualizer.plot_time_derivative_against_time()
