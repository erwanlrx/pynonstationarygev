from multiprocessing.pool import Pool

import matplotlib as mpl
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranPrecipitation1Day, \
    SafranPrecipitation3Days, SafranSnowfall3Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_trend.abstract_gev_trend_test import AbstractGevTrendTest
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.plot_selection_curves_paper2 import \
    plot_selection_curves_paper2
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.shape_plot import shape_plot
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.snowfall_plot import \
    plot_snowfall_mean, plot_snowfall_change_mean
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values_with_mean_aic import \
    StudyVisualizerForMeanValuesWithMeanAic
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.validation_plot import \
    validation_plot
from projects.exceeding_snow_loads.section_results.plot_selection_curves import plot_selection_curves
from projects.exceeding_snow_loads.section_results.plot_trend_curves import plot_trend_map

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_trend.visualizers.utils import load_altitude_to_visualizer
from projects.exceeding_snow_loads.section_results.plot_uncertainty_curves import plot_uncertainty_massifs
from projects.exceeding_snow_loads.utils import paper_study_classes, paper_altitudes
from root_utils import NB_CORES

import matplotlib.pyplot as plt


def minor_result(altitude):
    """Plot trends for a single altitude to be fast"""
    visualizer = StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude), multiprocessing=True,
                                                       )
    visualizer.plot_trends()
    plt.show()


def compute_minimized_aic(visualizer):
    if isinstance(visualizer, StudyVisualizerForMeanValuesWithMeanAic):
        _ = visualizer.massif_name_and_trend_test_class_to_trend_test
    else:
        _ = visualizer.massif_name_to_trend_test_that_minimized_aic
    return True


def intermediate_result(altitudes, massif_names=None,
                        model_subsets_for_uncertainty=None,
                        uncertainty_methods=None,
                        study_class=SafranSnowfall1Day,
                        multiprocessing=False, study_visualizer_class=StudyVisualizerForMeanValues,
                        season=Season.annual):
    """
    Plot all the trends for all altitudes
    And enable to plot uncertainty plot for some specific massif_names, uncertainty methods to be fast
    :param altitudes:
    :param massif_names:
    :param non_stationary_uncertainty:
    :param uncertainty_methods:
    :param study_class:
    :return:
    """
    # Load altitude to visualizer
    altitude_to_visualizer = load_altitude_to_visualizer(altitudes, massif_names, model_subsets_for_uncertainty,
                                                         study_class, uncertainty_methods,
                                                         study_visualizer_class=study_visualizer_class,
                                                         season=season)
    # Load variable object efficiently
    for v in altitude_to_visualizer.values():
        _ = v.study.year_to_variable_object
    # Compute minimized value efficiently
    visualizers = list(altitude_to_visualizer.values())
    if multiprocessing:
        with Pool(4) as p:
            _ = p.map(compute_minimized_aic, visualizers)
    else:
        for visualizer in visualizers:
            _ = compute_minimized_aic(visualizer)
    # Aggregate the choice for the minimizer
    aggregate(visualizers)

    # Plots
    validation_plot(altitude_to_visualizer, order_derivative=0)
    # validation_plot(altitude_to_visualizer, order_derivative=1)
    plot_snowfall_mean(altitude_to_visualizer)
    # plot_selection_curves_paper2(altitude_to_visualizer)
    plot_snowfall_change_mean(altitude_to_visualizer)
    shape_plot(altitude_to_visualizer)


def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.ci_mle][:]
    # massif_names = ['Beaufortain', 'Vercors']
    massif_names = None
    study_classes = [SafranSnowfall1Day, SafranPrecipitation1Day][::-1][:]
    # study_classes = [SafranSnowfall3Days, SafranPrecipitation3Days][::-1]
    model_subsets_for_uncertainty = None
    altitudes = paper_altitudes
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    # altitudes = [900, 1200, 1500, 1800][:2]
    # altitudes = [1800, 2100, 2400, 2700][:3]
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    # altitudes = draft_altitudes
    # for significance_level in [0.1, 0.05][]:
    AbstractGevTrendTest.SIGNIFICANCE_LEVEL = 0.05
    study_visualizer_class = [StudyVisualizerForMeanValues, StudyVisualizerForMeanValuesWithMeanAic][0]
    season = [Season.annual, Season.winter_extended][1]
    for study_class in study_classes:
        intermediate_result(altitudes, massif_names, model_subsets_for_uncertainty,
                            uncertainty_methods, study_class, multiprocessing=False,
                            study_visualizer_class=study_visualizer_class,
                            season=season)


def aggregate(visualizers):
    visualizer = visualizers[0]
    if not isinstance(visualizer, StudyVisualizerForMeanValuesWithMeanAic):
        return
    massif_names = set.union(*[set(v.massif_names) for v in visualizers])
    massif_names = list(massif_names)
    trend_tests = visualizer.non_stationary_trend_test

    massif_name_to_trend_test_to_aic_list = {m: {t: [] for t in trend_tests}
                                             for m in massif_names}
    for v in visualizers:
        for (m, t), t2 in v.massif_name_and_trend_test_class_to_trend_test.items():
            massif_name_to_trend_test_to_aic_list[m][t].append(t2.aic)

    massif_name_to_trend_test_with_minimial_mean_aic = {}
    for m in massif_names:
        trend_test_and_mean_aic = [(t, np.array(massif_name_to_trend_test_to_aic_list[m][t]).mean())
                                   for t in trend_tests]
        sorted_l = sorted(trend_test_and_mean_aic, key=lambda e: e[1])
        trend_test_with_minimial_mean_aic = sorted_l[0][0]
        massif_name_to_trend_test_with_minimial_mean_aic[m] = trend_test_with_minimial_mean_aic
    print(massif_name_to_trend_test_with_minimial_mean_aic)
    for v in visualizers:
        v.massif_name_to_trend_test_with_minimial_mean_aic = {}
        for m, t in massif_name_to_trend_test_with_minimial_mean_aic.items():
            if (m, t) in v.massif_name_and_trend_test_class_to_trend_test:
                v.massif_name_to_trend_test_with_minimial_mean_aic[m] = v.massif_name_and_trend_test_class_to_trend_test[(m, t)]


if __name__ == '__main__':
    major_result()
    # intermediate_result(altitudes=[300], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     multiprocessing=True)
