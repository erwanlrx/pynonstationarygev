import datetime
import time
from collections import OrderedDict

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from extreme_trend.one_fold_fit.plots.plot_histogram_altitude_studies import plot_nb_massif_on_upper_axis
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.get_nb_linear_pieces import run_selection, \
    eliminate_massif_name_with_too_much_zeros
from projects.projected_extreme_snowfall.results.part_3.projection_elevation_plot_utils import \
    plot_piechart_scatter_plot, plot_relative_change_at_massif_level, \
    plot_relative_change_at_massif_level_sensitivity_to_frequency, plot_contour_changes_values, plot_transition_lines
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load
import matplotlib.pyplot as plt

import numpy as np

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    start = time.time()

    fast = None
    snowfall = True
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]
    # altitudes = [900, 1200, 1500, 1800, 2100][:]
    altitudes = [2100, 2400, 2700, 3000, 3300, 3600][:]

    all_massif_names = AbstractStudy.all_massif_names()[:]
    # all_massif_names = ["Mont-Blanc", "Vanoise", "Oisans", "Grandes-Rousses"]

    if fast:
        altitudes = altitudes[-2:]
        all_massif_names = ["Mont-Blanc", "Vanoise", "Oisans", "Grandes-Rousses"]

    if fast is None:
        altitudes = altitudes[-2:]
        all_massif_names = ["Mont-Blanc", "Vanoise", "Oisans", "Grandes-Rousses"]

    visualizers = []
    for altitude in altitudes:
        print('altitude', altitude)
        altitudes_list = [[altitude]]

        ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][1:]
        # massif_names = ['Mercantour', 'Thabor', 'Devoluy', 'Parpaillon', 'Haut_Var-Haut_Verdon'][:2]

        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(
            all_massif_names,
            altitude,
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall,
        season=season)

        print(altitude, massif_names)

        massif_name_to_param_name_to_climate_coordinates_with_effects = {}
        for massif_name, parametrization_number in massif_name_to_parametrization_number.items():
            combination = (parametrization_number, parametrization_number, 0)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination)
            massif_name_to_param_name_to_climate_coordinates_with_effects[
                massif_name] = param_name_to_climate_coordinates_with_effects

        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, season, scenario,
            model_classes=massif_name_to_model_class,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            fit_method=fit_method,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
            safran_study_class=safran_study_class,
            linear_effects=linear_effects,
            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
            param_name_to_climate_coordinates_with_effects=massif_name_to_param_name_to_climate_coordinates_with_effects,
        )

        sub_visualizers = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in visualizer.ensemble_fits(TogetherEnsembleFit)]
        print(len(sub_visualizers))
        sub_visualizer = sub_visualizers[0]
        visualizers.append(sub_visualizer)

    return_periods = [None, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000][:-4]

    if snowfall is True:
        # Illustrate the percentage of massifs
        covariates = [1.5, 2, 2.5, 3, 3.5, 4][:]
        if len(visualizers) == 10:
            visualizers_list = [visualizers[:5], visualizers[5:]]
        else:
            visualizers_list = [visualizers]
        for visualizers_local in visualizers_list:
            for relative_change in [True, False][:1]:
                for return_period in [OneFoldFit.return_period, None]:
                    plot_piechart_scatter_plot(visualizers_local, all_massif_names, covariates, relative_change, return_period, snowfall)

        # Illustrate the contour with all elevation
        elevations_for_contour_plot = [2100, 2400, 2700, 3000, 3300, 3600]
        for relative_change in [True, False][:1]:
            return_period_to_paths = OrderedDict()
            visualizers_for_contour_plot = [v for v in visualizers if v.study.altitude in elevations_for_contour_plot]
            for return_period in return_periods[:]:
                paths = plot_contour_changes_values(visualizers_for_contour_plot, relative_change, return_period, snowfall)
                return_period_to_paths[return_period] = paths

            # Plot transition line together
            for return_periods_for_plots in [[None, 2, 5, 10, 20, 50, 100], [100, 200, 500, 1000, 2000]][:1]:
                local_return_period_to_paths = OrderedDict()
                for r in return_periods_for_plots:
                    local_return_period_to_paths[r] = return_period_to_paths[r]
                plot_transition_lines(visualizers[0], local_return_period_to_paths, relative_change)

    if snowfall:
        all_massif_names = [None]
    else:
        all_massif_names = [None]

    # Illustrate the trend of each massif
    with_significance = False
    for relative_change in [True, False][:1]:
        for massif_name in all_massif_names:
            # for visualizer in visualizers:
            #     plot_relative_change_at_massif_level_sensitivity_to_frequency(visualizer, massif_name,
            #                                                                   with_significance, relative_change,
            #                                                                   return_periods)
            # plot_relative_change_at_massif_level(visualizers, massif_name, False,
            #                                      with_significance, relative_change, None)
            if snowfall in [True, None]:
                return_periods_for_plots = [return_periods[0], return_periods[-1]]
                return_period_to_categories_list = {
                    return_periods[-1]: [[900, 1200, 1500, 1800, 2100, 2400], [2700, 3000], [3300, 3600]],
                    return_periods[0]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000], [3300], [3600]],
                }
            else:
                return_periods_for_plots = [return_periods[-2]]
                return_period_to_categories_list = {
                    return_periods[-2]: [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
                }
            for return_period in return_periods_for_plots:
                categories_list = return_period_to_categories_list[return_period]
                for categories in categories_list:
                    categories = set(categories)
                    visualizers_categories = [v for v in visualizers if v.study.altitude in categories]
                    if len(visualizers_categories) > 0:
                        plot_relative_change_at_massif_level(visualizers_categories, massif_name, True,
                                                             with_significance, relative_change, return_period,
                                                             snowfall)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main()
