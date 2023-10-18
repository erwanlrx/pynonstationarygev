from collections import OrderedDict
from typing import Any

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.utils import Season

show = False
if show in [None, True]:
    matplotlib.use('Agg')
    import matplotlib as mpl

    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from projected_extremes.section_results.utils.average_bias import compute_average_bias, load_study, plot_average_bias
from projected_extremes.section_results.utils.setting_utils import set_up_and_load
from projected_extremes.section_results.validation_experiment.model_as_truth_experiment import ModelAsTruthExperiment


def main_preliminary_projections():
    # Set parameters

    # fast = False considers all ensemble members and all elevations,
    # fast = None considers all ensemble members and 1 elevation,
    # fast = True considers only 6 ensemble mmebers and 1 elevation

    # snowfall=True corresponds to daily snowfall
    # snowfall=False corresponds to accumulated ground snow load
    # snowfall=None corresponds to daily winter precipitation
    fast = False
    snowfall = None
    nb_days = 1

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, model_classes_list, scenario, study_class, \
    temporal_covariate_for_fit, remove_physically_implausible_models, display_only_model_that_pass_gof_test,\
    safran_study_class, fit_method, season = set_up_and_load(fast, snowfall, nb_days)

    altitudes_list = [[900], [1200], [1500], [1800], [2100], [2400], [2700], [3000], [3300], [3600]][:]
    # altitudes_list = [[1500], [1800], [2100], [2400]]
    # altitudes_list = [[1500], [1800], [2100], [2400], [2700]]
    # massif_names = ['Mercantour']

    # Run a model as truth experiment
    # for each altitude and for each model_class (number of pieces for the piecewise linear functions)
    for season in [Season.spring]:
        for altitudes in altitudes_list:
            for model_class in model_classes_list[:]:
                model_classes = [model_class]
                run_mas(altitudes, display_only_model_that_pass_gof_test, gcm_rcm_couples, massif_names,
                        model_classes, remove_physically_implausible_models, safran_study_class, scenario, show,
                        study_class, temporal_covariate_for_fit, fit_method, season)


def run_mas(altitudes, display_only_model_that_pass_gof_test, gcm_rcm_couples, massif_names,
            model_classes, remove_physically_implausible_models, safran_study_class, scenario, show,
            study_class, temporal_covariate_for_fit, fit_method, season):

    # Load the data that correspond to the altitude of interest
    altitude = altitudes[0]
    print('Altitude={}'.format(altitude))
    gcm_rcm_couple_to_study, safran_study = load_study(altitude, gcm_rcm_couples, safran_study_class, scenario,
                                                       study_class, season)

    # Loop on the massifs
    for massif_name in massif_names:
        if massif_name in safran_study.study_massif_names:
            print(massif_name)

            average_bias, _ = compute_average_bias(gcm_rcm_couple_to_study, massif_name, safran_study, show=show)
            gcm_rcm_couples_sampled_for_experiment, gcm_rcm_couple_to_average_bias, gcm_rcm_couple_to_gcm_rcm_couple_to_biases = plot_average_bias(
                gcm_rcm_couple_to_study, massif_name, average_bias,
                alpha=1000, show=show)

            print("Number of couples for the model as truth experiment:", len(gcm_rcm_couples_sampled_for_experiment))
            # Loop on the gcm_rcm_couples (one loop for each gcm_rcm_couple that we set as pseudo-truth)
            for gcm_rcm_couple in gcm_rcm_couples_sampled_for_experiment:

                # Loop on the potential parameterization for the three parameters of the GEV distribution
                # 0 represents the parameterization without adjustment coefficients
                # We do not rely on adjustment coefficients for the model as truth experiment
                combination = (0, 0, 0)

                xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples,
                                            safran_study_class,
                                            study_class, season=season,
                                            scenario=scenario,
                                            selection_method_names=['aic'],
                                            model_classes=model_classes,
                                            massif_names=[massif_name],
                                            fit_method=fit_method,
                                            temporal_covariate_for_fit=temporal_covariate_for_fit,
                                            remove_physically_implausible_models=remove_physically_implausible_models,
                                            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                            gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment,
                                            combination=combination,
                                            year_max_for_gcm=2100,
                                            year_max_for_pseudo_obs=2019,
                                            linear_effects=(False, False, False)
                                            ,
                                            )
                xp.run_one_experiment(gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)


if __name__ == '__main__':
    main_preliminary_projections()
