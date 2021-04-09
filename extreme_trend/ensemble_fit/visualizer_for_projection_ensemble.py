import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.altitude_group import get_altitude_class_from_altitudes
from extreme_trend.one_fold_fit.plots.plot_histogram_altitude_studies import \
    plot_histogram_all_trends_against_altitudes, plot_shoe_plot_changes_against_altitude
from extreme_trend.one_fold_fit.utils_altitude_studies_visualizer import compute_and_assign_max_abs


class VisualizerForProjectionEnsemble(object):

    def __init__(self, altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 gcm_to_year_min_and_year_max=None,
                 interval_str_prefix='',
                 ):
        self.interval_str_prefix = interval_str_prefix
        self.altitudes_list = altitudes_list
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.scenario = scenario
        self.gcm_rcm_couples = gcm_rcm_couples
        self.massif_names = massif_names
        self.ensemble_fit_classes = ensemble_fit_classes

        # Some checks
        if gcm_to_year_min_and_year_max is not None:
            for gcm, years in gcm_to_year_min_and_year_max.items():
                assert isinstance(gcm, str), gcm
                assert len(years) == 2, years

        # Load all studies
        altitude_class_to_gcm_couple_to_studies = OrderedDict()
        for altitudes in altitudes_list:
            altitude_class = get_altitude_class_from_altitudes(altitudes)
            gcm_rcm_couple_to_studies = {}
            for gcm_rcm_couple in gcm_rcm_couples:
                if gcm_to_year_min_and_year_max is None:
                    kwargs_study = {}
                else:
                    gcm = gcm_rcm_couple[0]
                    if gcm not in gcm_to_year_min_and_year_max:
                        # It means that for this gcm and scenario,
                        # there is not enough data (less than 30 years) for the fit
                        continue
                    year_min, year_max = gcm_to_year_min_and_year_max[gcm]
                    kwargs_study = {'year_min': year_min, 'year_max': year_max}
                studies = AltitudesStudies(study_class, altitudes, season=season,
                                           scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                                           **kwargs_study)
                gcm_rcm_couple_to_studies[gcm_rcm_couple] = studies
            if len(gcm_rcm_couple_to_studies) == 0:
                print('No valid studies for the following couples:', self.gcm_rcm_couples)
            altitude_class_to_gcm_couple_to_studies[altitude_class] = gcm_rcm_couple_to_studies

        # Load ensemble fit
        self.altitude_class_to_ensemble_class_to_ensemble_fit = OrderedDict()
        for altitude_class, gcm_rcm_couple_to_studies in altitude_class_to_gcm_couple_to_studies.items():
            ensemble_class_to_ensemble_fit = {}
            for ensemble_fit_class in ensemble_fit_classes:
                ensemble_fit = ensemble_fit_class(massif_names, gcm_rcm_couple_to_studies, model_classes,
                                                  fit_method, temporal_covariate_for_fit,
                                                  display_only_model_that_pass_gof_test,
                                                  confidence_interval_based_on_delta_method,
                                                  remove_physically_implausible_models)
                ensemble_class_to_ensemble_fit[ensemble_fit_class] = ensemble_fit
            self.altitude_class_to_ensemble_class_to_ensemble_fit[altitude_class] = ensemble_class_to_ensemble_fit

    def plot_for_visualizer_list(self, visualizer_list):
        with_significance = False
        for v in visualizer_list:
            v.plot_moments()
        plot_histogram_all_trends_against_altitudes(self.massif_names, visualizer_list,
                                                    with_significance=with_significance)
        for relative in [True, False]:
            plot_shoe_plot_changes_against_altitude(self.massif_names, visualizer_list, relative=relative,
                                                    with_significance=with_significance)

    def plot(self):
        # Set limit for the plot
        visualizer_list = []
        for ensemble_fit_class in self.ensemble_fit_classes:
            for ensemble_fit in self.ensemble_fits(ensemble_fit_class):
                visualizer_list.extend(ensemble_fit.visualizer_list)
        # compute_and_assign_max_abs(visualizer_list)
        # Plot
        if IndependentEnsembleFit in self.ensemble_fit_classes:
            self.plot_independent()
        if TogetherEnsembleFit in self.ensemble_fit_classes:
            self.plot_together()

    def plot_independent(self):
        # Aggregated at gcm_rcm_level plots
        merge_keys = [AbstractEnsembleFit.Median_merge, AbstractEnsembleFit.Mean_merge]
        keys = self.gcm_rcm_couples + merge_keys
        # Only plot Mean for speed
        # keys = [AbstractEnsembleFit.Mean_merge]
        for key in keys:
            visualizer_list = [independent_ensemble_fit.gcm_rcm_couple_to_visualizer[key]
                               if key in self.gcm_rcm_couples
                               else independent_ensemble_fit.merge_function_name_to_visualizer[key]
                               for independent_ensemble_fit in self.ensemble_fits(IndependentEnsembleFit)
                               ]
            if key in merge_keys:
                for v in visualizer_list:
                    v.studies.study.gcm_rcm_couple = ("{} {}".format(key, "merge"), self.interval_str_prefix)
            self.plot_for_visualizer_list(visualizer_list)

    def plot_together(self):
        visualizer_list = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in self.ensemble_fits(TogetherEnsembleFit)]
        for v in visualizer_list:
            v.studies.study.gcm_rcm_couple = ("together merge", self.interval_str_prefix)
        self.plot_for_visualizer_list(visualizer_list)

    def ensemble_fits(self, ensemble_class):
        """Return the ordered ensemble fit for a given ensemble class (in the order of the altitudes)"""
        return [ensemble_class_to_ensemble_fit[ensemble_class]
                for ensemble_class_to_ensemble_fit
                in self.altitude_class_to_ensemble_class_to_ensemble_fit.values()]

    def plot_preliminary_first_part(self):
        if self.massif_names is None:
            massif_names = AbstractStudy.all_massif_names()
        else:
            massif_names = self.massif_names
        assert isinstance(massif_names, list)
        # Plot for all parameters
        for param_name in GevParams.PARAM_NAMES:
            for degree in [0, 1]:
                for massif_name in massif_names:
                    self.plot_preliminary_first_part_for_one_massif(massif_name, param_name, degree)

    def plot_preliminary_first_part_for_one_massif(self, massif_name, param_name, degree):
        # Retrieve the data
        ensemble_fit: IndependentEnsembleFit
        gcm_rcm_couple_to_data = {
            c: [] for c in self.gcm_rcm_couples
        }
        for ensemble_fit in self.ensemble_fits(IndependentEnsembleFit):
            for gcm_rcm_couple in self.gcm_rcm_couples:
                visualizer = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple]
                if massif_name in visualizer.massif_name_to_one_fold_fit:
                    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
                    coef = one_fold_fit.best_coef(param_name, 0, degree)
                    altitude = visualizer.altitude_group.reference_altitude
                    gcm_rcm_couple_to_data[gcm_rcm_couple].append((altitude, coef))
        # Plot
        ax = plt.gca()
        for gcm_rcm_couple, data in gcm_rcm_couple_to_data.items():
            altitudes, coefs = list(zip(*data))
            color = gcm_rcm_couple_to_color[gcm_rcm_couple]
            label = gcm_rcm_couple_to_str(gcm_rcm_couple)
            ax.plot(coefs, altitudes, color=color, label=label, marker='o')
        ax.legend()
        visualizer.plot_name = '{}/{}_{}'.format(param_name, degree, massif_name)
        visualizer.show_or_save_to_file(no_title=True)
        plt.close()