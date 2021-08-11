import datetime
import os
import random
import time
from typing import List

import numpy as np
from rpy2.rinterface import RRuntimeError

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.utils import DATA_PATH
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh, NllhIsInfException, \
    compute_nllh_for_list_of_pair
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import SafeRunException
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble
from projects.projected_extreme_snowfall.results.combination_utils import load_combination, load_combination_name
from projects.projected_extreme_snowfall.results.part_2.v2.utils import is_already_done, update_csv
from root_utils import get_display_name_from_object_type
import os.path as op


class AbstractExperiment(object):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel],
                 selection_method_names: List[str],
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 ):
        self.selection_method_names = selection_method_names
        self.fit_method = fit_method
        self.massif_names = massif_names
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_gof_test = display_only_model_that_pass_gof_test
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.model_classes = model_classes
        self.scenario = scenario
        self.season = season
        self.study_class = study_class
        self.safran_study_class = safran_study_class
        self.gcm_rcm_couples = gcm_rcm_couples
        self.altitudes = altitudes

    def load_studies_obs_for_train(self, **kwargs) -> AltitudesStudies:
        raise NotImplementedError

    def load_studies_obs_for_test(self, **kwargs) -> AltitudesStudies:
        raise NotImplementedError

    def load_gcm_rcm_couple_to_studies(self, **kwargs):
        raise NotImplementedError

    def run_one_experiment(self, **kwargs):
        gcm_couple = ("", "") if len(kwargs) == 0 else kwargs['gcm_rcm_couple_as_pseudo_truth']
        if not is_already_done(self.excel_filepath, self.combination_name, self.experiment_name, gcm_couple):
            start = time.time()
            try:
                nllh_list = self._run_one_experiment(kwargs)
            except (NllhIsInfException, SafeRunException) as e:
                print(e.__repr__())
                nllh_list = [np.nan for _ in self.selection_method_names]

            duration = str(datetime.timedelta(seconds=time.time() - start))
            print('Total duration for one experiment', duration)
            nllh_value = np.array(nllh_list)
            update_csv(self.excel_filepath, self.combination_name, self.experiment_name, gcm_couple, nllh_value)

    def _run_one_experiment(self, kwargs):
        # Load gcm_rcm_couple_to_studies
        gcm_rcm_couple_to_studies = self.load_gcm_rcm_couple_to_studies(**kwargs)
        # Load ensemble fit
        visualizer = VisualizerNonStationaryEnsemble(gcm_rcm_couple_to_studies=gcm_rcm_couple_to_studies,
                                                     model_classes=self.model_classes,
                                                     show=False,
                                                     massif_names=self.massif_names,
                                                     fit_method=self.fit_method,
                                                     temporal_covariate_for_fit=self.temporal_covariate_for_fit,
                                                     display_only_model_that_pass_anderson_test=self.display_only_model_that_pass_gof_test,
                                                     confidence_interval_based_on_delta_method=False,
                                                     remove_physically_implausible_models=self.remove_physically_implausible_models,
                                                     param_name_to_climate_coordinates_with_effects=self.param_name_to_climate_coordinates_with_effects,
                                                     **kwargs)
        # Get the best margin function for the selection method name
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
        assert len(one_fold_fit.fitted_estimators) == 1, 'for the model as truth they should not be any combinations'
        assert len(self.selection_method_names) == 1
        best_estimator = one_fold_fit._sorted_estimators_with_method_name("aic")[0]
        # Compute the average nllh for the test data
        studies_for_test = self.load_studies_obs_for_test(**kwargs)
        dataset_test = self.load_spatio_temporal_dataset(studies_for_test, **kwargs)
        nllh_list = []
        df_coordinates_temp_for_test = best_estimator.load_coordinates_temp(dataset_test.coordinates)
        print('Model={}'.format(get_display_name_from_object_type(best_estimator.margin_model)))
        for time, maxima in zip(df_coordinates_temp_for_test.values, dataset_test.observations.maxima_gev):
            list_of_pair = [(maxima, time)]
            args = True, list_of_pair, best_estimator.margin_function_from_fit, True
            nllh_for_test = compute_nllh_for_list_of_pair(args)
            nllh_list.append(-nllh_for_test)
        return nllh_list

    def load_altitude_studies(self, gcm_rcm_couple=None, year_min=None, year_max=None):
        kwargs = {}
        if year_min is not None:
            kwargs['year_min'] = year_min
        if year_max is not None:
            kwargs['year_max'] = year_max
        if gcm_rcm_couple is None:
            return AltitudesStudies(self.safran_study_class, self.altitudes, season=self.season, **kwargs)
        else:
            return AltitudesStudies(self.study_class, self.altitudes, season=self.season,
                                    scenario=self.scenario, gcm_rcm_couple=gcm_rcm_couple, **kwargs)

    # Utils

    def load_spatio_temporal_dataset(self, studies, **kwargs):
        return studies.spatio_temporal_dataset(self.massif_name, **kwargs)

    @property
    def excel_filename(self):
        study_name = get_display_name_from_object_type(self.study_class)
        altitude = str(self.altitude)
        nb_couples = len(self.gcm_rcm_couples)
        goodness_of_fit = self.display_only_model_that_pass_gof_test
        model_name = get_display_name_from_object_type(self.model_class)
        return "{}_{}m_{}couples_test{}_{}".format(study_name, altitude, nb_couples, goodness_of_fit, model_name)

    @property
    def excel_filepath(self):
        path = op.join(DATA_PATH, "abstract_experiments", get_display_name_from_object_type(self))
        if not op.exists(path):
            os.makedirs(path, exist_ok=True)
        return op.join(path, self.excel_filename + '.xlsx')

    @property
    def massif_name(self):
        assert len(self.massif_names) == 1
        return self.massif_names[0]

    @property
    def model_class(self):
        assert len(self.model_classes) == 1
        return self.model_classes[0]

    @property
    def altitude(self):
        assert len(self.altitudes) == 1
        return self.altitudes[0]

    @property
    def experiment_name(self):
        return str(self.altitude) + self.massif_name

    @property
    def combination_name(self):
        return load_combination_name(self.param_name_to_climate_coordinates_with_effects)
