from typing import List

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontDepth
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall, AdamontPrecipitation, \
    AdamontSnowfall3days, AdamontSnowfall5days, AdamontRainfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019, CrocusDepth2022, \
    CrocusDepth2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_precipf import SafranPrecipitation2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_rainf import SafranRainfall2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2019, \
    SafranSnowfall3Days2022, SafranSnowfall5Days2022, SafranSnowfall3Days2019, SafranSnowfall5Days2019
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel, \
    NonStationaryThreeLinearLocationAndScaleAndShapeModel, \
    NonStationaryFourLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.utils import set_seed_for_test
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


def set_up_and_load(fast, snowfall=True, nb_days=1,     season = Season.annual):
    safran_study_class, study_class = load_study_classes(snowfall, nb_days)
    OneFoldFit.multiprocessing = False

    remove_physically_implausible_models, display_only_model_that_pass_gof_test = False, False

    fit_method = MarginFitMethod.evgam

    model_classes_list = [NonStationaryLocationAndScaleAndShapeTemporalModel,
                          NonStationaryTwoLinearLocationAndScaleAndShapeModel,
                          NonStationaryThreeLinearLocationAndScaleAndShapeModel,
                          NonStationaryFourLinearLocationAndScaleAndShapeModel][:]

    if snowfall is True:
        return_period = 100
    elif snowfall is None:
        return_period = 100
    else:
        return_period = 50
    OneFoldFit.return_period = return_period

    temporal_covariate_for_fit = [TimeTemporalCovariate,
                                  AnomalyTemperatureWithSplineTemporalCovariate][1]
    set_seed_for_test()
    scenario = AdamontScenario.rcp85_extended
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)
    print('Scenario is', scenario)
    print('Covariate is {}'.format(temporal_covariate_for_fit))
    if fast is None:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = AbstractStudy.all_massif_names()
        altitudes_list = [2100]
    elif fast:
        gcm_rcm_couples = gcm_rcm_couples[:3] + gcm_rcm_couples[-3:]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 1000
        massif_names = ['Vanoise']
        altitudes_list = [1800]
    else:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 1000
        massif_names = AbstractStudy.all_massif_names()
        if snowfall:
            # altitudes_list = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
            altitudes_list = [2100, 2400, 2700, 3000, 3300, 3600]
        else:
            altitudes_list = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]

    assert isinstance(gcm_rcm_couples, list)
    altitudes_list = [[a] for a in altitudes_list]
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    assert (safran_study_class is None) or (issubclass(safran_study_class, AbstractStudyMaxFiles))

    print('Altitudes', altitudes_list)
    print('Return period:', OneFoldFit.return_period)
    print('Significance level:', OneFoldFit.SIGNIFICANCE_LEVEL)
    for altitudes in altitudes_list:
        assert len(altitudes) == 1
    print('number of models', len(model_classes_list))
    print('first model', get_display_name_from_object_type(model_classes_list[0]))
    print('number of gcm rcm couples', len(gcm_rcm_couples))

    print('only models that pass gof:', display_only_model_that_pass_gof_test)
    print('remove physically implausible models:', remove_physically_implausible_models)

    print('observation class:', get_display_name_from_object_type(safran_study_class))
    print('Take into account the observations: {}'.format(safran_study_class is not None))

    return (altitudes_list, gcm_rcm_couples, massif_names, model_classes_list, scenario, study_class,
            temporal_covariate_for_fit, remove_physically_implausible_models, display_only_model_that_pass_gof_test,
            safran_study_class, fit_method, season)





def load_study_classes(snowfall, nb_days=1):
    print(f'number of days for study classes: {nb_days}')
    if snowfall is True:
        if nb_days == 1:
            safran_study_class = SafranSnowfall2019
            study_class = AdamontSnowfall
        elif nb_days == 3:
            safran_study_class = SafranSnowfall3Days2019
            study_class = AdamontSnowfall3days
        elif nb_days == 5:
            safran_study_class = SafranSnowfall5Days2019
            study_class = AdamontSnowfall5days
        else:
            raise NotImplementedError
    elif snowfall is None:
        # study_class = AdamontPrecipitation
        # safran_study_class = SafranPrecipitation2019
        study_class = AdamontRainfall
        safran_study_class = SafranRainfall2019
    else:
        study_class = AdamontDepth
        safran_study_class = CrocusDepth2019
        # study_class = AdamontSnowLoad
        # safran_study_class = CrocusSnowLoad2019
    return safran_study_class, study_class


def get_last_year_for_the_train_set(percentage):
    last_year_for_the_train_set = 1959 + round(percentage * 61) - 1
    return last_year_for_the_train_set

def get_variable_name(safran_study_class):
    if safran_study_class is CrocusSnowLoad2019:
        return "snow load"
    elif safran_study_class is SafranSnowfall2019:
        return "snowfall"
    elif safran_study_class is SafranPrecipitation2019:
        return "precipitation"
    elif safran_study_class is SafranSnowfall3Days2022:
        return "snowfall3days2022"
    elif safran_study_class is SafranSnowfall5Days2022:
        return "snowfall5days2022"
    elif safran_study_class is CrocusDepth2022:
        return "snowdepth2022"
    elif safran_study_class is SafranSnowfall3Days2019:
        return "snowfall3days2019"
    elif safran_study_class is SafranSnowfall5Days2019:
        return "snowfall5days2019"
    elif safran_study_class is CrocusDepth2019:
        return "snowdepth2019"
    elif safran_study_class is SafranRainfall2019:
        return "rainfall2019"
    else:
        raise NotImplementedError
