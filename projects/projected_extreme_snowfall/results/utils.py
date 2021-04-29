from typing import List

import matplotlib

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel, \
    NonStationaryScaleAndShapeTemporalModel, NonStationaryLocationAndScaleAndShapeTemporalModel, \
    NonStationaryLocationAndShapeTemporalModel, NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel, NonStationaryTwoLinearScaleOneLinearShapeModel, \
    NonStationaryTwoLinearScaleAndShapeModel, NonStationaryTwoLinearShapeOneLinearLocAndScaleModel, \
    NonStationaryTwoLinearScaleOneLinearLocAndShapeModel, NonStationaryTwoLinearShapeOneLinearLocModel, \
    NonStationaryTwoLinearScaleOneLinearLocModel, NonStationaryTwoLinearScaleAndShapeOneLinearLocModel, \
    NonStationaryTwoLinearLocationOneLinearScaleModel, NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel, \
    NonStationaryTwoLinearLocationOneLinearShapeModel, NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel, \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel, \
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel, NonStationaryTwoLinearLocationAndScaleModel, \
    NonStationaryTwoLinearLocationAndShape
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearShapeModel, NonStationaryTwoLinearShapeOneLinearScaleModel, NonStationaryTwoLinearScaleModel
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates

from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate

from extreme_fit.model.utils import set_seed_for_test

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel


def set_up_and_load(fast):
    study_class = AdamontSnowfall

    temporal_covariate_for_fit = [TimeTemporalCovariate,
                                  AnomalyTemperatureWithSplineTemporalCovariate][1]
    set_seed_for_test()
    scenario = AdamontScenario.rcp85_extended
    model_classes = SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE
    AltitudesStudiesVisualizerForNonStationaryModels.consider_at_least_two_altitudes = False
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)
    print('Scenario is', scenario)
    print('Covariate is {}'.format(temporal_covariate_for_fit))
    if fast is None:
        gcm_rcm_couples = gcm_rcm_couples[:]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        altitudes_list = [2100]
    elif fast:
        gcm_rcm_couples = gcm_rcm_couples[:2] + gcm_rcm_couples[-2:]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        altitudes_list = [2100][:]
        model_classes = model_classes[:4]
    else:
        altitudes_list = [1200, 2100, 3000]
    assert isinstance(gcm_rcm_couples, list)
    altitudes_list = [[a] for a in altitudes_list]
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    for altitudes in altitudes_list:
        assert len(altitudes) == 1
    massif_names = ['Vanoise']
    if fast in [None, False]:
        assert len(set(model_classes)) == 27
    print('number of models', len(model_classes))
    remove_physically_implausible_models, display_only_model_that_pass_gof_test = True, True

    print('only models that pass gof:', display_only_model_that_pass_gof_test)
    print('remove physically implausible models:', remove_physically_implausible_models)

    return altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, study_class, temporal_covariate_for_fit, remove_physically_implausible_models, display_only_model_that_pass_gof_test


climate_coordinates_with_effects_list = [None,
                                         [AbstractCoordinates.COORDINATE_GCM],
                                         [AbstractCoordinates.COORDINATE_RCM],
                                         [AbstractCoordinates.COORDINATE_GCM, AbstractCoordinates.COORDINATE_RCM]
                                         ]  # None means we do not create any effect


def load_combination_name_for_tuple(combination):
    return load_combination_name(load_param_name_to_climate_coordinates_with_effects(combination))


def load_param_name_to_climate_coordinates_with_effects(combination):
    param_name_to_climate_coordinates_with_effects = {param_name: climate_coordinates_with_effects_list[idx]
                                                      for param_name, idx in
                                                      zip(GevParams.PARAM_NAMES, combination)}
    return param_name_to_climate_coordinates_with_effects


def load_combination_name(param_name_to_climate_coordinates_with_effects):
    param_name_to_effect_name = {p: '+'.join([e.replace('coord_', '') for e in l])
                                 for p, l in param_name_to_climate_coordinates_with_effects.items() if
                                 l is not None}
    combination_name = ' '.join(
        [param_name + '_' + param_name_to_effect_name[param_name] for param_name in GevParams.PARAM_NAMES
         if param_name in param_name_to_effect_name])
    if combination_name == '':
        combination_name = 'no effect'
    return combination_name


SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE = [

    # Models with a constant Location parameter
    StationaryTemporalModel,
    # Simple linearity for the others
    NonStationaryScaleTemporalModel,
    NonStationaryShapeTemporalModel,
    NonStationaryScaleAndShapeTemporalModel,
    # Double linearity for the others
    NonStationaryTwoLinearScaleModel,
    NonStationaryTwoLinearShapeModel,
    NonStationaryTwoLinearShapeOneLinearScaleModel,
    NonStationaryTwoLinearScaleOneLinearShapeModel,
    NonStationaryTwoLinearScaleAndShapeModel,

    # Models with a linear location parameter
    NonStationaryLocationTemporalModel,
    # Simple linearity for the others
    NonStationaryLocationAndScaleTemporalModel,
    NonStationaryLocationAndShapeTemporalModel,
    NonStationaryLocationAndScaleAndShapeTemporalModel,
    # Double linearity for the others
    NonStationaryTwoLinearScaleOneLinearLocModel,
    NonStationaryTwoLinearShapeOneLinearLocModel,
    NonStationaryTwoLinearScaleOneLinearLocAndShapeModel,
    NonStationaryTwoLinearShapeOneLinearLocAndScaleModel,
    NonStationaryTwoLinearScaleAndShapeOneLinearLocModel,

    # Models with linear location parameter with double linearity
    NonStationaryTwoLinearLocationModel,
    # Simple linearity for the others
    NonStationaryTwoLinearLocationOneLinearScaleModel,
    NonStationaryTwoLinearLocationOneLinearShapeModel,
    NonStationaryTwoLinearLocationOneLinearScaleAndShapeModel,
    # Double Linearity for the others
    NonStationaryTwoLinearLocationAndScaleModel,
    NonStationaryTwoLinearLocationAndScaleOneLinearShapeModel,
    NonStationaryTwoLinearLocationAndShape,
    NonStationaryTwoLinearLocationAndShapeOneLinearScaleModel,
    NonStationaryTwoLinearLocationAndScaleAndShapeModel,

]

if __name__ == '__main__':
    print(len(set(SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE)))
