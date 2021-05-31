import numpy as np
from cached_property import cached_property

from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_simulation.abstract_simulation_with_effect import AbstractSimulationWithEffects
from extreme_trend.one_fold_fit.altitude_group import DefaultAltitudeGroup
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.setting_utils import LINEAR_MODELS_FOR_PROJECTION_ONE_ALTITUDE


class AbstractSimulationFitForEnsemble(object):
    RMSE_METRIC = 'rmse'
    CRPSS_METRIC = 'crpss'
    METRICS = [RMSE_METRIC, CRPSS_METRIC]

    def __init__(self, simulation: AbstractSimulationWithEffects,
                 year_list_to_test,
                 return_period,
                 model_classes,
                 with_effects=True, with_observation=True):
        self.simulation = simulation
        self.return_period = return_period
        self.year_list_to_test = year_list_to_test
        self.x_list_to_test = [self.simulation.get_x_from_year(year) for year in self.year_list_to_test]

        self.with_effects = with_effects
        self.with_observation = with_observation
        assert with_observation, 'create dataset without obs to remove this assertion'
        assert not ((not with_observation) and with_effects)

        # Default parameters
        self.model_classes = model_classes
        self.fit_method = MarginFitMethod.evgam
        self.altitude_group = DefaultAltitudeGroup(altitudes=[0])
        self.only_models_that_pass_goodness_of_fit_test = False
        self.confidence_interval_based_on_delta_method = False
        self.remove_physically_implausible_models = False
        combination = (2, 2, 0) if self.with_effects else (0, 0, 0)
        self.param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(combination)

    def plot_mean_metric(self, ax, metric_name):
        assert metric_name in self.METRICS
        mean_crpss = np.mean(self.metric_name_to_all_list[metric_name], axis=0)
        ax.set_xlabel('Years')
        ax.set_xlim((self.year_list_to_test[0], self.year_list_to_test[-1]))
        ax.set_ylabel(metric_name)
        ax.plot(self.year_list_to_test, mean_crpss, label=self.name)

    @property
    def name(self):
        raise NotImplementedError

    @cached_property
    def metric_name_to_all_list(self):
        all_dict = [self.compute_metric_name_to_list(i) for i in self.simulation.simulation_ids]
        return {metric_name: [d[metric_name] for d in all_dict] for metric_name in self.METRICS}

    def compute_metric_name_to_list(self, simulation_id):
        raise NotImplementedError

    def compute_return_levels(self, margin_function, x):
        gev_params = margin_function.get_params(x)
        return gev_params.return_level(self.return_period)




