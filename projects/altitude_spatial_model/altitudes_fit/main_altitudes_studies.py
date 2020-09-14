import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_simulation_study import SimulationStudy
from extreme_data.meteo_france_data.adamont_data.snowfall_simulation import SafranSnowfallSimulationRCP85
from extreme_fit.model.margin_model.polynomial_margin_model.utils import ALTITUDINAL_GEV_MODELS, \
    ALTITUDINAL_GEV_MODELS_LOCATION_QUADRATIC_MINIMUM, ALTITUDINAL_GEV_MODELS_LOCATION_ONLY_SCALE_ALTITUDES, \
    ALTITUDINAL_GEV_MODELS_LOCATION, ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.plot_total_aic import plot_total_aic, \
    plot_individual_aic
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import MeanAlpsTemperatureCovariate

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit
from projects.exceeding_snow_loads.utils import dpi_paper1_figure

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranPrecipitation1Day, SafranPrecipitation3Days, \
    SafranPrecipitation5Days, SafranPrecipitation7Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def plot_time_series(studies, massif_names=None):
    studies.plot_maxima_time_series(massif_names=massif_names)


def plot_moments(studies, massif_names=None):
    for std in [True, False][:]:
        for change in [True, False, None]:
            studies.plot_mean_maxima_against_altitude(massif_names=massif_names, std=std, change=change)


def plot_altitudinal_fit(studies, massif_names=None):
    # model_classes = ALTITUDINAL_GEV_MODELS_LOCATION_ONLY_SCALE_ALTITUDES
    # model_classes = ALTITUDINAL_GEV_MODELS_LOCATION_QUADRATIC_MINIMUM
    model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    # model_classes = ALTITUDINAL_GEV_MODELS_LOCATION
    # model_classes = ALTITUDINAL_GEV_MODELS_LOCATION_CUBIC_MINIMUM
    # model_classes = ALTITUDINAL_GEV_MODELS_QUADRATIC
    visualizer = AltitudesStudiesVisualizerForNonStationaryModels(studies=studies,
                                                                  model_classes=model_classes,
                                                                  massif_names=massif_names,
                                                                  show=False,
                                                                  temporal_covariate_for_fit=None,
                                                                  # temporal_covariate_for_fit=MeanAlpsTemperatureCovariate,
                                                                  top_n_values_to_remove=None,
                                                                  )
    # Plot the results for the model that minimizes the individual aic
    plot_individual_aic(visualizer)
    # Plot the results for the model that minimizes the total aic
    # plot_total_aic(model_classes, visualizer)


def main():
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][4:6]
    # todo: l ecart  pour les saisons de l automne ou de sprint
    #  vient probablement de certains zéros pas pris en compte pour le fit,
    # mais pris en compte pour le calcul de mon aic
    # altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][:]
    # altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900][:]
    # altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][:]
    # altitudes = [600, 900, 1200, 1500, 1800, 2100][:]
    altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800][:]
    # altitudes = [1500, 1800][:]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days][:2]
    study_classes = [SafranPrecipitation1Day, SafranPrecipitation3Days, SafranPrecipitation5Days,
                     SafranPrecipitation7Days][:]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranPrecipitation1Day
                     , SafranPrecipitation3Days][:1]
    altitudes = [1800, 2100, 2400]
    study_classes = [SafranSnowfall1Day, SafranSnowfall3Days][:1]

    # Common parameters
    # altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    massif_names = None
    seasons = [Season.annual, Season.winter, Season.spring, Season.automn][:1]

    all_groups = altitudes_for_groups[:]
    for altitudes in all_groups:
        main_loop(altitudes, massif_names, seasons, study_classes)


def main_loop(altitudes, massif_names, seasons, study_classes):
    for season in seasons:
        for study_class in study_classes:
            if issubclass(study_class, SimulationStudy):
                for ensemble_idx in list(range(14))[:1]:
                    studies = AltitudesStudies(study_class, altitudes, season=season,
                                               ensemble_idx=ensemble_idx)
                    plot_studies(massif_names, season, studies, study_class)
            else:
                studies = AltitudesStudies(study_class, altitudes, season=season)
                plot_studies(massif_names, season, studies, study_class)


def plot_studies(massif_names, season, studies, study_class):
    print('inner loop', season, study_class)
    # plot_time_series(studies, massif_names)
    # plot_moments(studies, massif_names)
    plot_altitudinal_fit(studies, massif_names)


if __name__ == '__main__':
    main()
