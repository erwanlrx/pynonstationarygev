from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_variables import \
    SafranSnowfallSimulationVariable, SafranPrecipitationSimulationVariable, SafranSnowfall3daysSimulationVariable, \
    SafranSnowfall5daysSimulationVariable, SafranRainfallSimulationVariable
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion


class AdamontSnowfall(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfallSimulationVariable, *args, **kwargs)

class AdamontSnowfall3days(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfall3daysSimulationVariable, *args, **kwargs)


class AdamontSnowfall5days(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfall5daysSimulationVariable, *args, **kwargs)

class AdamontRainfall(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranRainfallSimulationVariable, *args, **kwargs)

class AdamontPrecipitation(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranPrecipitationSimulationVariable, *args, **kwargs)



if __name__ == '__main__':
    gcm_rcm_couples = list(gcm_rcm_couple_to_color.keys())
    for scenario in [AdamontScenario.histo, AdamontScenario.rcp85]:
        print('\n')
        for gcm_rcm_couple in gcm_rcm_couples[:]:
            normal_size = len(AdamontRainfall(altitude=1800, gcm_rcm_couple=gcm_rcm_couple,
                                    scenario=scenario,
                                    season=Season.annual).year_to_annual_maxima)
            for season in [Season.annual, Season.winter, Season.summer, Season.spring, Season.autumn][:]:
                study = AdamontRainfall(altitude=1800, gcm_rcm_couple=gcm_rcm_couple,
                                scenario=scenario,
                                season=season)
                size = len(study.year_to_annual_maxima)
                if size != normal_size:
                    str_diff = f" {size} != {normal_size}"
                    print(study.nc_filename_adamont_v2(scenario, maxima_date=False), str_diff)
