from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation1Day, SafranRainfall1Day
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles
from extreme_data.meteo_france_data.scm_models_data.utils import Season, season_to_str, season_to_suffix


class AbstractSafranRainfallMaxFiles(AbstractStudyMaxFiles, SafranRainfall1Day):

    def __init__(self, safran_year, **kwargs):
        if 'season' in kwargs:
            season = kwargs['season']
        else:
            season = Season.annual
        prefix = "max-1day-rainf"
        if season is Season.annual:
            keyword = f"{prefix}-year"
        elif season in [Season.winter, Season.summer, Season.spring, Season.autumn]:
            keyword = f"{prefix}-{season_to_str(season)}-{season_to_suffix(season)}"
        else:
            raise NotImplementedError('data not available for this season')
        super().__init__(safran_year, keyword, **kwargs)


class SafranRainfall2019(AbstractSafranRainfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)


if __name__ == '__main__':
    for season in [Season.annual, Season.winter, Season.summer, Season.spring, Season.autumn]:
        print(season)
        study = SafranRainfall2019(altitude=1800, season=season)
        print(len(study.year_to_annual_maxima))
