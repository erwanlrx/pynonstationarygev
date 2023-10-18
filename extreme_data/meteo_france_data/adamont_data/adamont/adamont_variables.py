import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable, \
    TotalSnowLoadVariable, CrocusTotalSweVariable, CrocusDepthVariable
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable, \
    SafranTotalPrecipVariable, SafranRainfallVariable
from extreme_data.meteo_france_data.scm_models_data.utils import Season, season_to_str, season_to_suffix
from root_utils import classproperty


class AbstractAdamontVariable(AbstractVariable):

    # Adamont v1

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        raise NotImplementedError

    @classmethod
    def keyword(cls):
        raise NotImplementedError

    @classmethod
    def indicator_name_for_maxima(cls):
        raise NotImplementedError

    @classmethod
    def indicator_name_for_maxima_date(cls):
        raise NotImplementedError

    @classmethod
    def get_folder_name_from_indicator_name(cls, indicator_name):
        return indicator_name.replace('-', '_').capitalize()

    @classmethod
    def transform_annual_maxima(cls, annual_maxima_data, maxima_date):
        if maxima_date:
            return annual_maxima_data.astype(int)
        else:
            return annual_maxima_data

    @classproperty
    def season_to_indicator_name_for_maxima(cls):
        raise NotImplementedError

    @classproperty
    def season_to_indicator_name_for_maxima_date(cls):
        raise NotImplementedError


class SafranSnowfallSimulationVariable(AbstractAdamontVariable):
    UNIT = SafranSnowfallVariable.UNIT
    NAME = SafranSnowfallVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        if annual_maxima_date:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima_date)
        else:
            return 'Snow'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'SNOW'

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-1day-snowf-year'

class SafranSnowfall3daysSimulationVariable(SafranSnowfallSimulationVariable):

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-3day-consec-snowf-year'

class SafranSnowfall5daysSimulationVariable(SafranSnowfallSimulationVariable):

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-5day-consec-snowf-year'

class SafranRainfallSimulationVariable(AbstractAdamontVariable):
    UNIT = SafranRainfallVariable.UNIT
    NAME = SafranRainfallVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        if annual_maxima_date:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima_date)
        else:
            return 'Rain'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-1day-rainf-year'
    @classproperty
    def season_to_indicator_name_for_maxima(cls):
        d = {}
        prefix = "max-1day-rainf"
        for season in [Season.winter, Season.summer, Season.spring, Season.autumn]:
            keyword = f"{prefix}-{season_to_str(season)}-{season_to_suffix(season)}"
            d[season] = keyword
        return d

class SafranPrecipitationSimulationVariable(AbstractAdamontVariable):
    UNIT = SafranTotalPrecipVariable.UNIT
    NAME = SafranTotalPrecipVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        if annual_maxima_date:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima_date)
        else:
            return 'Precip'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'Prec'

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-1day-precipf-year'

    @classproperty
    def season_to_indicator_name_for_maxima(cls):
        return {
            Season.winter: "max-1day-precipf-winter-12-02",
        }


class CrocusSweSimulationVariable(AbstractAdamontVariable):
    UNIT = CrocusTotalSweVariable.UNIT
    NAME = CrocusTotalSweVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        if annual_maxima_date:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima_date)
        else:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima)

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'swe-max-year-NN'

    @classproperty
    def indicator_name_for_maxima_date(cls):
        return 'date-swe-max-year-NN'

class CrocusDepthSimulationVariable(AbstractAdamontVariable):
    UNIT = CrocusDepthVariable.UNIT
    NAME = CrocusDepthVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima)

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'sd-max-year-NN'


class CrocusTotalSnowLoadVariable(CrocusSweSimulationVariable):
    NAME = TotalSnowLoadVariable.NAME
    UNIT = TotalSnowLoadVariable.UNIT

    @classmethod
    def transform_annual_maxima(cls, annual_maxima_data, maxima_date):
        annual_maxima_data = super().transform_annual_maxima(annual_maxima_data, maxima_date)
        if maxima_date:
            return annual_maxima_data
        else:
            return AbstractSnowLoadVariable.transform_swe_into_snow_load(annual_maxima_data)
