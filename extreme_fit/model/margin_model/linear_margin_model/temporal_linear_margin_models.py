from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import r
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class StationaryTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({})


class NonStationaryLocationTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1


class NonStationaryScaleTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def sigl(self):
        return 1


class NonStationaryLogScaleTemporalModel(NonStationaryScaleTemporalModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def siglink(self):
        return r('exp')


class NonStationaryShapeTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def shl(self):
        return 1


class NonStationaryLocationAndScaleTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [self.coordinates.idx_temporal_coordinates],
                                             GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1

    @property
    def sigl(self):
        return 1


class NonStationaryLocationAndShapeTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [self.coordinates.idx_temporal_coordinates],
                                             GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1

    @property
    def shl(self):
        return 1


class NonStationaryScaleAndShapeTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates],
                                             GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def shl(self):
        return 1

    @property
    def sigl(self):
        return 1


class NonStationaryLocationAndScaleAndShapeTemporalModel(AbstractTemporalLinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [self.coordinates.idx_temporal_coordinates],
                                             GevParams.SCALE: [self.coordinates.idx_temporal_coordinates],
                                             GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1

    @property
    def sigl(self):
        return 1

    @property
    def shl(self):
        return 1


class GumbelTemporalModel(StationaryTemporalModel):

    def __init__(self, coordinates: AbstractCoordinates,
                 params_user=None, starting_point=None, fit_method=MarginFitMethod.is_mev_gev_fit,
                 nb_iterations_for_bayesian_fit=5000, params_initial_fit_bayesian=None):
        super().__init__(coordinates, params_user, starting_point, fit_method,
                         nb_iterations_for_bayesian_fit, params_initial_fit_bayesian, type_for_MLE="Gumbel")

    @property
    def nb_params(self):
        return super().nb_params - 1


class NonStationaryLocationGumbelModel(GumbelTemporalModel, NonStationaryLocationTemporalModel):
    pass


class NonStationaryScaleGumbelModel(GumbelTemporalModel, NonStationaryScaleTemporalModel):
    pass


class NonStationaryLocationAndScaleGumbelModel(GumbelTemporalModel, NonStationaryLocationAndScaleTemporalModel):
    pass
