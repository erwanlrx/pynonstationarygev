from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import PolynomialMarginModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel


class AbstractAltitudinalModel(AbstractSpatioTemporalPolynomialModel):
    DISTRIBUTION_STR = 'Gev'

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function(self.param_name_to_list_dim_and_degree_for_margin_function)

    @property
    def param_name_to_list_dim_and_degree_for_margin_function(self):
        return self.param_name_to_list_dim_and_degree

    @property
    def param_name_to_list_dim_and_degree(self):
        raise NotImplementedError

    def dims(self, param_name):
        return [d for d, _ in self.param_name_to_list_dim_and_degree[param_name]]

    def dim_to_str_number(self, param_name, dim):
        list_dim_and_degree = self.param_name_to_list_dim_and_degree[param_name]
        dims = [d for d, _ in list_dim_and_degree]
        if dim not in dims:
            return '0'
        else:
            idx = dims.index(dim)
            return str(list_dim_and_degree[idx][1])

    @property
    def name_str(self):
        name = self.DISTRIBUTION_STR
        name += self.dim_to_str_number(GevParams.LOC, self.coordinates.idx_temporal_coordinates)
        name += self.dim_to_str_number(GevParams.SCALE, self.coordinates.idx_temporal_coordinates)
        if isinstance(self, AbstractAddCrossTermForLocation):
            name += 'x'
        if isinstance(self, AbstractAddCrossTermForScale):
            name += 's'
        return name


class StationaryAltitudinal(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalScaleLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)]
        }


class NonStationaryAltitudinalLocationLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalLocationQuadratic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }

class NonStationaryAltitudinalLocationCubic(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 3)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1)]
        }


class NonStationaryAltitudinalLocationLinearScaleLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
        }


class NonStationaryAltitudinalLocationQuadraticScaleLinear(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree(self):
        return {
            GevParams.LOC: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 2)],
            GevParams.SCALE: [(self.coordinates.idx_x_coordinates, 1), (self.coordinates.idx_temporal_coordinates, 1)],
        }


# Add cross terms

class AbstractAddCrossTermForLocation(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree_for_margin_function(self):
        d = self.param_name_to_list_dim_and_degree
        assert 1 <= len(d[GevParams.LOC]) <= 2
        assert self.coordinates.idx_x_coordinates == d[GevParams.LOC][0][0]
        d[GevParams.LOC].insert(1, ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1))
        return d

class AbstractAddCrossTermForScale(AbstractAltitudinalModel):

    @property
    def param_name_to_list_dim_and_degree_for_margin_function(self):
        d = self.param_name_to_list_dim_and_degree
        # The two insert below enable to check that the insert_index should indeed be 1
        assert 1 <= len(d[GevParams.SCALE]) <= 2
        assert self.coordinates.idx_x_coordinates == d[GevParams.SCALE][0][0]
        insert_index = 1
        d[GevParams.SCALE].insert(insert_index, ((self.coordinates.idx_x_coordinates, self.coordinates.idx_temporal_coordinates), 1))
        return d



class NonStationaryCrossTermForLocation(AbstractAddCrossTermForLocation, StationaryAltitudinal):
    pass


class NonStationaryAltitudinalLocationLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                 NonStationaryAltitudinalLocationLinear):
    pass


class NonStationaryAltitudinalLocationQuadraticCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                    NonStationaryAltitudinalLocationQuadratic):
    pass

class NonStationaryAltitudinalLocationCubicCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                NonStationaryAltitudinalLocationCubic,
                                                                ):
    pass

class NonStationaryAltitudinalLocationLinearScaleLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                            NonStationaryAltitudinalLocationLinearScaleLinear):
    pass


class NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                                               NonStationaryAltitudinalLocationQuadraticScaleLinear):
    pass


class NonStationaryAltitudinalScaleLinearCrossTermForLocation(AbstractAddCrossTermForLocation,
                                                              NonStationaryAltitudinalScaleLinear):
    pass

