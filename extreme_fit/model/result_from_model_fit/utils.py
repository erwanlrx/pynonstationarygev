from collections import OrderedDict

import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.param_function.linear_coef import LinearCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


def convertFloatVector_to_float(f):
    return np.array(f)[0]


def get_margin_coef_ordered_dict(param_name_to_dims, mle_values, type_for_mle="GEV", dim_to_coordinate_name=None):
    assert param_name_to_dims is not None
    # Build the Coeff dict from param_name_to_dim
    coef_dict = OrderedDict()
    i = 0
    for param_name in GevParams.PARAM_NAMES:
        # Add intercept (i.e. stationary parameter)
        intercept_coef_name = LinearCoef.coef_template_str(param_name, LinearCoef.INTERCEPT_NAME).format(1)
        if type_for_mle == "Gumbel" and param_name == GevParams.SHAPE:
            coef_value = 0
        else:
            coef_value = mle_values[i]
        coef_dict[intercept_coef_name] = coef_value
        i += 1
        # Add a potential linear temporal trend
        if param_name in param_name_to_dims:
            dims = param_name_to_dims[param_name]
            if isinstance(dims[0], int):
                coef_name = LinearCoef.coef_template_str(param_name, AbstractCoordinates.COORDINATE_T).format(1)
                coef_dict[coef_name] = mle_values[i]
                i += 1
            else:
                # We found (thanks to the test) that time was the first parameter when len(param_name_to_dims) == 1
                # otherwise time is the second parameter in the order of the mle parameters
                inverted_dims = dims[::-1] if len(param_name_to_dims) == 1 else dims
                for dim, max_degree in inverted_dims:
                    coordinate_name = dim_to_coordinate_name[dim]
                    coef_template = LinearCoef.coef_template_str(param_name, coordinate_name)
                    for j in range(1, max_degree + 1):
                        k = j if coordinate_name == AbstractCoordinates.COORDINATE_T else j + 1
                        coef_name = coef_template.format(k)
                        coef_dict[coef_name] = mle_values[i]
                        i += 1
    return coef_dict
