import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.split import Split


class AbstractMarginFunction(object):
    """ Class of function mapping points from a space S (could be 1D, 2D,...) to R^3 (the 3 parameters of the GEV)"""

    def __init__(self, coordinates: AbstractCoordinates):
        self.coordinates = coordinates

        # Visualization parameters
        self.visualization_axes = None
        self.datapoint_display = False
        self.spatio_temporal_split = Split.all
        self.datapoint_marker = 'o'

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Main method that maps each coordinate to its GEV parameters"""
        pass

    # Extraction function

    @property
    def gev_params_for_coordinates(self):
        gev_params = [self.get_gev_params(coordinate).to_dict() for coordinate in self.coordinates.coordinates_values()]
        gev_param_name_to_serie = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            s = pd.Series(data=[p[gev_param_name] for p in gev_params], index=self.coordinates.index)
            gev_param_name_to_serie[gev_param_name] = s
        return gev_param_name_to_serie

    # Visualization function

    def set_datapoint_display_parameters(self, spatio_temporal_split, datapoint_marker):
        self.datapoint_display = True
        self.spatio_temporal_split = spatio_temporal_split
        self.datapoint_marker = datapoint_marker

    def visualize(self, axes=None, show=True, dot_display=False):
        self.datapoint_display = dot_display
        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex='col', sharey='row')
            fig.subplots_adjust(hspace=0.4, wspace=0.4, )
        self.visualization_axes = axes
        for i, gev_param_name in enumerate(GevParams.GEV_PARAM_NAMES):
            ax = axes[i]
            self.visualize_single_param(gev_param_name, ax, show=False)
            title_str = gev_param_name
            ax.set_title(title_str)
        if show:
            plt.show()

    def visualize_single_param(self, gev_param_name=GevParams.GEV_LOC, ax=None, show=True):
        if self.coordinates.nb_columns == 1:
            self.visualize_1D(gev_param_name, ax, show)
        elif self.coordinates.nb_columns == 2:
            self.visualize_2D(gev_param_name, ax, show)
        else:
            raise NotImplementedError('3D Margin visualization not yet implemented')

    def visualize_1D(self, gev_param_name=GevParams.GEV_LOC, ax=None, show=True):
        x = self.coordinates.x_coordinates
        grid, linspace = self.get_grid_1D(x)
        gev_param_idx = GevParams.GEV_PARAM_NAMES.index(gev_param_name)
        if ax is None:
            ax = plt.gca()
        if self.datapoint_display:
            ax.plot(linspace, grid[:, gev_param_idx], self.datapoint_marker)
        else:
            ax.plot(linspace, grid[:, gev_param_idx])

        if show:
            plt.show()

    def visualize_2D(self, gev_param_name=GevParams.GEV_LOC, ax=None, show=True):
        x = self.coordinates.x_coordinates
        y = self.coordinates.y_coordinates
        grid = self.get_grid_2D(x, y)
        gev_param_idx = GevParams.GEV_PARAM_NAMES.index(gev_param_name)
        if ax is None:
            ax = plt.gca()
        imshow_method = ax.imshow
        imshow_method(grid[..., gev_param_idx], extent=(x.min(), x.max(), y.min(), y.max()),
                      interpolation='nearest', cmap=cm.gist_rainbow)
        # todo: add dot display in 2D
        if show:
            plt.show()

    def get_grid_1D(self, x):
        # TODO: to avoid getting the value several times, I could cache the results
        if self.datapoint_display:
            linspace = self.coordinates.coordinates_values(self.spatio_temporal_split)[:, 0]
            resolution = len(linspace)
        else:
            resolution = 100
            linspace = np.linspace(x.min(), x.max(), resolution)

        grid = np.zeros([resolution, 3])
        for i, xi in enumerate(linspace):
            grid[i] = self.get_gev_params(np.array([xi])).to_array()
        return grid, linspace

    def get_grid_2D(self, x, y):
        resolution = 100
        grid = np.zeros([resolution, resolution, 3])
        for i, xi in enumerate(np.linspace(x.min(), x.max(), resolution)):
            for j, yj in enumerate(np.linspace(y.min(), y.max(), resolution)):
                grid[i, j] = self.get_gev_params(np.array([xi, yj])).to_array()
        return grid
