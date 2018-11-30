import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List

from extreme_estimator.estimator.full_estimator import AbstractFullEstimator
from extreme_estimator.estimator.margin_estimator import AbstractMarginEstimator
from extreme_estimator.extreme_models.margin_model.margin_function.utils import error_dict_between_margin_functions
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.slicer.split import Split, ALL_SPLITS_EXCEPT_ALL


class MarginFunction(object):

    def margin_function(self, gev_param: GevParams) -> float:
        pass


class LocFunction(MarginFunction):

    def margin_function(self, gev_param: GevParams) -> float:
        return gev_param.location


class SplitCurve(object):

    def __init__(self, dataset: FullSimulatedDataset, estimator: Union[AbstractFullEstimator, AbstractMarginEstimator],
                 margin_functions: List[MarginFunction]):
        # Dataset is already loaded and will not be modified
        self.dataset = dataset
        # Both split must be defined
        assert not self.dataset.slicer.some_required_ind_are_not_defined
        self.margin_function_sample = self.dataset.margin_model.margin_function_sample

        self.estimator = estimator
        # Fit the estimator and get the margin_function
        self.estimator.fit()
        # todo: potentially I will do the fit several times, and retrieve the mean error
        self.margin_function_fitted = estimator.margin_function_fitted

        self.error_dict = error_dict_between_margin_functions(self.margin_function_sample, self.margin_function_fitted)
        # todo: add quantile curve, additionally to the marginal curve

    def visualize(self):
        fig, axes = plt.subplots(3, 2, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.4, wspace=0.4, )
        for i, gev_param_name in enumerate(GevParams.GEV_PARAM_NAMES):
            self.margin_graph(axes[i, 0], gev_param_name)
            self.score_graph(axes[i, 1], gev_param_name)
        plt.show()

    def margin_graph(self, ax, gev_param_name):
        # Display the fitted curve
        self.margin_function_fitted.visualize_single_param(gev_param_name, ax, show=False)
        # Display train/test points
        for split, marker in [(self.dataset.train_split, 'o'), (self.dataset.test_split, 'x')]:
            self.margin_function_sample.set_datapoint_display_parameters(split, datapoint_marker=marker)
            self.margin_function_sample.visualize_single_param(gev_param_name, ax, show=False)
        title_str = gev_param_name
        ax.set_title(title_str)

    def score_graph(self, ax, gev_param_name):
        for split in self.dataset.splits:
            s = self.error_dict[gev_param_name]
        data = [1.5] * 7 + [2.5] * 2 + [3.5] * 8 + [4.5] * 3 + [5.5] * 1 + [6.5] * 8
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(data), bw=0.5, ax=ax)
        print()
