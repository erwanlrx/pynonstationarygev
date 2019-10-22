from typing import Dict, List

import matplotlib.pyplot as plt

from experiment.eurocode_data.eurocode_return_level_uncertainties import EurocodeLevelUncertaintyFromExtremes
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES
from experiment.eurocode_data.utils import EUROCODE_QUANTILE, EUROCODE_ALTITUDES
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes


def plot_model_name_to_dep_to_ordered_return_level_uncertainties(
        dep_to_model_name_to_ordered_return_level_uncertainties, altitudes, show=True):
    # Create a 9 x 9 plot
    axes = create_adjusted_axes(3, 3)
    axes = list(axes.flatten())
    ax6 = axes[5]
    ax9 = axes[8]

    axes.remove(ax6)
    axes.remove(ax9)
    ax_to_departement = dict(zip(axes, DEPARTEMENT_TYPES[::-1]))
    for ax, departement in ax_to_departement.items():
        plot_dep_to_model_name_dep_to_ordered_return_level_uncertainties(ax, departement,
                                                                         altitudes,
                                                                         dep_to_model_name_to_ordered_return_level_uncertainties[
                                                                             departement]
                                                                         )
    ax6.remove()
    ax9.remove()

    plt.suptitle('50-year return levels of snow load for all French Alps departements. \n'
                 'Comparison between the maximum EUROCODE in the departement\n'
                 'and the maximum return level found (in 2017 for the non-stationary model) for the massif in the departement')
    if show:
        plt.show()


def plot_dep_to_model_name_dep_to_ordered_return_level_uncertainties(ax, dep_class,
                                                                     altitudes,
                                                                     model_name_to_ordered_return_level_uncertainties:
                                                                     Dict[str, List[
                                                                         EurocodeLevelUncertaintyFromExtremes]]):
    colors = ['red', 'blue', 'green']
    alpha = 0.2
    # Display the EUROCODE return level
    dep_object = dep_class()
    dep_object.eurocode_region.plot_max_loading(ax, altitudes=altitudes)
    # Display the return level from model class
    for color, (model_name, ordered_return_level_uncertaines) in zip(colors,
                                                                     model_name_to_ordered_return_level_uncertainties.items()):
        mean = [r.posterior_mean for r in ordered_return_level_uncertaines]
        ax.plot(altitudes, mean, '-', color=color, label=model_name)
        lower_bound = [r.poster_uncertainty_interval[0] for r in ordered_return_level_uncertaines]
        upper_bound = [r.poster_uncertainty_interval[1] for r in ordered_return_level_uncertaines]
        ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=alpha)
    ax.legend(loc=2)
    ax.set_ylim([0.0, 4.0])
    ax.set_title(str(dep_object))
    ax.set_ylabel('50-year return level (N $m^-2$)')
    ax.set_xlabel('Altitude (m)')
