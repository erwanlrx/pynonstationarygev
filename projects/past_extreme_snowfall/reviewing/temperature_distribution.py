import numpy as np
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import load_plot
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups


def plot_safran_temperature_distribution(altitudes, j):
    studies = AltitudesStudies(study_class=SafranTemperature, altitudes=altitudes)
    massif_name_to_mean_temp = {}
    for massif_name in studies.study.all_massif_names():
        means = []
        for study in studies.altitude_to_study.values():
            if massif_name in study.study_massif_names:
                annual_mean = study.massif_name_to_annual_mean[massif_name]
                means.append(np.mean(annual_mean))
        if len(means) == len(studies.altitudes):
            massif_name_to_mean_temp[massif_name] = float(np.mean(means))
    label = "Mean temperature over 1959-2019 for the range {}".format(j) + " ($^o\mathrm{C}$)"
    massif_name_to_text = {m: str(np.round(v, 1)) for m, v in massif_name_to_mean_temp.items()}
    negative_and_positive_values = j == 4
    load_plot(cmap=plt.cm.coolwarm, graduation=0.5, label=label, massif_name_to_value=massif_name_to_mean_temp,
              altitude=None, add_x_label=False, negative_and_positive_values=negative_and_positive_values,
              massif_name_to_text=massif_name_to_text, min_ratio_equal_to_zero_for_positive_values=False)
    # plt.show()
    visualizer = StudyVisualizer(study, show=False, save_to_file=True)
    visualizer.plot_name = 'mean temperature for the range {}'.format(j)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()

if __name__ == '__main__':
    for altitudes, j in list(zip(altitudes_for_groups, [1, 2, 3, 4]))[2:]:
        print(j)
        plot_safran_temperature_distribution(altitudes, j)