import numpy as np
import os.path as op
import matplotlib

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import \
    get_year_min_and_year_max_used_to_compute_quantile
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, gcm_rcm_couple_to_color, \
    gcm_rcm_couple_to_str, load_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, \
    SafranSnowfallCenterOnDay1dayMeanRate, SafranSnowfallCenterOnDay1day
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020, \
    SafranSnowfall2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariableCenterOnDay

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    STUDY_CLASS_TO_ABBREVIATION
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


def compute_bias_and_display_it(ax,
                                altitude_studies_reanalysis: AltitudesStudies,
                                adamont_altitude_studies: AltitudesStudies,
                                gcm_rcm_couple,
                                massif_names=None,
                                            relative_bias=False,
                                ):
    bias_in_the_mean_maxima = []
    altitudes = []
    for altitude, study_reanalysis in altitude_studies_reanalysis.altitude_to_study.items():
        adamont_study = adamont_altitude_studies.altitude_to_study[altitude]
        can_compute_biais = (massif_names is None) or any([m in adamont_study.study_massif_names for m in massif_names])
        if can_compute_biais:

            altitudes.append(altitude)

            if massif_names is None:
                mean_maxima_adamont = adamont_study.mean_annual_maxima
                mean_maxima_reanalysis = study_reanalysis.mean_annual_maxima
            else:
                mean_maxima_reanalysis = np.mean(np.concatenate([study_reanalysis.massif_name_to_annual_maxima[m]
                                                                 for m in massif_names
                                                                 if m in study_reanalysis.massif_name_to_annual_maxima]))
                mean_maxima_adamont = np.mean(np.concatenate([adamont_study.massif_name_to_annual_maxima[m]
                                                              for m in massif_names
                                                              if m in adamont_study.massif_name_to_annual_maxima]))

            bias = mean_maxima_adamont - mean_maxima_reanalysis
            if relative_bias:
                bias *= 100 / mean_maxima_reanalysis
            bias_in_the_mean_maxima.append(bias)

    color = gcm_rcm_couple_to_color[gcm_rcm_couple]
    label = gcm_rcm_couple_to_str(gcm_rcm_couple)
    ax.plot(bias_in_the_mean_maxima, altitudes, label=label, color=color)

    return np.array(bias_in_the_mean_maxima), altitudes


def main_comparaison_plot():
    altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]

    for adamont_version in [1, 2][:]:
        print('version:', adamont_version)

        gcm_rcm_couples = load_gcm_rcm_couples(adamont_scenario=AdamontScenario.histo, adamont_version=adamont_version)

        study_class_for_adamont_v1 = SafranSnowfall1Day
        study_class = SafranSnowfall2020 if adamont_version == 2 else study_class_for_adamont_v1
        comparaison_study_class = 'SAFRAN 2020' if adamont_version == 2 else 'SAFRAN 2019'

        # Faster to load once the two cases
        reanalysis_altitude_studies_1981 = AltitudesStudies(study_class=study_class,
                                                            altitudes=altitudes,
                                                            year_min=1981,
                                                            year_max=2011)
        reanalysis_altitude_studies_1988 = AltitudesStudies(study_class=study_class,
                                                            altitudes=altitudes,
                                                            year_min=1988,
                                                            year_max=2011)

        if adamont_version == 1:
            list_of_massis_names = [None]
        else:
            list_of_massis_names = [None] + [[m] for m in reanalysis_altitude_studies_1981.study.all_massif_names()]

        for relative_bias in [True, False][:1]:
            for massif_names in list_of_massis_names[:]:
                ax = plt.gca()
                bias_in_the_mean = []
                list_altitudes_for_bias = []
                for gcm_rcm_couple in gcm_rcm_couples:
                    print(massif_names, gcm_rcm_couple)
                    gcm, rcm = gcm_rcm_couple
                    years_reanalysis, years_model = get_year_min_and_year_max_used_to_compute_quantile(gcm)
                    assert years_reanalysis[0] in [1981, 1988]
                    if years_reanalysis[0] == 1981:
                        reanalysis_altitude_studies = reanalysis_altitude_studies_1981
                    else:
                        reanalysis_altitude_studies = reanalysis_altitude_studies_1988
                    adamont_altitude_studies = AltitudesStudies(study_class=AdamontSnowfall,
                                                                altitudes=altitudes,
                                                                year_min=years_model[0],
                                                                year_max=years_model[1],
                                                                scenario=AdamontScenario.histo,
                                                                gcm_rcm_couple=gcm_rcm_couple,
                                                                adamont_version=adamont_version)
                    bias, altitudes_for_bias = compute_bias_and_display_it(ax, reanalysis_altitude_studies,
                                                       adamont_altitude_studies, gcm_rcm_couple, massif_names,
                                                                           relative_bias)
                    bias_in_the_mean.append(bias)
                    list_altitudes_for_bias.append(altitudes_for_bias)

                # Assert the all the bias have been computed for the same altitudes
                altitudes_for_bias = list_altitudes_for_bias[0]
                for alti in list_altitudes_for_bias:
                    assert alti == altitudes_for_bias

                bias_in_the_mean = np.array(bias_in_the_mean)
                min_bias, median_bias, max_bias = [f(bias_in_the_mean, axis=0) for f in [np.min, np.median, np.max]]

                # Plot the range for the bias, and the median
                ax.yaxis.set_ticks(altitudes)
                color = 'k'
                ax.plot(median_bias, altitudes_for_bias, label='Median bias', color=color, linewidth=4)
                # ax.fill_betweenx(altitudes, min_bias, max_bias, label='Range for the bias', alpha=0.2, color='whitesmoke')
                ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k', linestyles='dashed')
                study_str = STUDY_CLASS_TO_ABBREVIATION[type(reanalysis_altitude_studies.study)]
                ax.set_ylim(top=altitudes[-1] + 1300)
                study = adamont_altitude_studies.study
                ax.legend(ncol=3, prop={'size': 7})
                ax.set_ylabel('Altitude (m)', fontsize=10)
                massif_str = 'all massifs' if massif_names is None else 'the {} massif'.format(massif_names[0])
                unit = '%' if relative_bias else study.variable_unit
                bias_name = 'Relative bias' if relative_bias else 'Bias'
                title = '{} in the mean annual maxima of {} of {}\n' \
                                 'for ADAMONT v{}' \
                                 ' against {} on the quantile mapping period ({})'.format(bias_name,
                                                                                          study_str, massif_str,
                                                                                          adamont_version,
                                                                                          comparaison_study_class,
                                                                                          unit)
                folder = 'relative bias' if relative_bias else 'absolute bias'
                plot_name = op.join(folder, title)
                ax.set_xlabel(title, fontsize=10)
                reanalysis_altitude_studies.show_or_save_to_file(plot_name=plot_name, no_title=True)
                plt.close()


if __name__ == '__main__':
    main_comparaison_plot()
