
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from projected_extremes.reviewing.reviewing_utils import load_csv_filepath_gof, mode_to_name


def plot_pvalue_test():
    ax = plt.gca()
    all_massif = False
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
    for mode in range(1):
        percentages = []
        for altitude in altitudes:
            csv_filepath = load_csv_filepath_gof(mode, altitude, all_massif)
            s = pd.read_csv(csv_filepath, index_col=0)
            pvalues = s.iloc[:, 0].values
            count_above_5_percent = [int(m >= 0.05) for m in pvalues]
            percentage_above_5_percent = 100 * sum(count_above_5_percent) / len(count_above_5_percent)
            print(percentage_above_5_percent)
            percentages.append(percentage_above_5_percent)
        ax.plot(percentages, altitudes, label=mode_to_name[mode])
    ax.set_xlim((0, 100))
    ylim = ax.get_ylim()
    ax.vlines(5, ymin=ylim[0], ymax=ylim[1], color='k', linestyles='dashed', label='5\% significance level')
    ax.set_xlabel('% of positive Anderson-Darling test for a 5% significance level')
    ax.set_ylabel('Elevation (m)')
    ax.legend()
    plt.show()



def quantitative(pvalues, visualizer):
    # Create an histogram for the metric
    ax = plt.gca()
    count_above_5_percent = [int(m >= 0.05) for m in pvalues]
    percentage_above_5_percent = 100 * sum(count_above_5_percent) / len(count_above_5_percent)
    print("Percentage above 5 percent", percentage_above_5_percent)
    count_above_1_percent = [int(m >= 0.01) for m in pvalues]
    percentage_above_1_percent = 100 * sum(count_above_1_percent) / len(count_above_1_percent)
    print("Percentage above 1 percent", percentage_above_1_percent)
    ax.hist(pvalues, bins=20, range=[0, 1], weights=np.ones(len(pvalues)) / len(pvalues))
    ax.set_xlim((0, 1))
    ylim = ax.get_ylim()
    ax.vlines(0.05, ymin=ylim[0], ymax=ylim[1], color='k', linestyles='dashed', label='0.05 significance level')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('p-value for the Anderson-Darling test')
    ax.set_ylabel('Percentage')
    ax.legend()
    visualizer.plot_name = 'All pvalues'
    visualizer.show_or_save_to_file()

if __name__ == '__main__':
    plot_pvalue_test()