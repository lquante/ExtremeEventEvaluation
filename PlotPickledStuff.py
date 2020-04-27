#!/usr/bin/env python

# TODO: extend to multiple variables
import argparse
import os
# Imports
import pickle

import cartopy
import iris
import iris.coord_categorisation
import iris.plot as iplt
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import ruamel

# define some plotting methods:

# define colormap whitefilling 0 values

colors_no_extreme = plt.cm.viridis(np.linspace(0, 0.05, 1))
colors_extreme = plt.cm.viridis(np.linspace(0.1, 1, 20))
all_colors = np.vstack((colors_no_extreme, colors_extreme))
# make color value for 0 white
all_colors[0] = (1, 1, 1, 1.0)

extremes_cmap = clr.LinearSegmentedColormap.from_list('extremes_map',
                                                      all_colors)
colormap = plt.get_cmap(extremes_cmap, 30)


# plot map cubes:

def plot_cube(cube, startyear, finalyear, threshold, vmin, vmax, label):
    pcm = iris.plot.pcolormesh(cube, cmap=extremes_cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, extend='both', orientation='horizontal', label=label)

    plt.title(str(startyear) + " to " + str(finalyear), fontsize=10)
    plt.gca().coastlines()


def plot_difference_cube(cube, scenario_data, startyear_data, finalyear_data, scenario_comparison, startyear_comparison,
                         finalyear_comparison, threshold, vmin, vmax, label):
    colormap = plt.get_cmap('RdBu_r', 30)
    ax = plt.axes(projection=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    ax.coastlines()
    pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(pcm, extend='both', orientation='horizontal', label=label)
    cbar.minorticks_on()

    plt.title(scenario_data + ": " + str(startyear_data) + " to " + str(finalyear_data) + "\n -" +
              scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_ratio_cube(cube, scenario_data, startyear_data, finalyear_data, scenario_comparison, startyear_comparison,
                    finalyear_comparison, specification, label):
    colormap = plt.get_cmap('RdBu_r', 30)
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')

    pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=0, vmax=2)

    cbar = plt.colorbar(pcm, extend='both', orientation='horizontal', label=label)
    cbar.minorticks_on()

    plt.title(scenario_data + ": " + str(startyear_data) + " to " + str(finalyear_data) + "\n -" +
              scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_ensemble_average_quantile_baseline(file, quantile, scenario, start_years,
                                            timeperiod_length, reference_scenario, reference_start_year,
                                            reference_final_year, areaname, maps=True,
                                            quantile_ratios=True):
    label_frequency = 'days with daily snowfall > baseline'
    label_frequency_diff = 'difference of ' + label_frequency
    label_es = 'expected excess snowfall > baseline (mm)'
    label_es_diff = 'difference of ' + label_es

    label_es_diff_relative = 'baseline relative ' + label_es_diff

    label_quantile_ratio = 'ratio of percentile / baseline percentile'
    label_es_ratio = 'ratio of expected extreme snowfall (EES) / baseline EES'

    label_ratio_diff = "difference of percentile and expected snofall to resp. baseline ratio"

    modelname = 'ISIMIP_3b_primary_model_average'

    data_to_plot = pickle.load(file)
    # dict_to_plot[i_start_year, 'diff_frequency'] = diff_frequency_average[i_start_year]
    # dict_to_plot[i_start_year, 'diff_es'] = diff_expected_snowfall_average[i_start_year]
    # dict_to_plot[i_start_year, 'diff_relative'] = diff_expected_snowfall_average_relative[i_start_year]

    # dict_to_plot[i_start_year, 'quantile_ratio'] = quantile_baseline_ratio[i_start_year]
    # dict_to_plot[i_start_year, 'es_ratio'] = expected_snowfall_ratio[i_start_year]
    # dict_to_plot[i_start_year, 'diff_ratios'] = diff_ratios[i_start_year]

    number_of_timeperiods = len(start_years)

    diff_freq_key = 'diff_frequency'
    diff_es_key = 'diff_es'
    diff_rel_es_key = 'diff_relative'

    quantile_ratio_key = 'quantile_ratio'
    es_ratio_key = 'es_ratio'
    diff_ratio_key = 'diff_ratios'
    if (quantile_ratios):

        fig, fig_axs = plt.subplots(ncols=3, nrows=number_of_timeperiods, figsize=(18, 4 * number_of_timeperiods))
        gs = fig_axs[0, 0].get_gridspec()
        # remove the underlying axes
        for i in range(0, 3):
            for ax in fig_axs[0:, i]:
                ax.remove()
        for i in range(0, number_of_timeperiods):
            final_year = start_years[i] + timeperiod_length - 1
            fig.add_subplot(gs[i, 0])
            plot_ratio_cube(data_to_plot[quantile, start_years[i], quantile_ratio_key], scenario, start_years[i],
                            final_year,
                            reference_scenario, reference_start_year, reference_final_year, quantile,
                            label_quantile_ratio)

            fig.add_subplot(gs[i, 1])

            plot_ratio_cube(data_to_plot[quantile, start_years[i], es_ratio_key], scenario, start_years[i], final_year,
                            reference_scenario, reference_start_year, reference_final_year, quantile, label_es_ratio)

            fig.add_subplot(gs[i, 2])

            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_ratio_key], scenario, start_years[i],
                                 final_year, reference_scenario,
                                 reference_start_year, reference_final_year, quantile, -2, 2, label_ratio_diff)

        plt.tight_layout()
        filename = 'quantile_baseline_ratio_maps_' + str(quantile) + '_' + str(
            areaname) + '_' + reference_scenario + '_vs_' + scenario + '_' + str(start_years[0]) + '_' + str(final_year)
        suptitle = str(modelname) + " - " + str(
            areaname) + '- \n' + scenario + ' versus ' + reference_scenario + '\n  ratio to baseline ' + str(
            quantile) + ' percentile'
        fig.subplots_adjust(top=0.875)
        plt.suptitle(suptitle, y=0.95)

        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    if (maps):
        fig, fig_axs = plt.subplots(ncols=3, nrows=number_of_timeperiods, figsize=(18, 4 * number_of_timeperiods))
        gs = fig_axs[0, 0].get_gridspec()
        # remove the underlying axes
        for i in range(0, 3):
            for ax in fig_axs[0:, i]:
                ax.remove()
        for i in range(0, number_of_timeperiods):
            # fig.add_subplot(gs[i, 0])
            # plot_cube(baseline_average[i_start_year],1851,1880,quantile,vmin_base,vmax_base,label_baseline)
            # plt.title(str(modelname)+' - '+ 'baseline '+str(quantile)+' percentile (historical, 1851-1880)')
            final_year = start_years[i] + timeperiod_length - 1
            fig.add_subplot(gs[i, 0])
            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_freq_key], scenario, start_years[i],
                                 final_year,
                                 reference_scenario, reference_start_year, reference_final_year, quantile,
                                 -50, 50, label_frequency_diff)

            fig.add_subplot(gs[i, 1])

            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_es_key], scenario, start_years[i],
                                 final_year,
                                 reference_scenario, reference_start_year, reference_final_year, quantile, -500,
                                 500, label_es_diff)

            fig.add_subplot(gs[i, 2])

            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_rel_es_key], scenario, start_years[i],
                                 final_year, reference_scenario, reference_start_year, reference_final_year, quantile,
                                 -1, 1, label_es_diff_relative)

        plt.tight_layout()
        filename = 'quantile_ensemble_average_maps_' + str(quantile) + '_' + str(
            areaname) + '_' + reference_scenario + '_vs_' + scenario + '_' + str(start_years[0]) + '_' + str(final_year)
        suptitle = str(modelname) + " - " + str(
            areaname) + '- \n' + scenario + ' versus ' + reference_scenario + '\n comparison  of daily snowfall > \n baseline ' + str(
            quantile) + ' percentile'
        fig.subplots_adjust(top=0.875)
        plt.suptitle(suptitle, y=0.95)

        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


# add settings argument

parser = argparse.ArgumentParser(description="Calculate some analysis metrics on specified data files")

# settings file import
# argument parser definition
parser = argparse.ArgumentParser(description="Generate a movie on the time dimension of Geo-referenced netcdf file(s)")
# path to *.yml file with settings to be used
parser.add_argument(
    "--settings"
    , type=str,
    help="Path to the settingsfile (default: CURRENT/settings.yml)"
)

args = parser.parse_args()

# default settings
if not args.settings:
    args.settings = os.path.join(os.getcwd(), 'settings.yml')

# load settings
# load settings file
yaml = ruamel.yaml.YAML()
with open(args.settings, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

filelist = settings['files']
outputdir = settings['outputdir']

areaname = settings['areaname']
quantiles = settings['quantiles']
start_years = settings['start_years']
# set outputpath for plots etc
os.chdir(outputdir)

for i_file in filelist:
    for i_quantile in quantiles:
        for i_start_years in start_years:
            scenario = 'ssp585'
            timeperiod_length = 10
            reference_scenario = 'historical'
            reference_start_year = 1851
            reference_final_year = 1880
            plot_ensemble_average_quantile_baseline(i_file, i_quantile, scenario, i_start_years, timeperiod_length,
                                                    reference_scenario, reference_start_year,
                                                    reference_final_year, areaname, maps=True, quantile_ratios=True)
