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
import matplotlib.path as mpath
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

    pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.gca().add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='w')
    plt.gca().coastlines('110m')
    plt.gca().gridlines()

    # https: // scitools.org.uk / cartopy / docs / latest / gallery / always_circular_stereo.html?highlight = circular
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    plt.gca().set_boundary(circle, transform=plt.gca().transAxes)

    plt.colorbar(pcm, shrink=0.7, extend='both', orientation='horizontal', label=label)

    plt.title(scenario_data + ": " + str(startyear_data) + " to " + str(finalyear_data) + "\n -" +
              scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_ratio_cube(cube, scenario_data, startyear_data, finalyear_data, scenario_comparison, startyear_comparison,
                    finalyear_comparison, specification, label):
    colormap = plt.get_cmap('RdBu_r', 30)

    pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=0, vmax=2)
    plt.gca().add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='w')
    plt.gca().coastlines('110m')
    plt.gca().gridlines()

    # https: // scitools.org.uk / cartopy / docs / latest / gallery / always_circular_stereo.html?highlight = circular
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    plt.gca().set_boundary(circle, transform=plt.gca().transAxes)

    plt.colorbar(pcm, shrink=0.7, extend='both', orientation='horizontal', label=label)

    plt.title(scenario_data + ": " + str(startyear_data) + " to " + str(finalyear_data) + "\n -" +
              scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_ensemble_average_quantile_baseline(file, scenario, start_years,
                                            timeperiod_length, reference_scenario, reference_start_year,
                                            reference_final_year, areaname, maps=True,
                                            quantile_ratios=True):
    label_frequency = 'days with daily snowfall > baseline'
    label_frequency_diff = 'difference of ' + label_frequency
    label_es = 'expected excess snowfall > baseline (mm)'
    label_es_diff = 'difference of ' + label_es

    label_es_rel_diff = label_es_diff + '\n normed to expected number of events'

    label_es_diff_relative = 'baseline relative ' + label_es_diff

    label_quantile_ratio = 'ratio of percentile / baseline percentile'
    label_es_ratio = 'ratio of expected extreme snowfall (EES) / baseline EES'

    label_mean_ratio = 'ratio of mean daily snowfall / baseline '

    modelname = 'ISIMIP_3b_primary_model_average'
    with open(file, 'rb') as stream:
        data_to_plot = (pickle.load(stream))

    quantile = (list(data_to_plot.keys())[1][0])

    number_of_timeperiods = len(start_years)

    diff_freq_key = 'diff_frequency'
    diff_es_key = 'diff_es'
    diff_rel_es_key = 'diff_es_baseline'

    quantile_ratio_key = 'quantile_ratio'
    es_ratio_key = 'es_ratio'
    mean_ratio_key = 'mean_ratio'
    projection_crs = cartopy.crs.AzimuthalEquidistant(central_latitude=90)
    if (quantile_ratios):
        number_of_columns = 3
        fig, fig_axs = plt.subplots(ncols=number_of_columns, nrows=number_of_timeperiods,
                                    figsize=(5.5 * number_of_columns, 5 * number_of_timeperiods))
        gs = fig_axs[0, 0].get_gridspec()
        # remove the underlying axes
        for i in range(0, number_of_columns):
            for ax in fig_axs[0:, i]:
                ax.remove()
        for i in range(0, number_of_timeperiods):
            final_year = start_years[i] + timeperiod_length - 1

            # mean ratio
            fig.add_subplot(gs[i, 0], projection=projection_crs)
            plot_ratio_cube(data_to_plot[quantile, start_years[i], mean_ratio_key], scenario, start_years[i],
                            final_year, reference_scenario,
                            reference_start_year, reference_final_year, quantile, label_mean_ratio)

            # quantile ratio
            fig.add_subplot(gs[i, 1], projection=projection_crs)

            plot_ratio_cube(data_to_plot[quantile, start_years[i], quantile_ratio_key], scenario, start_years[i],
                            final_year,
                            reference_scenario, reference_start_year, reference_final_year, quantile,
                            label_quantile_ratio)

            # es ratio
            fig.add_subplot(gs[i, 2], projection=projection_crs)

            plot_ratio_cube(data_to_plot[quantile, start_years[i], es_ratio_key], scenario, start_years[i], final_year,
                            reference_scenario, reference_start_year, reference_final_year, quantile, label_es_ratio)



        plt.tight_layout()
        filename = 'baseline_ratio_maps_' + str(modelname) + '_' + str(quantile) + '_' + str(
            areaname) + '_' + reference_scenario + '_vs_' + scenario + '_' + str(start_years[0]) + '_' + str(final_year)
        suptitle = str(modelname) + " - " + str(
            areaname) + '- \n' + scenario + ' versus ' + reference_scenario + '\n  ratio to baseline ' + str(
            quantile) + ' percentile'
        fig.subplots_adjust(top=0.85)
        plt.suptitle(suptitle, y=0.95)


        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    if (maps):
        if (quantile == 99.9):
            freq_max_value = 20
            es_max_value = 100
            es_rel_max_value = 200
        else:
            if (quantile == 99):
                freq_max_value = 100
                es_max_value = 50
                es_rel_max_value = 100
            else:
                freq_max_value = 100
                es_max_value = 50
                es_rel_max_value = 100
        number_of_columns = 3
        fig, fig_axs = plt.subplots(ncols=number_of_columns, nrows=number_of_timeperiods,
                                    figsize=(5.5 * number_of_columns, 5 * number_of_timeperiods))
        gs = fig_axs[0, 0].get_gridspec()
        for i in range(0, number_of_columns):
            for ax in fig_axs[0:, i]:
                ax.remove()

        for i in range(0, number_of_timeperiods):
            # fig.add_subplot(gs[i, 0])
            # plot_cube(baseline_average[i_start_year],1851,1880,quantile,vmin_base,vmax_base,label_baseline)
            # plt.title(str(modelname)+' - '+ 'baseline '+str(quantile)+' percentile (historical, 1851-1880)')

            final_year = start_years[i] + timeperiod_length - 1
            fig.add_subplot(gs[i, 0], projection=projection_crs)
            # frequency_difference

            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_freq_key], scenario, start_years[i],
                                 final_year,
                                 reference_scenario, reference_start_year, reference_final_year, quantile,
                                 -freq_max_value, freq_max_value, label_frequency_diff)

            # es_difference
            fig.add_subplot(gs[i, 1], projection=projection_crs)
            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_es_key], scenario, start_years[i],
                                 final_year,
                                 reference_scenario, reference_start_year, reference_final_year, quantile,
                                 -es_max_value,
                                 es_max_value, label_es_diff)

            # baseline_normed_es_difference
            fig.add_subplot(gs[i, 2], projection=projection_crs)
            print(data_to_plot[quantile, start_years[i], diff_rel_es_key])
            plot_difference_cube(data_to_plot[quantile, start_years[i], diff_rel_es_key], scenario, start_years[i],
                                 final_year, reference_scenario, reference_start_year, reference_final_year, quantile,
                                 -es_max_value,
                                 es_max_value, label_es_rel_diff)

        plt.tight_layout()
        filename = 'quantile_ensemble_average_maps_' + str(quantile) + '_' + str(
            areaname) + '_' + reference_scenario + '_vs_' + scenario + '_' + str(start_years[0]) + '_' + str(final_year)
        suptitle = str(modelname) + " - " + str(
            areaname) + '- \n' + scenario + ' versus ' + reference_scenario + '\n comparison  of daily snowfall > \n baseline ' + str(
            quantile) + ' percentile'
        fig.subplots_adjust(top=0.85)
        plt.suptitle(suptitle, y=0.95)

        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


#
# settings file import
# argument parser definition
parser = argparse.ArgumentParser(description="Plot maps from pickled analysis data.")
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

areanames = settings['areanames']
quantiles = settings['quantiles']
start_years = settings['start_years']
# set outputpath for plots etc
os.chdir(outputdir)
i=0
for i_file in filelist:
    scenario = 'ssp585'
            timeperiod_length = 10
            reference_scenario = 'historical'
            reference_start_year = 1851
            reference_final_year = 1880
    areaname = areanames[i]
    i_start_years = start_years[i]
            plot_ensemble_average_quantile_baseline(i_file, scenario, i_start_years, timeperiod_length,
                                                    reference_scenario, reference_start_year,
                                                    reference_final_year, areaname, maps=True, quantile_ratios=True)
    i += 1
