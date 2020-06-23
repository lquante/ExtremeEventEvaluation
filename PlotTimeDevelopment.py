#!/usr/bin/env python


import argparse
import os
# Imports
import pickle
import string
from collections import OrderedDict
from datetime import datetime

import cartopy
import cartopy.io
import cf_units
import iris
import iris.analysis
import iris.coord_categorisation
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.colors as clr
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from iris.util import unify_time_units
from ruamel.yaml import ruamel
from shapely.geometry import Point
# define colormap whitefilling 0 values
from tqdm import tqdm

# define some plotting methods:

colors_no_extreme = plt.cm.viridis(np.linspace(0, 0.05, 1))
colors_extreme = plt.cm.viridis(np.linspace(0.1, 1, 20))
all_colors = np.vstack((colors_no_extreme, colors_extreme))
# make color value for 0 white
all_colors[0] = (1, 1, 1, 1.0)

extremes_cmap = clr.LinearSegmentedColormap.from_list('extremes_map',
                                                      all_colors)
colormap = plt.get_cmap(extremes_cmap, 30)


# plot methods for time series cubes
# function to check whether data is within shapefile - source: http://bit.ly/2pKXnWa

def get_mask(cube, geom):
    mask = np.ones(cube.data.shape)
    p = -1
    for i_geom in tqdm(geom):
        for i in (np.ndindex(cube.data.shape)):
            if i[0] != p:
                p = i[0]
            this_cube = cube[i]
            this_lat = this_cube.coord('latitude').points[0]
            this_lon = iris.analysis.cartography.wrap_lons(this_cube.coord('longitude').points[0], -180, 360)
            this_point = Point(this_lon, this_lat)
            mask[i] = this_point.within(i_geom)
    mask = mask.astype(bool)
    return mask


# methods to calculate area averages for cubes and concatenate them

def cube_from_bounding_box(cube, bounding_box):
    longitude_min = np.min((bounding_box[1], bounding_box[3]))
    longitude_max = np.max((bounding_box[1], bounding_box[3]))

    latitude_min = np.min((bounding_box[0], bounding_box[2]))
    latitude_max = np.max((bounding_box[0], bounding_box[2]))

    longitude_constrained = cube.intersection(latitude=(latitude_min, latitude_max))
    return longitude_constrained.intersection(longitude=(longitude_min, longitude_max))


def cube_area_average(cube, boundingbox):
    area_cube = cube_from_bounding_box(cube, boundingbox)
    return area_cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN)


def concatenate_cube_dict(cubedict, maskfile):
    keys = list(cubedict.keys())
    initial_cube = cubedict[keys[0]]

    if individual_mask:
        ocean_shapefile = '/p/tmp/quante/ExtremeEventEvaluation/ne_110m_ocean/ne_110m_ocean.shp'
        reader = cartopy.io.shapereader.Reader(ocean_shapefile)
        oceans = reader.geometries()
        cube_mask = get_mask(initial_cube, oceans)
        file = open(maskfile, 'wb')
        pickle.dump(cube_mask, file)
    else:
        file = open(maskfile, 'rb')
        cube_mask = pickle.load(file)
    cubelist = iris.cube.CubeList([iris.util.mask_cube(initial_cube, cube_mask)])

    # cubelist = iris.cube.CubeList([initial_cube])
    number_cubes = len(keys)
    for i in range(1, number_cubes):
        cubelist.append(iris.util.mask_cube(cubedict[keys[i]], cube_mask))
        # cubelist.append(cubedict[keys[i]])

    return cubelist


# plotting of maps of cubes


def plot_difference_cube(cube, scenario_data, model, quantile, startyear_data, finalyear_data, scenario_comparison,
                         startyear_comparison,
                         finalyear_comparison, projection, vmin, vmax, label):
    colormap = plt.get_cmap('RdBu_r', 30)

    ax = plt.axes(projection=projection)
    pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=vmin, vmax=vmax)

    ax.coastlines('110m')
    ax.gridlines()
    # https: // scitools.org.uk / cartopy / docs / latest / gallery / always_circular_stereo.html?highlight = circular
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=plt.gca().transAxes)

    plt.colorbar(pcm, shrink=0.7, extend='both', orientation='horizontal', label=label)

    plt.title(
        model + " percentile: " + str(quantile) + "\n" + scenario_data + ": " + str(startyear_data) + " to " + str(
            finalyear_data) +
        scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_ratio_cube(cube, scenario_data, modelname, quantile, startyear_data, finalyear_data, scenario_comparison,
                    startyear_comparison,
                    finalyear_comparison, projection, label, vmin=0, vmax=2):
    colormap = plt.get_cmap('RdBu_r', 30)
    ax = plt.axes(projection=projection)
    pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=vmin, vmax=vmax)

    ax.coastlines('110m')
    ax.gridlines()

    # https: // scitools.org.uk / cartopy / docs / latest / gallery / always_circular_stereo.html?highlight = circular
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=plt.gca().transAxes)

    plt.colorbar(pcm, shrink=0.7, extend='both', orientation='horizontal', label=label)

    plt.title(modelname + " percentile: " + str(quantile) + "\n" + " " + scenario_data + ": " + str(
        startyear_data) + " to " + str(finalyear_data) + ' ' +
              scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_cubelist_ratio_maps(cubelist, filename, scenario, modelname, quantile):
    for i_cube in cubelist:
        scenario_year_coord = i_cube.coord('scenario_year')
        for i_timepoint in scenario_year_coord.cells():
            constraint = iris.Constraint(scenario_year=i_timepoint)

            year = i_timepoint[0].year
            time_cube = i_cube.extract(constraint)
            start_year = year - 4
            final_year = year + 5

            projection = cartopy.crs.AzimuthalEquidistant(central_latitude=90)
            plot_ratio_cube(time_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                            reference_start_year, reference_final_year,
                            projection,
                            time_cube.var_name)
            plt.savefig(
                filename + "_" + scenario + "_" + str(start_year) + "_" + str(final_year) + "_" + str(
                    time_cube.var_name) + '_' + str(quantile) + '.png',
                dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()


def plot_cubelist_diff_maps(cubelist, filename, scenario, modelname, quantile):
    for i_cube in cubelist:
        scenario_year_coord = i_cube.coord('scenario_year')
        for i_timepoint in scenario_year_coord.cells():
            constraint = iris.Constraint(scenario_year=i_timepoint)

            year = i_timepoint[0].year
            time_cube = i_cube.extract(constraint)
            start_year = year - 4
            final_year = year + 5

            projection = cartopy.crs.AzimuthalEquidistant(central_latitude=90)
            diff_freq_key = 'diff_frequency'
            diff_es_key = 'diff_es'
            diff_rel_es_key = 'diff_es_baseline'
            diff_mean_key = 'diff_mean'
            if (time_cube.var_name == diff_es_key):
                vmax = 150
            if (time_cube.var_name == diff_freq_key):
                if (quantile == 99):

                    vmax = 90
                else:
                    vmax = 30

            if (time_cube.var_name == diff_rel_es_key):
                vmax = 250
            if (time_cube.var_name == diff_mean_key):
                vmax = 20

            vmin = -vmax

            plot_difference_cube(time_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                                 reference_start_year,
                                 reference_final_year, projection,
                                 vmin, vmax, time_cube.var_name)
            plt.savefig(
                filename + "_" + scenario + "_" + str(start_year) + "_" + str(final_year) + "_" + str(
                    time_cube.var_name) + '_' + str(quantile) + '.png',
                dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()


def plot_contribution_maps(cubelist, filename, scenario, modelname, quantile):
    for i_cube in cubelist:
        scenario_year_coord = i_cube.coord('scenario_year')
        for i_timepoint in scenario_year_coord.cells():
            constraint = iris.Constraint(scenario_year=i_timepoint)

            year = i_timepoint[0].year
            time_cube = i_cube.extract(constraint)
            start_year = year - 4
            final_year = year + 5

            projection = cartopy.crs.AzimuthalEquidistant(central_latitude=90)
            vmax = 1

            vmin = 0
            plot_ratio_cube(time_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                            reference_start_year,
                            reference_final_year, projection,
                            time_cube.var_name, vmin=vmin, vmax=vmax)
            plt.savefig(
                filename + "_" + scenario + "_" + str(start_year) + "_" + str(final_year) + "_" + str(
                    time_cube.var_name) + '_' + str(quantile) + '.png',
                dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()


# plotting of time series of cube data
def plot_cubelist_average_single(cubelist, filename):
    i = 0
    if (cubelist[0].coord('latitude').has_bounds() != True):
        cubelist[0].coord('latitude').guess_bounds()
        cubelist[0].coord('longitude').guess_bounds()
    area_weights = iris.analysis.cartography.area_weights(cubelist[0])
    for i_cube in cubelist:
        average = i_cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN, weights=area_weights)

        qplt.plot(average)
        filename_individual = filename + "_" + average.var_name
        plt.savefig(filename_individual + str(i) + '.png', dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()
        i = i + 1


def plot_cubelist_average_single_scenario(ylims, cubelist, filename, var_names, modelname, areaname, scenario,
                                          datapoint,
                                          population=False, temperature=False):
    plot_cubelist_average(ylims, cubelist, var_names, population=population)

    title_str = (modelname + " percentile: " + str(
        datapoint) + '\n' + scenario + " for " + areaname + ' ratios to baseline: ' + str(
        reference_start_year) + ' to ' + str(reference_final_year))
    if temperature:
        title_str = modelname + " snow between: " + str(
            datapoint) + "K" + '\n' + scenario + " for " + areaname + ' ratios to baseline: ' + str(
            reference_start_year) + ' to ' + str(reference_final_year)

    plt.title(title_str)

    plt.savefig(filename + '_' + scenario + '_' + str(datapoint) + '.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_weights(cube, population=False):
    if (cube.coord('latitude').has_bounds() != True):
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
    number_of_timesteps = cube.shape[0]
    print(cube.shape)
    print(number_of_timesteps)
    weights = iris.analysis.cartography.area_weights(cube)
    print(weights.shape)
    if population:
        total_population = population.collapsed(('latitude', 'longitude'), iris.analysis.MEAN).data
        percentage_population = population / total_population * 100
        weights = np.tile(percentage_population.data, (number_of_timesteps, 1, 1))
    return weights


def plot_cubelist_average(cubelist, var_names, color,
                          population=False, label="label", fmt="x"):
    for i_cube in cubelist:
        if i_cube.var_name in var_names:
            weights = generate_weights(i_cube, population=population)
            average = i_cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN, weights=weights)
            # compute change of ratio compared to baseline ==1
            one_cube = average.copy(data=np.ones(average.data.shape))
            change_value = average - one_cube
            if 'models' in label:
                iplt.plot(change_value, fmt, lw='0.35', label=label, color=color, linestyle='solid')
            else:
                iplt.plot(change_value, fmt, lw='0.35', label=label, color=color)
            plt.ylabel("Change in " + i_cube.var_name + " (% baseline)")
            plt.xlabel('Year')


# methods to create figures with data from multiple scenarios
def add_plot_summary_statistics(axes, ylims):
    # create list with all y datapoints:
    minmaxcolor = 'steelblue'
    y_data = []
    for i_line in axes.lines:
        if not ('models' in i_line.get_label()):
            print(i_line.get_data()[1])
            y_data.append(i_line.get_data()[1])
    x_data = axes.lines[0].get_data()[0]
    # crude way to extraxt datetime from calendar datetime object:
    x_datetime = []
    for i_data in x_data:
        x_datetime.append(i_data.datetime)
    dataframe = pd.DataFrame(y_data, columns=x_datetime)
    print(dataframe)
    # add shaded area between a lower and upper bound, e.g. min / max TODO: extend to percentiles etc.
    lower_bound = dataframe.quantile(0.167)
    upper_bound = dataframe.quantile(0.833)
    x = x_data
    plt.fill_between(x, lower_bound, upper_bound, color=minmaxcolor, alpha=0.5)
    # add plot of the mean

    mean = dataframe.mean().to_numpy()
    median = dataframe.median().to_numpy()

    plt.plot(x, median, color='navy', marker='o', linestyle='solid', lw=1, label="ensemble_median")
    # axes.set_ylim(ylims)
    ylim_max = axes.get_ylim()[1]
    ylim_min = ylim_max - 0.5
    axes.set_ylim([ylim_min, ylim_max])

    y_tick_spacing = 0.05
    axes.yaxis.set_major_locator(mticker.MultipleLocator(y_tick_spacing))
    plt.axhline(y=0, ls='dotted', lw=0.25, c='k')


def plot_variable_multiple_scenarios(ylims, cubelist_dict_by_scenario, var_name, scenarios, scenario_colors,
                                     population=False):
    for i_scenario in scenarios:
        scenario_color = scenario_colors[i_scenario]
        plot_cubelist_average(cubelist_dict_by_scenario[i_scenario], var_name, scenario_color, population=population,
                              label=i_scenario)


def plot_multiple_variables_multiple_scenario(ylims, cubelist_dict_by_scenario, filename, variablenames, modelname,
                                              areaname, scenarios, scenario_colors, datapoint,
                                              population=False, temperature=False):
    number_of_variables = len(variablenames)
    number_of_columns = 1
    number_of_rows = int(number_of_variables / number_of_columns)

    fig = plt.figure(figsize=(12, 16))

    ax = []
    ax.append(fig.add_subplot(number_of_rows, number_of_columns, 1))
    plot_variable_multiple_scenarios(ylims, cubelist_dict_by_scenario, variablenames[0], scenarios, scenario_colors,
                                     population=population)
    add_plot_summary_statistics(plt.gca(), ylims)

    position = range(1, number_of_variables + 1)

    for k in range(1, number_of_variables):
        ax.append(fig.add_subplot(number_of_rows, number_of_columns, position[k], sharex=ax[k - 1]))
        plot_variable_multiple_scenarios(ylims, cubelist_dict_by_scenario, variablenames[k], scenarios, scenario_colors,
                                         population=population)
        add_plot_summary_statistics(plt.gca(), ylims)

    title_str = (modelname + " percentile: " + str(
        datapoint) + " " + areaname)
    if temperature:
        title_str = modelname + " snow between: " + str(
            datapoint) + "K" + areaname + ' ratios to baseline: ' + str(
            reference_start_year) + ' to ' + str(reference_final_year)

    axs = fig.axes
    plt.suptitle(title_str, y=0.98)
    for n, ax in enumerate(axs):
        ax.text(-0.1, 1.1, string.ascii_lowercase[n], transform=ax.transAxes,
                size=10, weight='bold')
    # get rid of duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    years = mdates.YearLocator(5, 2, 1)
    # ax.xaxis.set_major_locator(years)
    fig.autofmt_xdate()
    plt.figtext(.02, .02, "shaded area: 16.7 to 83.3 percentile of single models")
    plt.tight_layout()

    plt.savefig(filename + '_' + "multi_scenario" + '_' + str(datapoint) + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def import_unify(file, cubelists, data_to_plot, maskpath):
    cubelists[file] = concatenate_cube_dict(data_to_plot[file], maskpath)
    extended_list = iris.cube.CubeList()
    for i_cube in cubelists[file]:
        coord = i_cube.coord('scenario_year')
        coord.convert_units(cf_units.Unit('days since 1850-01-01', calendar='proleptic_gregorian'))
        i_cube = iris.util.new_axis(i_cube, scalar_coord=coord)
        extended_list.append(i_cube)
    extended_list = extended_list.concatenate()

    unify_time_units(extended_list)
    return extended_list


def plot_development_multiple_scenarios(ylims, modelname, filelist_dict, arealist, areanames, scenarios, datapoint
                                        , temperature=False, maps=False):
    label_frequency = 'days with daily snowfall > baseline'
    label_es = 'expected excess snowfall > baseline (mm)'
    label_es_diff = 'difference of ' + label_es
    scenario_data = {}

    for i_scenario in scenarios:
        data_to_plot = {}
        for i_file in filelist_dict[i_scenario]:
            with open(i_file, 'rb') as stream:
                data_to_plot[i_file] = (pickle.load(stream))
        scenario_data[i_scenario] = data_to_plot

    frequency_ratio_key = 'frequency'
    quantile_ratio_key = 'percentile'
    es_ratio_key = 'EES'
    mean_ratio_key = 'mean'

    if temperature:
        days_between_temperature_key = 'days_between_temperature'
        mean_snow_between_temperature_key = 'mean_snow_between_temperature'
        mean_pr_ratio_key = 'mean_precipitation'
        percentile_pr_ratio_key = 'pr_percentile_99.9'
        snow_between_percentile_key = 'snow_between_' + str(np.min(datapoint)) + '_' + str(
            np.max(datapoint)) + 'K_percentile_99.9'

    # prepare time series of interesting phenomenoms by concatening all cubes:
    scenario_cubelists = {}
    cubelists = {}

    for i_scenario in scenarios:
        all_files = iris.cube.CubeList()
        results_cache = []
        for i_file in filelist_dict[i_scenario]:
            results_cache.append(import_unify(i_file, cubelists, scenario_data[i_scenario], maskfiles[i_scenario]))

        for i_extended_list in results_cache:
            for i_cube in i_extended_list:
                all_files.append(i_cube)

        all_files_concatenated = all_files.concatenate()
        scenario_cubelists[i_scenario] = all_files_concatenated

    population_cubes = {}

    area_cubes = {}
    i = 0
    for i_area in arealist:
        scenario_area_data = {}
        for i_scenario in scenarios:
            scenario_area_data[i_scenario] = iris.cube.CubeList()
            for i_cube in scenario_cubelists[i_scenario]:
                area_cube = cube_from_bounding_box(i_cube, i_area)
                population_cube = cube_from_bounding_box(current_population, i_area)
                scenario_area_data[i_scenario].append(area_cube)
                population_cubes[areanames[i]] = (population_cube)
        area_cubes[areanames[i]] = scenario_area_data
        i = i + 1
    if temperature:
        ratios = [days_between_temperature_key, mean_snow_between_temperature_key,
                  percentile_pr_ratio_key, snow_between_percentile_key]
    else:
        ratios = [mean_ratio_key, quantile_ratio_key, es_ratio_key]

    for i_area in areanames:
        # test population weighting
        print(i_area)
        if populationweighting:
            plot_multiple_variables_multiple_scenario(ylims, area_cubes[i_area], str(i_area + '_population_weighted'),
                                                      ratios,
                                                      modelname + "_population_weighted",
                                                      i_area, scenarios, scenario_colors, datapoint,
                                                      population=population_cubes[i_area],
                                                      temperature=temperature)
            print(i_area + "population finished")
        plot_multiple_variables_multiple_scenario(ylims, area_cubes[i_area], str(i_area + '_ratios'), ratios, modelname,
                                                  i_area, scenarios, scenario_colors, datapoint,
                                                  temperature=temperature)

    if maps:
        for i_scenario in scenarios:
            plot_cubelist_ratio_maps(scenario_cubelists[i_scenario][4:8], 'NORTHERN_HEMISPHERE', i_scenario, modelname,
                                     datapoint)


def plot_development_single_scenario(ylims, modelname, filelist, arealist, areanames, scenario, datapoint
                                     , maps=True, temperature=False):
    label_frequency = 'days with daily snowfall > baseline'
    label_es = 'expected excess snowfall > baseline (mm)'
    label_es_diff = 'difference of ' + label_es

    data_to_plot = {}
    for i_file in filelist:
        with open(i_file, 'rb') as stream:
            data_to_plot[i_file] = (pickle.load(stream))
    # TODO?!: cleaner way to extract quantile / tempearature which is plotted
    frequency_ratio_key = 'frequency'
    quantile_ratio_key = 'percentile'
    es_ratio_key = 'EES'
    mean_ratio_key = 'mean'

    if temperature:
        days_between_temperature_key = 'days_between_temperature'
        mean_snow_between_temperature_key = 'mean_snow_between_temperature'
        mean_pr_ratio_key = 'mean_precipitation'
        percentile_pr_ratio_key = 'pr_percentile_99.9'
        snow_between_percentile_key = 'snow_between_' + str(np.min(datapoint)) + '_' + str(
            np.max(datapoint)) + 'K_percentile_99.9'

    # prepare time series of interesting phenomenoms by concatening all cubes:
    cubelists = {}
    all_files = iris.cube.CubeList()

    results_cache = []
    for i_file in filelist:
        results_cache.append(import_unify(i_file, cubelists, data_to_plot))

    for i_extended_list in results_cache:
        for i_cube in i_extended_list:
            all_files.append(i_cube)

    all_files_concatenated = all_files.concatenate()
    print(all_files_concatenated)

    area_cubes = {}
    population_cubes = {}
    i = 0
    for i_area in arealist:
        area_cubes[areanames[i]] = iris.cube.CubeList()
        for i_cube in all_files_concatenated:
            area_cube = cube_from_bounding_box(i_cube, i_area)
            population_cube = cube_from_bounding_box(current_population, i_area)
            area_cubes[areanames[i]].append(area_cube)
            population_cubes[areanames[i]] = (population_cube)
        i = i + 1

    if temperature:
        ratios = [days_between_temperature_key, mean_snow_between_temperature_key,
                  percentile_pr_ratio_key, snow_between_percentile_key]
    else:
        ratios = [frequency_ratio_key, mean_ratio_key, es_ratio_key, quantile_ratio_key]

    for i_area in areanames:
        # test population weighting
        print(i_area)
        if populationweighting:
            plot_cubelist_average_single_scenario(ylims, area_cubes[i_area], str(i_area + '_population_weighted'),
                                                  ratios,
                                                  modelname + "_population_weighted",
                                                  i_area, scenario, datapoint, population=population_cubes[i_area],
                                                  temperature=temperature)
            print(i_area + "population finished")
        plot_cubelist_average_single_scenario(ylims, (area_cubes[i_area]), str(i_area + '_ratios'), ratios, modelname,
                                              i_area, scenario, datapoint, temperature=temperature)

    # TODO: adjust for temperature data

    if maps:
        plot_cubelist_ratio_maps(all_files_concatenated[0:4], 'full_NH_ratios', scenario, modelname, datapoint)

    model_contribution = False
    if model_contribution:
        plot_contribution_maps(all_files_concatenated[8:28], 'model_contributions', scenario, modelname, datapoint)


# population weighting module

def population_weighting(cubelist, population_data):
    weighted_cubelist = iris.cube.CubeList()
    for iterator_cube in cubelist:
        # adjust grid
        population_data_regridded = population_data.regrid(iterator_cube, iris.analysis.Linear(
            extrapolation_mode='extrapolate'))
        scenario_year = iterator_cube.coord('scenario_year')
        iterator_cube.remove_coord('scenario_year')

        # # intersect cubes
        # intersection = iris.analysis.maths.intersection_of_cubes(iterator_cube, population_data_regridded)
        # weight cube
        weighted_cube = iterator_cube * population_data_regridded
        iterator_cube.add_dim_coord(scenario_year, 0)
        weighted_cube.add_dim_coord(scenario_year, 0)
        weighted_cubelist.append(weighted_cube)
        weighted_cube.var_name = iterator_cube.var_name + '_population_weighted'

    return weighted_cubelist


def plot_areaboxes(arealist, testcube):
    # TODO: figure out a way to transform to a
    coordinate_projection = cartopy.crs.PlateCarree()
    projection = cartopy.crs.AzimuthalEquidistant(central_latitude=90)
    ax = plt.axes(projection=projection)

    uniform_grey_start = plt.cm.gray(np.linspace(0, 0.5, 1))
    uniform_grey_end = plt.cm.gray(np.linspace(0.5, 1, 1))

    all_grey = np.vstack((uniform_grey_start, uniform_grey_end))

    all_grey[0] = (0.5, 0.5, 0.5, 1)
    all_grey[1] = (0.5, 0.5, 0.5, 1)

    grey_cmap = clr.LinearSegmentedColormap.from_list('grey_cmap',
                                                      all_grey)

    iris.plot.pcolormesh(testcube, cmap=grey_cmap)

    num_colors = len(arealist)
    cm = plt.get_cmap('tab10')
    i = 0

    for i_area in arealist:
        # TODO: fix this very shaky calculation in a more elegant, general way
        max_lon = i_area[3]
        min_lon = i_area[1]

        max_lat = np.max((i_area[0], i_area[2]))
        min_lat = np.min((i_area[0], i_area[2]))

        height = max_lat - min_lat
        if (max_lon + 180 > min_lon + 180):
            width = abs(max_lon - min_lon)
        else:
            width = 360 - (abs(max_lon) + abs(min_lon))
        xy = [min_lon, min_lat]

        ax.add_patch(mpatches.Rectangle(xy=xy, width=width, height=height,
                                        facecolor=(cm(i / num_colors)),
                                        alpha=0.5,
                                        transform=coordinate_projection)
                     )
        ax.text(xy[0] + 10, xy[1] + 10, str(i + 1), fontsize=6,
                verticalalignment='top', color='red', transform=coordinate_projection)
        i = i + 1
    ax.gridlines()
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=plt.gca().transAxes)
    ax.coastlines()

    plt.savefig('region_map.png', dpi=150, bbox_inches='tight')
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

scenarios = settings['scenarios']
outputdir = settings['outputdir']
arealist = settings['arealist']
areanames = settings['areanames']
datapoints = settings['datapoints']
modelname = settings['modelname']
ylims = settings['ylims']
maskpath = settings['maskpath']
populationfile = settings["population"]
temperature = bool(settings['temperature'])
single_scenario = bool(settings['single_scenario'])
multi_model = settings['multi_model']
individual_mask = bool(settings['individual_mask'])
universal_mask = bool(settings['universal_mask'])
maskfiles = {}
populationweighting = bool(settings['population_weighting'])
with open(settings['scenario_colors'], 'r') as stream:
    try:
        scenario_colors = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

population = iris.load_cube(populationfile)
# extract year 2020, TODO: better way to constraint on year? Problem: bad format of population file
current_population = (population.extract(iris.Constraint(raster=5)))

current_population.remove_coord('raster')
scaling_factor = 100000
current_population.units = scaling_factor
current_population = current_population / scaling_factor
plot_areaboxes(arealist, cube_from_bounding_box(current_population, [30, -180, 90, 180]))

if (single_scenario):
    for i_data in datapoints:
        date = datetime.now()
        dirname = "data_" + str(i_data)
        os.mkdir(dirname)
        os.chdir(dirname)
        filelists = {}
        for i_scenario in scenarios:
            os.mkdir(i_scenario)
            os.chdir(i_scenario)
            if (temperature):
                i_data = tuple(i_data)
            filelist = scenarios[i_scenario][(i_data)]
            timeperiod_length = 10
            reference_start_year = 1851
            reference_final_year = 1920

            plot_development_single_scenario(ylims, modelname, filelist, arealist, areanames, i_scenario, (i_data),
                                             maps=False, temperature=temperature)
            os.chdir("..")
        os.chdir("..")
if multi_model:
    modellist = settings['modellist']
    for i_data in datapoints:
        date = datetime.now()
        dirname = "data_" + str(i_data)
        os.mkdir(dirname)
        os.chdir(dirname)
        filelists = {}
        for i_model in modellist:
            if universal_mask:
                maskfiles[i_model] = settings['maskpath']
            else:
                maskfiles[i_model] = settings['maskidentifier'] + i_model
            if (temperature):
                i_data = tuple(i_data)
            filelists[i_model] = scenarios[i_model][(i_data)]
            timeperiod_length = 10
            reference_start_year = 1851
            reference_final_year = 1920
        plot_development_multiple_scenarios(ylims, modelname, filelists, arealist, areanames, modellist, (i_data),
                                            temperature=temperature, maps=False)
        os.chdir("..")
else:
    for i_data in datapoints:
        date = datetime.now()
        dirname = "data_" + str(i_data)
        os.mkdir(dirname)
        os.chdir(dirname)
        filelists = {}
        for i_scenario in scenarios:
            print(i_scenario)
            if (temperature):
                i_data = tuple(i_data)
            filelists[i_scenario] = scenarios[i_scenario][(i_data)]
            timeperiod_length = 10
            reference_start_year = 1851
            reference_final_year = 1920

        plot_development_multiple_scenarios(ylims, modelname, filelists, arealist, areanames, scenarios, (i_data),
                                            temperature=temperature)
        os.chdir("..")
