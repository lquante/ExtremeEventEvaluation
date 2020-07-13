#!/usr/bin/env python


import argparse
import os
# Imports
import pickle
import re
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

    latitude_constrained = cube.intersection(latitude=(latitude_min, latitude_max))
    return latitude_constrained.intersection(longitude=(longitude_min, longitude_max))


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
    # # https: // scitools.org.uk / cartopy / docs / latest / gallery / always_circular_stereo.html?highlight = circular
    # # Compute a circle in axes coordinates, which we can use as a boundary
    # # for the map. We can pan/zoom as much as we like - the boundary will be
    # # permanently circular.
    # theta = np.linspace(0, 2 * np.pi, 100)
    # center, radius = [0.5, 0.5], 0.5
    # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # circle = mpath.Path(verts * radius + center)
    # ax.set_boundary(circle, transform=plt.gca().transAxes)

    plt.colorbar(pcm, shrink=0.7, extend='both', orientation='horizontal') # label + title to be added in tex

    # plt.title(
    #     model + " percentile: " + str(quantile) + "\n" + scenario_data + ": " + str(startyear_data) + " to " + str(
    #         finalyear_data) +
    #     scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_ratio_cube(cube, scenario_data, modelname, quantile, startyear_data, finalyear_data, scenario_comparison,
                    startyear_comparison,
                    finalyear_comparison, projection, label, vmin=0, vmax=2,contour=False):
    colormap = plt.get_cmap('RdBu_r', 30)
    ax = plt.axes(projection=projection)
    if contour:
        levels = range(-30,35,5)
        pcm = iris.plot.contourf(cube,levels=levels,cmap=colormap,extend='both')
    else:
        pcm = iris.plot.pcolormesh(cube, cmap=colormap, vmin=vmin, vmax=vmax)

    ax.coastlines('110m')
    ax.gridlines()

    # https: // scitools.org.uk / cartopy / docs / latest / gallery / always_circular_stereo.html?highlight = circular
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    # theta = np.linspace(0, 2 * np.pi, 100)
    # center, radius = [0.5, 0.5], 0.5
    # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # circle = mpath.Path(verts * radius + center)
    # ax.set_boundary(circle, transform=plt.gca().transAxes)

    plt.colorbar(pcm, shrink=0.7, extend='both', orientation='horizontal')# label + title to be added in tex

    # plt.title(modelname + " percentile: " + str(quantile) + "\n" + " " + scenario_data + ": " + str(
    #     startyear_data) + " to " + str(finalyear_data) + ' ' +
    #           scenario_comparison + ": " + str(startyear_comparison) + " to " + str(finalyear_comparison), fontsize=10)


def plot_cubelist_ratio_maps(cubelist,varnames, filename, scenario, modelname, quantile,mask=None):
    for i_cube in cubelist:
        scenario_year_coord = i_cube.coord('scenario_year')
        if i_cube.var_name in varnames:
            for i_timepoint in scenario_year_coord.cells():
                constraint = iris.Constraint(scenario_year=i_timepoint)

                year = i_timepoint[0].year
                time_cube = i_cube.extract(constraint)
                start_year = year - 4
                final_year = year + 5

                if mask is None:
                    z = 1 + 1
                else:
                    time_cube = iris.util.mask_cube(time_cube, mask)

                plot_ratio_cube(time_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                                reference_start_year, reference_final_year,
                                projection,
                                time_cube.var_name,vmin=-100,vmax=100)
                plt.savefig(
                    filename + "_" + scenario + "_" + str(start_year) + "_" + str(final_year) + "_" + str(
                        time_cube.var_name) + '_' + str(quantile) + '.png',
                    dpi=300, bbox_inches='tight')
                plt.close()

def plot_min_max_maps(cubelist,varnames, filename, scenario, modelname, quantile,mask=None):
    for i_cube in cubelist:
        if i_cube.var_name in varnames:
            max_cube = i_cube.collapsed ('scenario_year',iris.analysis.MAX)
            min_cube = i_cube.collapsed ('scenario_year',iris.analysis.MIN)
            diff_cube = (max_cube-min_cube)*100
            diff_cube.var_name = i_cube.var_name+"_max-min"

            # mask with mask data
            if mask is None:
                z = 1 + 1
            else:
                diff_cube = iris.util.mask_cube(diff_cube, mask)


            start_year = 2021
            final_year = 2100
            plot_ratio_cube(diff_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                            reference_start_year, reference_final_year,
                            projection,
                            diff_cube.var_name,vmin=-100,vmax=100)
            plt.savefig(
                filename + "_" + scenario + "_" + str(start_year) + "_" + str(final_year) + "_" + str(
                    diff_cube.var_name) + '_' + str(quantile) + '.png',
                dpi=300, bbox_inches='tight')
            plt.close()

def plot_start_relative_maps(cubelist,varnames, filename, scenario, modelname, quantile,mask=None):
    for i_cube in cubelist:
        scenario_year_coord = i_cube.coord('scenario_year')
        if i_cube.var_name in varnames:
            first = last = next(scenario_year_coord.cells(), 2020)
            first_timepoint = iris.Constraint(scenario_year=first)
            start_year = first[0].year
            start_cube = i_cube.extract(first_timepoint)
            for last in scenario_year_coord.cells():
                last_timepoint = iris.Constraint (scenario_year=last)
                final_cube = i_cube.extract(last_timepoint)
                final_year = last[0].year
                diff_cube = (final_cube-start_cube)*100
                diff_cube.var_name = i_cube.var_name + "_linear_trend_"+str(final_year)+"-"+str(start_year)

                #mask with mask data
                if mask is None:
                    z=1+1
                else:
                    diff_cube = iris.util.mask_cube(diff_cube,mask)





                if "EES" in i_cube.var_name:
                    vmax = 25
                else:
                    vmax = 100

                plot_ratio_cube(diff_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                                reference_start_year, reference_final_year,
                                projection,
                                diff_cube.var_name,vmin=-vmax,vmax=vmax)
                plt.savefig(
                    filename + "_" + scenario + "_" + str(
                        diff_cube.var_name) + '_' + str(quantile) + '.png',
                    dpi=300, bbox_inches='tight')
                plt.close()

def plot_contour_start_relative_maps(cubelist,varnames, filename, scenario, modelname, quantile,mask=None):
    for i_cube in cubelist:
        scenario_year_coord = i_cube.coord('scenario_year')
        if i_cube.var_name in varnames:
            first = last = next(scenario_year_coord.cells(), 2020)
            first_timepoint = iris.Constraint(scenario_year=first)
            start_year = first[0].year
            start_cube = i_cube.extract(first_timepoint)
            for last in scenario_year_coord.cells():
                last_timepoint = iris.Constraint (scenario_year=last)
                final_cube = i_cube.extract(last_timepoint)
                final_year = last[0].year
                diff_cube = (final_cube-start_cube)*100
                diff_cube.var_name = i_cube.var_name + "_linear_trend_"+str(final_year)+"-"+str(start_year)

                #mask with mask data
                if mask is None:
                    z=1+1
                else:
                    diff_cube = iris.util.mask_cube(diff_cube,mask)





                if "EES" in i_cube.var_name:
                    vmax = 25
                else:
                    vmax = 100

                plot_ratio_cube(diff_cube, scenario, modelname, quantile, start_year, final_year, 'historical',
                                reference_start_year, reference_final_year,
                                projection,
                                diff_cube.var_name,vmin=-vmax,vmax=vmax,contour=True)
                plt.savefig(
                    filename + "_" + scenario + "_" + str(
                        diff_cube.var_name) + '_' + str(quantile) + '.png',
                    dpi=300, bbox_inches='tight')
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

    #plt.title(title_str)

    plt.savefig(filename + '_' + scenario + '_' + str(datapoint) + '.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_weights(cube, population=False):
    if (cube.coord('latitude').has_bounds() != True):
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
    number_of_timesteps = cube.shape[0]
    weights = iris.analysis.cartography.area_weights(cube)
    if population:
        total_population = population.collapsed(('latitude', 'longitude'), iris.analysis.MEAN).data
        percentage_population = population / total_population * 100
        weights = np.tile(percentage_population.data, (number_of_timesteps, 1, 1))
    return weights


def plot_cubelist_average(cubelist, var_names, color,
                          population=False, label="label", fmt="x",mask=None):
    for i_cube in cubelist:
        if i_cube.var_name in var_names:
            weights = generate_weights(i_cube, population=population)
            if mask is None:
                pi=3.14
            else:
                # set weights of masked areas to 0
                weights = weights * np.invert(mask)
            average = i_cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN, weights=weights)

            # compute change of ratio compared to baseline ==1
            one_cube = average.copy(data=np.ones(average.data.shape))
            change_value = (average - one_cube)*100 #scale to percent
            if 'models' in label:
                iplt.plot(change_value, fmt, lw='0.35', label=label, color=color, linestyle='solid')
            else:
                iplt.plot(change_value, fmt, lw='0.35', label=label, color=color)
            plt.ylabel("Change in " + re.search('(\w+)(\d+_\d+)(.*)',i_cube.var_name).group(1) + " (% baseline)")
            plt.xlabel('Year')


# methods to create figures with data from multiple scenarios
def add_plot_summary_statistics(axes, ylims):
    # create list with all y datapoints:
    minmaxcolor = 'steelblue'

    x_data = axes.lines[0].get_data()[0]
    y_data = []
    for i_line in axes.lines:
        if not ('models' in i_line.get_label()):
            y_data.append(i_line.get_data()[1])
    removal_counter = 0
    for iterator_line_number in range(0,len(axes.lines)):
            if not 'models' in axes.lines[iterator_line_number-removal_counter].get_label():
                axes.lines.pop(iterator_line_number-removal_counter)
                removal_counter +=1

    # crude way to extraxt datetime from calendar datetime object:
    x_datetime = []
    for i_data in x_data:
        x_datetime.append(i_data.datetime)
    dataframe = pd.DataFrame(y_data, columns=x_datetime)
    print(dataframe)
    # add shaded area between a lower and upper bound, e.g. min / max TODO: extend to percentiles etc.
    lower_bound = dataframe.apply(np.percentile, args=[16.7])
    upper_bound = dataframe.apply(np.percentile, args=[83.3])
    x = x_data
    plt.fill_between(x, lower_bound, upper_bound, color=minmaxcolor, alpha=0.5,linewidth=0.0)
    # add plot of the mean

    mean = dataframe.mean().to_numpy()
    median = dataframe.median().to_numpy()

    plt.plot(x, median, color='navy', marker='o', linestyle='solid', lw=1, label="ensemble_median")
    plt.draw()
    # #axes.set_ylim(ylims)
    #y_tick_spacing = 10
    # ylim_min = np.min(lower_bound)
    # if np.min(lower_bound) > 0:
    #     ylim_min = 0
    # ylim_max = ylim_min + 0.35
    # axes.set_ylim([ylim_min, ylim_max])
    #axes.yaxis.set_major_locator(mticker.MultipleLocator(y_tick_spacing))

    plt.axhline(y=0, ls='dotted', lw=0.25, c='k')


def plot_variable_multiple_scenarios(ylims, cubelist_dict_by_scenario, var_name, scenarios, scenario_colors,
                                     population=False,elevation_mask_level=None):
    for i_scenario in scenarios:
        scenario_color = scenario_colors[i_scenario]
        # generate mask of elevation of 1000
        if (elevation_mask_level is None):
            elevation_mask=None
        else:
            elevation_mask = generate_elevation_mask(elevation_cube, cubelist_dict_by_scenario[i_scenario][0], elevation_mask_level)
        plot_cubelist_average(cubelist_dict_by_scenario[i_scenario], var_name, scenario_color, population=population,
                              label=i_scenario,mask=elevation_mask)


def plot_multiple_variables_multiple_scenario(ylims, cubelist_dict_by_scenario, filename, variablenames, modelname,
                                              areaname, scenarios, scenario_colors, datapoint,
                                              population=False, temperature=False,elevation_mask_level=None):
    number_of_variables = len(variablenames)
    number_of_rows = 1
    number_of_columns = int(number_of_variables / number_of_rows)

    fig = plt.figure(figsize=(12, 4))

    ax = []
    ax.append(fig.add_subplot(number_of_rows, number_of_columns, 1))
    print(variablenames[0])
    plot_variable_multiple_scenarios(ylims, cubelist_dict_by_scenario, variablenames[0], scenarios, scenario_colors,
                                     population=population,elevation_mask_level=elevation_mask_level)
    add_plot_summary_statistics(plt.gca(), ylims)

    position = range(1, number_of_variables + 1)

    for k in range(1, number_of_variables):
        ax.append(fig.add_subplot(number_of_rows, number_of_columns, position[k], sharex=ax[k - 1]))
        print(variablenames[k])
        plot_variable_multiple_scenarios(ylims, cubelist_dict_by_scenario, variablenames[k], scenarios, scenario_colors,
                                         population=population,elevation_mask_level=elevation_mask_level)
        add_plot_summary_statistics(plt.gca(), ylims)

    title_str = (modelname + " percentile: " + str(
        datapoint) + " " + areaname)
    if temperature:
        title_str = modelname + " snow between: " + str(
            datapoint) + "K" + areaname + ' ratios to baseline: ' + str(
            reference_start_year) + ' to ' + str(reference_final_year)

    axs = fig.axes
    #plt.suptitle(title_str, y=1.05)
    # add lowercase letters to figure for labelling
    for n, ax in enumerate(axs):
        ax.text(-0.1, 1.02, string.ascii_lowercase[n], transform=ax.transAxes,
                size=11, weight='bold')
    # get rid of duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # fig.legend(by_label.values(), by_label.keys())
    years = mdates.YearLocator(10, 2, 1)
    # ax.xaxis.set_major_locator(years)
    ax.margins(0)
    plt.tight_layout()

    plt.savefig(modelname+ '_' + filename + '_' + "multi_scenario" + '_' + str(datapoint) + '.png', dpi=300, bbox_inches='tight')
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


def plot_development_multiple_scenarios(ylims, modelname, filelist_dict, arealist, areanames, scenarios, datapoints
                                        ,temperature=False, maps=False,EES_level_curves=False,country_codes=None):

        # prepare time series of interesting phenomenoms by concatening all cubes:
        scenario_cubelists = {}
        cubelists = {}
        scenario_data = {}
        scenario_cubelists_elevation_masked = {}
        print(filelist_dict)
        for i_scenario in scenarios:
            data_to_plot = {}
            for i_file in filelist_dict[i_scenario]:
                with open(i_file, 'rb') as stream:
                    data_to_plot[i_file] = (pickle.load(stream))
            scenario_data[i_scenario] = data_to_plot
        for i_scenario in scenarios:
            all_files = iris.cube.CubeList()
            results_cache = []
            for i_file in filelist_dict[i_scenario]:
                results_cache.append(import_unify(i_file, cubelists, scenario_data[i_scenario], maskfiles[i_scenario]))

            for i_extended_list in results_cache:
                for i_cube in i_extended_list:
                    all_files.append(i_cube)
            all_files_concatenated = all_files.concatenate()
            print(all_files_concatenated)
            scenario_cubelists[i_scenario] = all_files_concatenated
            # elevation_masked_cubelist = iris.cube.CubeList()
            # # generate elvation mask
            # elevation_mask = generate_elevation_mask(elevation_cube, all_files_concatenated[0], 1000)
            # number_of_timesteps = i_cube.shape[0]
            # time_scaled_mask = np.tile(elevation_mask, (number_of_timesteps, 1, 1))
            # # apply mask to all cubes from this scenario
            # for i_cube in all_files_concatenated:
            #     elevation_masked_cubelist.append(iris.util.mask_cube(i_cube,time_scaled_mask))
            # scenario_cubelists_elevation_masked[i_scenario] = elevation_masked_cubelist

        #get data for submitted country codes like extracting for areas
        country_population_cubes = {}
        country_cubes = {}
        country_cubes_elevation_masked = {}
        i = 0
        for i_country in country_codes:
            scenario_country_data = {}
            scenario_country_data_elevation_masked = {}
            for i_scenario in scenarios:
                scenario_country_data[i_scenario] = iris.cube.CubeList()
                for i_cube in scenario_cubelists[i_scenario]:
                    country_cube = filter_for_country(iso_country_cube, i_country,i_cube)
                    population_cube = filter_for_country(iso_country_cube, i_country,current_population)
                    scenario_country_data[i_scenario].append(country_cube)
                    country_population_cubes[i_country] = (population_cube)
                country_cubes[i_country] = scenario_country_data
                # scenario_area_data_elevation_masked[i_scenario]= iris.cube.CubeList()
                # for i_cube in scenario_cubelists_elevation_masked[i_scenario]:
                #     area_cube = cube_from_bounding_box(i_cube, i_area)
                #     scenario_area_data_elevation_masked[i_scenario].append(area_cube)
                # area_cubes_elevation_masked[areanames[i]] = scenario_area_data_elevation_masked
            i = i + 1


        population_cubes = {}
        area_cubes = {}
        area_cubes_elevation_masked = {}
        i = 0
        for i_area in arealist:
            scenario_area_data = {}
            scenario_area_data_elevation_masked = {}
            for i_scenario in scenarios:
                scenario_area_data[i_scenario] = iris.cube.CubeList()
                for i_cube in scenario_cubelists[i_scenario]:
                    area_cube = cube_from_bounding_box(i_cube, i_area)
                    population_cube = cube_from_bounding_box(current_population, i_area)
                    scenario_area_data[i_scenario].append(area_cube)
                    population_cubes[areanames[i]] = (population_cube)
                area_cubes[areanames[i]] = scenario_area_data
                # scenario_area_data_elevation_masked[i_scenario]= iris.cube.CubeList()
                # for i_cube in scenario_cubelists_elevation_masked[i_scenario]:
                #     area_cube = cube_from_bounding_box(i_cube, i_area)
                #     scenario_area_data_elevation_masked[i_scenario].append(area_cube)
                # area_cubes_elevation_masked[areanames[i]] = scenario_area_data_elevation_masked
            i = i + 1

        for i_datapoint in datapoints:
            frequency_ratio_key = 'frequency_' + str(i_datapoint)
            quantile_ratio_key = 'percentile_' + str(i_datapoint)
            es_ratio_key = 'EES_' + str(i_datapoint)
            mean_ratio_key = 'mean'

            if temperature:
                days_between_temperature_key = 'days_between_temperature'
                mean_snow_between_temperature_key = 'mean_snow_between_temperature'
                mean_pr_ratio_key = 'mean_precipitation'
                percentile_pr_ratio_key = 'pr_percentile_99.9'
                snow_between_percentile_key = 'snow_between_' + str(np.min(i_datapoint)) + '_' + str(
                    np.max(i_datapoint)) + 'K_percentile_99.9'

            if temperature:
                ratios = [days_between_temperature_key, mean_snow_between_temperature_key,
                          percentile_pr_ratio_key, snow_between_percentile_key]
            else:

                variable_keys = [frequency_ratio_key,mean_ratio_key, es_ratio_key, quantile_ratio_key]

                percentile_bins = [(0,100)]

            # level_bins = [(0,100)]
            # # plot summary map for levels of bins:
            # level_sample_key = 'mean'
            # levelnames = []
            # for i_bin in level_bins:
            #     levelnames.append(level_sample_key + "_" + str(i_bin[0]) + "_" + str(i_bin[1]))
            # for i_scenario in scenarios:
            #     levelcubes = {}
            #     for i_cube in scenario_cubelists[i_scenario]:
            #         if (i_cube.var_name in levelnames):
            #             levelcubes[i_cube.var_name] = i_cube
            #     print(levelcubes)
            #     plot_levels('mean_decentiles', levelcubes, i_scenario)


            for i_bin in percentile_bins:
                ratios = []
                for i_variable in variable_keys:
                        ratios.append(i_variable + "_" + str(i_bin[0]) + "_" + str(i_bin[1]))

                for i_area in areanames:

                    # test population weighting
                    print(i_area)
                    if populationweighting:
                        plot_multiple_variables_multiple_scenario(ylims, area_cubes[i_area], str(i_area + '_population_weighted'),
                                                                  ratios,
                                                                  modelname + "_population_weighted",
                                                                  i_area, scenarios, scenario_colors, i_datapoint,
                                                                  population=population_cubes[i_area],
                                                                  temperature=temperature)
                        print(i_area + "population finished")
                    plot_multiple_variables_multiple_scenario(ylims, area_cubes[i_area], str(i_area + '_ratios_<1000m_'+ str(i_bin[0]) + "_" + str(i_bin[1])), ratios, modelname,
                                                              i_area, scenarios, scenario_colors, i_datapoint,
                                                              temperature=temperature,elevation_mask_level=1000)
                    plot_multiple_variables_multiple_scenario(ylims, area_cubes[i_area], str(
                        i_area + '_ratios_' + str(i_bin[0]) + "_" + str(i_bin[1])), ratios, modelname,
                                                              i_area, scenarios, scenario_colors, i_datapoint,
                                                              temperature=temperature)

                    for i_country in country_codes:

                        # test population weighting
                        print(i_country)
                        if populationweighting:
                            plot_multiple_variables_multiple_scenario(ylims, country_cubes[i_country],
                                                                      str(i_country + '_population_weighted'),
                                                                      ratios,
                                                                      modelname + "_population_weighted",
                                                                      i_country, scenarios, scenario_colors, i_datapoint,
                                                                      population=country_population_cubes[i_country],
                                                                      temperature=temperature)
                            print(i_country + "population finished")
                        plot_multiple_variables_multiple_scenario(ylims, country_cubes[i_country], str(
                            i_country + '_ratios_<1000m_' + str(i_bin[0]) + "_" + str(i_bin[1])), ratios, modelname,
                                                                  i_country, scenarios, scenario_colors, i_datapoint,
                                                                  temperature=temperature, elevation_mask_level=1000)
                        plot_multiple_variables_multiple_scenario(ylims, country_cubes[i_country], str(
                            i_country + '_ratios_' + str(i_bin[0]) + "_" + str(i_bin[1])), ratios, modelname,
                                                                  i_country, scenarios, scenario_colors, i_datapoint,
                                                                  temperature=temperature)

                if maps:

                    for i_scenario in scenarios:
                        elevation_mask = generate_elevation_mask(elevation_cube,scenario_cubelists[i_scenario][0], 1000)
                        # min-max maps for ssp scenario
                        plot_min_max_maps(scenario_cubelists[i_scenario],ratios, 'NORTHERN_HEMISPHERE_elevation<1000m', i_scenario, modelname,
                                                 i_datapoint,mask=elevation_mask)
                        plot_min_max_maps(scenario_cubelists[i_scenario], ratios, 'NORTHERN_HEMISPHERE_elevation>=1000m',
                                          i_scenario, modelname,
                                          i_datapoint, mask=np.invert(elevation_mask))
                        # intra-scenario trend maps
                        plot_start_relative_maps(scenario_cubelists[i_scenario], ratios,
                                                 'NORTHERN_HEMISPHERE_elevation<1000m', i_scenario, modelname,
                                                 i_datapoint, mask=elevation_mask)

                        plot_start_relative_maps(scenario_cubelists[i_scenario], ratios,
                                                 'NORTHERN_HEMISPHERE_elevation>=1000m', i_scenario, modelname,
                                                 i_datapoint, mask=np.invert(elevation_mask))
                        # just baseline relative maps
                        plot_cubelist_ratio_maps(scenario_cubelists[i_scenario],ratios, 'NORTHERN_HEMISPHERE', i_scenario,
                                                 modelname,i_datapoint)
                        plot_cubelist_ratio_maps(scenario_cubelists[i_scenario], ratios, 'NORTHERN_HEMISPHERE',
                                                 i_scenario,
                                                 modelname, i_datapoint,mask=elevation_mask)

                    elevation_mask = generate_elevation_mask(elevation_cube, scenario_cubelists['isimip_ensemble_all_models'][0],
                                                                 1000)
                    plot_contour_start_relative_maps(scenario_cubelists['isimip_ensemble_all_models'], ratios,'NORTHERN_HEMISPHERE_elevation<1000m', i_scenario, modelname,
                                      i_datapoint,mask=elevation_mask)



        # optional analysis of development of different levels of ees in parallel to identify switch of sign of EES trend
        EES_level_curves = False
        if EES_level_curves:
            levels = datapoints # possible to extend by excluding specific levels
            keys = []
            percentile_bin = (25,100)
            for i_level in levels:
                es_ratio_key = 'EES_' + str(i_level) +"_" + str(percentile_bin[0])+"_"+str(percentile_bin[1])
                keys.append(es_ratio_key)
            for i_scenario in scenarios:
                ees_cubes = {}
                for i_cube in scenario_cubelists[i_scenario]:
                    if (i_cube.var_name in keys):
                        print(i_cube.var_name)
                        percentile_part = re.search('(EES_)(\d+)(.*)',i_cube.var_name,)
                        print(percentile_part)
                        print(percentile_part.group(2))
                        varname = percentile_part.group(2) # use regex to get percentile key
                        ees_cubes[varname]=i_cube # get percentile as key for dict
                print(ees_cubes)
                percentile_keys = list(ees_cubes.keys())
                timepoints = ees_cubes[percentile_keys[0]].coord('scenario_year').cells()
                for i_timepoint in timepoints:
                    year_constraint = iris.Constraint(scenario_year=i_timepoint)
                    x_data = []
                    y_data = []
                    # calculate datapoints for plotting
                    for i_key in ees_cubes.keys():
                        timecube = ees_cubes[i_key].extract(year_constraint)
                        weights = generate_weights(timecube, population=False)
                        average = timecube.collapsed(('latitude','longitude'),iris.analysis.MEAN,weights=weights).data
                        x_data.append(int(i_key))
                        y_data.append(average)
                    year = str(i_timepoint[0].year)
                    plt.plot(x_data,y_data)
                    plt.title("EES trend curve "+year)
                    plt.savefig(i_scenario+ '_' + year + '_' + "EES_trend_curve"+'.png', dpi=300, bbox_inches='tight')
                    plt.close()


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
def plot_levels(name, level_cubes, scenario):

    coordinate_projection = cartopy.crs.PlateCarree()

    ax = plt.axes(projection=projection)

    num_colors = len(level_cubes)
    cm = plt.get_cmap('coolwarm')
    uniform_color_start = plt.cm.gray(np.linspace(0, 0.5, 1))
    uniform_color_end = plt.cm.gray(np.linspace(0.5, 1, 1))

    uniform_color = np.vstack((uniform_color_start, uniform_color_end))


    i = 0
    for i_key in level_cubes.keys():
        uniform_color[0] = cm(i / num_colors)
        uniform_color[1] = uniform_color[0]
        uniform_cmap = clr.LinearSegmentedColormap.from_list('uniform_cmap',
                                                             uniform_color)
        i_cube = level_cubes[i_key]
        # constrain on 1 timepoint
        scenario_year_coord = i_cube.coord('scenario_year')
        constraint = iris.Constraint(scenario_year=scenario_year_coord.cells().next())
        time_cube = i_cube.extract(constraint)
        plot = iris.plot.contourf(time_cube,cmap=uniform_cmap,axes=ax)

        i= i+1
    ax.gridlines()
    # theta = np.linspace(0, 2 * np.pi, 100)
    # center, radius = [0.5, 0.5], 0.5
    # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # circle = mpath.Path(verts * radius + center)
    # ax.set_boundary(circle, transform=plt.gca().transAxes)
    ax.coastlines()

    plt.savefig(scenario+"_"+name+'_level_map.png', dpi=150, bbox_inches='tight')
    plt.close()

def guess_non_existing_bounds(cube):
    if (cube.coord('latitude').has_bounds() != True):
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()

def generate_elevation_mask (elevation_cube,grid_cube,height):
    guess_non_existing_bounds(grid_cube)
    elevation_cube = cube_from_bounding_box(elevation_cube, [30, -180, 90, 180])
    elevation_cube = elevation_cube.regrid(grid_cube,iris.analysis.AreaWeighted())
    return elevation_cube.data[0,:,:] > height

def filter_for_country (iso_country_cube,iso_code_index,data_cube):
    print(iso_country_cube)

    iso_code_array = iso_country_cube.data

    iso_country_cube.data = iso_code_array*(iso_code_array==iso_code_index)
    # regrid to data cube grid
    guess_non_existing_bounds(data_cube)
    iso_country_cube= iso_country_cube.regrid(data_cube,iris.analysis.Nearest(extrapolation_mode='mask'))
    iso_new_grid_data = iso_country_cube.data
    print(data_cube.data.shape)
    print(iso_new_grid_data.shape)
    iris.quickplot.contourf(iso_country_cube,levels=[0,iso_code_index-1,iso_code_index+1])
    plt.savefig(str(iso_code_index)+".png")
    plt.close()

    return iris.util.mask_cube(data_cube,(iso_new_grid_data!=iso_code_index))


def plot_areaboxes(arealist, testcube):
    # TODO: figure out a way to transform to a
    coordinate_projection = cartopy.crs.PlateCarree()

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
    # theta = np.linspace(0, 2 * np.pi, 100)
    # center, radius = [0.5, 0.5], 0.5
    # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # circle = mpath.Path(verts * radius + center)
    # ax.set_boundary(circle, transform=plt.gca().transAxes)
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

with open(settings['areas'], 'r') as stream:
    try:
        areas = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
arealist = areas['arealist']
areanames = areas['areanames']

elevation_file = settings['elevation']

elevation_cube = iris.load_cube(elevation_file)
elevation_cube.remove_coord('time')
guess_non_existing_bounds(elevation_cube)

iso_country_file = settings['iso_country']
iso_country_cube = iris.load_cube(iso_country_file)
guess_non_existing_bounds(iso_country_cube)

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
single_model_display = bool(settings['single_model_display'])
#TODO: set projection to be used globally
projection = cartopy.crs.PlateCarree()

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

# TODO: transfer to settingsfile after test

country_iso_codes = [66,61,157] #66 GER, 61 FRA, 157 ESP


if multi_model:
    modellist = settings['modellist']

    date = datetime.now()
    dirname = "data_" + str(datapoints)
    os.mkdir(dirname)
    os.chdir(dirname)
    filelists = {}
    for i_model in modellist:
        if universal_mask:
            maskfiles[i_model] = settings['maskpath']
        else:
            maskfiles[i_model] = settings['maskidentifier'] + i_model
        filelists[i_model] = scenarios[i_model]
        timeperiod_length = 10
        reference_start_year = 1851
        reference_final_year = 1920

    plot_development_multiple_scenarios(ylims, modelname, filelists, arealist, areanames, modellist, datapoints,
                                 temperature=temperature, maps=True,countrycodes=country_iso_codes)
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
