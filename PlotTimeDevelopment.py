#!/usr/bin/env python


import argparse
import os
# Imports
import pickle
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
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
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
            this_lon = this_cube.coord('longitude').points[0]
            this_point = Point(this_lon, this_lat)
            mask[i] = this_point.within(i_geom)

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

    file = open(maskfile, 'rb')
    cube_mask = pickle.load(file)

    cubelist = iris.cube.CubeList([iris.util.mask_cube(initial_cube, cube_mask)])
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
        startyear_data) + " to " + str(finalyear_data) +
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


def plot_cubelist_average_group(ylims, cubelist, filename, var_names, modelname, areaname, scenario, datapoint,
                                population=False, temperature=False):
    if (cubelist[0].coord('latitude').has_bounds() != True):
        cubelist[0].coord('latitude').guess_bounds()
        cubelist[0].coord('longitude').guess_bounds()
    weights = iris.analysis.cartography.area_weights(cubelist[0])
    if (population != False):
        total_population = population.collapsed(('latitude', 'longitude'), iris.analysis.MEAN).data
        percentage_population = population / total_population * 100
        number_of_timesteps = cubelist[0].shape[0]

        weights = np.tile(percentage_population.data, (number_of_timesteps, 1, 1))

    for i_cube in cubelist:
        plt.ylim(ylims)
        if i_cube.var_name in var_names:
            print(i_cube.data)
            average = i_cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN, weights=weights)
            iplt.plot(average, linestyle='solid', lw='0.35', label=i_cube.var_name, )
    plt.title(modelname + " percentile: " + str(
        datapoint) + '\n' + scenario + " for " + areaname + ' ratios to baseline: ' + str(
        reference_start_year) + ' to ' + str(reference_final_year))
    if temperature:
        plt.title(modelname + " snow below: " + str(
            datapoint) + "K" + '\n' + scenario + " for " + areaname + ' ratios to baseline: ' + str(
            reference_start_year) + ' to ' + str(reference_final_year))

    # start_date = datetime(1855,1,1,0,0,0,0)
    # end_date = datetime(1925, 1, 1, 0, 0, 0, 0)
    # delta = timedelta(days=3653)
    # plt.xticks(mdates.drange(start_date, end_date, delta))
    plt.axhline(y=1, ls='dotted', lw=0.25, c='k')
    plt.legend()
    plt.savefig(filename + '_' + scenario + '_' + str(datapoint) + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def import_unify(file, cubelists, data_to_plot):
    # # define ocean mask
    # ocean_shapefile = '/p/tmp/quante/ExtremeEventEvaluation/ne_110m_ocean/ne_110m_ocean.shp'
    #
    # reader = cartopy.io.shapereader.Reader(ocean_shapefile)
    # oceans = reader.geometries()

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


def plot_data_development(ylims, modelname, filelist, arealist, areanames, scenario
                          , maps=True, temperature=False):
    label_frequency = 'days with daily snowfall > baseline'
    label_es = 'expected excess snowfall > baseline (mm)'
    label_es_diff = 'difference of ' + label_es

    data_to_plot = {}
    for i_file in filelist:
        with open(i_file, 'rb') as stream:
            data_to_plot[i_file] = (pickle.load(stream))
    # TODO?!: cleaner way to extract quantile / tempearature which is plotted
    datapoint = (list(data_to_plot[filelist[0]].keys())[1][0])

    frequency_ratio_key = 'frequency_ratio'
    quantile_ratio_key = 'percentile_ratio'
    es_ratio_key = 'EES_ratio'
    mean_ratio_key = 'mean_ratio'

    if temperature:
        days_below_temperature_key = 'days_below_temperature_ratio'
        mean_snow_below_temperature_key = 'mean_snow_below_temperature_ratio'
        mean_pr_ratio_key = 'mean_precipitation_ratio'
        percentile_pr_ratio_key = 'pr_percentile_99.9_ratio'
        snow_below_percentile_key = 'snow_below_temp_percentile_99.9_ratio'
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
        ratios = [days_below_temperature_key, mean_snow_below_temperature_key, mean_pr_ratio_key,
                  percentile_pr_ratio_key, snow_below_percentile_key]
    else:
        ratios = [frequency_ratio_key, mean_ratio_key, es_ratio_key, quantile_ratio_key]

    for i_area in areanames:
        # test population weighting
        plot_cubelist_average_group(ylims, area_cubes[i_area], str(i_area + '_population_weighted'), ratios,
                                    modelname + "_population_weighted",
                                    i_area, scenario, datapoint, population=population_cubes[i_area],
                                    temperature=temperature)

        plot_cubelist_average_group(ylims, (area_cubes[i_area]), str(i_area + '_ratios'), ratios, modelname,
                                    i_area, scenario, datapoint, temperature=temperature)

    # TODO: adjust for temperature data
    if maps:
        plot_cubelist_ratio_maps(all_files_concatenated[4:8], 'full_NH_ratios', scenario, modelname, datapoint)
        plot_cubelist_diff_maps(all_files_concatenated[0:4], 'full_NH_differences', scenario, modelname, datapoint)
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
temperature = settings['temperature']
if temperature == 1:
    temperature = True
else:
    temperature = False

population = iris.load_cube(populationfile)
# extract year 2020, TODO: better way to constraint on year? Problem: bad format of population file
current_population = (population.extract(iris.Constraint(raster=5)))

current_population.remove_coord('raster')
scaling_factor = 100000
current_population.units = scaling_factor
current_population = current_population / scaling_factor

# set outputpath for plots etc
for i_data in datapoints:
    date = datetime.now()
    dirname = str(date) + "_data_" + str(i_data)
    os.mkdir(dirname)
    os.chdir(dirname)
    for i_scenario in scenarios:
        os.mkdir(i_scenario)
        os.chdir(i_scenario)
        filelist = scenarios[i_scenario][i_data]
        timeperiod_length = 10
        reference_start_year = 1851
        reference_final_year = 1920
        plot_data_development(ylims, modelname, filelist, arealist, areanames, i_scenario,
                              maps=False, temperature=temperature)
        os.chdir("..")
    os.chdir("..")
