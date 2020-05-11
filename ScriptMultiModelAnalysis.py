#!/usr/bin/env python

# Imports
import argparse
import os
import pickle
import warnings

import iris
import iris.analysis
import iris.coord_categorisation
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
from ruamel.yaml import ruamel
from tqdm import tqdm


def filepath_generator(basepath, scenario, model):
    return basepath + '/' + scenario + '/output/data_' + model + '.yml'


# functions to filter for specific variables, add seasons etc

def filter_cube(cube, variablename):
    return cube.extract(variablename)


def latitude_constraint(latitude, cube):
    # restrict on latitudes above northern tropic
    latitudeConstraint = iris.Constraint(latitude=lambda v: latitude <= v)
    return cube.extract(latitudeConstraint)


def add_seasons(cubelist):
    for i_cube in cubelist:
        iris.coord_categorisation.add_season(i_cube, 'time', name='season')
        iris.coord_categorisation.add_season_year(i_cube, 'time', name='season_year')
    return cubelist


def unify_concatenate(cubelist):
    unify_time_units(cubelist)
    equalise_attributes(cubelist)

    return cubelist.concatenate_cube()


# add area bounds to enable area weighted mean:
def add_lon_lat_bounds(cube):
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()


# filter for specific season
def filter_season(cube, season):
    season_constr = iris.Constraint(season=season)
    return cube.extract(season_constr)


def prepare_season_stats(cubedict, season):
    keys = cubedict.keys()
    season_dict = {}
    for i_key in keys:
        season_cube = filter_season(cubedict[i_key], season)
        season_dict[i_key] = season_cube
    return season_dict


# generate extra cube of decade 2011 to 2020 (with ssp585 data)
def unify_current_decade(cubelist, model, scenario):
    cubes_to_concatenate = iris.cube.CubeList()
    cubes_to_concatenate.append(cubelist[model, 'historical'][17])
    cubes_to_concatenate.append(cubelist[model, scenario][0])

    cube_2011_2020 = unify_concatenate(cubes_to_concatenate)
    cubelist[model, 'historical'][17] = cube_2011_2020


def generate_quantile_exceedance_development(cubelist, baselinequantiles, quantiles, startyear, finalyear):
    results = {}
    cuberesults = {}
    number_of_cubes = len(cubelist)
    for i_key in cubelist.keys():
        # calculate some metrics per cube from cubelist
        data = {}
        cube = cubelist[i_key]
        data_timeperiod = extract_dates(cube, startyear, finalyear)
        for i_quantile in quantiles:
            number_of_timepoints = data_timeperiod.data.shape[0]
            print('timepoints for model: ' + str(i_key) + " " + str(number_of_timepoints))
            # calculate mean for comparison to quantile development
            mean_data = data_timeperiod.collapsed('time', iris.analysis.MEAN)

            mean_data.var_name = ('mean_daily_snowfall')
            # calculate exceedance of quantiles for each gridcell to use as thresholds:
            # N.B. baselinequantile cubes to be submitted as dictionary of quantile values

            quantile_baseline = baselinequantiles[i_quantile]

            exceedance_data = data_timeperiod.data - quantile_baseline.data
            exceedance = data_timeperiod.copy(data=exceedance_data)

            exceedance.var_name = ('snowfall_above_' + str(i_quantile) + 'percentile')

            # consider only positve values of exceedance
            exceedance_array = exceedance.data
            exceedance_indicators = (exceedance_array > 0)

            exceedance.data = exceedance_array * exceedance_indicators

            number_exceedances = exceedance.collapsed('time', iris.analysis.COUNT, function=lambda x: x > 0, )

            number_exceedances.var_name = ('days_snowfall_above' + str(i_quantile) + 'percentile')
            sum_exceedances = exceedance.collapsed('time', iris.analysis.SUM)

            sum_exceedances.var_name = ('sum_snowfall_above_' + str(i_quantile) + 'percentile')

            # TODO: fix exceedance count for multi-model-ensemble
            # data['exceedance', i_quantile] = exceedance
            data['number_exceedances', i_quantile] = number_exceedances
            data['sum_exceedance', i_quantile] = sum_exceedances
            data['mean'] = mean_data
        cuberesults[i_key] = data
    # sum up results into a single cube
    keys = list(cubelist.keys())
    for i_key in cuberesults[keys[0]].keys():
        results[i_key] = cuberesults[keys[0]][i_key]
        for i_cube in range(1, number_of_cubes):
            results[i_key] = results[i_key] + cuberesults[keys[i_cube]][i_key]

    # adjust average measures:
    results['mean'] = results['mean'] / number_of_cubes
    for i_quantile in quantiles:
        results['quantile', i_quantile] = percentile_from_cubedict(cubelist, i_quantile, startyear, finalyear)
        results['quantile_baseline', i_quantile] = baselinequantiles[i_quantile]
        results['mean_exceedance', i_quantile] = results['sum_exceedance', i_quantile] / results[
            'number_exceedances', i_quantile]
        results['mean_exceedance', i_quantile].var_name = 'mean_exceedance_' + str(i_quantile) + 'percentile'
        expected_exceedances = number_of_timepoints * (100 - i_quantile) / 100 * number_of_cubes
        results['baseline_relative_mean_exceedance', i_quantile] = results[
                                                                       'sum_exceedance', i_quantile] / expected_exceedances
        results['baseline_relative_mean_exceedance', i_quantile].var_name = 'baseline_normed_mean_exceedance_' + str(
            i_quantile) + 'percentile'
    return results


def calculate_quantile_exceedance_measure(historical_cubelist, ssp_cubelist, ssp_scenario, baseline_quantiles,
                                          quantiles,
                                          number_of_years_to_compare, number_of_timeperiods, historical_start,
                                          ssp_start, historical=True):
    historical_start_list = []

    historical_start_list.append(historical_start)

    ssp_start_list = []

    ssp_start_list.append(ssp_start)

    for index in range(1, number_of_timeperiods):
        historical_start_list.append(historical_start_list[index - 1] + number_of_years_to_compare)

        ssp_start_list.append(ssp_start_list[index - 1] + number_of_years_to_compare)

    data = {}
    if (historical):
        for i_historical_start in historical_start_list:
            i_historical_end = i_historical_start + number_of_years_to_compare - 1
            data[
                'historical_quantile', i_historical_start, i_historical_end] = generate_quantile_exceedance_development(
                historical_cubelist, baseline_quantiles, quantiles, i_historical_start, i_historical_end)

    for i_ssp_start in ssp_start_list:
        i_ssp_end = i_ssp_start + number_of_years_to_compare - 1
        ssp_key = ssp_scenario + "_quantile"
        data[ssp_key, i_ssp_start, i_ssp_end] = generate_quantile_exceedance_development(ssp_cubelist,
                                                                                         baseline_quantiles,
                                                                                         quantiles,
                                                                                         i_ssp_start,
                                                                                         i_ssp_end)
    return (data)


# function to restrict cubes on bounding box

def cube_from_bounding_box(cube, bounding_box):
    return cube.intersection(longitude=(bounding_box[0], bounding_box[2])).intersection(
        latitude=(bounding_box[1], bounding_box[3]))


# method to create list of country limited cubes

def country_cubelist(cubelist, country_box):
    country_cubes = iris.cube.CubeList()
    for i_cube in cubelist:
        country_cubes.append(cube_from_bounding_box(i_cube, country_box))
    return unify_concatenate(country_cubes)


# filter all scenarios:
def country_filter(cubedict, country_box):
    dict_keys = cubedict.keys()
    country_cubes = {}
    for i_key in dict_keys:
        country_cubes[i_key] = country_cubelist(cubedict[i_key], country_box)
    return country_cubes


def extract_dates(cube, startyear, finalyear):
    # filter for certain years (inclusive for start and final year)
    # add auxilary year coordinate

    iris.coord_categorisation.add_year(cube, 'time', name='year')
    year_constraint = iris.Constraint(year=lambda t: startyear <= t <= finalyear)
    limited_cube = cube.extract(year_constraint)
    cube.remove_coord('year')
    return limited_cube


def percentile_from_cubedict(cubedict, percentile, startyear, finalyear):
    # extract data from cubes and merge
    keys = list(cubedict.keys())
    dataarray = extract_dates(cubedict[keys[0]], startyear, finalyear).data
    for i in range(1, len(cubedict)):
        np.concatenate((dataarray, extract_dates(cubedict[keys[i]], startyear, finalyear).data), axis=0)
    # calculate percentile
    percentile_data = np.percentile(dataarray, percentile, axis=0)

    # construct percentile cube to return
    percentile_cube = extract_dates(cubedict[keys[0]], startyear, finalyear).collapsed('time', iris.analysis.PERCENTILE,
                                                                                       percent=percentile)
    # replace data with data from all cubes
    percentile_cube.data = percentile_data

    return percentile_cube


def comparison_threshold(modellist, basic_cubelist, ssp_scenario, number_of_decades, area_box, start_historical,
                         start_ssp,
                         intensity=False, historical=True):
    # filter data for country box:
    area_data = country_filter(basic_cubelist, area_box)

    # filter for season to reduce computational load, winter snowfalls anyway top percentile most likely
    # country_data = prepare_season_stats (country_data,season)

    # merge model cubes into ensemble cube

    historical_ensemble_cubelist = {}
    ssp_ensemble_cubelist = {}
    baseline_ensemble_cubelist = {}
    for i_model in modellist:
        historical_ensemble_cubelist[i_model] = (area_data[i_model, 'historical'])
        ssp_ensemble_cubelist[i_model] = (area_data[i_model, ssp_scenario])
        baseline_ensemble_cubelist[i_model] = (area_data[i_model, 'historical'])

    quantiles = [99, 99.9]
    baseline_quantiles = {}
    for i_quantile in quantiles:
        baseline_quantiles[i_quantile] = percentile_from_cubedict(baseline_ensemble_cubelist, i_quantile, 1851, 1880)
    return calculate_quantile_exceedance_measure(historical_ensemble_cubelist, ssp_ensemble_cubelist, ssp_scenario,
                                                 baseline_quantiles, quantiles,
                                                 10, number_of_decades, start_historical, start_ssp,
                                                 intensity=intensity, historical=historical)


def multi_region_threshold_analysis_preindustrial(modellist, arealist, ssp_scenario, start_ssp, start_historical,
                                                  number_of_decades,
                                                  intensity=True, historical=True):
    for i_area in tqdm(arealist.keys()):
        results = {}
        results['ensemble', 'preindustrial', i_area] = comparison_threshold(modellist, cubelist, ssp_scenario,
                                                                            number_of_decades,
                                                                            arealist[i_area], start_historical,
                                                                            start_ssp, intensity=intensity)
        filename = str('ensemble') + '_' + str(i_area) + '_preindustrial_baseline_hist_from_' + str(
            start_historical) + "_" + ssp_scenario + '_from_' + str(
            start_ssp) + '_' + str(number_of_decades) + "_"

        file = open(filename, 'wb')
        pickle.dump(results, file)



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

models = settings['models']
basepath = settings['basepath']
variablename = settings['variablename']
outputdir = settings['outputdir']
scenarios = settings['scenarios']
data_dictionary = {}

for i_model in models:
    data_dictionary[i_model, 'historical'] = filepath_generator(basepath, 'historical', i_model)
    for i_scenario in scenarios:
        data_dictionary[i_model, i_scenario] = filepath_generator(basepath, i_scenario, i_model)

# generate dictionary of  data [model,scenario]

# get all keys from dictionary
data_keys = data_dictionary.keys()
filepaths = {}
for i_key in data_keys:
    with open(data_dictionary[i_key], 'r') as stream:
        try:
            filepaths[i_key] = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
# load data

cubelist = {}
# restrict latitudes
for i_key in data_keys:
    cubelist[i_key] = (latitude_constraint(23, iris.load(filepaths[i_key], variablename)))

for i_model in models:
    unify_current_decade(cubelist, i_model, 'ssp585')

# ignore warnings
warnings.simplefilter("ignore")

os.chdir(outputdir)

# define boxes for northern america and northern europe:

northern_america = (-140, 30, -50, 60)

northern_europe = (-25, 42, 44, 68)

lat_lower_bound = 30
lat_upper_bound = 90
northern_hemisphere_first_quarter = (0, lat_lower_bound, 90, lat_upper_bound)
northern_hemisphere_second_quarter = (90, lat_lower_bound, 180, lat_upper_bound)
northern_hemisphere_third_quarter = (-180, lat_lower_bound, -90, lat_upper_bound)
northern_hemisphere_fourth_quarter = (-90, lat_lower_bound, 0, lat_upper_bound)
arealist = {}
# arealist['NORTHERN AMERICA'] = northern_america
# arealist['NORTHERN EUROPE'] = northern_europe
arealist['NORTHERN_HEMISPHERE_1'] = northern_hemisphere_first_quarter
arealist['NORTHERN_HEMISPHERE_2'] = northern_hemisphere_second_quarter
arealist['NORTHERN_HEMISPHERE_3'] = northern_hemisphere_third_quarter
arealist['NORTHERN_HEMISPHERE_4'] = northern_hemisphere_fourth_quarter

for i_scenario in tqdm(scenarios):
    multi_region_threshold_analysis_preindustrial(models, arealist, i_scenario, 2021, 1851, 3)
    multi_region_threshold_analysis_preindustrial(models, arealist, i_scenario, 2051, 1851, 3)
    multi_region_threshold_analysis_preindustrial(models, arealist, i_scenario, 2081, 1851, 2)

# historical analysis for comparison

multi_region_threshold_analysis_preindustrial(models, arealist, 'historical', 1881, 1851, 3)
multi_region_threshold_analysis_preindustrial(models, arealist, 'historical', 1911, 1851, 3)
multi_region_threshold_analysis_preindustrial(models, arealist, 'historical', 1941, 1851, 3)
multi_region_threshold_analysis_preindustrial(models, arealist, 'historical', 1971, 1851, 3)
multi_region_threshold_analysis_preindustrial(models, arealist, 'historical', 2000, 1851, 2)
