#!/usr/bin/env python

# Imports
import argparse
import datetime
import os
import pickle
import warnings

import iris
import iris.analysis
import iris.coord_categorisation
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
def unify_current_decade(model):
    cubes_to_concatenate = iris.cube.CubeList()
    cubes_to_concatenate.append(cubelist[model, 'historical'][17])
    cubes_to_concatenate.append(cubelist[model, 'ssp585'][0])

    cube_2011_2020 = unify_concatenate(cubes_to_concatenate)
    cubelist[model, 'historical'][17] = cube_2011_2020


def extract_dates(cube, startyear, finalyear):
    # filter for certain years (inclusive for start and final year)
    # add auxilary year coordinate

    iris.coord_categorisation.add_year(cube, 'time', name='year')
    year_constraint = iris.Constraint(year=lambda t: startyear <= t <= finalyear)
    limited_cube = cube.extract(year_constraint)
    cube.remove_coord('year')
    return limited_cube


def generate_quantile_exceedance_development(cube, baselinequantiles, quantiles, startyear, finalyear, intensity=False):
    data_timeperiod = extract_dates(cube, startyear, finalyear)

    # calculate exceedance of quantiles for each gridcell to use as thresholds:
    # N.B. baselinequantile cubes to be submitted as dictionary of quantile values
    data = {}
    for i_quantile in quantiles:
        quantile_baseline = baselinequantiles[i_quantile]
        exceedance_data = data_timeperiod.data - quantile_baseline.data
        exceedance = data_timeperiod.copy(data=exceedance_data)

        var_name = ('expected_snowfall_above_' + str(i_quantile) + 'percentile')
        exceedance.var_name = var_name

        # consider only positve values of exceedance
        exceedance_array = exceedance.data
        exceedance_indicators = (exceedance_array > 0)

        exceedance.data = exceedance_array * exceedance_indicators

        number_exceedances = exceedance.collapsed('time', iris.analysis.COUNT, function=lambda x: x > 0)
        sum_exceedances = exceedance.collapsed('time', iris.analysis.SUM)

        mean_exceedance = sum_exceedances / number_exceedances

        data['quantile_baseline', i_quantile] = quantile_baseline
        data['number_exceedances', i_quantile] = number_exceedances
        data['mean_exceedance', i_quantile] = mean_exceedance
    return data


def calculate_quantile_exceedance_measure(historical_cube, ssp126_cube, ssp370_cube, ssp585_cube, baseline_quantiles,
                                          quantiles,
                                          number_of_years_to_compare, number_of_timeperiods, historical_start,
                                          ssp_start, intensity=False):
    historical_start_list = []

    historical_start_list.append(historical_start)

    ssp_start_list = []

    ssp_start_list.append(ssp_start)

    for index in range(1, number_of_timeperiods):
        historical_start_list.append(historical_start_list[index - 1] + number_of_years_to_compare)

        ssp_start_list.append(ssp_start_list[index - 1] + number_of_years_to_compare)

    data = {}
    for i_historical_start in historical_start_list:
        i_historical_end = i_historical_start + number_of_years_to_compare - 1
        data['historical_quantile', i_historical_start, i_historical_end] = generate_quantile_exceedance_development(
            historical_cube, baseline_quantiles, quantiles, i_historical_start, i_historical_end, intensity=intensity)

    for i_ssp_start in ssp_start_list:
        i_ssp_end = i_ssp_start + number_of_years_to_compare - 1
        data['ssp126_quantile', i_ssp_start, i_ssp_end] = generate_quantile_exceedance_development(ssp126_cube,
                                                                                                   baseline_quantiles,
                                                                                                   quantiles,
                                                                                                   i_ssp_start,
                                                                                                   i_ssp_end,
                                                                                                   intensity=intensity)
        data['ssp370_quantile', i_ssp_start, i_ssp_end] = generate_quantile_exceedance_development(ssp370_cube,
                                                                                                   baseline_quantiles,
                                                                                                   quantiles,
                                                                                                   i_ssp_start,
                                                                                                   i_ssp_end,
                                                                                                   intensity=intensity)
        data['ssp585_quantile', i_ssp_start, i_ssp_end] = generate_quantile_exceedance_development(ssp585_cube,
                                                                                                   baseline_quantiles,
                                                                                                   quantiles,
                                                                                                   i_ssp_start,
                                                                                                   i_ssp_end,
                                                                                                   intensity=intensity)

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


def comparison_threshold(model, basic_cubelist, number_of_decades, area_box, start_historical, start_ssp, areaname,
                         intensity=False):
    # filter data for country box:
    area_data = country_filter(basic_cubelist, area_box)

    # filter for season to reduce computational load, winter snowfalls anyway top percentile most likely
    # country_data = prepare_season_stats (country_data,season)

    historical_cube = area_data[model, 'historical']
    ssp126_cube = area_data[model, 'ssp126']
    ssp370_cube = area_data[model, 'ssp370']
    ssp585_cube = area_data[model, 'ssp585']
    baseline_cube = extract_dates(historical_cube, 1851, 1880)

    quantiles = [99, 99.73, 99.9, 99.99]
    baseline_quantiles = {}
    for i_quantile in quantiles:
        baseline_quantiles[i_quantile] = baseline_cube.collapsed('time', iris.analysis.PERCENTILE, percent=i_quantile)
    return calculate_quantile_exceedance_measure(historical_cube, ssp126_cube, ssp370_cube, ssp585_cube,
                                                 baseline_quantiles, quantiles,
                                                 10, number_of_decades, start_historical, start_ssp,
                                                 intensity=intensity)


def multi_region_threshold_analysis_preindustrial(modellist, arealist, intensity=True):
    results = {}
    for i_key in tqdm(arealist.keys()):
        for i_model in tqdm(modellist):
            results[i_model, 'preindustrial', i_key] = comparison_threshold(i_model, cubelist, 8, arealist[i_key], 1851,
                                                                            2021, i_key, intensity=True)
    filename = str(i_model) + '_preindustrial_baseline_'
    date = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    filename = filename + str(date)

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
data_dictionary = {}

for i_model in models:
    data_dictionary[i_model, 'historical'] = filepath_generator(basepath, 'historical', i_model)
    data_dictionary[i_model, 'ssp126'] = filepath_generator(basepath, 'ssp126', i_model)
    data_dictionary[i_model, 'ssp370'] = filepath_generator(basepath, 'ssp370', i_model)
    data_dictionary[i_model, 'ssp585'] = filepath_generator(basepath, 'ssp585', i_model)

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

for i_key in data_keys:
    cubelist[i_key] = (latitude_constraint(23, iris.load(filepaths[i_key], variablename)))

for i_model in models:
    unify_current_decade(i_model)

# ignore warnings
warnings.simplefilter("ignore")

os.chdir(outputdir)

# define boxes for northern america and northern europe:

northern_america = (-140, 30, -50, 60)

northern_europe = (-25, 42, 44, 68)

arealist = {}
arealist['NORTHERN AMERICA'] = northern_america
arealist['NORTHERN EUROPE'] = northern_europe

multi_region_threshold_analysis_preindustrial(models, arealist)
