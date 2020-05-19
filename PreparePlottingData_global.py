#!/usr/bin/env python

import argparse
import os
import pickle

import cf_units
import iris
import iris.analysis
import iris.coord_categorisation
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
from ruamel.yaml import ruamel
from tqdm import tqdm


def specify_results(resultsdict, modelname, comparisonname, regionname, multimodel=True):
    if multimodel:
        output = resultsdict[modelname, comparisonname, regionname]
    else:
        output = resultsdict[comparisonname, regionname]
    return output


def data_collection(results, scenario, start_year, final_year, timeperiod_length):
    data_timeperiod = final_year - start_year + 1
    number_of_timeperiods = data_timeperiod / timeperiod_length
    reference_data = []

    for year in range(start_year, final_year, timeperiod_length):
        reference_data.append(results[scenario, year, year + timeperiod_length - 1])

    exceedance_arrays = []
    intensity_arrays = []
    maps = []
    expected_snowfall_maps = []

    # prepare reference arrays
    for i_reference_timeperiod in reference_data:
        exceedance_arrays.append(i_reference_timeperiod['exceedance_array'][1])
        intensity_arrays.append(i_reference_timeperiod['intensity_array'][1])
        reference_thresholds = i_reference_timeperiod['intensity_array'][0]
        maps.append(i_reference_timeperiod['map_cubes'])

    intensity = np.sum(intensity_arrays, axis=0)

    exceedances = np.sum(exceedance_arrays, axis=0)

    reference_exceedances = np.vstack((reference_thresholds, exceedances))
    reference_intensity = np.vstack((reference_thresholds, intensity))

    # prepare reference cubes
    dict_cubelists = {}
    for i_key in (maps[0].keys()):
        cubelist = iris.cube.CubeList()
        for i_data in maps:
            cubelist.append(i_data[i_key])
        dict_cubelists[i_key] = cubelist

    reference_cubes = {}
    for i_key in (dict_cubelists):
        reference_cube = dict_cubelists[i_key][0] - dict_cubelists[i_key][0]
        for i_cube in dict_cubelists[i_key]:
            reference_cube += i_cube
        reference_cubes[i_key] = reference_cube

    for i_key in reference_cubes.keys():
        reference_cubes[i_key] = reference_cubes[i_key] / number_of_timeperiods

    # prepare quantile cubes

    reference_collection = {}
    reference_collection['exceedances'] = reference_exceedances
    reference_collection['intensity'] = reference_intensity
    reference_collection['cubes'] = reference_cubes

    return reference_collection


def quantile_collection(results, scenario, start_year, final_year, timeperiod_length, rolling_window=False):
    data_timeperiod = final_year - start_year + 1
    number_of_timeperiods = data_timeperiod / timeperiod_length
    quantile_data = []
    for year in range(start_year, final_year + 1, timeperiod_length):
        quantile_data.append(results[scenario, year, year + timeperiod_length - 1])
    # extension if rolling window data
    if rolling_window:
        quantile_data = []
        number_of_timeperiods = data_timeperiod - timeperiod_length + 1
        for year in range(start_year, final_year - timeperiod_length + 1 + 1, 1):
            quantile_data.append(results[scenario, year, year + timeperiod_length - 1])

    # prepare quantile cubes
    quantile_cubelists = {}

    for i_key in (quantile_data[0].keys()):
        cubelist = iris.cube.CubeList()
        for i_data in quantile_data:
            cubelist.append(i_data[i_key])

        quantile_cubelists[i_key] = cubelist

    quantile_cubes = {}

    for i_key in (quantile_cubelists.keys()):
        print(i_key)
        reference_cube = quantile_cubelists[i_key][0] - quantile_cubelists[i_key][0]
        if (reference_cube.coords(long_name='year')):
            reference_cube.remove_coord('year')
        for i_cube in quantile_cubelists[i_key]:
            reference_cube += i_cube
        time_coord = iris.coords.AuxCoord(start_year + data_timeperiod / 2 - 1,
                                          long_name='year', units='1', bounds=(start_year, final_year))
        reference_cube.add_aux_coord(time_coord)
        quantile_cubes[i_key] = reference_cube
        print(reference_cube)

    for i_key in quantile_cubelists.keys():
        quantile_cubes[i_key] = quantile_cubes[i_key] / number_of_timeperiods

    reference_collection = {}
    reference_collection['quantiles'] = quantile_cubes
    return reference_collection


def ensemble_average(modellist, data):
    average = data[modellist[0]]
    number_of_models = len(modellist)
    for i in range(1, number_of_models):
        average += data[modellist[i]]
    return average / number_of_models


def concatenate_cube_dict(cubedict):
    keys = list(cubedict.keys())
    start_cube = cubedict[keys[0]]
    number_cubes = len(keys)
    for i in range(1, number_cubes):
        cube_list = iris.cube.CubeList([start_cube, cubedict[keys[i]]])
        start_cube = cube_list.concatenate()[0]

    return start_cube


def mask_close_to_0_values(cube):
    zero_mask = np.where(cube.data < 10, True, False)
    cube_masked = cube.copy()
    cube_masked.data = np.ma.array(cube.data, mask=zero_mask)
    return cube_masked


def ensemble_average_quantile_baseline(quantile, modellist, results, ref_results, scenario, start_years,
                                       timeperiod_length, reference_scenario, reference_start_year,
                                       reference_final_year, rolling_window=False):
    dict_to_plot = {}
    reference_data = {}
    data = {}
    baseline = {}

    quantile_dict = {}
    ref_exceedances = {}
    ref_mean = {}
    ref_frequency = {}
    ref_expected_snowfall = {}
    ref_expected_snowfall_rel_baseline = {}
    data_exceedances = {}
    data_mean = {}
    data_frequency = {}
    data_expected_snowfall = {}
    data_expected_snowfall_rel_baseline = {}
    diff_frequency = {}
    diff_expected_snowfall = {}
    diff_ratios = {}

    baseline_key = "quantile_baseline"
    quantile_key = "quantile"
    exceedance_key = "exceedance"
    frequency_key = 'number_exceedances'
    mean_exceedance_key = 'mean_exceedance'
    relative_mean_exceedance_key = 'baseline_relative_mean_exceedance'

    baseline_average = {}
    quantile_average = {}
    quantile_baseline_ratio = {}

    expected_snowfall_ratio = {}
    ref_frequency_average = {}
    ref_expected_snowfall_average = {}
    ref_expected_snowfall_rel_baseline_average = {}

    data_frequency_average = {}
    data_expected_snowfall_average = {}
    data_expected_snowfall_rel_baseline_average = {}

    ref_mean_average = {}
    data_mean_average = {}
    diff_mean_average = {}
    mean_ratios = {}
    frequency_ratios = {}
    diff_frequency_average = {}
    diff_expected_snowfall_average = {}
    diff_es_baseline_relative = {}
    diff_es_baseline_relative_average = {}
    diff_expected_snowfall_average_relative = {}
    ref_ensemble_exceedances = {}
    data_ensemble_exceedances = {}
    for i_start_year in start_years:
        start_year = i_start_year
        final_year = start_year + timeperiod_length - 1
        print(start_year)
        print(final_year)
        for i_model in modellist:
            # TODO: balanced approach to get reference period with equally weighted rolling window decades, for now: just average the consecutive decades
            reference_data[i_model] = quantile_collection(ref_results[i_model],
                                                          reference_scenario, reference_start_year,
                                                          reference_final_year, timeperiod_length, rolling_window=False)
            data[i_model] = quantile_collection(results[i_model], scenario, start_year, final_year,
                                                timeperiod_length, rolling_window=rolling_window)
            baseline[i_model] = reference_data[i_model]['quantiles'][quantile_key, quantile]
            ref_mean[i_model] = reference_data[i_model]['quantiles']['mean']
            data_mean[i_model] = data[i_model]['quantiles']['mean']
            # reference_year_coordinate = ref_mean[i_model].coord('year')
            data_year_coordinate = data_mean[i_model].coord('year')
            quantile_dict[i_model] = data[i_model]['quantiles'][quantile_key, quantile]

            # ref_exceedances[i_model] = reference_data[i_model]['quantiles'][exceedance_key, quantile]

            ref_frequency[i_model] = reference_data[i_model]['quantiles'][frequency_key, quantile]
            # ref_frequency[i_model].add_aux_coord(reference_year_coordinate)

            ref_expected_snowfall[i_model] = reference_data[i_model]['quantiles'][mean_exceedance_key, quantile]
            ref_expected_snowfall_rel_baseline[i_model] = reference_data[i_model]['quantiles'][
                relative_mean_exceedance_key, quantile]
            # data_exceedances[i_model] = data[i_model]['quantiles'][exceedance_key, quantile]

            data_frequency[i_model] = data[i_model]['quantiles'][frequency_key, quantile]
            data_expected_snowfall[i_model] = data[i_model]['quantiles'][mean_exceedance_key, quantile]
            data_expected_snowfall_rel_baseline[i_model] = data[i_model]['quantiles'][
                relative_mean_exceedance_key, quantile]

            # add time coordinate to identify scenario decade from which the difference stems, to prepare timeplots of development
            time_coord = iris.coords.AuxCoord(start_year + timeperiod_length / 2 - 1,
                                              long_name='scenario_year',
                                              units=cf_units.Unit('years since 0000-01-01', calendar='gregorian'),
                                              bounds=(start_year, final_year))

            diff_frequency[i_model] = data_frequency[i_model] - ref_frequency[i_model]
            diff_frequency[i_model].add_aux_coord(time_coord)
            diff_expected_snowfall[i_model] = data_expected_snowfall[i_model] - ref_expected_snowfall[i_model]
            diff_expected_snowfall[i_model].add_aux_coord(time_coord)
            diff_es_baseline_relative[i_model] = data_expected_snowfall_rel_baseline[i_model] - \
                                                 ref_expected_snowfall_rel_baseline[i_model]
            diff_es_baseline_relative[i_model].add_aux_coord(time_coord)
        baseline_average[i_start_year] = ensemble_average(modellist, baseline)
        # mask data where quantile ==0
        baseline_average[i_start_year] = mask_close_to_0_values(baseline_average[i_start_year])
        ref_mean_average[i_start_year] = ensemble_average(modellist, ref_mean)

        data_mean_average[i_start_year] = ensemble_average(modellist, data_mean)

        # add time coordinate to identify scenario decade from which the difference stems, to prepare timeplots of development
        time_coord = iris.coords.AuxCoord(start_year + timeperiod_length / 2 - 1,
                                          long_name='scenario_year',
                                          units=cf_units.Unit('years since 0000-01-01 00:00:00', calendar='gregorian'))
        # , bounds=(start_year, final_year))

        diff_mean_average[i_start_year] = data_mean_average[i_start_year] - ref_mean_average[i_start_year]
        diff_mean_average[i_start_year].add_aux_coord(time_coord)
        quantile_average[i_start_year] = ensemble_average(modellist, quantile_dict)

        # mask data where quantile ==0

        quantile_baseline_ratio[i_start_year] = quantile_average[i_start_year] / baseline_average[i_start_year]
        quantile_baseline_ratio[i_start_year].add_aux_coord(time_coord)
        ref_frequency_average[i_start_year] = ensemble_average(modellist, ref_frequency)

        ref_expected_snowfall_average[i_start_year] = ensemble_average(modellist, ref_expected_snowfall)
        ref_expected_snowfall_rel_baseline_average[i_start_year] = ensemble_average(modellist,
                                                                                    ref_expected_snowfall_rel_baseline)
        data_frequency_average[i_start_year] = ensemble_average(modellist, data_frequency)
        data_expected_snowfall_average[i_start_year] = ensemble_average(modellist, data_expected_snowfall)
        data_expected_snowfall_rel_baseline_average[i_start_year] = ensemble_average(modellist,
                                                                                     data_expected_snowfall_rel_baseline)
        diff_frequency_average[i_start_year] = data_frequency_average[i_start_year] - ref_frequency_average[
            i_start_year]
        diff_frequency_average[i_start_year].add_aux_coord(time_coord)
        frequency_ratios[i_start_year] = data_frequency_average[i_start_year] / ref_frequency_average[i_start_year]

        frequency_ratios[i_start_year].add_aux_coord(time_coord)
        diff_expected_snowfall_average[i_start_year] = (
                data_expected_snowfall_average[i_start_year] - ref_expected_snowfall_average[i_start_year])

        diff_expected_snowfall_average[i_start_year].add_aux_coord(time_coord)

        diff_expected_snowfall_average_relative[i_start_year] = diff_expected_snowfall_average[i_start_year] / \
                                                                baseline_average[i_start_year]
        diff_expected_snowfall_average_relative[i_start_year].add_aux_coord(time_coord)
        diff_es_baseline_relative_average[i_start_year] = data_expected_snowfall_rel_baseline_average[i_start_year] - \
                                                          ref_expected_snowfall_rel_baseline_average[i_start_year]
        diff_es_baseline_relative_average[i_start_year].add_aux_coord(time_coord)
        expected_snowfall_ratio[i_start_year] = data_expected_snowfall_average[i_start_year] / \
                                                ref_expected_snowfall_average[i_start_year]
        expected_snowfall_ratio[i_start_year].add_aux_coord(time_coord)
        mean_ratios[start_year] = data_mean_average[i_start_year] / ref_mean_average[i_start_year]

        mean_ratios[start_year].add_aux_coord(time_coord)
        # copy cube to avoid problems with units

        # ref_ensemble_exceedances[i_start_year] = concatenate_cube_dict(ref_exceedances)
        # data_ensemble_exceedances[i_start_year] = concatenate_cube_dict(data_exceedances)

        # mask outliers from relative values caused by almost 0 baseline:
        to_mask = diff_expected_snowfall_average_relative[i_start_year].data
        diff_expected_snowfall_average_relative[i_start_year].data = np.ma.masked_where(np.abs(to_mask) >= 2, to_mask)

        dict_to_plot[i_start_year, 'diff_mean'] = diff_mean_average[i_start_year]
        dict_to_plot[quantile, i_start_year, 'diff_frequency'] = diff_frequency_average[i_start_year]
        dict_to_plot[quantile, i_start_year, 'diff_es'] = diff_expected_snowfall_average[i_start_year]
        dict_to_plot[quantile, i_start_year, 'diff_es_baseline'] = diff_es_baseline_relative_average[i_start_year]

        dict_to_plot[quantile, i_start_year, 'mean_ratio'] = mean_ratios[i_start_year]
        dict_to_plot[quantile, i_start_year, 'frequency_ratio'] = frequency_ratios[i_start_year]
        dict_to_plot[quantile, i_start_year, 'percentile_ratio'] = quantile_baseline_ratio[i_start_year]
        dict_to_plot[quantile, i_start_year, 'es_ratio'] = expected_snowfall_ratio[i_start_year]

        dict_to_plot[i_start_year, 'diff_mean'].var_name = 'diff_mean'
        dict_to_plot[quantile, i_start_year, 'diff_frequency'].var_name = 'diff_frequency'
        dict_to_plot[quantile, i_start_year, 'diff_es'].var_name = 'diff_es'
        dict_to_plot[quantile, i_start_year, 'diff_es_baseline'].var_name = 'diff_es_baseline'
        dict_to_plot[quantile, i_start_year, 'frequency_ratio'].var_name = 'frequency_ratio'
        dict_to_plot[quantile, i_start_year, 'percentile_ratio'].var_name = 'percentile_ratio'
        dict_to_plot[quantile, i_start_year, 'es_ratio'].var_name = 'es_ratio'
        dict_to_plot[quantile, i_start_year, 'mean_ratio'].var_name = 'mean_ratio'



    return dict_to_plot


def ensemble_quantile_average(modellist, scenarios, quantiles, all_results, all_ref_results, comparisonname, areaname,
                              start_years, timeperiod_length, rolling_window=False):
    ensemble_results = {}
    ref_ensemble_results = {}
    for i_model in modellist:
        ensemble_results[i_model] = specify_results(all_results, i_model, comparisonname, areaname)
        ref_ensemble_results[i_model] = specify_results(all_ref_results, i_model, comparisonname, areaname)
    # adjust start years for rolling window:
    if (rolling_window):
        min_start_year = np.min(start_years)
        max_start_year = np.max(start_years)
        stepsize = 1
        start_years = range(min_start_year, max_start_year + 1, stepsize)

    reference_scenarios = ['historical']
    plotting_data = {}
    for i_quantile in quantiles:
        for i_scenario in scenarios:
            for i_reference_scenario in reference_scenarios:
                plotting_data[i_quantile, i_scenario, i_reference_scenario] = ensemble_average_quantile_baseline(
                    i_quantile, modellist, ensemble_results, ref_ensemble_results, i_scenario,
                    start_years, timeperiod_length,
                    i_reference_scenario, 1851, 1880, rolling_window=rolling_window)

    return plotting_data


# add settings argument

parser = argparse.ArgumentParser(
    description="Calculate some analysis metrics on specified data files and pickle them for use in a plotting module")

# settings file import
# argument parser definition
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
reference_files = settings['reference_files']
outputdir = settings['outputdir']
areanames = settings['areanames']
scenarios = settings['scenarios']
quantiles = settings['quantiles']
start_years = settings['start_years']
timeperiod_length = settings['timeperiod_length']
rolling_window = settings['rolling_window']
# set outputpath for plots etc
os.chdir(outputdir)

# load results data
partial_results = []
for i_file in filelist:
    with open(i_file, 'rb') as stream:
        partial_results.append(pickle.load(stream))
number_models = len(partial_results)
print(number_models)
all_results = dict(partial_results[0])
for i in range(1, number_models):
    all_results.update(partial_results[i])

# load reference results data
partial_ref_results = []
for i_file in reference_files:
    with open(i_file, 'rb') as stream:
        partial_ref_results.append(pickle.load(stream))
number_models = len(partial_ref_results)
print(number_models)
all_ref_results = dict(partial_ref_results[0])
for i in range(1, number_models):
    all_ref_results.update(partial_ref_results[i])

# define models to be handled separatly
modellist = [
    'ensemble']
# loop over all time periods
for i_start_years in tqdm(start_years):
    cache_plotting_data = {}

    for i_areaname in tqdm(areanames):
        cache_plotting_data[i_areaname] = ensemble_quantile_average(modellist, scenarios, quantiles, all_results,
                                                                    all_ref_results, 'preindustrial',
                                                                    i_areaname, i_start_years, timeperiod_length,
                                                                    rolling_window=rolling_window)
    for i_key in tqdm(cache_plotting_data[areanames[0]].keys()):
        quantile = i_key[0]
        scenario = i_key[1]
        ref_scenario = i_key[2]
        global_plotting_data = {}
        for i_second_key in tqdm(cache_plotting_data[areanames[0]][i_key]):
            partial_data_cubelist = iris.cube.CubeList()
            for i_areaname in tqdm(areanames):
                partial_data_cubelist.append(cache_plotting_data[i_areaname][i_key][i_second_key])
            equalise_attributes(partial_data_cubelist)
            unify_time_units(partial_data_cubelist)
            global_plotting_data[i_second_key] = partial_data_cubelist.concatenate_cube(check_aux_coords=False)
        min_start_year = np.min(start_years)
        max_start_year = np.max(start_years)

        filename = 'frequency_es_quantile_ratios_' + str('global') + "_" + str('preindustrial') + "_" + str(
            scenario) + "_" + str(ref_scenario) + "_" + str(quantile) + "_" + str(min_start_year) + "_" + str(
            max_start_year) + "_" + str(rolling_window)
        file = open(filename, 'wb')
        pickle.dump(global_plotting_data, file)
