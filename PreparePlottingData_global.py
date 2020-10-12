#!/usr/bin/env python

import argparse
import os
import pickle
from datetime import datetime

import cf_units
import iris
import iris.analysis
import iris.coord_categorisation
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
from ruamel.yaml import ruamel
from tqdm import tqdm


# method to get results from dictionary classified by scenario, region, and optionally modelname
def specify_results(resultsdict, modelname, comparisonname, regionname, multimodel=True):
    if multimodel:
        output = resultsdict[modelname, comparisonname, regionname]
    else:
        output = resultsdict[comparisonname, regionname]
    return output


# method to collect data for specific timeperiods from data collection
def results_collection(results, data_scenario, start_year, final_year, length_timeperiod, rolling_window=False):
    data_timeperiod = final_year - start_year + 1
    number_of_timeperiods = data_timeperiod / length_timeperiod
    data = []
    for year in range(start_year, final_year + 1, length_timeperiod):
        data.append(results[data_scenario, year, year + length_timeperiod - 1])
    # extension if rolling window data
    if rolling_window:
        data = []
        number_of_timeperiods = data_timeperiod - length_timeperiod + 1
        for year in range(start_year, final_year - length_timeperiod + 1 + 1, 1):
            data.append(results[data_scenario, year, year + length_timeperiod - 1])

    # prepare data cubes
    data_cubelists = {}

    for iterator_key in (data[0].keys()):
        cubelist = iris.cube.CubeList()
        for i_data in data:
            cubelist.append(i_data[iterator_key])

        data_cubelists[iterator_key] = cubelist

    data_cubes = {}

    for iterator_key in (data_cubelists.keys()):
        print(iterator_key)
        reference_cube = data_cubelists[iterator_key][0] - data_cubelists[iterator_key][0]
        if reference_cube.coords(long_name='year'):
            reference_cube.remove_coord('year')

            for i_cube in data_cubelists[iterator_key]:
                reference_cube += i_cube
            time_coord = iris.coords.AuxCoord(start_year + data_timeperiod / 2 - 1,
                                              long_name='year', units='1', bounds=(start_year, final_year))
            reference_cube.add_aux_coord(time_coord)
            data_cubes[iterator_key] = reference_cube

            for i_cube in data_cubelists[iterator_key]:
                reference_cube += i_cube
            time_coord = iris.coords.AuxCoord(start_year + data_timeperiod / 2 - 1,
                                              long_name='year', units='1', bounds=(start_year, final_year))
            if reference_cube.coords(long_name='year'):
                reference_cube.remove_coord('year')
            reference_cube.add_aux_coord(time_coord)
            data_cubes[iterator_key] = reference_cube

    for iterator_key in data_cubelists.keys():
        data_cubes[iterator_key] = data_cubes[iterator_key] / number_of_timeperiods

    reference_collection = {'data': data_cubes}
    return reference_collection


# calculates average for a list of models and a dictionary with single model data
def ensemble_average(models, data):
    average = data[models[0]]
    number_of_models = len(models)
    for iterator in range(1, number_of_models):
        average += data[models[iterator]]
    return average / number_of_models


# concatenates all cubes from a cubedict
def concatenate_cube_dict(cubedict):
    keys = list(cubedict.keys())
    start_cube = cubedict[keys[0]]
    number_cubes = len(keys)
    cube_list = iris.cube.CubeList(start_cube)
    for iterator in range(1, number_cubes):
        cube_list.append(cubedict[keys[iterator]])
    return cube_list.concatenate_cube()


# calculates baseline relative change values for specified quantiles
def ensemble_average_quantile_baseline(quantiles_to_calculate, models, ensemblename, results, ref_results,
                                       data_scenario,
                                       starting_years,
                                       length_timeperiod, reference_scenario, reference_start_year,
                                       reference_final_year, rolling_window=False):
    dict_to_plot = {}
    for quantile_to_calculate in quantiles_to_calculate:
        reference_data = {}
        data = {}
        baseline = {}

        quantile_dict = {}
        ref_exceedances = {}
        ref_mean = {}
        ref_relevance_mask = {}
        ref_frequency = {}
        ref_expected_snowfall = {}
        data_exceedances = {}
        data_mean = {}
        data_frequency = {}
        data_expected_snowfall = {}
        diff_frequency = {}
        diff_expected_snowfall = {}

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

        data_frequency_average = {}
        data_expected_snowfall_average = {}

        ref_mean_average = {}
        data_mean_average = {}
        diff_mean_average = {}
        mean_ratios = {}
        frequency_ratios = {}
        diff_frequency_average = {}
        diff_expected_snowfall_average = {}

        diff_expected_snowfall_average_relative = {}
        ref_ensemble_exceedances = {}
        data_ensemble_exceedances = {}
        for i_start_year in starting_years:
            start_year = i_start_year
            final_year = start_year + length_timeperiod - 1
            print(start_year)
            print(final_year)

            # add time coordinate to identify scenario decade from which the difference stems, to prepare timeplots of development
            datetime_year = datetime(int(start_year + length_timeperiod / 2 - 1), 1, 1, 0, 0, 0, 0).toordinal()
            datetime_start_year = datetime(int(start_year), 1, 1, 0, 0, 0, 0).toordinal()
            datetime_final_year = datetime(int(final_year), 1, 1, 0, 0, 0, 0).toordinal()
            time_coord = iris.coords.AuxCoord(datetime_year,
                                              long_name='scenario_year',
                                              units=cf_units.Unit('days since 1-01-01', calendar='proleptic_gregorian'),
                                              bounds=(datetime_start_year, datetime_final_year))

            for i_model in models:
                # TODO: balanced approach to get reference period with equally weighted rolling window decades, for now: just average the consecutive decades
                reference_data[i_model] = results_collection(ref_results[i_model],
                                                             reference_scenario, reference_start_year,
                                                             reference_final_year, length_timeperiod,
                                                             rolling_window=False)
                data[i_model] = results_collection(results[i_model], data_scenario, start_year, final_year,
                                                   length_timeperiod, rolling_window=rolling_window)
                baseline[i_model] = reference_data[i_model]['data'][quantile_key, quantile_to_calculate]
                ref_mean[i_model] = reference_data[i_model]['data']['mean']
                data_mean[i_model] = data[i_model]['data']['mean']

                quantile_dict[i_model] = data[i_model]['data'][quantile_key, quantile_to_calculate]

                ref_frequency[i_model] = reference_data[i_model]['data'][frequency_key, quantile_to_calculate]

                ref_expected_snowfall[i_model] = reference_data[i_model]['data'][
                    mean_exceedance_key, quantile_to_calculate]
                ref_expected_snowfall[i_model].units = baseline[i_model].units
                ref_expected_snowfall[i_model] = reference_data[i_model]['data'][
                                                     mean_exceedance_key, quantile_to_calculate] + baseline[i_model]

                data_frequency[i_model] = data[i_model]['data'][frequency_key, quantile_to_calculate]
                data_expected_snowfall[i_model] = data[i_model]['data'][mean_exceedance_key, quantile_to_calculate]
                data_expected_snowfall[i_model].units = baseline[i_model].units
                data_expected_snowfall[i_model] = data[i_model]['data'][
                                                      mean_exceedance_key, quantile_to_calculate] + baseline[i_model]

            unmasked_ref_mean = ensemble_average(models, ref_mean)
            unmasked_data_mean = ensemble_average(models, data_mean)

            # get ocean mask TODO: flexible per model
            maskfile = open('/p/tmp/quante/ExtremeEventEvaluation/ocean_mask_110', 'rb')
            ocean_mask = pickle.load(maskfile)
            ocean_masked_ref_mean_average = iris.util.mask_cube(unmasked_ref_mean, ocean_mask)
            np.ma.filled(ocean_masked_ref_mean_average, np.nan)
            # generate mask for values in decentiles of meanTODO: generalization might be useful
            # percentile_thresholds = [(0,100),(0,50),(33,100),(25,100),(0,10),(10,20),(30,40),(40,50),(50,100),(50,60),(60, 70),(70,80),(80,90),(90,100)]
            percentile_thresholds = [(0, 100)]
            for i_percentile_threshold in percentile_thresholds:
                unmasked_baseline = ensemble_average(models, baseline)
                unmasked_percentile = ensemble_average(models, quantile_dict)

                unmasked_ref_mean = ensemble_average(models, ref_mean)
                unmasked_data_mean = ensemble_average(models, data_mean)

                unmasked_ref_frequency = ensemble_average(models, ref_frequency)
                unmasked_data_frequency = ensemble_average(models, data_frequency)

                unmasked_ref_expected_snowfall = ensemble_average(models, ref_expected_snowfall)
                unmasked_data_expected_snowfall = ensemble_average(models, data_expected_snowfall)

                lower_bound = np.nanpercentile(ocean_masked_ref_mean_average.data, i_percentile_threshold[0])
                upper_bound = np.nanpercentile(ocean_masked_ref_mean_average.data, i_percentile_threshold[1])
                ref_relevance_mask = np.logical_or(ocean_masked_ref_mean_average.data <= lower_bound,
                                                   ocean_masked_ref_mean_average.data > upper_bound)

                ref_mean_average[i_start_year] = iris.util.mask_cube(unmasked_ref_mean, ref_relevance_mask)
                data_mean_average[i_start_year] = iris.util.mask_cube(unmasked_data_mean, ref_relevance_mask)

                mean_ratios[start_year] = data_mean_average[i_start_year] / ref_mean_average[i_start_year]
                mean_ratios[start_year].add_aux_coord(time_coord)

                quantile_average[i_start_year] = iris.util.mask_cube(unmasked_percentile, ref_relevance_mask)
                baseline_average[i_start_year] = iris.util.mask_cube(unmasked_baseline, ref_relevance_mask)

                quantile_baseline_ratio[i_start_year] = quantile_average[i_start_year] / baseline_average[i_start_year]
                quantile_baseline_ratio[i_start_year].add_aux_coord(time_coord)

                ref_frequency_average[i_start_year] = iris.util.mask_cube(unmasked_ref_frequency, ref_relevance_mask)
                data_frequency_average[i_start_year] = iris.util.mask_cube(unmasked_data_frequency, ref_relevance_mask)
                frequency_ratios[i_start_year] = data_frequency_average[i_start_year] / ref_frequency_average[
                    i_start_year]
                frequency_ratios[i_start_year].add_aux_coord(time_coord)

                ref_expected_snowfall_average[i_start_year] = iris.util.mask_cube(unmasked_ref_expected_snowfall,
                                                                                  ref_relevance_mask)
                data_expected_snowfall_average[i_start_year] = iris.util.mask_cube(unmasked_data_expected_snowfall,
                                                                                   ref_relevance_mask)

                expected_snowfall_ratio[i_start_year] = data_expected_snowfall_average[i_start_year] / \
                                                        ref_expected_snowfall_average[i_start_year]
                expected_snowfall_ratio[i_start_year].add_aux_coord(time_coord)

                dict_to_plot[quantile_to_calculate, i_start_year, 'mean', i_percentile_threshold] = mean_ratios[
                    i_start_year]
                dict_to_plot[quantile_to_calculate, i_start_year, 'frequency', i_percentile_threshold] = \
                    frequency_ratios[
                        i_start_year]
                dict_to_plot[quantile_to_calculate, i_start_year, 'percentile', i_percentile_threshold] = \
                    quantile_baseline_ratio[i_start_year]
                dict_to_plot[quantile_to_calculate, i_start_year, 'es', i_percentile_threshold] = \
                    expected_snowfall_ratio[
                        i_start_year]

                dict_to_plot[
                    quantile_to_calculate, i_start_year, 'frequency', i_percentile_threshold].var_name = 'frequency_' + str(
                    quantile_to_calculate) + '_' + str(i_percentile_threshold[0]) + "_" + str(i_percentile_threshold[1])
                dict_to_plot[
                    quantile_to_calculate, i_start_year, 'percentile', i_percentile_threshold].var_name = 'percentile_' + str(
                    quantile_to_calculate) + '_' + str(i_percentile_threshold[0]) + "_" + str(i_percentile_threshold[1])
                dict_to_plot[quantile_to_calculate, i_start_year, 'es', i_percentile_threshold].var_name = 'EES_' + str(
                    quantile_to_calculate) + '_' + str(i_percentile_threshold[0]) + "_" + str(i_percentile_threshold[1])
                dict_to_plot[
                    quantile_to_calculate, i_start_year, 'mean', i_percentile_threshold].var_name = 'mean_' + str(
                    i_percentile_threshold[0]) + "_" + str(i_percentile_threshold[1])
    return dict_to_plot


# calculates baseline relative change values for specified temperatures
def ensemble_average_temperature(temperatures, models, ensemblename, results, ref_results, data_scenario,
                                 starting_years,
                                 length_timeperiod, reference_scenario, reference_start_year,
                                 reference_final_year, rolling_window=False):
    dict_to_plot = {}
    for temperature in temperatures:
        reference_data = {}
        data = {}
        baseline = {}

        ref_days_between_temperature = {}
        ref_snow_between_temperature = {}
        ref_mean_snow_between_temperature = {}
        ref_mean_precipitation = {}

        data_days_between_temperature = {}
        data_snow_between_temperature = {}
        data_mean_snow_between_temperature = {}
        data_mean_precipitation = {}

        days_between_temperature_key = 'temperature_days'
        mean_snow_between_temperature_key = "mean_snow_between_temperature"
        snow_between_temperature_key = "snow_between_temperature"

        mean_precipitation_key = "mean_precipitation"

        ensemble_ref_days_between_temperature = {}
        ensemble_ref_snow_between_temperature = {}
        ensemble_ref_mean_snow_between_temperature = {}
        ensemble_ref_mean_precipitation = {}

        ensemble_data_days_between_temperature = {}
        ensemble_data_snow_between_temperature = {}
        ensemble_data_mean_snow_between_temperature = {}
        ensemble_data_mean_precipitation = {}

        days_between_temperature_ratio = {}
        mean_snow_between_temperature_ratio = {}
        snow_between_temperature_ratio = {}
        mean_precipitation_ratio = {}

        for i_start_year in starting_years:
            start_year = i_start_year
            final_year = start_year + length_timeperiod - 1
            print(start_year)
            print(final_year)
            for i_model in models:
                # TODO: balanced approach to get reference period with equally weighted rolling window decades, for now: just average the consecutive decades
                reference_data[i_model] = results_collection(ref_results[i_model],
                                                             reference_scenario, reference_start_year,
                                                             reference_final_year, length_timeperiod,
                                                             rolling_window=False)
                data[i_model] = results_collection(results[i_model], data_scenario, start_year, final_year,
                                                   length_timeperiod, rolling_window=rolling_window)

                ref_days_between_temperature[i_model] = reference_data[i_model]['data'][
                    days_between_temperature_key, str(temperature)]
                data_days_between_temperature[i_model] = data[i_model]['data'][
                    days_between_temperature_key, str(temperature)]

                ref_mean_snow_between_temperature[i_model] = reference_data[i_model]['data'][
                    mean_snow_between_temperature_key, str(temperature)]
                data_mean_snow_between_temperature[i_model] = data[i_model]['data'][
                    mean_snow_between_temperature_key, str(temperature)]

                ref_mean_precipitation[i_model] = reference_data[i_model]['data'][
                    mean_precipitation_key]
                data_mean_precipitation[i_model] = data[i_model]['data'][
                    mean_precipitation_key]

            # add time coordinate to identify scenario decade from which the difference stems, to prepare timeplots of development
            datetime_year = datetime(int(start_year + length_timeperiod / 2 - 1), 1, 1, 0, 0, 0, 0).toordinal()
            datetime_start_year = datetime(int(start_year), 1, 1, 0, 0, 0, 0).toordinal()
            datetime_final_year = datetime(int(final_year), 1, 1, 0, 0, 0, 0).toordinal()

            time_coord = iris.coords.AuxCoord(datetime_year,
                                              long_name='scenario_year',
                                              units=cf_units.Unit('days since 1-01-01', calendar='proleptic_gregorian'),
                                              bounds=(datetime_start_year, datetime_final_year))
            ensemble_data_days_between_temperature[i_start_year] = ensemble_average(models,
                                                                                    data_days_between_temperature)
            # ensemble_data_snow_between_temperature [i_start_year] = ensemble_average(models,data_snow_between_temperature)
            ensemble_data_mean_snow_between_temperature[i_start_year] = ensemble_average(models,
                                                                                         data_mean_snow_between_temperature)
            ensemble_data_mean_precipitation[i_start_year] = ensemble_average(models,
                                                                              data_mean_precipitation)

            ensemble_ref_days_between_temperature[i_start_year] = ensemble_average(models, ref_days_between_temperature)
            # ensemble_ref_snow_between_temperature [i_start_year] = ensemble_average(models, ref_snow_between_temperature)
            ensemble_ref_mean_snow_between_temperature[i_start_year] = ensemble_average(models,
                                                                                        ref_mean_snow_between_temperature)
            ensemble_ref_mean_precipitation[i_start_year] = ensemble_average(models,
                                                                             ref_mean_precipitation)

            days_between_temperature_ratio[start_year] = ensemble_data_days_between_temperature[i_start_year] / \
                                                         ensemble_ref_days_between_temperature[i_start_year]

            mean_snow_between_temperature_ratio[start_year] = ensemble_data_mean_snow_between_temperature[
                                                                  i_start_year] / \
                                                              ensemble_ref_mean_snow_between_temperature[i_start_year]

            days_between_temperature_ratio[start_year].add_aux_coord(time_coord)

            mean_precipitation_ratio[start_year] = ensemble_data_mean_precipitation[i_start_year] / \
                                                   ensemble_ref_mean_precipitation[i_start_year]

            mean_snow_between_temperature_ratio[start_year].add_aux_coord(time_coord)
            mean_precipitation_ratio[start_year].add_aux_coord(time_coord)

            # percentiles of snow between temperature

            dict_to_plot[str(temperature), i_start_year, 'days_between_temperature'] = days_between_temperature_ratio[
                i_start_year]
            dict_to_plot[str(temperature), i_start_year, 'mean_snow_between_temperature'] = \
                mean_snow_between_temperature_ratio[i_start_year]
            dict_to_plot[str(temperature), i_start_year, 'mean_precipitation'] = \
                mean_precipitation_ratio[i_start_year]

            lower_bound_temperature = np.min(temperature)
            upper_bound_temperature = np.max(temperature)

            dict_to_plot[
                str(
                    temperature), i_start_year, 'days_between_temperature'].var_name = 'days_between_temperature' + '_' + str(
                lower_bound_temperature) + '_' + str(upper_bound_temperature)
            dict_to_plot[
                str(
                    temperature), i_start_year, 'mean_snow_between_temperature'].var_name = 'mean_snow_between_temperature' + '_' + str(
                lower_bound_temperature) + '_' + str(upper_bound_temperature)
            dict_to_plot[
                str(temperature), i_start_year, 'mean_precipitation'].var_name = 'mean_precipitation' + '_' + str(
                lower_bound_temperature) + '_' + str(upper_bound_temperature)

            print(dict_to_plot.keys())
    return dict_to_plot


# initializes calculation of metrics for specified models, scenarios, scenario, area, starting years, timeperiods
def ensemble_plotting_average(models, data_scenarios, data_to_calculate, data_results, reference_results,
                              comparisonname, areaname,
                              starting_years, length_timeperiod, rolling_window=False, temperature=False):
    ensemble_results = {}
    ref_ensemble_results = {}
    for i_model in models:
        ensemble_results[i_model] = specify_results(data_results, i_model, comparisonname, areaname)
        ref_ensemble_results[i_model] = specify_results(reference_results, i_model, comparisonname, areaname)
    # adjust start years for rolling window:
    if rolling_window:
        min_start_year = np.min(starting_years)
        max_start_year = np.max(starting_years)
        stepsize = 1
        starting_years = range(min_start_year, max_start_year + 1, stepsize)

    reference_scenarios = ['historical']
    plotting_data = {}

    for i_scenario in data_scenarios:
        for i_reference_scenario in reference_scenarios:
            if not temperature:
                plotting_data[i_scenario, i_reference_scenario] = ensemble_average_quantile_baseline(
                    data_to_calculate, models, 'ensemble', ensemble_results, ref_ensemble_results, i_scenario,
                    starting_years, length_timeperiod,
                    i_reference_scenario, baseline_start, baseline_end, rolling_window=rolling_window)
            else:
                plotting_data[i_scenario, i_reference_scenario] = ensemble_average_temperature(
                    data_to_calculate, models, 'ensemble', ensemble_results, ref_ensemble_results, i_scenario,
                    starting_years, length_timeperiod,
                    i_reference_scenario, baseline_start, baseline_end, rolling_window=rolling_window)
    return plotting_data


# loads results from a list of pickle files
def load_results(filelist):
    partial_results = []
    for i_file in filelist:
        with open(i_file, 'rb') as stream:
            partial_results.append(pickle.load(stream))
    number_models = len(partial_results)
    all_results = dict(partial_results[0])
    for iterator in range(1, number_models):
        all_results.update(partial_results[iterator])
    return all_results


# generate settings parser

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

data_filelist = settings['files']
reference_files = settings['reference_files']
outputdir = settings['outputdir']
areanames = settings['areanames']
scenarios = settings['scenarios']
modellist = settings['modellist']

baseline_start = settings['baseline_start']
baseline_end = settings['baseline_end']

start_years = settings['start_years']
timeperiod_length = settings['timeperiod_length']
rolling_window = bool(settings['rolling_window'])

ensemble_members = settings['ensemble_members']
ensemble_name = settings['ensemble_name']
# if temperature data
temperature = bool(settings['temperature'])
if temperature:
    datapoints = settings['temperatures']
else:
    datapoints = settings['quantiles']
# set outputpath for plots etc
os.chdir(outputdir)

# load results
data_results = load_results(data_filelist)
ref_results = load_results(reference_files)

# loop over all time periods
for i_start_years in tqdm(start_years):
    cache_plotting_data = {}

    for i_areaname in tqdm(areanames):
        cache_plotting_data[i_areaname] = ensemble_plotting_average(modellist, scenarios, datapoints, data_results,
                                                                    ref_results, 'preindustrial',
                                                                    i_areaname, i_start_years, timeperiod_length,
                                                                    rolling_window=rolling_window,
                                                                    temperature=temperature)
    for i_key in tqdm(cache_plotting_data[areanames[0]].keys()):
        scenario = i_key[0]
        ref_scenario = i_key[1]
        global_plotting_data = {}
        for i_second_key in tqdm(cache_plotting_data[areanames[0]][i_key]):
            partial_data_cubelist = iris.cube.CubeList()
            for i_areaname in tqdm(areanames):
                partial_data_cubelist.append(cache_plotting_data[i_areaname][i_key][i_second_key])
            equalise_attributes(partial_data_cubelist)
            unify_time_units(partial_data_cubelist)
            for i_cube in partial_data_cubelist:
                i_cube.coord('longitude').bounds = None
                i_cube.coord('latitude').bounds = None

            global_plotting_data[i_second_key] = partial_data_cubelist.concatenate_cube(check_aux_coords=False)

        min_start_year = np.min(start_years)
        max_start_year = np.max(start_years)

        filename = 'snow_stats_' + str(ensemble_name) + "_" + str('global') + "_" + str('preindustrial') + "_" + str(
            scenario) + "_" + str(ref_scenario) + "_" + str(datapoints) + "_" + str(min_start_year) + "_" + str(
            max_start_year) + "_" + str(rolling_window)
        if temperature:
            filename = 'snow_temperature_stats_' + str(ensemble_name) + "_" + str('global') + "_" + str(
                'preindustrial') + "_" + str(scenario) + "_" + str(ref_scenario) + "_" + str(datapoints) + "_" + str(
                min_start_year) \
                       + "_" + str(max_start_year) + "_" + str(rolling_window)
            if (len(datapoints) > 5):
                filename = 'snow_temperature_stats_' + str(ensemble_name) + "_" + str('global') + "_" + str(
                    'preindustrial') + "_" + str(scenario) + "_" + str(ref_scenario) + "_" + str(
                    'many_temperatures') + "_" + str(
                    min_start_year) \
                           + "_" + str(max_start_year) + "_" + str(rolling_window)
        file = open(filename, 'wb')
        pickle.dump(global_plotting_data, file)
