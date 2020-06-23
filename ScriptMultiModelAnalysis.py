#!/usr/bin/env python

# Imports
import argparse
import multiprocessing
import os
import pickle
import warnings

import iris
import iris.analysis
import iris.coord_categorisation
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
from joblib import Parallel, delayed
from ruamel.yaml import ruamel
from tqdm import tqdm


# function to generate path to each individual model data


def filepath_generator(path, scenario, model):
    return path + '/' + scenario + '/output/data_' + model + '.yml'


# help functions to filter for specific variables, add seasons etc

def filter_cube(cube, variable):
    return cube.extract(variable)


# restrict on latitudes
def latitude_constraint(latitude, cube):
    latitudeConstraint: iris.Constraint = iris.Constraint(latitude=lambda v: latitude <= v)
    return cube.extract(latitudeConstraint)


def unify_concatenate(cubelist):
    unify_time_units(cubelist)
    equalise_attributes(cubelist)
    return cubelist.concatenate_cube()


# add area bounds to enable area weighted mean:
def add_lon_lat_bounds(cube):
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()


# generate extra cube of decade 2011 to 2020 (with scenario data)
def unify_current_decade(cubelist, model, scenario):
    cubes_to_concatenate = iris.cube.CubeList()
    cubes_to_concatenate.append(cubelist[model, 'historical'][17])
    cubes_to_concatenate.append(cubelist[model, scenario][0])

    cube_2011_2020 = unify_concatenate(cubes_to_concatenate)
    cubelist[model, 'historical'][17] = cube_2011_2020


# main method to calculate statistics of percentile exceedances from a list of cubes representing a model ensemble
def generate_quantile_exceedance_development(cubelist, baselinequantiles, quantiles, startyear, finalyear):
    results = {}
    cuberesults = {}
    number_of_cubes = len(cubelist)
    number_of_timepoints = 0
    for iterator_key in cubelist.keys():
        # calculate some metrics per cube from cubelist
        data = {}
        cube = cubelist[iterator_key]
        data_timeperiod = extract_dates(cube, startyear, finalyear)
        number_of_timepoints = data_timeperiod.data.shape[0]
        mean_data = data_timeperiod.collapsed('time', iris.analysis.MEAN)
        mean_data.var_name = 'mean_daily_snowfall'
        for i_quantile in quantiles:
            print('timepoints for model: ' + str(iterator_key) + " " + str(number_of_timepoints))

            # calculate exceedance of percentiles for each gridcell to use as thresholds:
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

            data['number_exceedances', i_quantile] = number_exceedances
            data['sum_exceedance', i_quantile] = sum_exceedances
        data['mean'] = mean_data
        cuberesults[iterator_key] = data
    # sum up results from separate model cubes into a single cube
    keys = list(cubelist.keys())
    for iterator_key in cuberesults[keys[0]].keys():
        results[iterator_key] = cuberesults[keys[0]][iterator_key]
        for i_cube in range(1, number_of_cubes):
            results[iterator_key] = results[iterator_key] + cuberesults[keys[i_cube]][iterator_key]

    # adjust average measures:
    results['mean'] = results['mean'] / number_of_cubes
    for i_quantile in quantiles:
        # percentile calculated from all data points in all cubes
        results['quantile', i_quantile] = percentile_from_cubedict(cubelist, i_quantile, startyear, finalyear)
        results['quantile_baseline', i_quantile] = baselinequantiles[i_quantile]
        results['mean_exceedance', i_quantile] = results['sum_exceedance', i_quantile] / results[
            'number_exceedances', i_quantile]
        results['mean_exceedance', i_quantile].var_name = 'mean_exceedance_' + str(i_quantile) + 'percentile'
        # to check how much changing event frequency influences expected snowfall,
        # calculate "mean" based on expected number of events
        expected_exceedances = number_of_timepoints * (100 - i_quantile) / 100 * number_of_cubes
        results['baseline_relative_mean_exceedance', i_quantile] \
            = results['sum_exceedance', i_quantile] / expected_exceedances
        results['baseline_relative_mean_exceedance', i_quantile].var_name \
            = 'baseline_normed_mean_exceedance_' + str(i_quantile) + 'percentile'
        # calculate relative contribuition of each model to exceedance count:
        for iterator_model in cubelist.keys():
            results['exceedance_number_contribuition', i_quantile, iterator_model] = cuberesults[iterator_model][
                                                                                         'number_exceedances', i_quantile] / \
                                                                                     results[
                                                                                         'number_exceedances', i_quantile]
            results['exceedance_mean_contribuition', i_quantile, iterator_model] = cuberesults[iterator_model][
                                                                                       'sum_exceedance', i_quantile] / \
                                                                                   results['sum_exceedance', i_quantile]
    return results


def calculate_quantile_exceedance_measure(historical_cubelist, ssp_cubelist, ssp_scenario, baseline_quantiles,
                                          quantiles,
                                          number_of_years_to_compare, number_of_timeperiods, historical_start,
                                          ssp_start, historical=True, rolling_window=True):
    historical_start_list = [historical_start]
    ssp_start_list = [ssp_start]

    if rolling_window:
        total_timeperiod = number_of_years_to_compare * number_of_timeperiods

        historical_final_start_year = historical_start + total_timeperiod - number_of_years_to_compare
        ssp_final_start_year = ssp_start + total_timeperiod - number_of_years_to_compare

        for year in range(historical_start + 1, historical_final_start_year + 1, 1):
            historical_start_list.append(year)
        for year in range(ssp_start + 1, ssp_final_start_year + 1, 1):
            ssp_start_list.append(year)

    else:
        for index in range(1, number_of_timeperiods):
            historical_start_list.append(historical_start_list[index - 1] + number_of_years_to_compare)

            ssp_start_list.append(ssp_start_list[index - 1] + number_of_years_to_compare)

    num_cores = multiprocessing.cpu_count()
    if (ensemble_boolean):
        num_cores = int(num_cores / 8)
    historical_starts = tqdm(historical_start_list)
    data = {}

    if historical:
        # multi - processing for different start years
        results_cache = (Parallel(n_jobs=num_cores)(
            (delayed(percentile_dict_entry)(i_start, "historical", historical_cubelist, baseline_quantiles,
                                            quantiles, number_of_years_to_compare) for i_start in historical_starts)))
        for i_result in results_cache:
            data[i_result[0]] = i_result[1]

    ssp_starts = tqdm(ssp_start_list)
    # multi - processing for different start years
    results_cache = (Parallel(n_jobs=num_cores)(
        (delayed(percentile_dict_entry)(i_start, ssp_scenario, ssp_cubelist, baseline_quantiles, quantiles,
                                        number_of_years_to_compare) for i_start in ssp_starts)))
    for i_result in results_cache:
        data[i_result[0]] = i_result[1]
    return data


def percentile_dict_entry(i_historical_start, key, cubelist, baseline_percentiles, percentiles_to_calculate,
                          timeperiod_length):
    i_historical_end = i_historical_start + timeperiod_length - 1
    result = generate_quantile_exceedance_development(
        cubelist, baseline_percentiles, percentiles_to_calculate, i_historical_start, i_historical_end)

    return (key, i_historical_start, i_historical_end), result


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


def country_filter(cubedict, country_box):
    dict_keys = cubedict.keys()
    country_cubes = {}
    for iterator_key in dict_keys:
        country_cubes[iterator_key] = country_cubelist(cubedict[iterator_key], country_box)
    return country_cubes


def extract_dates(cube, startyear, finalyear, year=False):
    # filter for certain years (inclusive for start and final year)
    # add auxilary year coordinate
    try:
        cube.remove_coord('year')
    except iris.exceptions.CoordinateNotFoundError:
        pass
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    year_constraint = iris.Constraint(year=lambda t: startyear <= t <= finalyear)
    limited_cube = cube.extract(year_constraint)
    return limited_cube


def percentile_from_cubedict(cubedict, percentile, startyear, finalyear, year=False, nan_percentile=False):
    # extract data from cubes and merge
    keys = list(cubedict.keys())
    dataarray = extract_dates(cubedict[keys[0]], startyear, finalyear, year=year).data
    for i in range(1, len(cubedict)):
        dataarray = np.concatenate((dataarray, extract_dates(cubedict[keys[i]], startyear, finalyear, year=year).data),
                                   axis=0)

    if (nan_percentile):
        percentile_data = percentile_data = np.nanpercentile(dataarray, percentile, axis=0)
    else:
        percentile_data = np.percentile(dataarray, percentile, axis=0)
    # calculate percentile

    # construct percentile cube to return
    percentile_cube = extract_dates(cubedict[keys[0]], startyear, finalyear).collapsed('time', iris.analysis.PERCENTILE,
                                                                                       percent=percentile)
    # replace data with data from all cubes
    percentile_cube.data = percentile_data

    return percentile_cube


def comparison_threshold(modellist, basic_cubelist, ssp_scenario, number_of_decades, area_box, start_historical,
                         start_ssp, percentiles_to_calculate,
                         historical=True, rolling_window=True):
    # filter data for country box:
    area_data = country_filter(basic_cubelist, area_box)

    # merge model cubes into ensemble cube

    historical_ensemble_cubelist = {}
    ssp_ensemble_cubelist = {}
    baseline_ensemble_cubelist = {}
    for iterator_model in modellist:
        historical_ensemble_cubelist[iterator_model] = (area_data[iterator_model, 'historical'])
        ssp_ensemble_cubelist[iterator_model] = (area_data[iterator_model, ssp_scenario])
        baseline_ensemble_cubelist[iterator_model] = (area_data[iterator_model, 'historical'])

    baseline_percentiles = {}
    for i_percentile in percentiles_to_calculate:
        baseline_percentiles[i_percentile] = percentile_from_cubedict(baseline_ensemble_cubelist, i_percentile,
                                                                      historical_start,
                                                                      baseline_end)
    return calculate_quantile_exceedance_measure(historical_ensemble_cubelist, ssp_ensemble_cubelist, ssp_scenario,
                                                 baseline_percentiles, percentiles_to_calculate,
                                                 10, number_of_decades, start_historical, start_ssp,
                                                 historical=historical, rolling_window=rolling_window)


def multi_region_threshold_analysis_preindustrial(cubelist, modellist, areas_to_analyse, ssp_scenario, start_ssp,
                                                  start_historical,
                                                  number_of_decades, percentiles_to_calculate,
                                                  historical=True, rolling_window=True, ensemble=False,
                                                  ensemblename=''):
    identifier = modellist[0]
    if ensemble:
        identifier = ensemblename

    for i_area in tqdm(areas_to_analyse.keys()):
        results = {(identifier, 'preindustrial', i_area): comparison_threshold(modellist, cubelist, ssp_scenario,
                                                                               number_of_decades,
                                                                               areas_to_analyse[i_area],
                                                                               start_historical,
                                                                               start_ssp, percentiles_to_calculate,
                                                                               historical=historical,
                                                                               rolling_window=rolling_window)}
        filename = str(identifier) + '_' + str(i_area) + '_preindustrial_baseline_hist_from_' + str(
            start_historical) + "_" + ssp_scenario + '_from_' + str(
            start_ssp) + '_' + str(number_of_decades)

        file = open(filename, 'wb')
        pickle.dump(results, file)


# method to load stuff with lower bound for latitude from a dict of files giving *.nc data
def load_from_nc(filedict, variablename, latitude_lower_bound):
    filelist = []
    with open(filedict, 'r') as stream:
        try:
            filelist = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    loaded_cubes = iris.cube.CubeList()
    for i_file in filelist:
        cube = (latitude_constraint(latitude_lower_bound, iris.load(i_file, variablename)))

        if len(cube) == 1:
            loaded_cubes.append(cube[0])
    return loaded_cubes


def generate_temperature_snow_development(cubelist_temperature, cubelist_snow, cubelist_precipitation,
                                          threshold_temperatures, startyear, finalyear):
    results = {}
    cuberesults = {}
    number_of_cubes = len(cubelist_temperature)
    model_dict_snow_thresholded = {}
    temperature_dict_snow_thresholded = {}
    for model_key in cubelist_temperature.keys():
        # calculate some metrics per cube from cubelist
        data = {}
        temperature_cube = cubelist_temperature[model_key]
        snow_cube = cubelist_snow[model_key]
        precipitation_cube = cubelist_precipitation[model_key]

        temperature_timeperiod = extract_dates(temperature_cube, startyear, finalyear)
        snow_timeperiod = extract_dates(snow_cube, startyear, finalyear)
        precipitation_timeperiod = extract_dates(precipitation_cube, startyear, finalyear)

        number_of_timepoints_temp = temperature_timeperiod.data.shape[0]
        print('timepoints for model: ' + str(model_key) + " " + str(number_of_timepoints_temp))

        # calculate indicator of threshold temperature:

        temperature_data = temperature_timeperiod.data

        for i_temperature_bin in threshold_temperatures:
            lower_bound_temperature = float(i_temperature_bin[0])
            upper_bound_temperature = float(i_temperature_bin[1])

            lower_bound_temperature_indicator_data = np.int8(lower_bound_temperature <= temperature_data)
            upper_bound_temperature_indicator_data = np.int8(temperature_data < upper_bound_temperature)
            sum_of_indicators = lower_bound_temperature_indicator_data + upper_bound_temperature_indicator_data
            temperature_indicator_data = np.int8(sum_of_indicators == 2)

            temperature_below_threshold = temperature_timeperiod.copy(data=temperature_indicator_data)
            temperature_below_threshold.var_name = (
                    'indicator_temperature_between_' + str(lower_bound_temperature) + '_and_' + str(
                upper_bound_temperature) + 'degree_K')

            temperature_days = temperature_below_threshold.collapsed('time', iris.analysis.COUNT,
                                                                     function=lambda x: x > 0)
            temperature_days.var_name = ('days_temperature_between_' + str(lower_bound_temperature) + '_and_' + str(
                upper_bound_temperature) + 'degree_K')
            temperature_days.remove_coord('time')

            # calculate snowfall below temperature

            snow_between_temperature = snow_timeperiod * temperature_below_threshold
            snow_between_temperature.var_name = ('prsn_between_' + str(lower_bound_temperature) + '_and_' + str(
                upper_bound_temperature) + 'degree_K')

            snow_data_percentile = snow_timeperiod.data
            snow_data_percentile[temperature_indicator_data == 0] = np.nan

            model_dict_snow_thresholded[str(i_temperature_bin), model_key] = snow_between_temperature.copy(
                data=snow_data_percentile)

            total_snow_below_temperature = snow_between_temperature.collapsed('time', iris.analysis.SUM)

            mean_snow_below_temperature = total_snow_below_temperature / temperature_days
            mean_snow_below_temperature.var_name = ('mean_prsn_between_' + str(lower_bound_temperature) + '_and_' + str(
                upper_bound_temperature) + 'degree_K')

            data['temperature_days', str(i_temperature_bin)] = temperature_days
            # data['snow_between_temperature', str(i_temperature_bin)] = snow_between_temperature
            data['mean_snow_between_temperature', str(i_temperature_bin)] = mean_snow_below_temperature

        # get precipitation stats for comparison
        mean_precipitation = precipitation_timeperiod.collapsed('time', iris.analysis.MEAN)
        mean_precipitation.var_name = ('mean_pr')
        data['mean_precipitation'] = mean_precipitation
        cuberesults[model_key] = data
    # sum up results from separate model cubes into a single cube
    model_keys = list(cubelist_temperature.keys())

    for i_temperature_bin in threshold_temperatures:
        only_model_dict_snow_thresholded = {}
        for model_key in model_keys:
            only_model_dict_snow_thresholded[model_key] = model_dict_snow_thresholded[str(i_temperature_bin), model_key]
        temperature_dict_snow_thresholded[str(i_temperature_bin)] = only_model_dict_snow_thresholded
    for data_key in cuberesults[model_keys[0]].keys():
        results_data = cuberesults[model_keys[0]][data_key].data
        for i_model in range(1, number_of_cubes):
            results_data = results_data + cuberesults[model_keys[i_model]][data_key].data
        # adjust to average measures:
        results_data = results_data / number_of_cubes
        results[data_key] = cuberesults[model_keys[0]][data_key].copy(data=results_data)
    # transform snow_between temp dict to temperature bin wise results:

    # NB: just using percentiles in a crude way directly from settingsfile, to get the same as used for EEI of snow etc.
    for i_percentile in percentiles:
        results['precipitation_percentile', i_percentile] = percentile_from_cubedict(cubelist_precipitation,
                                                                                     i_percentile, startyear,
                                                                                     finalyear)
        for i_temperature_bin in threshold_temperatures:
            results[
                'snow_between_temperature_percentile', str(i_temperature_bin), i_percentile] = percentile_from_cubedict(
                temperature_dict_snow_thresholded[str(i_temperature_bin)],
                i_percentile, startyear,
                finalyear, nan_percentile=True)

    return results


# temperature analysis function
def calculate_snowfall_below_temperature_measures(historical_ensemble_cubelist_temperature,
                                                  historical_ensemble_cubelist_snow,
                                                  historical_ensemble_cubelist_precipitation,
                                                  ssp_ensemble_cubelist_temperature, ssp_ensemble_cubelist_snow,
                                                  ssp_ensemble_cubelist_precipitation,
                                                  scenario,
                                                  threshold_temperatures,
                                                  number_of_years_to_compare,
                                                  number_of_timeperiods, historical_start, scenario_start, historical,
                                                  rolling_window):
    scenario_start_list = [scenario_start]
    historical_start_list = [historical_start]
    if rolling_window:
        total_timeperiod = number_of_years_to_compare * number_of_timeperiods

        historical_final_start_year = historical_start + total_timeperiod - number_of_years_to_compare
        ssp_final_start_year = scenario_start + total_timeperiod - number_of_years_to_compare

        for year in range(historical_start + 1, historical_final_start_year + 1, 1):
            historical_start_list.append(year)
        for year in range(scenario_start + 1, ssp_final_start_year + 1, 1):
            scenario_start_list.append(year)

    else:
        for index in range(1, number_of_timeperiods):
            historical_start_list.append(historical_start_list[index - 1] + number_of_years_to_compare)

            scenario_start_list.append(scenario_start_list[index - 1] + number_of_years_to_compare)

    num_cores = int(multiprocessing.cpu_count() / 4)
    historical_starts = tqdm(historical_start_list)
    data = {}

    if historical:
        # multi - processing for different start years
        results_cache = (Parallel(n_jobs=num_cores)(
            (delayed(temperature_dict_entry)(i_start, "historical", historical_ensemble_cubelist_temperature,
                                             historical_ensemble_cubelist_snow,
                                             historical_ensemble_cubelist_precipitation,
                                             threshold_temperatures, number_of_years_to_compare) for i_start in
             historical_starts)))
        for i_result in results_cache:
            data[i_result[0]] = i_result[1]

    scenario_starts = tqdm(scenario_start_list)
    # results_cache = []
    # # for i_start in scenario_starts:
    # #     results_cache.append(temperature_dict_entry(i_start, scenario, ssp_ensemble_cubelist_temperature,
    # #                                       ssp_ensemble_cubelist_snow,
    # #                                       ssp_ensemble_cubelist_precipitation,
    # #                                       threshold_temperatures,
    # #                                       number_of_years_to_compare))
    # multi - processing for different start years

    results_cache = (Parallel(n_jobs=num_cores)(
        (delayed(temperature_dict_entry)(i_start, scenario, ssp_ensemble_cubelist_temperature,
                                         ssp_ensemble_cubelist_snow,
                                         ssp_ensemble_cubelist_precipitation,
                                         threshold_temperatures,
                                         number_of_years_to_compare) for i_start in scenario_starts)))
    for i_result in results_cache:
        data[i_result[0]] = i_result[1]
    return data


def temperature_dict_entry(start, key, cubelist_temperature, cubelist_snow, cubelist_precipitation,
                           threshold_temperatures,
                           timeperiod_length):
    end = start + timeperiod_length - 1
    result = generate_temperature_snow_development(
        cubelist_temperature, cubelist_snow, cubelist_precipitation, threshold_temperatures,
        start, end)

    return (key, start, end), result


def analyse_temperature_snow(modellist, temperature_cubes, snow_cubes, precipitation_cubes, scenario, number_of_decades,
                             area,
                             historical_start,
                             scenario_start, threshold_temperatures,
                             historical=True,
                             rolling_window=True):
    # filter data for country box:
    area_data_temperature = country_filter(temperature_cubes, area)
    area_data_snow = country_filter(snow_cubes, area)
    area_data_precipitation = country_filter(precipitation_cubes, area)
    # merge model cubes into ensemble cube
    historical_ensemble_cubelist_temperature = {}
    ssp_ensemble_cubelist_temperature = {}

    historical_ensemble_cubelist_snow = {}
    ssp_ensemble_cubelist_snow = {}

    historical_ensemble_cubelist_precipitation = {}
    ssp_ensemble_cubelist_precipitation = {}

    for iterator_model in modellist:
        historical_ensemble_cubelist_temperature[iterator_model] = (area_data_temperature[iterator_model, 'historical'])
        ssp_ensemble_cubelist_temperature[iterator_model] = (area_data_temperature[iterator_model, scenario])

        historical_ensemble_cubelist_snow[iterator_model] = (area_data_snow[iterator_model, 'historical'])
        ssp_ensemble_cubelist_snow[iterator_model] = (area_data_snow[iterator_model, scenario])

        historical_ensemble_cubelist_precipitation[iterator_model] = (
            area_data_precipitation[iterator_model, 'historical'])
        ssp_ensemble_cubelist_precipitation[iterator_model] = (area_data_precipitation[iterator_model, scenario])

    return calculate_snowfall_below_temperature_measures(historical_ensemble_cubelist_temperature,
                                                         historical_ensemble_cubelist_snow,
                                                         historical_ensemble_cubelist_precipitation,
                                                         ssp_ensemble_cubelist_temperature, ssp_ensemble_cubelist_snow,
                                                         ssp_ensemble_cubelist_precipitation,
                                                         scenario
                                                         , threshold_temperatures, 10, number_of_decades,
                                                         historical_start, scenario_start,
                                                         historical=historical, rolling_window=rolling_window)


def temperature_analysis(temperature_cubes, snow_cubes, precipitation_cubes, modellist, areas_to_analyse, scenario,
                         scenario_start,
                         historical_start, number_of_decades, threshold_temperatures, historical=True,
                         rolling_window=True):
    for i_area in tqdm(areas_to_analyse.keys()):
        results = {
            ('ensemble', 'preindustrial', i_area): analyse_temperature_snow(modellist, temperature_cubes, snow_cubes,
                                                                            precipitation_cubes,
                                                                            scenario,
                                                                            number_of_decades,
                                                                            areas_to_analyse[i_area],
                                                                            historical_start,
                                                                            scenario_start, threshold_temperatures,
                                                                            historical=historical,
                                                                            rolling_window=rolling_window)}
        filename = str('temperature_snow_ensemble') + '_' + str(i_area) + '_preindustrial_baseline_hist_from_' + str(
            historical_start) + "_" + scenario + '_from_' + str(scenario_start) + '_' + str(number_of_decades)

        file = open(filename, 'wb')
        pickle.dump(results, file)


# TODO: more elegant parametrization
def simulation_runner(models):
    if (temperature):
        if (baseline == 1):
            # baseline
            temperature_analysis(tas_cubes, prsn_cubes, pr_cubes, models, arealist, 'historical', historical_start,
                                 historical_start,
                                 baseline_decades, threshold_temperatures, historical=False,
                                 rolling_window=rolling_window)
        if (full_historical == 1):
            # generate full historical data for comparison
            temperature_analysis(tas_cubes, prsn_cubes, pr_cubes, models, arealist, 'historical', 1931, 1851, 9,
                                 threshold_temperatures, rolling_window=rolling_window)
        if (scenario_analysis == 1):

            for i_scenario in tqdm(ssp_scenarios):
                temperature_analysis(tas_cubes, prsn_cubes, pr_cubes, models, arealist, i_scenario, 2021, 1851, 8,
                                     threshold_temperatures,
                                     historical=False, rolling_window=rolling_window)



    else:
        if (baseline == 1):
            multi_region_threshold_analysis_preindustrial(prsn_cubes, models, arealist, 'historical', historical_start,
                                                          historical_start,
                                                          baseline_decades, percentiles, historical=False,
                                                          rolling_window=rolling_window, ensemble=ensemble_boolean,
                                                          ensemblename=identifier_ensemble)
        if (scenario_analysis == 1):
            # generate ssp data (without historical leg, since done in next step
            for i_scenario in tqdm(ssp_scenarios):
                multi_region_threshold_analysis_preindustrial(prsn_cubes, models, arealist, i_scenario, 2021, 1851, 8,
                                                              percentiles,
                                                              historical=False, rolling_window=rolling_window,
                                                              ensemble=ensemble_boolean,
                                                              ensemblename=identifier_ensemble)
        if (full_historical == 1):
            # generate full historical data for comparison
            multi_region_threshold_analysis_preindustrial(prsn_cubes, models, arealist, 'historical', 1931, 1851, 9,
                                                          percentiles, rolling_window=rolling_window,
                                                          ensemble=ensemble_boolean, ensemblename=identifier_ensemble)


# actual script, starting with settingsparser

parser = argparse.ArgumentParser(
    description="Calculate some analysis metrics regarding extreme snowfall on specified data")

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

temperature = bool(settings['temperature'])
threshold_temperatures = settings['threshold_temperatures']
models = settings['models']

outputdir = settings['outputdir']
scenarios = settings['scenarios']
ssp_scenarios = settings['ssp_scenarios']
percentiles = settings['percentiles']
historical_start = int(settings['baseline_start'])
baseline_end = int(settings['baseline_end'])
baseline = int(settings['baseline'])
full_historical = int(settings['full_historical'])
scenario_analysis = int(settings['scenario_analysis'])
rolling_window = bool(settings['rolling_window'])

ensemble_boolean = bool(settings['ensemble'])
identifier_ensemble = settings['identifier_ensemble']

data_dictionary = {}

# filelists to be given sorted by scenarios and models
filelist_tas = settings['files']
filelist_prsn = settings['files']
filelist_pr = settings['files']

# modified data import directly from isimip *.nc files
if (temperature):

    # load data
    tas_cubes = {}
    prsn_cubes = {}
    pr_cubes = {}

    for i_scenario in scenarios:
        for i_model in models:
            tas_cubes[i_model, i_scenario] = load_from_nc(filelist_tas[i_scenario][i_model], 'air_temperature', 30)
            prsn_cubes[i_model, i_scenario] = load_from_nc(filelist_prsn[i_scenario][i_model], 'snowfall_flux', 30)
            pr_cubes[i_model, i_scenario] = load_from_nc(filelist_pr[i_scenario][i_model], 'precipitation_flux', 30)
    # for i_model in models:
    # unify_current_decade(tas_cubes, i_model, 'ssp585')
    # unify_current_decade(prsn_cubes, i_model, 'ssp585')

else:
    prsn_cubes = {}

    for i_scenario in scenarios:
        for i_model in models:
            prsn_cubes[i_model, i_scenario] = load_from_nc(filelist_prsn[i_scenario][i_model], 'snowfall_flux', 30)
    # TODO: flexible extension of historical data with respective ssp, but should not make a big difference
    # for i_model in models:
    #     unify_current_decade(prsn_cubes, i_model, 'ssp585')

# ignore warnings
warnings.simplefilter("ignore")

os.chdir(outputdir)

# define 4 boxes for northern hemisphere to reduce memory requirement of each run

lat_lower_bound = 30
lat_upper_bound = 90

northern_hemisphere = (0, lat_lower_bound, 360, lat_upper_bound)
northern_hemisphere_first_quarter = (0, lat_lower_bound, 90, lat_upper_bound)
northern_hemisphere_second_quarter = (90, lat_lower_bound, 180, lat_upper_bound)
northern_hemisphere_third_quarter = (-180, lat_lower_bound, -90, lat_upper_bound)
northern_hemisphere_fourth_quarter = (-90, lat_lower_bound, 0, lat_upper_bound)

arealist = {'NORTHERN_HEMISPHERE': northern_hemisphere}

# '_1': northern_hemisphere_first_quarter,
# 'NORTHERN_HEMISPHERE_2': northern_hemisphere_second_quarter,
# 'NORTHERN_HEMISPHERE_3': northern_hemisphere_third_quarter,
# 'NORTHERN_HEMISPHERE_4': northern_hemisphere_fourth_quarter}

# generate reference results for baseline:
baseline_decades = int((baseline_end - historical_start + 1) / 10)

# run on all models if ensemble

if (ensemble_boolean):

    simulation_runner(models)

else:
    for i_model in models:
        print([i_model])
        simulation_runner([i_model])
