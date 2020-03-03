#!/usr/bin/env python

# script to analyse data for a certain specified grid point (Lat,Lon) from output of snow_processing:

# imports

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iris
import iris.coord_categorisation
import iris.plot as iplt

import numpy as np
from iris.analysis import Aggregator
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import rolling_window, unify_time_units
from ruamel.yaml import ruamel


# function definition,
# Based on code adjusted from Scitools Iris,
# https://scitools.org.uk/iris/docs/latest/examples/General/custom_aggregation.html


# check for CONSECUTIVLY REPEATED exceedance of threshholds:
# adjusted code example from https://scitools.org.uk/iris/docs/latest/examples/General/custom_aggregation.html


# Define a function to perform the custom statistical operation.
# Note: in order to meet the requirements of iris.analysis.Aggregator, it must
# do the calculation over an arbitrary (given) data axis.
def count_spells(data, threshold, axis, spell_length):
    """
    Function to calculate the number of points in a sequence where the value
    has exceeded a threshold value for at least a certain number of timepoints.

    Generalised to operate on multiple time sequences arranged on a specific
    axis of a multidimensional array.

    Args:

    * data (array):
        raw data to be compared with value threshold.

    * threshold (float):
        threshold point for 'significant' datapoints.

    * axis (int):
        number of the array dimension mapping the time sequences.
        (Can also be negative, e.g. '-1' means last dimension)

    * spell_length (int):
        number of consecutive times at which value > threshold to "count".

    """
    if axis < 0:
        # just cope with negative axis numbers
        axis += data.ndim
    # Threshold the data to find the 'significant' points.
    data_hits = data > threshold
    # Make an array with data values "windowed" along the time axis.
    hit_windows = rolling_window(data_hits, window=spell_length, axis=axis)
    # Find the windows "full of True-s" (along the added 'window axis').
    full_windows = np.all(hit_windows, axis=axis + 1)
    # Count points fulfilling the condition (along the time axis).
    spell_point_counts = np.sum(full_windows, axis=axis, dtype=int)
    return spell_point_counts


# check for CUMULATIVE exceedance of threshhold:
# adjusted code example from https://scitools.org.uk/iris/docs/latest/examples/General/custom_aggregation.html


def count_accumulated_exceedance(data, threshold, axis, accumulation_length):
    """
    Function to calculate the number of points in a sequence where the cumulative value
    has exceeded a threshold within a certain number of timepoints (accumulation_length).

    Generalised to operate on multiple time sequences arranged on a specific
    axis of a multidimensional array.

    Args:

    * data (array):
        raw data to be compared with value threshold.

    * threshold (float):
        threshold point for accumulation over datapoints.

    * axis (int):
        number of the array dimension mapping the time sequences.
        (Can also be negative, e.g. '-1' means last dimension)

    * accumulation_length (int):
        number of timepoint for which the average shall be calculated.

    """
    if axis < 0:
        # just cope with negative axis numbers
        axis += data.ndim

    # Make an array with data values "windowed" along the time axis.
    accumulation_windows = rolling_window(data, window=accumulation_length, axis=axis)
    # Find the windows exceeding the accumulation threshold (along the added 'window axis').
    exceeding_windows = np.sum(accumulation_windows, axis=axis + 1) > threshold
    # Count points fulfilling the condition (along the time axis).
    exceeding_count = np.sum(exceeding_windows, axis=axis, dtype=int)
    return exceeding_count


def calculate_data_above_threshold_for_x_days(data, threshold, numberOfDays, relativeValue):
    # Make an aggregator from the user function.
    SPELL_COUNT = Aggregator('spell_count',
                             count_spells,
                             units_func=lambda units: 1)

    # Calculate the statistic
    data_above_threshold = data.collapsed('time', SPELL_COUNT,
                                          threshold=threshold,
                                          spell_length=numberOfDays)
    # TODO: customize label
    data_above_threshold.rename(
        ' Percentage of days with consecutive ' + str(numberOfDays) + '-day snow falls above ' + str(
            threshold) + 'mm in timeperiod')

    # relative result
    if (relativeValue == True):
        total_days = data.coords("time")[0].shape[0]
        data_above_threshold.data = data_above_threshold.data / numberOfDays * 100
        data_above_threshold.rename(
            ' Percentage of days with consecutive ' + str(numberOfDays) + '-day snow falls above ' + str(
                threshold) + 'mm in timeperiod')
    else:
        data_above_threshold.rename(' Number of days with ' + str(numberOfDays) + '-day snow falls above ' + str(
            threshold) + 'mm for each day in timeperiod')

    return data_above_threshold


def calculate_data_above_threshold_in_x_days(data, threshold, numberOfDays, relativeValue):
    # Make an aggregator from the user function.
    ACCUMULATION_COUNT = Aggregator('accumulation_exceedance_count',
                                    count_accumulated_exceedance,
                                    units_func=lambda units: 1)

    # Calculate the statistic
    data_above_threshold = data.collapsed('time', ACCUMULATION_COUNT,
                                          threshold=threshold,
                                          accumulation_length=numberOfDays)

    # relative result
    if (relativeValue == True):
        data_above_threshold.rename(
            ' Percentage of days with accumulated snow  falls in ' + str(numberOfDays) + ' days above ' + str(
                threshold) + 'mm')
        total_days = data.coords("time")[0].shape[0]
        data_above_threshold.data = data_above_threshold.data / numberOfDays * 100
    else:
        data_above_threshold.rename(
            ' Number of days with accumulated snow falls in ' + str(numberOfDays) + ' days above ' + str(
                threshold) + 'mm')
    return data_above_threshold


def data_analysis_accumulated_threshold(filepath, depth_threshold, time_threshold):
    # retrieve data
    data = load_data_from_netcdf(filepath)
    cube_daily = get_cube_from_cubelist(data, 'approx_fresh_daily_snow_height')
    analysed_data = calculate_data_above_threshold_in_x_days(cube_daily, depth_threshold, time_threshold, False)

    return analysed_data  # returns analysed data


def data_analysis_each_day_threshold(filepath, depth_threshold, time_threshold):
    # retrieve data
    data = load_data_from_netcdf(filepath)
    cube_daily = get_cube_from_cubelist(data, 'approx_fresh_daily_snow_height')
    analysed_data = calculate_data_above_threshold_for_x_days(cube_daily, depth_threshold, time_threshold, False)

    return analysed_data  # returns analysed data


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def get_cube_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist[0]


# model identification based on structure of filename: str("output_" + i_model + "_" + timeindex + ".nc") from Pathname Collection Helper
def model_identification_re(filepath):
    model = re.search('(.*/)(output_)(.*)(_)(\d{4})(\d{4})(.nc)$', filepath)
    model_properties = {}
    if (model):

        model_properties["name"] = model.group(3)
        model_properties["start"] = int(model.group(5))
        model_properties["end"] = int(model.group(6))
    else:
        model_properties["name"] = "model_not_identified"
        model_properties["start"] = 0000
        model_properties["end"] = 0000
    return model_properties


# save data to nc file in outputlocation
def safe_data_nc(data, analysisidentifier):
    iris.save(data, outputdir + '/data_analysis_' + analysisidentifier + '.nc')


# function to concatenate cube for specified variable
def concatenate_variables(data, variable):
    # concatenate all input files in one cube (NB: assumes data unique wrt to time)

    # import all single data files into one cubelist

    complete_data_cube = iris.load(data)

    # filter for variable of interest

    filtered_data_cube = complete_data_cube.extract(variable)

    # equalise attributes of cubes and unify time units

    equalise_attributes(filtered_data_cube)

    unify_time_units(filtered_data_cube)

    # concatenate cubes

    return filtered_data_cube.concatenate_cube()


# argument parser definition
parser = argparse.ArgumentParser(description="Calculate some analysis metrics on specified data files")
# path to *.yml file with settings to be used
parser.add_argument(
    "--areasettings",
    nargs="+"
    , type=str,
    help="Path to the area analysis settingsfiles"
)

parser.add_argument(
    "--globalsettings",
    nargs="+"
    , type=str,
    help="Path to the global analysis settingsfiles"
)

parser.add_argument(
    "--data",
    nargs="+"
    , type=str,
    help="YML list of datafile(s)"
)

parser.add_argument(
    "--resolution"
    , type=str,
    help="CMIP_HR, CMIP_LR or halfDegree resolution for areal analysis"
)

parser.add_argument(
    "--latitude_lower_bound"
    , type=int,
    help="lower_bound_for_latitudes_to_analyze"
)

parser.add_argument(
    "--latitude_upper_bound"
    , type=int,
    default=90,
    help="upper_bound_for_latitudes_to_analyze"
)

parser.add_argument(
    "--settings"
    , type=str,
    help="argument to submit settingsfile with all arguments included"
)

args = parser.parse_args()

if (args.settings):

    yaml = ruamel.yaml.YAML()
    with open(args.settings, 'r') as stream:
        try:
            settings = yaml.load(stream)
        except ruamel.yaml.YAMLError as exc:
            print(exc)

    args.areasettings = settings["area_settings"]
    args.globalsettings = settings["global_settings"]
    args.data = settings["data"]
    args.resolution = settings["resolution"]
    args.latitude_lower_bound = settings["latitude_lower_bound"]
    args.latitude_upper_bound = settings["latitude_upper_bound"]

for i_data in args.data:
    # load data file
    yaml = ruamel.yaml.YAML()
    with open(i_data, 'r') as stream:
        try:
            data = yaml.load(stream)
        except ruamel.yaml.YAMLError as exc:
            print(exc)

    # get model properties for plots
    # TODO: resolve workaround for year recognition
    start_year = 50000
    end_year = 0

    for i_datafile in data:
        model_properties = model_identification_re(i_datafile)
        model_name = (model_properties["name"])
        partial_start_year = (model_properties["start"])
        partial_end_year = (model_properties["end"])
        if (partial_start_year < start_year):
            start_year = partial_start_year
        if (partial_end_year > end_year):
            end_year = partial_end_year

    string_start_year = str(start_year)
    string_end_year = str(end_year)

    data_identifier = model_name + "_" + string_start_year + "_" + string_end_year
    # main part of script

    # GLOBAL STATISTICS

    for i_settings in args.globalsettings:
        # load settings file
        yaml = ruamel.yaml.YAML()
        with open(i_settings, 'r') as stream:
            try:
                settings = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        variable = settings["variable"]
        depth_thresholds = settings["depth_thresholds"]
        time_thresholds = settings["time_thresholds"]
        quantiles = settings["quantiles"]
        outputdir = settings["output"] + "/global"

        if (os.path.exists(outputdir) == False):
            os.mkdir(outputdir)

        concatenated_data = concatenate_variables(data, variable)

        # apply latitudinal bounds

        # restrict on target coords depending on resolution setting

        latitude_constraint = iris.Constraint(
            latitude=lambda v: args.latitude_lower_bound <= v <= args.latitude_upper_bound)

        bounded_data = concatenated_data.extract(latitude_constraint)

        # quantile analysis

        quantiles = bounded_data.collapsed('time', iris.analysis.PERCENTILE, percent=quantiles)

        iris.save(quantiles, outputdir + '/' + data_identifier + '_quantile_analysis.nc')

        # threshhold analysis

        for i_depth in depth_thresholds:
            for i_days in time_thresholds:
                thresholded_data = calculate_data_above_threshold_in_x_days(bounded_data, i_depth, i_days, False)
                threshold_identifier = outputdir + '/' + data_identifier + '_threshold_analysis_depth_' + str(
                    i_depth) + '_days_' + str(i_days)
                iris.save(thresholded_data, threshold_identifier + '.nc')

    # AREA STATISTICS AND GRAPHS

    for i_settings in args.areasettings:
        # load settings file
        yaml = ruamel.yaml.YAML()
        with open(i_settings, 'r') as stream:
            try:
                settings = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        variable = settings["variable"]
        area_lat = settings["area"]["lat"]
        area_lon = settings["area"]["lon"]
        area_name = settings["area"]["name"]
        outputdir = (settings["output"] + "/" + area_name)

        if (os.path.exists(outputdir) == False):
            os.mkdir(outputdir)
            # define plotname part

        plotname = model_name + "_" + area_name + "_" + string_start_year + "_" + string_end_year
        plotdir = outputdir + "/" + plotname

        concatenated_data = concatenate_variables(data, variable)



        if (args.resolution == "CMIP_HR"):
            area_constraint = iris.Constraint(
                latitude=lambda v: area_lat - 90 / 192 <= v <= area_lat + 90 / 192,
                longitude=lambda v: area_lon - 180 / 384 <= v <= area_lon + 180 / 384)
        # restrict on target coords (for low resolution models)
        if (args.resolution == "CMIP_LR"):
            area_constraint = iris.Constraint(
                latitude=lambda v: area_lat - 90 / 144 <= v <= area_lat + 90 / 144,
                longitude=lambda v: area_lon - 180 / 192 <= v <= area_lon + 180 / 192)
        else:
            area_constraint = iris.Constraint(
                latitude=lambda v: area_lat - 90 / 360 <= v <= area_lat + 90 / 360,
                longitude=lambda v: area_lon - 180 / 720 <= v <= area_lon + 180 / 720)

        target_area_data = concatenated_data.extract(area_constraint)

        # plot time series of snow fall

        iris.plot.plot(target_area_data)
        plt.xlabel("time")
        plt.ylabel("daily snowfall (mm)")
        plt.title(
            area_name)
        plt.suptitle(model_name + " from " + string_start_year + " to " + string_end_year)
        plt.savefig((plotdir + "_timeseries"))
        plt.close()
        # plot histogram
        np_data = target_area_data.data

        # TODO: generalize diagram labels

        bins = np.arange(10, np_data.max() + 10, 5)
        plt.xlabel("daily snowfall (mm)")
        plt.ylabel("number of days in resp. bin")
        plt.title(
            area_name)
        plt.suptitle(model_name + " from " + string_start_year + " to " + string_end_year)
        plt.hist(target_area_data.data, bins, range=(10, np_data.max()))
        plt.savefig((plotdir + "_histogramm"))
        plt.close()
        # plot density histogram
        np_data = target_area_data.data

        bins = np.arange(10, np_data.max() + 10, 5)
        plt.xlabel("daily snowfall (mm)")
        plt.ylabel("share of days in resp. bin")
        plt.title(
            area_name)
        plt.suptitle(model_name + "  from " + string_start_year + " to " + string_end_year)

        plt.hist(target_area_data.data, bins, range=(10, np_data.max()), density=True)
        plt.savefig((plotdir + "_density"))
        plt.close()

        iris.save(target_area_data,
                  outputdir + '/' + model_name + "_" + area_name + "_" + string_start_year + "_" + string_end_year + '_area_analysis.nc')

        # calculate yearly, seasonal year and seasonal mean snowfall:

        iris.coord_categorisation.add_season(target_area_data, 'time', name='clim_season')
        iris.coord_categorisation.add_season_year(target_area_data, 'time', name='season_year')
        iris.coord_categorisation.add_year(target_area_data, 'time', name='year')

        annual_seasonal_max = target_area_data.aggregated_by(
            ['clim_season', 'season_year'], iris.analysis.MAX)

        annual_max = target_area_data.aggregated_by(
            ['year'], iris.analysis.MAX)

        # plot the yearly / seasonal maximum time series

        iris.plot.plot(annual_max)
        plt.xlabel("time")
        plt.ylabel("max annual snowfall (mm)")
        plt.title(
            area_name)
        plt.suptitle(model_name + " from " + string_start_year + " to " + string_end_year)
        plt.savefig((plotdir + "_timeseries_annual_max"))
        plt.close()


        iris.save(target_area_data,
                  outputdir + '/' + model_name + "_" + area_name + "_" + string_start_year + "_" + string_end_year + '_area_analysis.nc')
