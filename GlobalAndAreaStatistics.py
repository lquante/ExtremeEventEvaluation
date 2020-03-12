#!/usr/bin/env python

# script to analyse data for a certain specified grid point (Lat,Lon) from output of snow_processing:

# imports

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")

import iris
import iris.coord_categorisation

import numpy as np
from iris.analysis import Aggregator
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import rolling_window, unify_time_units
from ruamel.yaml import ruamel

from tqdm import tqdm


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
    return np.sum(exceeding_windows, axis=axis, dtype=int)


def flag_average_exceedance(data, threshold, axis, accumulation_length):
    """
    Function to mark the number of points in a sequence where the average
    has exceeded a threshold for a certain number of timepoints (accumulation_length).

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
    exceeding_windows = np.average(accumulation_windows, axis=axis + 1) > threshold
    return exceeding_windows


def count_average_exceedance(data, threshold, axis, accumulation_length):
    return np.sum(flag_average_exceedance(data, threshold, axis, accumulation_length), axis=axis, dtype=int)


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


def calculate_average_above_threshold_in_x_days(data, threshold, numberOfDays, relativeValue):
    # Make an aggregator from the user function.
    AVERAGE_COUNT = Aggregator('average_exceedance_count',
                               count_accumulated_exceedance,
                               units_func=lambda units: 1)

    # Calculate the statistic
    average_above_threshold = data.collapsed('time', AVERAGE_COUNT,
                                             threshold=threshold,
                                             accumulation_length=numberOfDays)

    # relative result
    if (relativeValue == True):
        average_above_threshold.rename(
            ' Percentage of days with average snow  falls in ' + str(numberOfDays) + ' days above ' + str(
                threshold) + 'mm')
        average_above_threshold.data = average_above_threshold.data / numberOfDays * 100
    else:
        average_above_threshold.rename(
            ' Number of days with average snow falls in ' + str(numberOfDays) + ' days above ' + str(
                threshold) + 'mm')
    return average_above_threshold


# experimental tools to calculate custom stats


def average_stats(data, numberOfDays, original_varname):
    extended_varname = original_varname + "_average" + str(numberOfDays)
    data.var_name = extended_varname
    average_cube = data.rolling_window('time', iris.analysis.MEAN, numberOfDays)
    return average_cube


def max_stats(data, numberOfDays, original_varname):
    extended_varname = original_varname + "_maximum" + str(numberOfDays)
    data.var_name = extended_varname
    max_cube = data.rolling_window('time', iris.analysis.MAX, numberOfDays)
    return max_cube




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

for i_data in tqdm(args.data):
    # load data file
    yaml = ruamel.yaml.YAML()
    with open(i_data, 'r') as stream:
        try:
            data = yaml.load(stream)
        except ruamel.yaml.YAMLError as exc:
            print(exc)

        # main part of script

        # GLOBAL STATISTICS

        for i_settings in tqdm(args.globalsettings):
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

            # get model properties for plots

        for i_datafile in tqdm(data):
            model_properties = model_identification_re(i_datafile)
            model_name = (model_properties["name"])
            start_year = (model_properties["start"])
            end_year = (model_properties["end"])

            string_start_year = str(start_year)
            string_end_year = str(end_year)

            data_identifier = model_name + "_" + string_start_year + "_" + string_end_year

            modeldir = outputdir + "/" + model_name
            if (os.path.exists(modeldir) == True):
                os.chdir(modeldir)
            else:
                os.mkdir(modeldir)
                os.chdir(modeldir)

            # load data of relevant variable
            variable_data_cube = iris.load_cube(i_datafile, variable)

            # apply latitudinal bounds
            latitude_constraint = iris.Constraint(
                latitude=lambda v: args.latitude_lower_bound <= v <= args.latitude_upper_bound)

            bounded_data = variable_data_cube.extract(latitude_constraint)

            # average / maximum analysis for rolling window

            original_varname = bounded_data.var_name
            for i_days in tqdm(time_thresholds):

                averagedir = modeldir + "/average"
                if (os.path.exists(averagedir) == True):
                    os.chdir(averagedir)
                else:
                    os.mkdir(averagedir)
                    os.chdir(averagedir)

                average_data = average_stats(bounded_data, i_days, original_varname)
                average_identifier = averagedir + '/' + data_identifier + '_average_analysis_days_' + str(i_days)
                iris.save(average_data, average_identifier + '.nc')

                maxdir = modeldir + "/maximum"
                if (os.path.exists(maxdir) == False):
                    os.mkdir(maxdir)
                os.chdir(maxdir)

                maximum_data = max_stats(bounded_data, i_days, original_varname)
                maximum_identifier = maxdir + '/' + data_identifier + '_maximum_analysis_days_' + str(i_days)
                iris.save(maximum_data, maximum_identifier + '.nc')

    # AREA STATISTICS AND GRAPHS

    for i_settings in tqdm(args.areasettings):
        # load settings file
        yaml = ruamel.yaml.YAML()
        with open(i_settings, 'r') as stream:
            try:
                settings = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        start_year = 5000
        end_year = 0
        for i_datafile in tqdm(data):
            model_properties = model_identification_re(i_datafile)
            model_name = (model_properties["name"])
            partial_start_year = (model_properties["start"])
            partial_end_year = (model_properties["end"])
            if (start_year > partial_start_year):
                start_year = partial_start_year
            if (end_year < partial_end_year):
                end_year = partial_end_year
        string_start_year = str(start_year)
        string_end_year = str(end_year)

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

        iris.coord_categorisation.add_season(target_area_data, 'time', name='clim_season')
        iris.coord_categorisation.add_season_year(target_area_data, 'time', name='season_year')
        iris.coord_categorisation.add_year(target_area_data, 'time', name='year')

        iris.save(target_area_data,
                  outputdir + '/' + model_name + "_" + area_name + "_" + string_start_year + "_" + string_end_year + '_area_analysis.nc')
