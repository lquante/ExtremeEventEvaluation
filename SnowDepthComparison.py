#!/usr/bin/env python

# script to prepare comparison data from output of snow_processing:

# imports

import argparse
import os
import re

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
from iris.analysis import Aggregator
from iris.util import rolling_window
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


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def get_cube_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist[0]


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


def data_analysis(filepath, depth_threshold, time_threshold):
    # retrieve data
    data = load_data_from_netcdf(filepath)
    cube_daily = get_cube_from_cubelist(data, 'approx_fresh_daily_snow_height')
    # TODO: choince of variables to analyse via argpaser , e.g. change and accumulation analysis might be included later
   # TODO: chocie of analysis method
    # if args.method[0] == 1:
    #     analysed_data = calculate_data_above_threshold_for_x_days(cube_daily, depth_threshold, time_threshold, False)

    analysed_data = calculate_data_above_threshold_in_x_days(cube_daily, depth_threshold, time_threshold, False)

    return analysed_data  # returns analysed data


# def plot function

def contour_plot_intensity_data(data, contour_levels, filename=''):
    # Plot the results.
    qplt.contourf(data, contour_levels, cmap='GnBu')
    plt.gca().coastlines()

    plt.savefig(filename)
    iplt.show()


def contour_plot_compare_intensity_data(data, contour_levels, filename=''):
    # Plot the results.
    qplt.contourf(data, contour_levels, cmap='coolwarm')
    plt.gca().coastlines()

    plt.savefig(filename)
    iplt.show()

# argument parser definition
parser = argparse.ArgumentParser(description="Calculate some analysis metrics on specified files")
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


# main script to conduct analysis

# load settings file
yaml = ruamel.yaml.YAML()
with open(args.settings, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
depth_thresholds = settings["thresholds"]["depth"]
time_thresholds = settings ["thresholds"]["time"]

inputfile = settings ["input"]
outputdir = settings ["output"]


# some analysis help methods

# model identification based on structure of outpufilename: str("output_" + i_model + "_" + timeindex + ".nc") from Pathname Collection Helper
def model_identification_re(filepath):
    model = re.search('(.*/)(output_)(.*_\d{4}\d{4})(.nc)$', filepath)
    if (model):
        model_string = model.group(3)
    else:
        model_string = "model_not_identified"
    return model_string


# save data to nc file in outputlocation
def safe_data_nc(data, analysisidentifier):
    iris.save(data, outputdir + '/data_analysis_' + analysisidentifier + '.nc')

# main part of script
# loop over all input files and thresholds

for i_depth in depth_thresholds:

    for i_time in time_thresholds:
        model_identifier = model_identification_re(inputfile)
        threshold_identifier = "depth_" + str(i_depth) + "_time_" + str(i_time)
        analysis_identifier = model_identifier + "_" + threshold_identifier
        analysed_data = data_analysis(inputfile, i_depth, i_time)
        safe_data_nc(analysed_data, analysis_identifier)

