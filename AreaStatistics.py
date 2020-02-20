#!/usr/bin/env python

# script to analyse data for a certain specified grid point (Lat,Lon) from output of snow_processing:

# imports

import argparse
import re

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import ruamel


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def get_cube_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist[0]


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
    required=True,
    help="Path to the settingsfile"
)

parser.add_argument(
    "--data"
    , type=str,
    required=True,
    help="Path to datafile"
)

args = parser.parse_args()

# load settings file
yaml = ruamel.yaml.YAML()
with open(args.settings, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

area_lat = settings["area"]["lat"]
area_lon = settings["area"]["lon"]
area_name = settings["area"]["name"]
outputdir = settings["output"]

data = args.data

# model identification based on structure of filename: str("output_" + i_model + "_" + timeindex + ".nc") from Pathname Collection Helper
def model_identification_re(filepath):
    model = re.search('(.*/)(output_)(.*)(.nc)$', filepath)
    if (model):
        model_string = model.group(3)
    else:
        model_string = "model_not_identified"
    return model_string


# save data to nc file in outputlocation
def safe_data_nc(data, analysisidentifier):
    iris.save(data, outputdir + '/data_analysis_' + analysisidentifier + '.nc')


# main part of script
# concatenate all input files in one cube (NB: assumes data unique wrt to time) TODO: generalize recognition of identical model


# import all single data files into one cubelist

complete_data_cube = iris.load(data)

# filter for daily snowfall

filtered_data_cube = complete_data_cube.extract('approx_fresh_daily_snow_height')

# concatenate cubes

concatenated = filtered_data_cube.concatenate_cube()

# restrict on target coords (for low resolution models)

area_constraint = iris.Constraint(
    latitude=lambda v: area_lat - 90 / 144 <= v <= area_lat + 90 / 144,
    longitude=lambda v: area_lon - 180 / 192 <= v <= area_lon + 180 / 192)

target_area_data = concatenated.extract(area_constraint)

# get modelname for plots
data_identifier = model_identification_re(data)

# define plotfilenamestump

plotname = outputdir +"/"+ data_identifier + "_" + area_name

# plot time series of snow fall

iris.plot.plot(target_area_data)
plt.xlabel("time")
plt.ylabel("daily snowfall (mm)")
plt.title(
    "Time series of daily snowfall for approx. Lat. " + str(area_lat) + " and lon. " + str(area_lon) + ", " + area_name)
plt.suptitle(data_identifier)
plt.savefig((plotname + "_timeseries"))
plt.close()
# plot histogram
np_data = target_area_data.data

bins = np.arange(10, np_data.max() + 10, 5)
plt.xlabel("daily snowfall (mm)")
plt.ylabel("number of days in resp. bin")
plt.title(
    "Histogramm of daily snowfall approx. for Lat. " + str(area_lat) + " and Lon. " + str(area_lon) + ", " + area_name)
plt.suptitle(data_identifier)
plt.hist(target_area_data.data, bins, range=(10, np_data.max()))
plt.savefig((plotname + "_histogramm"))
plt.close()
# plot density histogram
np_data = target_area_data.data

bins = np.arange(10, np_data.max() + 10, 5)
plt.xlabel("daily snowfall (mm)")
plt.ylabel("share of days in resp. bin")
plt.title(
    "Histogramm of daily snowfall approx. for Lat. " + str(area_lat) + " and Lon. " + str(area_lon) + ", " + area_name)
plt.suptitle(data_identifier)

plt.hist(target_area_data.data, bins, range=(10, np_data.max()), density=True)
plt.savefig((plotname + "_density"))
plt.close()

iris.save(target_area_data, outputdir + '/' + data_identifier + '_area_analysis.nc')
