#!/usr/bin/env python

# script to analyse  quantiles of netcdf geodata

# imports

import argparse
import os
import re

import iris
from iris.analysis import Aggregator
from iris.experimental.equalise_cubes import equalise_attributes
from ruamel.yaml import ruamel


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def get_cube_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist[0]


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


# load settings file
yaml = ruamel.yaml.YAML()
with open(args.settings, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
data_input = settings["data"]
quantiles = settings ["quantiles"]
outputdir = settings ["output"]


# load data file
yaml = ruamel.yaml.YAML()
with open(data_input, 'r') as stream:
    try:
        data = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# some analysis help methods

# model identification based on structure of outpufilename: str("output_" + i_model + "_" + timeindex + ".nc") from Pathname Collection Helper
def model_identification_re(filepath):
    model = re.search('(.*/)(settings_)(.*)(.nc)$', filepath)
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

# equalize attributes to enable concatenating into a single cube

equalise_attributes(filtered_data_cube)

# concatenate cubes

concatenated = filtered_data_cube.concatenate_cube()


quantiles = concatenated.collapsed('time', iris.analysis.PERCENTILE, percent=quantiles)


settings_identifier = model_identification_re(args.settings)

iris.save(quantiles, outputdir + '/' + settings_identifier+'_quantile_analysis.nc')