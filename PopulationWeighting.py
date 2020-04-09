#!/usr/bin/env python

# script to weight netcdf file of geo data with population data from
# https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11
# NB: resolution of grids should match to utilize this script, otherwise interpolation will be applied

# imports

import argparse
import os
import re

import iris.analysis
from ruamel.yaml import ruamel
# argument parser definition
from tqdm import tqdm

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

# load settings file
yaml = ruamel.yaml.YAML()
with open(args.settings, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


populationfile = settings["population"]
data = settings["data"]
outputdir = settings["output"]

population = iris.load_cube(populationfile)
# extract year 2020
current_population = (population.extract(iris.Constraint(raster=5)))

print(current_population.var_name)
current_population.remove_coord('raster')
current_population.units = 1
# scale down population to avoid overflow isssues
scaling_factor = 1000000
current_population = current_population / scaling_factor
current_population.units = scaling_factor
print(current_population.summary())

for i_datalists in tqdm(data):

    yaml = ruamel.yaml.YAML()
    with open(i_datalists, 'r') as stream:
        try:
            datalist = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for i_data in tqdm(datalist):
        datacubes = iris.load(i_data)

        data_file = re.search('(.*/)(.*)(.nc)$', i_data)
        if (data_file):
            data_name = data_file.group(2)
        else:
            data_name = "name_not_identified"
        weighted_list = []
        for i_cube in datacubes:
            # i_cube = i_cube.collapsed("time", iris.analysis.MEAN)  # TODO: find a way to broadcast cubes
            current_population_regridded = current_population.regrid(i_cube, iris.analysis.Linear(
                extrapolation_mode='extrapolate'))

            i_cube.coord('latitude').attributes = {}
            current_population_regridded.coord('latitude').attributes = {}
            current_population_regridded.coord('longitude').attributes = {}
            i_cube.coord('longitude').attributes = {}

            weighted_data = i_cube.data * current_population_regridded.data

            weighted_data_cube = i_cube
            weighted_data_cube.data = weighted_data

            var_name = i_cube.var_name
            var_name_extended = (var_name + '_population_weighted')
            weighted_data_cube.var_name = var_name_extended
            weighted_list.append(weighted_data_cube)

        os.chdir(outputdir)
        iris.save(weighted_list, data_name + "_weighted.nc")
