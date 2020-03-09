#!/usr/bin/env python

# script to weight netcdf file of geo data with population data from
# https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11
# NB: resolution of grids should match to utilize this script, otherwise interpolation will be applied

# imports

import argparse
import os
import re

import iris
import iris.analysis
import xarray as xr
from netCDF4 import Dataset

# argument parser definition
parser = argparse.ArgumentParser(description="Calculate some analysis metrics on specified data files")
# path to *.yml file with data to be used
parser.add_argument(
    "--data",
    nargs="+"
    , type=str,
    help="Path to the data file(s)"
)

parser.add_argument(
    "--population"
    , type=str,
    required=True,
    help="Path to the population file to be used for weighting"
)

args = parser.parse_args()

population = iris.load_cube(args.population)
# extract year 2020
current_population = (population.extract(iris.Constraint(raster=5)))
# scale down to avoid overflow
current_population_downscaled = current_population / 100000

for i_data in args.data:
    datacubes = iris.load(i_data)

    data_file = re.search('(.*/)(.*)(.nc)$', i_data)
    if (data_file):
        data_name = data_file.group(2)
    else:
        data_name = "name_not_identified"
    weighted_list = []
    for i_cube in datacubes:
        i_cube = i_cube.collapsed("time", iris.analysis.MEAN)  # TODO: find a way to broadcast cubes
        current_population_downscaled = current_population_downscaled.regrid(i_cube, iris.analysis.Linear(
            extrapolation_mode='extrapolate'))
        var_name = i_cube.var_name
        weighted_cube = current_population_downscaled * i_cube
        var_name_extended = (var_name + '_population_weighted')
        print(var_name_extended)
        weighted_cube.var_name = var_name_extended
        weighted_list.append(weighted_cube)
    iris.save(weighted_list, data_name + "_weighted.nc")
