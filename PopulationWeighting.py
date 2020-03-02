#!/usr/bin/env python

# script to weight netcdf file of geo data with population data from
# https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11
# NB: resolution of grids should match to utilize this script, otherwise interpolation will be applied

# imports

import argparse
import re

import iris

# argument parser definition
parser = argparse.ArgumentParser(description="Calculate some analysis metrics on specified data files")
# path to *.yml file with settings to be used
parser.add_argument(
    "--data",
    nargs="+"
    , type=str,
    required=True,
    help="Path to the data file(s)"
)

parser.add_argument(
    "--population"
    , type=str,
    required=True,
    help="Path to the population file to be used for weighting"
)

args = parser.parse_args()

# weight each data file
population = iris.load_cubes(args.population, 'Population_Count,_v4.11_(2000,_2005,_2010,_2015,_2020):_30_arc-minutes')
print(population)
for i_data in args.data:
    datacubes = iris.load_cubes(i_data)
    data_file = re.search('(.*/)(.*)(.nc)$', i_data)
    if (data_file):
        data_name = data_file.group(2)
    else:
        data_name = "name_not_identified"
    weighted_list = iris.cube.CubeList
    for i_cube in datacubes:
        weighted_cube = i_cube * population
        weighted_list.append(weighted_cube)
    iris.save(weighted_list, data_name + "_weighted.nc")
