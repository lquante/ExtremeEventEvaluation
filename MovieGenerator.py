#!/usr/bin/env python

import argparse
import os
import time
from datetime import datetime

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from iris.analysis import Aggregator
from iris.experimental.equalise_cubes import equalise_attributes
from matplotlib.animation import FuncAnimation
from ruamel.yaml import ruamel
from tqdm import tqdm_gui


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def filter_cubes_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist


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

data_input = settings["data"]
variable_to_plot = settings ["variable"]
data_identifier = settings ["data_identifier"]
outputdir = settings ["output"]

# load data from (list of) filepaths
basic_data = load_data_from_netcdf(data_input)

filtered_data = filter_cubes_from_cubelist(basic_data, variable_to_plot)

if(len(filtered_data)>1):
# unify attributes of cubes
    equalise_attributes(filtered_data)
# concatenate cubes
    variable_data = filtered_data.concatenate_cube()
else:
    variable_data = filtered_data[0]

contours = np.arange(0, 1000, 10)


def contour_plot_geodata(data, contour_levels):
    # Plot the results.
    iris.plot.contourf(data, contour_levels, cmap='GnBu')


def pcolor_geodata(data):
    # Plot the results.
    iris.plot.pcolor(data)


def points_geodata(data):
    # Plot the results.
    iris.plot.points(data)


# extract time dimension
time = variable_data.coord('time')
number_of_timepoints = time.points.size
# define timeshift from 1-1-1 to 1850-1-1
start_date = datetime.strptime('18500101T0000Z', '%Y%m%dT%H%MZ')
start_shift = start_date.toordinal()
# guess bounds for lat and lon
variable_data.coord('latitude').guess_bounds()
variable_data.coord('longitude').guess_bounds()


def animate(frame):
    points_geodata(variable_data[frame])
    print(str(frame))
    # initialize plot with world map data


start_data = variable_data[0]
titled_world_map = plt.figure()
first_date = datetime.fromordinal(int(time.points[0]) + start_shift)
last_date = datetime.fromordinal(int(time.points[number_of_timepoints - 1]) + start_shift)
plt.title("From " + str(first_date) + " to " + str(last_date))
plt.suptitle(variable_to_plot + " , " + data_identifier)
qplt.points(start_data)
plt.gca().coastlines()
# Set up formatting for the movie files
writer = animation.FFMpegWriter(fps=25)

movie = FuncAnimation(
    # Your Matplotlib Figure object
    titled_world_map,
    # The function that does the updating of the Figure
    animate,
    # Frame information (here just frame number)
    number_of_timepoints - 1
)
movie.save(outputdir + "/movie_" + variable_to_plot + "_" + data_identifier + "_" + str(first_date) + "_" + str(
    last_date) + "test_100.mp4", writer=writer)
