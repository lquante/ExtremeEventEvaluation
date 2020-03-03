#!/usr/bin/env python

import argparse
import os
from datetime import datetime

import iris
import iris.plot as iplt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from iris.analysis import Aggregator
from iris.experimental.equalise_cubes import equalise_attributes
from matplotlib.animation import FuncAnimation
from ruamel.yaml import ruamel


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
    variable_data = filtered_data [0]


contours = np.arange(0, 1000, 10)

# def plot function
def contour_plot_geodata(data, contour_levels):
    # Plot the results.
    iris.plot.contourf(data, contour_levels, cmap='GnBu')

# extract time dimension
time = variable_data.coord('time')
number_of_timepoints = time.points.size
# define timeshift from 1-1-1 to 1850-1-1
start_date = datetime.strptime('18500101T0000Z', '%Y%m%dT%H%MZ')
start_shift = start_date.toordinal()

def plot_for_time_index(i_time):
    date_lower_bound = datetime.fromordinal(int(i_time) + start_shift)
    date_upper_bound = datetime.fromordinal(int(i_time + 1) + start_shift)
    datepoint = iris.Constraint(time=lambda cell: date_lower_bound <= cell.point <= date_upper_bound)
    contour_plot_geodata(variable_data.extract(datepoint), contours)

def animate(frame):
    plot_for_time_index(time.points[frame])

def zero_function(data,axis):
    return 0
EMPTY_AGGREGATOR = Aggregator("zero function",zero_function)


# initialize plot with world map data
empty_data = variable_data.collapsed('time', iris.analysis.SUM)
titled_world_map = plt.figure()
first_date = datetime.fromordinal(int(time.points[0]) + start_shift)
last_date = datetime.fromordinal(int(time.points[number_of_timepoints-1]) + start_shift)
plt.title("From " + str(first_date) +" to "+str(last_date))
plt.suptitle(variable_to_plot + " , "+ data_identifier)
iris.plot.contourf(empty_data,contours, cmap='GnBu')
plt.gca().coastlines()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1000)


movie = FuncAnimation(
    # Your Matplotlib Figure object
    titled_world_map,
    # The function that does the updating of the Figure
    animate,
    # Frame information (here just frame number)
    np.arange(1, number_of_timepoints, 1),
    fargs=[],
    # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
    interval=50
)

# Try to set the DPI to the actual number of pixels you're plotting
movie.save(outputdir+"/movie_"+variable_to_plot+"_"+data_identifier+"_"+ str(first_date) +"_"+str(last_date)+".mp4", writer= writer)
