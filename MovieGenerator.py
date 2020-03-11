#!/usr/bin/env python

import argparse
import gc
import multiprocessing
import os
from datetime import datetime

import cf_units
import iris
import iris.analysis
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
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


def animate(frame, variable_data, start_shift, colormap, datamax):
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()
    frame_data = variable_data[frame]

    plot = iris.plot.pcolormesh(frame_data, cmap=colormap, vmin=0, vmax=datamax)
    ordinal_date = (frame_data.coord('time').points[0])
    date = datetime.fromordinal(int(ordinal_date + start_shift)).date()
    plt.title(str(date), fontsize=10)
    if frame % 10 == 0:
        gc.collect()
    return plot


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


# loads data for each datafile: (needs to contain exactly one variable) TODO: generalize

def movie_from_data(i_data):
    basic_data = load_data_from_netcdf(i_data)
    concatenated_data = basic_data[0]
    variable_to_plot = concatenated_data.var_name
    variable_data = filter_cubes_from_cubelist(concatenated_data, variable_to_plot)

    # extract time unit & infos
    time = variable_data.coord('time')
    time_units = variable_data.coord('time').units
    number_of_timepoints = time.points.size
    # define timeshift from 1-1-1 to 1850-1-1
    fixed_time_unit = cf_units.Unit('days since 1850-1-1 00:00:00', calendar=time_units.calendar)
    variable_data.coord('time').convert_units(fixed_time_unit)
    start_date = datetime.strptime('18500101T0000Z', '%Y%m%dT%H%MZ')
    start_shift = start_date.toordinal()
    # guess bounds for lat and lon
    variable_data.coord('latitude').guess_bounds()
    variable_data.coord('longitude').guess_bounds()

    start_data = variable_data[0]
    titled_world_map = plt.figure()
    titled_world_map.set_size_inches(7.5, 3.5)
    first_date = datetime.fromordinal(int(time.points[0]) + start_shift).date()
    last_date = datetime.fromordinal(int(time.points[number_of_timepoints - 1]) + start_shift).date()
    datamax = (variable_data.collapsed(("time", "latitude", "longitude"), iris.analysis.MAX).data)
    datamax_hundred = datamax if datamax % 100 == 0 else datamax + 100 - datamax % 100
    colormap = 'plasma'
    qplt.pcolormesh(start_data, cmap=colormap, vmin=0, vmax=datamax_hundred)
    plt.title("From " + str(first_date) + " to " + str(last_date), fontsize=10)
    plt.suptitle(variable_to_plot + " , " + data_identifier, fontsize=12)
    plt.gca().coastlines()
    # Set up formatting for the movie files
    ffmpeg_writer = animation.FFMpegWriter(fps=30)

    movie = FuncAnimation(
        # Your Matplotlib Figure object
        titled_world_map,
        # The function that does the updating of the Figure
        animate,
        fargs=[variable_data, start_shift, colormap, datamax_hundred],
        # Frame information (here just frame number)
        frames=np.arange(1, number_of_timepoints, 1),
        save_count=None

    )
    movie.save(outputdir + "/movie_" + variable_to_plot + "_" + data_identifier + "_" + str(first_date) + "_" + str(
        last_date) + ".mp4", writer=ffmpeg_writer)


# load settings file
yaml = ruamel.yaml.YAML()
with open(args.settings, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

data_input = settings["data"]
data_identifier = settings["data_identifier"]
outputdir = settings["output"]
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=int(num_cores / 4))(delayed(movie_from_data)(i_data) for i_data in data_input)
