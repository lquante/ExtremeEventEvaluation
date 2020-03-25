#!/usr/bin/env python

import argparse
import gc
import os
from datetime import datetime

import cf_units
import iris
import iris.analysis
import iris.plot as iplt
import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from ruamel.yaml import ruamel
from tqdm import tqdm


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def filter_cubes_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist


def animate(frame, variable_data, start_shift, colormap, vmin, vmax):
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()
    frame_data = variable_data[frame]

    plot = iris.plot.pcolormesh(frame_data, cmap=colormap, vmin=vmin, vmax=vmax)
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

def movie_from_data(i_data, variablename):
    basic_data = load_data_from_netcdf(i_data)
    filtered_data = filter_cubes_from_cubelist(basic_data, variablename)
    concatenated_data = filtered_data[0]
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
    percentiles = (
        variable_data.collapsed(("time", "latitude", "longitude"), iris.analysis.PERCENTILE, percent=[99]).data)
    datamax = (variable_data.collapsed(("time", "latitude", "longitude"), iris.analysis.MAX).data)
    percentile99 = percentiles
    data_max_hundred = datamax if datamax % 100 == 0 else datamax + 100 - datamax % 100
    # make a colormap that has no snow and snow extremes (>95%tile) clearly delineated and of the
    # same length (256 + 256)
    colors_no_extreme = plt.cm.plasma(np.linspace(0, 0.1, 1))
    colors_extreme = plt.cm.plasma(np.linspace(0.1, 1, 512))
    all_colors = np.vstack((colors_no_extreme, colors_extreme))
    # make color value for 0 white
    all_colors[0] = (0.5, 0.5, 0.5, 1.0)

    extremes_map = matplotlib.colors.LinearSegmentedColormap.from_list('extremes_map',
                                                                       all_colors)
    # make the norm:

    bounds_zero = [0, 1]
    bounds_extremes = list(np.arange(percentile99, data_max_hundred, 5))
    bounds = bounds_zero + bounds_extremes

    colormap = extremes_map

    pcm = iris.plot.pcolormesh(start_data, cmap=colormap, vmin=percentile99, vmax=data_max_hundred)
    cbar = plt.colorbar(pcm, extend='both', orientation='horizontal')
    ticks = cbar.get_ticks()
    ticks_update = np.append(ticks, percentile99)
    sorted_ticks = np.sort(ticks_update)
    cbar.set_ticks(sorted_ticks, update_ticks=True)
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
        fargs=[variable_data, start_shift, colormap, percentile99, data_max_hundred],
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

data_file = settings["data"]
variablename = settings["variable"]
data_identifier = settings["data_identifier"]
outputdir = settings["output"]

# load data file
yaml = ruamel.yaml.YAML()
with open(data_file, 'r') as stream:
    try:
        data_input = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

print(data_input)
for i_data in tqdm(data_input):
    movie_from_data(i_data, variablename)
