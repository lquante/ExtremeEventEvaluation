#!/usr/bin/env python

# script to visualize - especially georeferenced -  data from NetCDF files

# imports

import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iris
import iris.analysis
import iris.coord_categorisation
import iris.quickplot as qplt

import numpy as np
from ruamel.yaml import ruamel


# import help methods

def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


# model identification based on structure of filename: str(i_model + "_" + timeindex + _ + phenomenom".nc") from data processing
def model_identification_re(filepath):
    model = re.search('(.*/)(.*)(_)(\d{4})_(\d{4})(_.*)(.nc)$', filepath)
    model_properties = {}
    if (model):
        model_properties["name"] = model.group(2)
        model_properties["start"] = int(model.group(4))
        model_properties["end"] = int(model.group(5))
    else:
        model_properties["name"] = "model_not_identified"
        model_properties["start"] = 0000
        model_properties["end"] = 0000
    return model_properties


# help methods for plotting, etc.

def contour_min_max(data):
    data_max = data.collapsed(["latitude", "longitude"], iris.analysis.MAX).data
    data_min = data.collapsed(["latitude", "longitude"], iris.analysis.MIN).data
    steps = 20
    stepsize = (data_max - data_min) / steps
    return np.arange(data_min, data_max, stepsize)


# argument parser definition
parser = argparse.ArgumentParser(description="Plot provided netcdf datafiles")

parser.add_argument(
    "--datadir"
    , type=str,
    help="directory to scrape for any *.nc data"
)

parser.add_argument(
    "--outputdir"
    , type=str,
    help="directory to save graphs"
)

parser.add_argument(
    "--latitude_lower_bound"
    , type=int,
    default=-90,
    help="lower_bound_for_latitudes"
)

parser.add_argument(
    "--latitude_upper_bound"
    , type=int,
    default=90,
    help="upper_bound_for_latitudes"
)

parser.add_argument(
    "--longitude_lower_bound"
    , type=int,
    default=0,
    help="lower_bound_for_longitudes"
)

parser.add_argument(
    "--longitude_upper_bound"
    , type=int,
    default=360,
    help="upper bound for latitudes"
)

args = parser.parse_args()

# scrape date directory for nc files


for i_datafile in Path(args.datadir).rglob("*.nc"):
    # get model properties for plots
    i_datafile = str(i_datafile)
    model_properties = model_identification_re(str(i_datafile))
    model_name = (model_properties["name"])
    start_year = (model_properties["start"])
    end_year = (model_properties["end"])

    datacube = iris.load_cube(i_datafile)
    os.chdir(args.outputdir)
    # get all variable names:
    var_name = datacube.var_name
    # ge coord names
    coord_names = [coord.name() for coord in datacube.coords()]
    # loop over all coordinates which are not lat lon or time
    coord_names.remove('latitude')
    coord_names.remove('longitude')
    coord_names.remove('time')
    # define timeshift from 1-1-1 to 1850-1-1
    start_date = datetime.strptime('18500101T0000Z', '%Y%m%dT%H%MZ')
    start_shift = start_date.toordinal()
    # loop over timepoints
    for i_time in datacube.coord('time').points:
        i_date = (datetime.fromordinal((int(i_time + start_shift))))

        timefiltered_data = datacube.extract(iris.Constraint(
            time=lambda timeinterval: i_date - timedelta(hours=12) <= (timeinterval.point) <= i_date + timedelta(
                hours=12)))  # TODO: fix workaround, works as long only daily data is used
        i_date = i_date.date()  # ignore clocktimes
        for i_coord in coord_names:
            for i_point in datacube.coord(i_coord).points:
                to_plot = timefiltered_data.extract(iris.Constraint(**{i_coord: i_point}))
                filename = (model_name + "_" + str(start_year) + "_" + str(end_year) + "_" + str(i_coord) + str(
                    i_date) + "_" + str(i_point) + ".png")
                # Plot the results.
                qplt.pcolor(to_plot)
                plt.title(i_coord + "=" + str(i_point) + " at " + str(i_date))
                plt.suptitle(var_name + " " + model_name + " from " + str(start_year) + " to " + str(end_year))
                plt.gca().coastlines()
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(6, 4)
                plt.savefig(filename)
                plt.close()
        # special case of scalar time value, that should be plotted:
        # if len(coord_names) == 0:
        #     to_plot = timefiltered_data
        #     filename = (model_name + "_" + str(start_year) + "_" + str(end_year) + "_" + str(var_name) + str(
        #         i_date) + ".png")
        #     # Plot the results.
        #     qplt.pcolor(to_plot)
        #     plt.title(var_name + " at " + str(i_date))
        #     plt.suptitle(model_name + " from " + str(start_year) + " to " + str(end_year))
        #     plt.gca().coastlines()
        #     fig = matplotlib.pyplot.gcf()
        #     fig.set_size_inches(6, 4)
        #     plt.savefig(filename)
        #     plt.close()
