#!/usr/bin/env python

# script to visualize - especially georeferenced -  data from NetCDF files

# imports

import argparse
import re
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iris
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


# argument parser definition
parser = argparse.ArgumentParser(description="Plot provided netcdf datafiles")

parser.add_argument(
    "--data",
    nargs="+"
    , type=str,
    help="YML list of datafile(s)"
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

# load list of datafiles for each suplied data list
for i_datalist in args.data:
    yaml = ruamel.yaml.YAML()
    with open(i_datalist, 'r') as stream:
        try:
            data = yaml.load(stream)
        except ruamel.yaml.YAMLError as exc:
            print(exc)

    for i_datafile in data:
        # get model properties for plots
        print(i_datafile)
        model_properties = model_identification_re(i_datafile)
        model_name = (model_properties["name"])
        start_year = (model_properties["start"])
        end_year = (model_properties["end"])

        datacube = iris.load_cube(i_datafile)

        # get all variable names:
        var_names = datacube.var_name
        # ge coord names
        coord_names = [coord.name() for coord in datacube.coords()]
        # loop over all coordinates which are not lat lon or time
        coord_names.remove('latitude')
        coord_names.remove('longitude')
        coord_names.remove('time')
        # plot all time scalar variables
        for i_coord in coord_names:
            for i_point in datacube.coord(i_coord).points:
                to_plot = datacube.extract(iris.Constraint(**{i_coord: i_point}))
                contours = np.arange(0, 1000, 50)
                filename = ('test' + str(i_point) + ".png")
                # Plot the results.

                qplt.contourf(to_plot, cmap='GnBu')
                plt.suptitle(model_name + " from " + str(start_year) + " to " + str(end_year))
                plt.title(i_coord + "=" + str(i_point) + " for " + var_names)
                plt.gca().coastlines()
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(6, 4)
                plt.savefig(filename)
                plt.close()

        # plot all over time changing variables
        # define timeshift from 1-1-1 to 1850-1-1
        start_date = datetime.strptime('18500101T0000Z', '%Y%m%dT%H%MZ')
        start_shift = start_date.toordinal()
        print(datacube.coord('time').points)

        for i_time in datacube.coord('time').points:
            i_date = (datetime.fromordinal((int(i_time + start_shift))))
            timefiltered_data = datacube.extract(iris.Constraint(
                time=lambda timeinterval: i_date - timedelta(hours=12) <= (timeinterval.point) <= i_date + timedelta(
                    hours=12)))  # TODO: fix workaround, works as long only daily data is used
            to_plot = timefiltered_data
            print(to_plot)
            contours = np.arange(0, 1000, 50)
            filename = ('test' + str(i_time) + ".png")
            # Plot the results.
            qplt.contourf(to_plot, contours, cmap='GnBu')
            plt.suptitle(model_name + " from " + str(start_year) + " to " + str(end_year))
            plt.title(var_names)
            plt.gca().coastlines()
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(6, 4)
            plt.savefig(filename)
            plt.close()
