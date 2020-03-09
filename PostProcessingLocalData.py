#!/usr/bin/env python

# script to visualize area restricted data from NetCDF files

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
    model = re.search('(.*/)(.*)(_.*_)(\d{4})_(\d{4})(_.*)(.nc)$', filepath)
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
    "--datafolder"
    , nargs="+"
    , type=str,
    help="directories with files to plot on similar scale"
)

args = parser.parse_args()

# scrape date directory for nc files


for i_datafolder in (args.datafolder):

    # get overall maximum of data for histogram scaling
    overall_max = 0
    overall_lower_ylim = 0
    overall_upper_ylim = 0
    for i_data in Path(i_datafolder).rglob("*.nc"):
        i_data = str(i_data)
        datacube = iris.load_cube(i_data)
        np_data = datacube.data
        overall_max = max(np_data.max(), overall_max)
        bins = np.arange(10, overall_max + 10, 5)
        plt.hist(np_data, bins, range=(10, overall_max), density=True)
        y_lim = plt.ylim()
        overall_lower_ylim = min(overall_lower_ylim, y_lim[0])
        overall_upper_ylim = max(overall_upper_ylim, y_lim[1])
        plt.close()
    for i_data in Path(i_datafolder).rglob("*.nc"):
        # get model properties for plots
        i_data = str(i_data)
        np_data = datacube.data
        model_properties = model_identification_re(i_data)
        model_name = (model_properties["name"])
        start_year = str((model_properties["start"]))
        end_year = str((model_properties["end"]))
        # recognize areaname on pattern *_*_AREA_YEAR_YEAR_area_analysis.nc
        area = re.search('(.*/)(.*_.*_)(.*)(_)(\d{4})_(\d{4})(_.*)(.nc)$', i_data)
        if (area):
            area_name = area.group(3)
        else:
            area_name = "area_not_identified"

        plotname = model_name + "_" + area_name + "_" + start_year + "_" + end_year
        datacube = iris.load_cube(i_data)
        os.chdir(i_datafolder)
        # plot density histogram
        bins = np.arange(10, overall_max + 10, 5)
        plt.xlabel("daily snowfall (mm)")
        plt.ylabel("share of days in resp. bin")
        plt.ylim(overall_lower_ylim, overall_upper_ylim)
        plt.title(
            area_name)
        plt.suptitle(model_name + "  from " + start_year + " to " + end_year)
        plt.hist(np_data, bins, range=(10, overall_max), density=True)
        plt.savefig((plotname + "_density"))
        plt.close()
