#!/usr/bin/env python

# script to analyse data for a certain specified grid point (Lat,Lon) from output of snow_processing:

# imports

import argparse
import re

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes
from ruamel.yaml import ruamel


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def get_cube_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist[0]


# def plot function

def contour_plot_intensity_data(data, contour_levels, filename=''):
    # Plot the results.
    qplt.contourf(data, contour_levels, cmap='GnBu')
    plt.gca().coastlines()

    plt.savefig(filename)
    iplt.show()


def contour_plot_compare_intensity_data(data, contour_levels, filename=''):
    # Plot the results.
    qplt.contourf(data, contour_levels, cmap='coolwarm')
    plt.gca().coastlines()

    plt.savefig(filename)
    iplt.show()


# model identification based on structure of filename: str("output_" + i_model + "_" + timeindex + ".nc") from Pathname Collection Helper
def model_identification_re(filepath):
    model = re.search('(.*/)(output_)(.*)(_)(\d{4})(\d{4})(.nc)$', filepath)
    model_properties = {}
    if (model):

        model_properties["name"] = model.group(3)
        model_properties["start"] = int(model.group(5))
        model_properties["end"] = int(model.group(6))
    else:
        model_properties["name"] = "model_not_identified"
        model_properties["start"] = 0000
        model_properties["end"] = 0000
    return model_properties


# save data to nc file in outputlocation
def safe_data_nc(data, analysisidentifier):
    iris.save(data, outputdir + '/data_analysis_' + analysisidentifier + '.nc')


# argument parser definition
parser = argparse.ArgumentParser(description="Calculate some analysis metrics on specified files")
# path to *.yml file with settings to be used
parser.add_argument(
    "--settings",
    nargs="+"
    , type=str,
    required=True,
    help="Path to the settingsfile"
)

parser.add_argument(
    "--data",
    nargs="+"
    , type=str,
    required=True,
    help="YML list of datafile(s)"
)

args = parser.parse_args()

for i_settings in args.settings:
    # load settings file
    yaml = ruamel.yaml.YAML()
    with open(i_settings, 'r') as stream:
        try:
            settings = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    variable = settings["variable"]
    area_lat = settings["area"]["lat"]
    area_lon = settings["area"]["lon"]
    area_name = settings["area"]["name"]
    outputdir = settings["output"]

    for i_data in args.data:
        # load data file
        yaml = ruamel.yaml.YAML()
        with open(i_data, 'r') as stream:
            try:
                data = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # get model properties for plots
        # TODO: resolve workaround for year recognition
        start_year = 50000
        end_year = 0

        for i_datafile in data:
            model_properties = model_identification_re(i_datafile)
            model_name = (model_properties["name"])
            partial_start_year = (model_properties["start"])
            partial_end_year = (model_properties["end"])
            if (partial_start_year < start_year):
                start_year = partial_start_year
            if (partial_end_year > end_year):
                end_year = partial_end_year

        string_start_year = str(start_year)
        string_end_year = str(end_year)
        # main part of script
        # concatenate all input files in one cube (NB: assumes data unique wrt to time) TODO: generalize recognition of identical model

        # import all single data files into one cubelist

        complete_data_cube = iris.load(data)

        # filter for variable of interest

        filtered_data_cube = complete_data_cube.extract(variable)

        # equalise attributes of cubes

        equalise_attributes(filtered_data_cube)

        # concatenate cubes

        concatenated = filtered_data_cube.concatenate_cube()

        # restrict on target coords (for high resolution models) TODO: auto recognition of high or low resolution model

        area_constraint_hr = iris.Constraint(
            latitude=lambda v: area_lat - 90 / 192 <= v <= area_lat + 90 / 192,
            longitude=lambda v: area_lon - 180 / 384 <= v <= area_lon + 180 / 384)

        # restrict on target coords (for low resolution models)

        area_constraint_lr = iris.Constraint(
            latitude=lambda v: area_lat - 90 / 144 <= v <= area_lat + 90 / 144,
            longitude=lambda v: area_lon - 180 / 192 <= v <= area_lon + 180 / 192)

        target_area_data = concatenated.extract(area_constraint_hr)
        print(target_area_data)

        # define plotfilenamestump

        plotname = outputdir + "/" + model_name + "_" + area_name + "_" + string_start_year + "_" + string_end_year

        # plot time series of snow fall

        iris.plot.plot(target_area_data)
        plt.xlabel("time")
        plt.ylabel("daily snowfall (mm)")
        plt.title(
            "Lat. " + str(area_lat) + " and lon. " + str(area_lon) + ", " + area_name)
        plt.suptitle(model_name + " from " + string_start_year + " to " + string_end_year)
        plt.savefig((plotname + "_timeseries"))
        plt.close()
        # plot histogram
        np_data = target_area_data.data

        # TODO: generalize diagram labels

        bins = np.arange(10, np_data.max() + 10, 5)
        plt.xlabel("daily snowfall (mm)")
        plt.ylabel("number of days in resp. bin")
        plt.title(
            "Lat. " + str(area_lat) + " and Lon. " + str(area_lon) + ", " + area_name)
        plt.suptitle(model_name + " from " + string_start_year + " to " + string_end_year)
        plt.hist(target_area_data.data, bins, range=(10, np_data.max()))
        plt.savefig((plotname + "_histogramm"))
        plt.close()
        # plot density histogram
        np_data = target_area_data.data

        bins = np.arange(10, np_data.max() + 10, 5)
        plt.xlabel("daily snowfall (mm)")
        plt.ylabel("share of days in resp. bin")
        plt.title(
            "Lat. " + str(area_lat) + " and Lon. " + str(area_lon) + ", " + area_name)
        plt.suptitle(model_name + "  from " + string_start_year + " to " + string_end_year)

        plt.hist(target_area_data.data, bins, range=(10, np_data.max()), density=True)
        plt.savefig((plotname + "_density"))
        plt.close()

        iris.save(target_area_data,
                  outputdir + '/' + model_name + "_" + area_name + "_" + string_start_year + "_" + string_end_year + '_area_analysis.nc')
