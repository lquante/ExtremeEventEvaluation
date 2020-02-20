# This script provides some utility to generate yml settingsfiles for an ensemble analysis
#  imports
import argparse
import os
import re
from pathlib import Path

from ruamel.yaml import ruamel

# Define parser
parser = argparse.ArgumentParser(description="Generate an ensemble of settings file for a specified list of underlying data files")


# Define  yaml settings blueprint to be amended with pathname
parser.add_argument(
    "--blueprint",
    type=str,
    help="Path to settings blueprint (default: CURRENT/settings_blueprint.yml)",
)


parser.add_argument(
    "--underlyingData",
    type=str,
    help="Path to list of underlying data  (default: CURRENT/datalist.yml)",
)

parser.add_argument(
    "--time_thresholds",
    nargs = '+', type=int,
    help="Time thresholds to be used",
)

parser.add_argument(
    "--depth_thresholds",
    nargs='+', type=int,
    help="Depth thresholds to be used",
)


parser.add_argument(
    "--outputdir"
    , type=str,
    help="Path  to output directory  (default: CURRENT)"
)

# path to store settingsfiles

parser.add_argument(
    "--settingsdir"
    , type=str,
    help="Path to directory for settingsfiles and a list of them and the outputfiles (default: CURRENT)"
)

args = parser.parse_args()

# default settings blueprint
if not args.blueprint:
    args.blueprint = os.path.join(os.getcwd(), "settings_blueprint.yml")

if not args.underlyingData:
    args.underlyingData = os.path.join(os.getcwd(), "datalist.yml")

# default settings output file
if not args.outputdir:
    args.outputdir = os.getcwd()
# default settings for path to put settingsfiles
if not args.settingsdir:
    args.settingsdir = os.getcwd()
# load settings blueprint file
yaml = ruamel.yaml.YAML()
with open(args.blueprint, 'r') as stream:
    try:
        settings = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# load underlying data file
yaml = ruamel.yaml.YAML()
with open(args.underlyingData, 'r') as stream:
    try:
        data = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)



settingspathcollection = []
for i_data in data:

        settings ["input"] = i_data
        settings ["thresholds"]["time"] = args.time_thresholds
        settings["thresholds"]["depth"] = args.depth_thresholds
        settings ["output"] = args.outputdir
        # save new settings file
        data_file = re.search('(.*/)(output_)(.*)(.nc)$', str(i_data))
        if (data_file):
            data_filename= data_file.group(3)
        else:
            data_filename = "data_not_identified"
        name_settings = "analysis_settings_"+data_filename+".yml"
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = None
        os.chdir(args.settingsdir)
        with open(name_settings, "w") as output:
            yaml.dump(settings, output)
        # collect paths to settings
        settingspathcollection.append(os.path.join(os.getcwd(), name_settings))
# create *.yml file of settingsfiles:
os.chdir(args.settingsdir)
yaml = ruamel.yaml.YAML()
yaml.default_flow_style = None
with open("list_of_settings.yml", "w") as output:
    yaml.dump(settingspathcollection, output)
