import argparse
import os

from ruamel.yaml import ruamel
# settings file import
# argument parser definition
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate a movie grid from movie files")
# path to *.yml file with movies to be used
parser.add_argument(
    "--movies",
    nargs="+"
    , type=str,
    required=True,
    help="Path to the movies file"
)

parser.add_argument(
    "--rows"
    , type=int,
    required=True,
    help="Rows of the output grid"
)

parser.add_argument(
    "--columns"
    , type=int,
    required=True,
    help="Columns of the output grid"
)

parser.add_argument(
    "--outputfile"
    , type=str,
    required=True,
    help="Filename for mp4 output"
)
args = parser.parse_args()

# load file with pathnames of movies
yaml = ruamel.yaml.YAML()
for i_movies in tqdm(args.movies):
    with open(i_movies, 'r') as stream:
        try:
            inputs = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # add up ffmpeg command with appropiate hstack and vstack
    cmd = ("ffmpeg")
    # add all input files
    for i_input in inputs:
        cmd += " -i {}".format(i_input)

    rows = range(1, args.rows + 1)
    columns = range(1, args.columns + 1)

    outputfile = args.outputfile + str(i_movies)

    cmd = cmd + ' -filter_complex "'
    for i in rows:
        cmd += "hstack=inputs={}[row{}];".format(len(columns), i)

    for i in rows:
        cmd += "[row{}]".format(i)
    cmd += 'vstack=inputs={}" {}.mp4'.format(len(rows), outputfile)

    print(cmd)
    os.system(cmd)
