from datetime import datetime

import iris
import numpy as np

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
from iris.analysis import Aggregator

from matplotlib.animation import FuncAnimation


def load_data_from_netcdf(filepath):
    # Load the whole data (as a list of cubes)
    file_path = (filepath)
    data = iris.load(file_path)
    return data


def get_cube_from_cubelist(data, variablename):
    filtered_cubelist = data.extract(variablename)
    return filtered_cubelist[0]


# Some global variables to define the whole run
basic_data = load_data_from_netcdf(
    "/home/quante/projects/extremesnowevents/Climate_Data_Central/output_MPI-ESM1-2-HR_ssp585_20952099.nc")
data_name = "Daily snowfall"
data_identifier = "MPI-ESM1-2-HR_ssp585"

variable_data = get_cube_from_cubelist(basic_data, "approx_fresh_daily_snow_height")
contours = np.arange(0, 1000, 10)

# def plot function
def contour_plot_geodata(data, contour_levels):
    # Plot the results.
    iris.plot.contourf(data, contour_levels, cmap='GnBu')

# extract time dimension
time = variable_data.coord('time')
number_of_timepoints = time.points.size
# define timeshift from 1-1-1 to 1850-1-1 => 365* = 674885
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
print(empty_data)
titled_world_map = plt.figure()
first_date = datetime.fromordinal(int(time.points[0]) + start_shift)
last_date = datetime.fromordinal(int(time.points[number_of_timepoints-1]) + start_shift)
plt.title(data_name+ " from " + str(first_date) +" to "+str(last_date))
plt.suptitle(data_identifier)
iris.plot.contourf(empty_data,contours, cmap='GnBu')
plt.gca().coastlines()


animation = FuncAnimation(
    # Your Matplotlib Figure object
    titled_world_map,
    # The function that does the updating of the Figure
    animate,
    # Frame information (here just frame number)
    np.arange(1, number_of_timepoints, 1),
    fargs=[],
    # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
    interval=40
)

# Try to set the DPI to the actual number of pixels you're plotting
animation.save("movie_"+data_name+"_"+data_identifier+"_"+ str(first_date) +"_"+str(last_date)+".mp4", dpi=512)
