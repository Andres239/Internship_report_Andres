#This script is based on another script coded by Elise Jouaiti, currently doctoral researcher at IPCMS

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.special import wofz
import scipy.constants as const


# Get data from file

def get_data(name):
    data = np.loadtxt(name, delimiter='\t', skiprows=0, encoding="iso-8859-1", unpack=True)
    return (data)

################################## Calibration

# Get x axis from calibration file

xaxis_nm = get_data('WSe2/WSe2_red/calibration_x_axis.txt')

################################## Map

filename = '/home/andres-rodriguez/TdG/WSe2/WSe2_red/Map/LRVI_H12P1_1-4__2024-12-18_16-28-37_tot.dat'
df = get_data(filename)

# Define data and noise matrices

data = []
noise = []

############################################################################
# The tot.dat file contains both data and noise. Even columns correspond to noise while odd columns correspond to data.
# The following part aims at inverting lines and columns while separating data and noise.
# Each list is constructed from the content of a given co2lumn and is added as a line to whether the data or to the noise matrix.

for k in range(int(len(df[0]) / 2)):
    list1 = []
    list2 = []
    for i in range(len(df)):
        list1.append(df[i][2 * k])
        list2.append(df[i][2 * k + 1])
    noise.append(list1)
    data.append(list2)

###############################################################
# Construction of a signal matrix by subtracting noise to data.
# This step may create negative spikes !

signal = []
for l in range(len(data)):
    list3 = []
    for m in range(len(data[0])):
        list3.append(data[l][m] - noise[l][m])
    signal.append(list3)

##########################################
# Construction of a 3D map matrix: 2D matrix (x and y).
# Each element of this matrix is a list corresponding to a spectrum at a given point.


# Map dimensions

print('Please, enter x total dimension')
x = 11

print('Please, enter y total dimension')
y = 11

# Initialization with zeros of the map matrix

carte = [[0 for i in range(x)] for j in range(y)]

# Initialization of a counter

a = 0

# Replacement of all 0 elements of the map matrix by lists from the signal matrix (spectra)

for o in range(y):
    for p in range(x):
        a = x * o
        carte[o][p] = signal[a + p]

# Zoom
print('Zoom. Be careful, indices for x and y start at 0.')

print('Please, enter xmin')
xmin = 0

print('Please, enter xmax')
xmax = 10
print('Please, enter ymin')
ymin = 0

print('Please, enter ymax')
ymax = 10

carte_new = [[0 for i in range(xmin, xmax + 1)] for j in range(ymin, ymax + 1)]


index_y = 0
for i in range(ymin, ymax + 1):
    index_x = 0

    for j in range(xmin, xmax + 1):
        carte_new[index_y][index_x] = carte[i][j]
        index_x = index_x + 1
    index_y = index_y + 1


# x axis converted in eV or raman shift

new_x_eV = (const.h * const.c) / (xaxis_nm * const.e * 10 ** -9)
#new_x_raman = 10 ** 7*(1 / 632.8 - 1 / xaxis_nm) 

min_limit = min(new_x_eV)
max_limit = max(new_x_eV)

# Ensure that noise is correctly removed

for c in range(ymax - ymin + 1):
    for d in range(xmax - xmin + 1):
        left_average = 0
        counter_left = 0
        for e in range(len(new_x_eV)):
            if min_limit <= new_x_eV[e] <= min_limit + 10:
                left_average += carte_new[c][d][e]
                counter_left = counter_left + 1
        left_average = left_average / counter_left

        right_average = 0
        counter_right = 0

        for t in range(len(new_x_eV)):
            if max_limit - 10 <= new_x_eV[t] <= max_limit:
                right_average += carte_new[c][d][t]
                counter_right += 1
        right_average = right_average / counter_right

        x_left = min_limit
        x_right = max_limit

        coefficient_director = (right_average - left_average) / (x_right - x_left)
        constant = right_average - coefficient_director * x_right

        carte_new[c][d] = carte_new[c][d] - coefficient_director * new_x_eV - constant

# Construction of the 2D matrix containing the intensity at each point of the map

# Initialization with zeros of the sum matrix

sum = [[0 for i in range(xmin, xmax + 1)] for j in range(ymin, ymax + 1)]
sum_bis = np.array(sum)

# Definition of the limits of integration:

print('Definition of the integration limits')
print('Please, enter the lower bound of integration (eV). For WSe2 use 1:2 eV')
energy_min = float(input())

print('Please, enter the upper bound of integration (eV)')
energy_max = float(input())

# Integration by doing a sum of all intensities within the integration limits.


'''
left_average = 0
counter_left = 0
for e in range(len(new_x_raman)):
    if min_limit <= new_x_raman[e] <= min_limit + 20:
        left_average += carte_new[4][7][e]
        counter_left = counter_left + 1
left_average = left_average / counter_left

right_average = 0
counter_right = 0

for t in range(len(new_x_raman)):
    if 3000 <= new_x_raman[t] <= 3060:
        right_average += carte_new[4][7][t]
        counter_right += 1
right_average = right_average / counter_right

x_left = min_limit
x_right = max_limit

coefficient_director = (right_average - left_average) / (x_right - x_left)
constant = right_average - coefficient_director * x_right

carte_new[4][7] = carte_new[4][7] - coefficient_director * new_x_raman - constant

'''
integrated_intensity_list = []

for t in range(len(new_x_eV)):
    if new_x_eV[t] >= energy_min and new_x_eV[t] <= energy_max:
        for r in range(ymax - ymin + 1):
            for s in range(xmax - xmin + 1):
                sum_bis[r][s] += carte_new[r][s][t]


for i in range(len(sum_bis[0])):
    for j in range(len(sum_bis)):
        if sum_bis[j][i] < 0:
            sum_bis[j][i] = 0


# Image the map and colorbar

fig, ax = plt.subplots()
cax = ax.imshow(sum_bis, interpolation='nearest', cmap=plt.cm.jet)

# Add a colorbar to the side of the map

cbar = fig.colorbar(cax)
cbar.set_label('Integrated intensity (a.u.)',  rotation=270, labelpad=20)

# Definition of a function that makes the map clickable

Last_marker = None

# Function that will be called when the map is clicked

def on_click(event):
    # Check if the mouse click was within the axes

    if event.inaxes is not None:
        global Last_marker
        # Get the x and y coordinates of the click

        x, y = event.xdata, event.ydata

        # Convert to integer indices so that it corresponds to the matrix indices

        ix, iy = int(round(x)), int(round(y))
        print(f"Clicked at coordinates: ({x:.2f}, {y:.2f})")
        print(f"Clicked at coordinates: ({ix}, {iy})")



        fig2, axs2 = plt.subplots(1, 1, figsize=(6, 3))
        spectrum = axs2.plot(new_x_eV, carte_new[iy][ix], color='purple')
        axs2.set_xlim(left=min_limit, right=max_limit)
        axs2.set_ylabel('Intensity (counts)')
        axs2.set_xlabel('Energy (eV)')
        fig2.show()



        if Last_marker is not None:
            Last_marker.remove()

        # 'ro' for red circle marker

        Last_marker = ax.plot(ix, iy, 'ro')[0]

        # Update the plot

        plt.draw()
    else:
        print("Clicked outside axes boundaries but still inside the plot window")


# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

