#!/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as optimize
import matplotlib.widgets as wdg
from matplotlib.widgets import Button, Slider
import functools


# Extract data from text file (function adapted for the original Neon calibration file, .spe converted to .dat)
def get_data(name):
    data = np.loadtxt(name, delimiter=' ', skiprows=0, encoding="iso-8859-1", unpack=True)
    return (data)


# Extract data from text file (function adapted for the calibration parameters saved in a text file)
def get_data2(name):
    data = np.loadtxt(name, delimiter='\t', skiprows=0, encoding="iso-8859-1", unpack=True)
    return (data)


# Definition of a quadratic function
def quad(x, a, b, c):
    y = a * x * x + b * x + c
    return y


# Definition of a linear function
def lin(x, a, b):
    y = a * x + b
    return y


# Definition a function that enables to fit the calibration line with a quadratic function
def fitacalib(pos, pos_real):
    p0 = [0.6, -1, 1]
    p1, error = optimize.curve_fit(quad, pos, pos_real, p0[:])
    return p1


# Definition of the lorentzian fit function
def lorentz(x, width, amp, pos):
    return amp / np.pi * (width / 2) / ((x - pos) ** 2 + (width / 2) ** 2)


# Definition of a function that enables to update the boundary limits of the fit and the position of the vertical lines
# according to the range slider.

def update(val):
    global z

    # Rescale the x axis considering the parameters of a quad function (a, b and c).
    # Parameters can be manually modified from the user interface

    calib_x = np.zeros(1340)
    for i in range(len(x)):
        calib_x[i] = quad(x[i], a.val, b.val, c.val)

    # Correct the boundary limits of the range slider considering the parameters of the quad function

    min_x = quad(slider.val[0], a.val, b.val, c.val)
    max_x = quad(slider.val[1], a.val, b.val, c.val)

    # Update the position of the vertical lines according to the information provided by the range slider

    lower_limit_line.set_xdata([min_x, min_x])
    upper_limit_line.set_xdata([max_x, max_x])

    # Update the x and y values that are considered for the fit

    new_x = [quad(x[i], a.val, b.val, c.val) for i in range(len(x)) if min_x < quad(x[i], a.val, b.val, c.val) < max_x]
    new_y = [y[i] for i in range(len(x)) if min_x < quad(x[i], a.val, b.val, c.val) < max_x]

    # Update the lorentzian fit according to the modified boundary limits

    z, err = sp.optimize.curve_fit(lorentz, new_x, new_y)
    graph[0].set_xdata(new_x)
    graph[0].set_ydata(lorentz(new_x, *z))

    spectrum[0].set_xdata(calib_x)

    # Redraw the figure to ensure it updates

    fig.canvas.draw_idle()


# Definition of a function that enables to select a range of the x and y axes to fit the spectrum on a restricted
# range

def selection(x, y, fit_min, fit_max):
    new_x = [x[i] for i in range(len(x)) if fit_min < x[i] < fit_max]
    new_y = [y[i] for i in range(len(x)) if fit_min < x[i] < fit_max]
    return new_x, new_y

# Definition of the diverse buttons that are used in the code and available for the user on the interface

class Boutons:
    # Definition of a function that stores the position of the fitted peak in a list if the button "Save peak position"
    # is clicked. The position of the peak is also stored in a table in another window.

    def peakposition(self, event):
        list_centers_fit_peaks.append(z[2])
        list1.append([z[2]])
        table3.add_cell(len(list1), 0, width=1, height=.06, text=list1[len(list1) - 1][0], loc="center")
        fig3.canvas.draw_idle()

    # Definition of a function that stores the selected calibration line if one of the buttons with a Neon wavelength
    # (on the left side of the interface) is clicked.

    def select_neon_ref(self, counter, event):
        list_peaks_neon.append(restricted_nist_list[counter])
        list0.append([restricted_nist_list[counter]])
        table0.add_cell(len(list0), 0, width=1, height=.06, text=list0[len(list0) - 1][0], loc="center")
        fig0.canvas.draw_idle()

    # Definition of a function that stores after calibration the position of the fitted peak in a list if the button "Save peak position"
    # is clicked. The aim is to compare the peak position of the Neon line after calibration to tabulated values and compute an error
    # value.

    def peakposition_after_calibration(self, event):
        list_centers_fit_peaks_after_calibration.append(z2[2])

        for j in range(1340):
            if new_x_calibrated[j] >= z2[2]:
                dif1 = abs(z2[2] - [x[j]])
                dif2 = abs(z2[2] - [x[j - 1]])
                if dif1 <= dif2:
                    pos_pix_after_calib = j - 1
                    break
                else:
                    pos_pix_after_calib = j - 2
                    break

        list_centers_fit_peaks_after_calibration_pos_pix.append(pos_pix_after_calib)

    # Definition of a function that computes the error (difference between the tabulated value of the Neon line and
    # the fitted peak position after calibration). The function also plots the spectrum with the correct x axis and
    # displays the error above each peak.

    def error(self, event):
        fig6, axs6 = plt.subplots(1, 1, figsize=(6, 4))
        for i in range(len(list_peaks_neon)):
            error_list.append(0)
            error_list[i] = list_peaks_neon[i] - list_centers_fit_peaks_after_calibration[i]
            error_list[i] = round(error_list[i], 4)

            plt.text(list_centers_fit_peaks_after_calibration[i], y[list_centers_fit_peaks_after_calibration_pos_pix[i]], str(error_list[i]) + " nm", fontsize=10,
                     ha='right')


        graph6 = axs6.plot(new_x_calibrated, y, color='purple')
        axs6.set_ylabel('Intensity (counts)')
        axs6.set_xlabel('Wavelength (nm)')
        axs6.set_xlim(left=min_limit2, right=max_limit2)

        fig6.show()

        print('error',error_list)


    # Definition of a function that saves the calibration by creating a file with two columns (col 1 and col 2)
    # respectively the positions of the fitted peaks before calibration (col 1) and the selected Neon calibration lines (col 2).

    def save_calibration(self, event):
        global list_centers_fit_peaks, list_peaks_neon, new_x_calibrated
        file_calibration = open("calibration.txt", "w")
        for i in range(len(list_centers_fit_peaks)):
            file_calibration.write(str(list_centers_fit_peaks[i]) + "\t" + str(list_peaks_neon[i]) + "\n")
        file_calibration.close()

        file_calibration_x = open("calibration_x_axis.txt", "w")
        for j in range(len(new_x_calibrated)):
            file_calibration_x.write(str(new_x_calibrated[j]) + "\n")
        file_calibration_x.close()

    # Definition of a function that extracts the saved peak positions and calibration lines from the calibration file
    # called "calibration.txt".

    def open_saved_calibration(self, event):
        global list_centers_fit_peaks, list_peaks_neon
        filename = 'calibration.txt'
        list_centers_fit_peaks, list_peaks_neon = get_data2(filename)

    # Definition of a function that empties the list of the peak positions and chosen calibration lines when the "Reset"
    # button is clicked.

    def reset1(self, event):
        global list_peaks_neon, list_centers_fit_peaks, table0, table3
        list_peaks_neon = []
        list_centers_fit_peaks = []
        list3 = [["Peak positions (nm)"]]
        list0 = [["neon lines (nm)"]]
        table3.remove()
        table0.remove()
        table3 = axs3.table(cellText=list1, cellLoc="center", loc="center")
        table0 = axs0.table(cellText=list0, cellLoc="center", loc="center")

        fig3.canvas.draw_idle()
        fig0.canvas.draw_idle()

    # Definition of a function that empties the list of the peak positions after calibration when the "Reset"
    # button is clicked.

    def reset2(self, event):
        global list_centers_fit_peaks_after_calibration, list_centers_fit_peaks_after_calibration_pos_pix
        list_centers_fit_peaks_after_calibration = []
        list_centers_fit_peaks_after_calibration_pos_pix = []
        error_list = []
        print('error list', error_list)

    # Calibration: creates a new x axis

    def real_calib(self, event):
        global new_x_calibrated

        #Convert the fitted peak positions to pixel positions

        pos_pix = np.zeros(len(list_centers_fit_peaks))
        for i in range(len(list_centers_fit_peaks)):
            for j in range(1340):
                if x[j] >= list_centers_fit_peaks[i]:
                    dif1 = abs(list_centers_fit_peaks[i] - [x[j]])
                    dif2 = abs(list_centers_fit_peaks[i] - [x[j - 1]])
                    if dif1 <= dif2:
                        pos_pix[i] = j
                        break
                    else:
                        pos_pix[i] = j - 1
                        break

        # Find the calibration function which is a quadratic function linking the pixel positions that were previously
        # calculated and the tabulated Neon lines

        results = fitacalib(pos_pix, list_peaks_neon)


        # Create the new x-axis using the calibration function

        new_x_calibrated = np.zeros(1340)
        for i in range(1340):
            new_x_calibrated[i] = results[0] * i ** 2 + results[1] * i + results[2]


        # 2 subplots

        fig4, axs4 = plt.subplots(2, 1, figsize=(6, 4))

        # Plot the spectrum with the calibrated x axis

        graph4 = axs4[0].plot(new_x_calibrated, y, color='purple')
        axs4[0].set_ylabel('Intensity (counts)')
        axs4[0].set_xlabel('Wavelength (nm)')
        axs4[0].set_xlim(left=min_limit, right=max_limit)

        # Plot the tabulated calibration lines as a function of peak positions (pixel). If the figure does not represent
        # a straight line, this means that the Neon peaks have not been chosen properly.

        straight_line = axs4[1].plot(pos_pix, list_peaks_neon, color='blue', marker='o')
        axs4[1].set_ylabel('Tabulated calibration lines (nm)')
        axs4[1].set_xlabel('Peak position (pixel)')

        fig4.show()


# Beginning of the code

if __name__ == "__main__":

    # Original Neon calibration file

    filename_original_calibration = r'C:\Users\Administrateur\Desktop\Sotos\Measurements2025_01_14\Neon_calibration_150gr_800nm_1.dat'
    x, y = get_data2(filename_original_calibration)


    # Noise removal based on finding an equation for a straight line that can be subtracted to the signal
    # An averaged signal is computed for each side of the spectrum.
    # Boundary limits for the average on each side can be directly modified in the code.

    left_average = 0
    counter_left = 0
    min_limit = min(x)
    max_limit = max(x)

    for e in range(len(x)):
        if min_limit <= x[e] <= min_limit + 10:
            left_average += y[e]
            counter_left += 1
    left_average = left_average / counter_left

    right_average = 0
    counter_right = 0

    for t in range(len(x)):
        if max_limit - 10 <= x[t] <= max_limit:
            right_average += y[t]
            counter_right += 1
    right_average = right_average / counter_right

    x_left = min_limit
    x_right = max_limit

    coefficient_director = (right_average - left_average) / (x_right - x_left)
    constant = right_average - coefficient_director * x_right

    y = y - coefficient_director * x - constant

    # Initialization of the data selection to fit a peak when the interface opens.
    # Boundary limits are defined as the minimum and maximum values of the original x axis.

    new_x, new_y = selection(x, y, min_limit ,max_limit)

    # Initialization of the fit when the interface opens.

    z, err = sp.optimize.curve_fit(lorentz, new_x, new_y)



    # 2 subplots

    fig, axs = plt.subplots(2, 1, figsize=(5, 3))

    # Enough space for the buttons on the left and the range sliders at the bottom

    fig.subplots_adjust(bottom=0.38)
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(top=0.95)

    # Plot the spectrum with the original x axis

    spectrum = axs[0].plot(x, y, color='purple')
    axs[0].set_xlim(left=min_limit, right=max_limit)
    axs[0].set_ylabel('Intensity (counts)')
    axs[1].set_xlabel('Wavelength (nm)')
    axs[1].set_ylabel('Tabulated calibration lines')

    # Plot the fit function

    graph = axs[0].plot(new_x, lorentz(new_x, *z), color='orange')


    # List containing the Neon lines in the visible and NIR range

    nist_calibration_lines_vis_NIR = [376.6259, 377.7133, 540.05618, 585.24879, 602.99969, 607.43377, 614.30626,
                                      616.35939, 621.72812, 626.64950, 638.29917, 640.2248, 650.65281, 659.89529,
                                      692.94673, 703.24131, 717.39381, 724.51666, 837.76080, 865.43831, 878.06226,
                                      878.37533, 1114.30200, 1117.75240, 1152.27459]

    restricted_nist_list = []
    counter = 0
    list_peaks_neon = []
    button_position = [0.9, 0.8, 0.7]
    axs[1].set_xlim(left=min_limit, right=max_limit)

    # Table containing the peak positions in a separate window

    list1 = [["Peak positions"]]
    fig3, axs3 = plt.subplots(1, 1)
    axs3.set_axis_off()
    table3 = axs3.table(cellText=list1, cellLoc="center", loc="center")
    fig3.show()

    # Table containing the selected Neon lines in a separate window

    list0 = [["Neon lines"]]
    fig0, axs0 = plt.subplots(1, 1)
    axs0.set_axis_off()
    table0 = axs0.table(cellText=list0, cellLoc="center", loc="center")
    fig0.show()

    # Create the RangeSlider

    slider_ax = fig.add_axes([0.249, 0.26, 0.64, 0.025])
    slider = wdg.RangeSlider(slider_ax, "Fitting range", x.min(), x.max())

    # Create the Vertical lines

    lower_limit_line = axs[0].axvline(slider.val[0], color='k')
    upper_limit_line = axs[0].axvline(slider.val[1], color='k')

    slider.on_changed(update)

    # Create the sliders to test a correction calibration function: ax²+bx+c. The a,b and c parameters can be modified
    # manually by the user from the interface.

    axa = fig.add_axes([0.20, 0.20, 0.65, 0.03])
    a = Slider(ax=axa, label='a', valmin=-.00001, valmax=.00001, valinit=0, )
    axb = fig.add_axes([0.20, 0.15, 0.65, 0.03])
    b = Slider(ax=axb, label='b', valmin=.9, valmax=1.1, valinit=1, )
    axc = fig.add_axes([0.20, 0.10, 0.65, 0.03])
    c = Slider(ax=axc, label='c', valmin=-30, valmax=30, valinit=0, )

    # The correction calibration function modifies the axis and spectrum of the upper figure.

    a.on_changed(update)
    b.on_changed(update)
    c.on_changed(update)

    # Button "Save peak position" on the first figure

    list_centers_fit_peaks = []
    callback2 = Boutons()
    axbouton = fig.add_axes([0.5, 0.015, 0.18, 0.065])
    cnext = Button(axbouton, 'Save peak position')
    cnext.on_clicked(callback2.peakposition)

    # Button "Reset" on the first figure

    callback8 = Boutons()
    axbouton8 = fig.add_axes([0.2, 0.015, 0.18, 0.065])
    cnext8 = Button(axbouton8, 'Reset')
    cnext8.on_clicked(callback8.reset1)

    # Button "Calibration" on the first figure

    callback3 = Boutons()
    axbouton3 = fig.add_axes([0.8, 0.015, 0.18, 0.065])
    cnext3 = Button(axbouton3, 'Calibration')
    cnext3.on_clicked(callback3.real_calib)

    # Button "Save calibration" on the first figure

    callback10 = Boutons()
    axbouton10 = fig.add_axes([0.905, 0.9, 0.092, 0.05])
    cnext10 = Button(axbouton10, 'Save calibration')
    cnext10.on_clicked(callback10.save_calibration)
    cnext10.label.set_fontsize(8)

    # Button "Use save calibration" to skip peak position selection as well as the selection of the corresponding Neon
    # lines

    callback11 = Boutons()
    axbouton11 = fig.add_axes([0.905, 0.8, 0.092, 0.05])
    cnext11 = Button(axbouton11, 'Use saved calibration')
    cnext11.on_clicked(callback11.open_saved_calibration)
    cnext11.label.set_fontsize(7.8)

    # This section of the script creates the button list on the left. It contains all the Neon line wavelengths within
    # the considered wavelength range.

    restricted_nist_list = []
    counter = 0
    list_peaks_neon = []
    axs[1].set_xlim(left=min_limit, right=max_limit)

    bnext = []
    callback = Boutons()
    button_names = []

    for i in range(len(nist_calibration_lines_vis_NIR)):
        if min_limit < nist_calibration_lines_vis_NIR[i] and nist_calibration_lines_vis_NIR[i] < max_limit:
            restricted_nist_list.append(nist_calibration_lines_vis_NIR[i])
            axs[1].axvline(x=nist_calibration_lines_vis_NIR[i], color='red')
            line_position = restricted_nist_list[counter]
            axbouton = fig.add_axes([0.015, 0.95 - .05 * counter, 0.12, 0.04])

            # Button names

            button_names.append(str(restricted_nist_list[counter]))
            bnext.append(Button(axbouton, button_names[counter] + " nm"))
            bnext[counter].on_clicked(functools.partial(callback.select_neon_ref, counter))
            counter += 1

    fig.show()
    plt.show()


    # The update2 function enables to update the fit and the positions of the vertical lines in a third major window
    # after closing the two first major ones. This enables to fit the peaks with the calibrated x axis to further compute
    # the error on the position for each peak.

    def update2(val):
        global slider2, lower_limit_line, upper_limit_line, graph5, fig5, z2
        global new_x_calibrated

        min_x = slider2.val[0]
        max_x = slider2.val[1]

        # Update the fit

        new_x2 = [new_x_calibrated[i] for i in range(len(new_x_calibrated)) if min_x < new_x_calibrated[i] < max_x]
        new_y2 = [y[i] for i in range(len(new_x_calibrated)) if min_x < new_x_calibrated[i] < max_x]

        # Update the position of the vertical lines

        lower_limit_line.set_xdata([min_x, min_x])
        upper_limit_line.set_xdata([max_x, max_x])

        z2, err2 = sp.optimize.curve_fit(lorentz, new_x2, new_y2)
        graph5[0].set_xdata(new_x2)
        graph5[0].set_ydata(lorentz(new_x2, *z2))

        # Redraw the figure to ensure it updates

        fig5.canvas.draw_idle()


    min_limit2 = min(new_x_calibrated)
    max_limit2 = max(new_x_calibrated)

    # Selection of a restricted wavelength to fit

    new_x2, new_y2 = selection(new_x_calibrated, y, min_limit2 , max_limit2)

    # fit

    z2, err2 = sp.optimize.curve_fit(lorentz, new_x2, new_y2)

    # Subplot of 1 plot

    fig5, axs5 = plt.subplots(1, 1, figsize=(6, 4))

    # Include some space for a slider

    fig5.subplots_adjust(bottom=0.35)

    # Third major window representing the spectrum with the calibrated x axis

    graph4 = axs5.plot(new_x_calibrated, y, color='purple')
    axs5.set_ylabel('Intensity (counts)')
    axs5.set_xlabel('Wavelength (nm)')
    axs5.set_xlim(left=min_limit2, right=max_limit2)

    graph5 = axs5.plot(new_x2, lorentz(new_x2, *z2), color='orange')

    # Create the RangeSlider

    slider2_ax = fig5.add_axes([0.15, 0.2, 0.67, 0.025])

    slider2 = wdg.RangeSlider(slider2_ax, "Fitting range", new_x_calibrated.min(), new_x_calibrated.max())

    # Create the Vertical lines

    lower_limit_line = axs5.axvline(slider2.val[0], color='k')
    upper_limit_line = axs5.axvline(slider2.val[1], color='k')

    slider2.on_changed(update2)

    # Button "Save peak position" for the fitted peaks

    list_centers_fit_peaks_after_calibration = []
    list_centers_fit_peaks_after_calibration_pos_pix = []
    callback3 = Boutons()
    axbouton3 = fig5.add_axes([0.4, 0.015, 0.18, 0.065])
    cnext3 = Button(axbouton3, 'Save peak position')
    cnext3.on_clicked(callback3.peakposition_after_calibration)

    # Button "Error" to compute the difference between the tabulated Neon lines and the fitted peak positions and
    # displays a plot with the error near to each fitted peak.

    error_list = []
    callback4 = Boutons()
    axbouton4 = fig5.add_axes([0.72, 0.015, 0.18, 0.065])
    cnext4 = Button(axbouton4, 'Error')
    cnext4.on_clicked(callback4.error)

    # Button "Reset" to empty the peak position list as well as the error list

    callback9 = Boutons()
    axbouton9 = fig5.add_axes([0.1, 0.015, 0.18, 0.065])
    cnext9 = Button(axbouton9, 'Reset')
    cnext9.on_clicked(callback9.reset2)

    fig5.show()

    plt.show()

