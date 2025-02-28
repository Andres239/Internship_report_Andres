import numpy as np
from scipy.optimize import curve_fit
import scipy as sp
import matplotlib.pyplot as plt

'''This script contains functions to analyze graphene samples, it is devided in two parts:
    1. Raman fitting: It fits the G and 2D peaks of a Raman spectrum
    2. Map processing: It processes Raman maps
    Coming soon: Functions to analyze the parameters obtained through the fitting and functions to plot the results'''

#------------------FITTING THE 2D AND G PEAKS------------------#


#Fitted functions: The G peak is fitted to a Lorentzian function and the 2D peak is fitted to a double Basko function
def lorentzian(x, offset, center, width, area):
    return offset + 2*(area/np.pi*width)/(4*(x-center)**2 + width**2)

def db_basko(x,  offset, center, width, area, rel_area, shift):
    return offset + (0.5*rel_area*area*width**2) / ((x-center)**2 + (width)**2)**(3/2) + (0.5*(1-rel_area)*area*width**2) / ((x-center-shift)**2 + (width)**2)**(3/2)

def basko(x, offset, center, width, area):
    return offset + 0.5*(area*width**2) / ((x-center)**2 + (width)**2)**(3/2)

#Treatment of the data
def select_ROI(x, y, center, width):
    ind = np.abs(x - center) < width
    return x[ind], y[ind]

def remove_spikes(x, y, spike1, spike2):
    #Spike 1 are really high values and spike 2 are really high variations
    y = y[y < spike1]
    delta = np.diff(y)
    y = y[np.insert(np.abs(delta) < spike2, 0, True)]
    return x[:len(y)], y

def find_mean_offset(y, n):
    return 0.5 * (np.mean(y[:n]) + np.mean(y[-n:]))

def remove_offset(x, y, n):
    a = (np.mean(y[-n:]) - np.mean(y[:n])) / (x[-1] - x[0])
    b = np.mean(y[:n])
    #print(a, b)
    offset = b + a * (x - x[0])

    return x, y - offset

def find_x0_y0(x, y, omega0):
    return x[np.argmin(np.abs(x - omega0))], y[np.argmin(np.abs(x - omega0))]

def convert_nm_to_raman(x_axis_nm, lambda_laser):
    return  1e7 / lambda_laser - 1e7 / x_axis_nm

#Fitting function
def raman_Gr_fit(x_data, y_data, omegaG, omega2D, n=10):

    n = 10  # Number of points to calculate the mean offset


    #Remove cosmic rays
    x, y = x_data, y_data
    
    #Create ROI around the G and 2D peaks

    x0G, y0G = find_x0_y0(x, y, omegaG)
    x02D, y02D = find_x0_y0(x, y, omega2D)
    
    
    xG, yG = select_ROI(x, y, x0G, 80)
    x2D, y2D = select_ROI(x, y, x02D, 120)

    #Find the mean offset of the whole data
    full_offset = find_mean_offset(y, n)

    #Define an intensity withou

    y0G -= full_offset
    y02D -= full_offset

    #Remove offset
    xG, yG = remove_offset(xG, yG, n)
    x2D, y2D = remove_offset(x2D, y2D, n)

    #Find the mean offset of the G and 2D peaks with the linear offset removed
    offsetG = find_mean_offset(yG, n)
    offset2D = find_mean_offset(y2D, n)
    
    #Initial guess for the parameters
    p0_G = [offsetG, x0G, 13, y0G * 15]
    pmin_G = (offsetG - 0.2 * np.abs(offsetG), x0G - 5, 5, y0G * 10)
    pmax_G = (offsetG + 0.2 * np.abs(offsetG), x0G + 15, 20, y0G * 400)

    p0_2D = [offset2D, x02D, 13, y02D * 15, 0.7, 12]
    pmin_2D = (offset2D - 0.2 * np.abs(offset2D), x02D - 5, 5, y02D * 10, 0.4, 6)
    pmax_2D = (offset2D + 0.2 * np.abs(offset2D), x02D + 5, 20, y02D * 400, 0.9, 20)
    
    #Fitting the data
    popt_Lorentz, pcov_Lorentz = curve_fit(lorentzian, xG, yG, p0_G)
    popt_DB_Basko, pcov_DB_Basko = curve_fit(db_basko, x2D, y2D, p0_2D)

    #Calculating the error

    perr_Lorentz = np.sqrt(np.diag(pcov_Lorentz))
    perr_DB_Basko = np.sqrt(np.diag(pcov_DB_Basko))


    p1_Basko = [popt_DB_Basko[0], popt_DB_Basko[1], popt_DB_Basko[2], popt_DB_Basko[3]*popt_DB_Basko[4]]
    p2_Basko = [popt_DB_Basko[0], popt_DB_Basko[1] + popt_DB_Basko[5], popt_DB_Basko[2], popt_DB_Basko[3]*(1 - popt_DB_Basko[4])]

    fit_Lorentz = lorentzian(xG, *popt_Lorentz)
    fit_1_Basko = basko(x2D, *p1_Basko)
    fit_2_Basko = basko(x2D, *p2_Basko)
    fit_DB_basko = db_basko(x2D, *popt_DB_Basko)

    results_G = np.column_stack((xG, yG, fit_Lorentz))
    results_2D = np.column_stack((x2D, y2D, fit_DB_basko, fit_1_Basko, fit_2_Basko))
    
    ratio_2D_G = popt_DB_Basko[3] / popt_Lorentz[3]

    params = [popt_Lorentz, popt_DB_Basko, perr_Lorentz, perr_DB_Basko, ratio_2D_G]


    return results_G, results_2D, params

def print_fit_results(params, n = 2):
    print(f'G peak: \n off_G = {params[0][0]:.{n}f} ± {params[2][0]:.{n}f} \n ω_G = {params[0][1]:.{n}f} ± {params[2][1]:.{n}f} cm⁻¹ \n Γ_G = {params[0][2]:.{n}f} ± {params[2][2]:.{n}f} cm⁻¹ \n I_G = {params[0][3]:.{n}f} ± {params[2][3]:.{n}f}')
    print(f'2D peak: \n off_2D = {params[1][0]:.{n}f} ± {params[3][0]:.{n}f} \n ω_2D- = {params[1][1]:.{n}f} ± {params[3][1]:.{n}f} cm⁻¹, ω_2D+ = {params[1][1] + params[1][5]:.{n}f} ± {params[3][1]:.{n}f} cm⁻¹ \n Γ_2D = {2 * np.sqrt(2 ** (2 / 3) - 1) * params[1][2]:.{n}f} ± {2 * np.sqrt(2 ** (2 / 3) - 1) * params[3][2]:.{n}f} cm⁻¹ \n I_2D = {params[1][3]:.{n}f} ± {params[3][3]:.{n}f} \n I_2D-/I_2D+ = {params[1][4] / (1 - params[1][4]):.{n}f} ± {params[3][4] / (1 - params[3][4]):.{n}f}')

#Plotting the fitting results
def plot_raman_fit(results_G, results_2D, params, x_data, y_data, name, omegaG, omega2D):

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x_data, y_data ,'k', linewidth=2)
    plt.title(name)
    plt.xlabel(r'wavenumber ($cm^{-1}$)')
    plt.ylabel('counts (arb.)')

    plt.subplot(3, 2, 3)
    plt.plot(results_G[:,0], results_G[:,1], 'ko')
    plt.plot(results_G[:,0], results_G[:,2], 'r')
    plt.xlabel(r'wavenumber ($cm^{-1}$)')
    plt.ylabel('counts (arb.)')
    plt.legend([f'Lorentz Σr² = {np.sum((results_G[:,1] - results_G[:,2]) ** 2):.3f}, 'r' $ω_G$ ='f' {params[0][1]:.1f} cm⁻¹,'r' Γ_G'f' = {params[0][2]:.1f} cm⁻¹,'r' $I_{2D}/I_G$'f' = {params[2][0]:.1f}'], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=1)

    plt.subplot(3, 2, 4)
    plt.plot(results_2D[:,0], results_2D[:,1], 'ko')
    plt.plot(results_2D[:,0], results_2D[:,2], 'b')
    plt.plot(results_2D[:,0], results_2D[:,3], 'r')
    plt.plot(results_2D[:,0], results_2D[:,4], 'g')
    plt.xlabel(r'wavenumber ($cm^{-1}$)')
    plt.ylabel('counts (arb.)')
    plt.legend([f'dbBasko Σr² = {np.sum((results_2D[:,1] - results_2D[:,2]) ** 2):.3f}'r' $ω_{2D-}$ =' f' {params[1][1]:.1f} cm⁻¹,' r' $ω_{2D+}$ =' f'{params[1][1] + params[1][5]:.1f} cm⁻¹,' r' $I_{2D-} / I_{2D+}$ =' f'{params[1][4] / (1 - params[1][4]):.1f},' r'$Γ_{2D}$ =' f'{2 * np.sqrt(2 ** (2 / 3) - 1) * params[1][2]:.1f} cm⁻¹'], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=1)

    plt.subplot(3, 2, 5)
    plt.plot(results_G[:,0], (results_G[:,2]-results_G[:,1]) / find_x0_y0(results_G[:,0], results_G[:,1], omegaG)[1])
    plt.xlabel(r'wavenumber ($cm^{-1}$)')
    plt.legend([r'residuals / $y_{max}$'], loc='upper left', ncol=1)

    plt.subplot(3, 2, 6)
    plt.plot(results_2D[:,0], (results_2D[:,2] - results_2D[:,1]) / find_x0_y0(results_2D[:,0], results_2D[:,1], omega2D)[1])
    plt.xlabel(r'wavenumber ($cm^{-1}$)')
    plt.legend([r'residuals / $y_{max}$'], loc='upper left', ncol=1)

    #plt.savefig(f'Fig_{name}.pdf')
    #plt.savefig(f'Fig_{name}.png')
    plt.show()

#------------------PROCESSING MAPS------------------#
def get_map_data(name):
    data = np.loadtxt(name, delimiter='\t', skiprows=0, encoding="iso-8859-1", unpack=True)
    return (data)


def separate_data_noise(df):
    #Even columns are noise, odd columns are data
    noise = np.zeros((len(df), int(len(df[0])/2)))
    data = np.zeros((len(df), int(len(df[0])/2)))

    for k in range(int(len(df[0]) / 2)):
        for i in range(len(df)):
            noise[i][k] = df[i][2 * k]
            data[i][k] = df[i][2 * k + 1]
    return noise, data

#Treatment of the data
def select_range(x, data, min_x, max_x):
    data_range = data[(x > min_x) & (x < max_x)]
    x_range = x[(x > min_x) & (x < max_x)]
    return x_range, data_range

def make_integration_array(data):
    integration_array = np.zeros(np.shape(data)[1])
    
    for i in range(len(integration_array)):
        integration_array[i] = np.sum(data[:, i])
    return integration_array

def remove_map_negatives(matrix):
    for i in range(len(matrix)):
        if matrix[i] < 0:
            matrix[i] = 0

def remove_map_offset(x_range, data_range, offset_points):
    for i in range(np.shape(data_range)[1]):
        data_range[:, i] = remove_offset(x_range, data_range[:, i], offset_points)[1]

'''def remove_map_spikes(x_range, data_range, spike1, spike2):
    new_data = np.zeros(np.shape(data_range))
    for i in range(np.shape(data_range)[1]):
        new_x, new_data[:,i] = rgf.Remove_spikes(x_range, data_range[:, i], spike1, spike2)
    return new_x, new_data
'''
def make_map(x, data, map_size, x_min, x_max, offset_points = 20):
    x_range, data_range = select_range(x, data, x_min, x_max)
    
    
    remove_map_offset(x_range, data_range, offset_points)
    integration_matrix = make_integration_array(data_range)
    remove_map_negatives(integration_matrix)
    map = np.transpose(np.reshape(integration_matrix, map_size))

    #For sample 12, position (5,6) is unusual, so it is changed to the mean of its neighbors, comment this line for other samples
    #map[6,5] = np.mean([map[5,7], map[5,5], map[6,4], map[6,6]])

    return x_range, data_range, map

#Plotting the map
def onclick(event, x_axis_raman, data, map_size, x_min, x_max, name, omegaG, omega2D):

    if event.inaxes is not None:

        x, y = event.xdata, event.ydata
        ix, iy = int(round(x)), int(round(y))
        print(f"Clicked at coordinates: ({x:.2f}, {y:.2f})")
        print(f"Clicked at indices: ({ix}, {iy})")

        x_range1, data_matrix, raman_fit = fit_from_map(x_axis_raman, data, map_size, x_min, x_max, omegaG, omega2D, (ix, iy))
        plot_raman_fit(raman_fit[0], raman_fit[1], raman_fit[2], x_range1, data_matrix[:,ix,iy], f'{name} at coordinates ({ix},{iy})', 1610, 2654)

        plt.draw()
    else:
        print('Clicked ouside axes bounds but inside plot window')

def draw_map(x_axis_raman, data, map_size, x_min, x_max, name, omegaG, omega2D):

    x_range, data_range, map = make_map(x_axis_raman, data, map_size, x_min, x_max)
    fig, ax = plt.subplots()
    cax = ax.imshow(map, interpolation='nearest', cmap=plt.cm.jet)
    #ax.set_title(f'{name} from {x_min} to {x_max} $cm^{-1}$')
    ax.set_xlabel(r'$\mu m$')
    ax.set_ylabel(r'$\mu m$')
    cbar = fig.colorbar(cax)
    cbar.set_label('Integrated intensity (a.u.)', rotation=270, labelpad=20)


    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, x_axis_raman, data, map_size, x_min, x_max, name, omegaG, omega2D))
    plt.show() 

def fit_from_map(x_axis, data, map_size, x_min, x_max, omegaG, omega2D, coord):
    x_range, data_range, map_matrix = make_map(x_axis, data, map_size, x_min, x_max)
    
    data_matrix = np.reshape(data_range, (len(x_range), map_size[0], map_size[1]))
    raman_fit = raman_Gr_fit(x_range, data_matrix[:, coord[0], coord[1]], omegaG, omega2D)
    
    return x_range, data_matrix, raman_fit


#------------------EXAMPLE------------------#
'''
lambda_laser = 632.8
expected_omegaG = 1580
expected_omega2D = 2630

x_axis_nm = get_map_data('Stephane_samples/06_12_2024_new/calibration_x_axis_300gr.txt')


x_axis_raman = convert_nm_to_raman(x_axis_nm, lambda_laser)

df = get_map_data('Stephane_samples/09_12_2024/Flake_2/LRVI_H12P1_1-4_map2_2024-12-09_19-51-12_tot.dat')

noise, data = separate_data_noise(df)

#data = data - noise

map_size = (15,15)

x_min = 1400
x_max = 2800


draw_map( x_axis_raman, data, map_size, x_min, x_max, 'Sample 6. Flake 2', expected_omegaG, expected_omega2D)

#x_test, data_test, raman_fit = fit_from_map(x_axis_raman, data, map_size, x_min, x_max, expected_omegaG, expected_omega2D, (5, 6))
'''