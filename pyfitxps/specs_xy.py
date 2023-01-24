#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:56:39 2022

@author: Julio C. Azcárate
@institution: Centro Atómico Bariloche
"""

###############################################################################
#
#   This library contain several functions to read and load spec's XY files 
#
###############################################################################


###############################################################################
# --- import libraries ---
import glob    # to find files in folders
import os

# --- import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt


###############################################################################
# - - - - - - - - -  Functions - - - - - - - - - - - - - - - - - - - - - - - -

def list_files_xy(path="samefolder"):
    """Función para cargar archivos dentro de una carpeta."""
    if path == "samefolder":
        XPS_files = glob.glob("*.xy")
    else:
        XPS_files = glob.glob(path + "/*.xy")
        print(path)
    # creo lista con los nombres de los archivos ordenados alfabéticamente
    list_XPS_files = XPS_files
    return list_XPS_files


# def which_subfolders(dirname):
#     """
#     list the subfolders
#     """
#     subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
#     for dirname in list(subfolders):
#         subfolders.extend(which_subfolders(dirname))
#     return subfolders



def load_one_file_xy(filename):
    """
    Read one .xy file and store info in a dictionary

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    dict with two elements: 'details' and 'data_orig':
        'details' is a dict wich store the info from adquisition parameters
        'data_orig' dict which contain tuples for orinigal data as two elements:
            keys:
                ['BE'] binding energy
                ['KE'] kinetic energy
                ['intensity'] intensity (counts or counst per second)
    
    for example:
        dict['details']['Pass Energy'] give the pass energy of the adquisition
        dict['data_orig']['BE'] give the x-axis for binding energy
        dict['data_orig']['intensity'] give the y-axis for intensity
    """
    
    file = open(filename, 'r')
    
    dict_details = {} # create the emty dict to store the details of experiment
    
    details_par = ( # parameters to read from comment lines in .xy file
        'Region',
        'Acquisition Date',
        'Analysis Method',
        'Analyzer',
        'Analyzer Lens',
        'Analyzer Slit',
        'Scan Mode',
        'Curves/Scan',
        'Values/Curve',
        'Dwell Time',
        'Excitation Energy',
        'Binding Energy',
        'Pass Energy',
        'Bias Voltage',
        'Detector Voltage',
        'Eff. Workfunction',
        'Source',
        'Comment',
        'Number of Scans',
        )
    
    for x in file:
        if '# ' in x:
            z = x.replace("# ", "")
            (k,v) = z.split(":", maxsplit=1) # splir the string after the first ":"
            if k in details_par: # just take the listed keys
                dict_details.update({k.strip(): v.strip()}) # add the k and v to dict

    dict_xy = {'details': dict_details, 
               } # create the final dict
    
    # key for parameter values to be converted to each proper type
    par_float = ('Dwell Time','Excitation Energy','Binding Energy','Pass Energy',
    'Bias Voltage','Detector Voltage','Eff. Workfunction','Number of Scans',)
    par_int = ('Curves/Scan','Values/Curve','Number of Scans')
    
    for keys in par_float:
        dict_xy['details'][keys] = float(dict_xy['details'][keys])
    for keys in par_int:
        dict_xy['details'][keys] = int(dict_xy['details'][keys])
    
    # closing the opened file
    file.close()
    
    # read file with numpy.genfromtxt and create one numpy.array pear each column
    # and store them as "unwritable" np.array.
    BE, intensity = genfromtxt(filename,comments='#', unpack=True)
    BE = BE * (-1)
    KE = dict_xy['details']['Excitation Energy'] + BE
    
    # convert BE, KE, intensity for original data as "unwritable" to perserve it
    BE.setflags(write=False)
    KE.setflags(write=False)
    intensity.setflags(write=False)
    
    data = {}
    data.update({'BE':BE})
    data.update({'KE':KE})
    data.update({'intensity':intensity})
    
    dict_xy.update({'data_orig': data})
    
    return dict_xy


def load_all_files_xy_in(folder_path):
    """
    Read all files .xy in the folder. Each file in converted in a 
    dictionary, and stored all together as elements of a dictionary.

    Parameters
    ----------
    folder_path : TYPE str
        DESCRIPTION.

    Returns
    -------
    Dict which each element is a dictionary
    dict with two elements: 'details' and 'data_orig':
        'details' is a dict wich store the info from adquisition parameters
        'data_orig' dict which contain tuples for orinigal data as two elements:
            keys:
                ['BE'] binding energy
                ['KE'] kinetic energy
                ['intensity'] intensity (counts or counst per second)
    
    for example:
        dict['details']['Pass Energy'] give the pass energy of the adquisition
        dict['data_orig']['BE'] give the x-axis for binding energy
        dict['data_orig']['intensity'] give the y-axis for intensity
    """
    
    files = os.listdir(folder_path)

    exper_dict = {}  

    for filename in files:
        dict_region = load_one_file_xy(folder_path + filename)
        exper_dict.update({dict_region['details']['Region']:dict_region})
    
    exper_dict.keys()

    return exper_dict


def plot_region(experiment_dict, region, energy_scale):
    """
    plot all spectra of the same XPS' region

    --------------------------------------------
    experiment: TYPE dict
    element   : TYPE str
    """

    res = []
    
    for i,j in enumerate(list(experiment_dict.keys())):
        if region in j:
            res.append(j)
    res
    
    for spectra in res:
        
        plt.plot(experiment_dict[spectra]['data_orig'][energy_scale],
                experiment_dict[spectra]['data_orig']['intensity'],
                label=experiment_dict[spectra]['details']['Region'],
                )
    plt.legend()
    plt.xlabel(energy_scale + ' [eV]')
    plt.ylabel('Intensity (cps)')

    plt.show()


def plot_all_regions_in(experiment_dict, all_region_to_plot, energy_scale):
    """
    Plot all regions listed in "all_region_to_plot" from the dictionary wich 
    contain the spectrum of the whole experiment "exper_dict".
    """
    
    if len(all_region_to_plot) > 1:
        cols = 2
    

        rows = round(len(all_region_to_plot)/2)
    
        fig, axs = plt.subplots(ncols=cols,nrows=rows,
                            figsize=( 8, 3 * rows ),
                            constrained_layout=True,
                            )
    
        axs = axs.flatten()
    
        for spectra in sorted(list(experiment_dict.keys())):
            for i,element in enumerate(all_region_to_plot):
                if element in spectra:
                    axs[i].plot(experiment_dict[spectra]['data_orig'][energy_scale],
                   experiment_dict[spectra]['data_orig']['intensity'],
                   label=experiment_dict[spectra]['details']['Region'],
                   )
                    axs[i].legend()
                    axs[i].set_xlabel(energy_scale+' [eV]')
                    axs[i].set_ylabel('Intensity [cps')
        plt.show()

    else:
        region = str(all_region_to_plot).replace("['","").replace("']","")
        plot_region(experiment_dict, region, energy_scale)

    
    