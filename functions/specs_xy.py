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

# --- import numpy as np
from numpy import genfromtxt


###############################################################################
# - - - - - - - - -  Functions - - - - - - - - - - - - - - - - - - - - - - - -

def list_files_xy(path="samefolder"):
    """Función para cargar archivos dentro de una carpeta."""
    if path == "samefolder":
        list_XPS_files = glob.glob("*.xy")
    else:
        list_XPS_files = glob.glob(path + "/*.xy")
        print(path)
    # creo lista con los nombres de los archivos ordenados alfabéticamente
    list_XPS_files.sort()
    print(list_XPS_files)



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