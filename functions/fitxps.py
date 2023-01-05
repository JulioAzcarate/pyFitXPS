#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:23:18 2022

@author: Julio C. Azcárate
@institution: Centro Atómico Bariloche
"""

###############################################################################
#
#   Functions to perform fitting of XPS regions ussing lmfit
#
###############################################################################


###############################################################################
# --- Numpy -------------------------------------------------------------------
from XPSdoniachs import XPSdoniachs_ext as dsg
import numpy as np

# --- Matplotlib --------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- LMFIT -------------------------------------------------------------------
from lmfit import Parameters, CompositeModel, Model
from lmfit.models import LinearModel

from lmfit.lineshapes import voigt

# --- scipy -------------------------------------------------------------------
from scipy.special import gamma


###############################################################################

from numpy import (copysign, exp, pi, real, sqrt, log, cos, arctan)

# - - - some general functions to be used in other complex function - - -
# here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
tiny = 1.0e-15
s2pi = sqrt(2 * pi)
ln2 = log(2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def convolve(arr, kernel):
    """Simple convolution of two arrays."""
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad * arr[0], arr, pad * arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts) / 2)
    return out[noff:noff + npts]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - Basic Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def not_zero(value):
    """Return value with a minimal absolute size of tiny, preserving the sign.
    This is a helper function to prevent ZeroDivisionError's.
    Parameters
    ----------
    value : scalar
        Value to be ensured not to be zero.
    Returns
    -------
    scalar
        Value ensured not to be zero.
    """
    return float(copysign(max(tiny, abs(value)), value))


def linear(x, m, b):
    """Linear function"""
    return m * x + b


def gauss_nor_h(x, center=0.0, gw=0.5):
    """Return a 1-dimensional Gaussian function normalized in peack height.
    gaussian(x, center, gw) =
        exp(-4 * ln2 * ((x - center) / gw)**2)
    """
    return exp(-4 * ln2 * ((x - center) / not_zero(gw))**2)

def gauss_nor_area(x,center, gw=0.5):
    """ 
    Return a 1-dimensional Gaussian function normalized in peack area.
    gaussian(x, center, gw) =
    """
    sigma = gw / 2.355

    return ((1 / (s2pi * not_zero(sigma)))
            * exp(-(1.0 * x - center)**2 / (2 * not_zero(sigma**2))))


def FL_LDOS(x, m, b, c, A, center, T):
    """Return a thermal distribution (FermiEdge) times Linear DOS

    - Fermi-Dirac distribution:
        thermal_distribution(x, center=0.0, T) =
            1/(exp((x - center)/kBT) + 1)

    - DOS (density of states) is considered as linear distribution close to
      the FermiEdge.

    Notes:
        kBT es in Bolztmann constant (kB = 0.025852 at 300 K) times temperature
    """

    kBT = 0.025852 / 300.0 * T

    FE = c + A * (b + m * (x - center)) * \
        real(1 / (exp((x - center) / not_zero(kBT)) + 1))
    return FE


# - - - Composed Function - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def FermiEdge(region, energy_scale, xmin, xmax, A, c, b, m, T, gw, center):
    """
    Return the fit of FermiEdge times linear DOS convoluted with a
    gaussian function

    Parameters
    ----------
    region : variable - dict name
        name of dictionary to containg the VB region to be fitted
    energy_scale : string
        which energy scale be selected: 'BE' or 'KE'
    xmin : float
        min for range data to fit
    xmax : float
        max for range data to fit
    c : float
        background noise right to the FermiEdge
    A : float
        pre-exponentian factor
    b : float
        linear intercept left to the FermiEdge for the linear DOS
    m : float
        slope for linear DOS
    T : float
        temperature in Kelvin
    gw : float
        gaussian with
    center : float
        position of the FermiEdge

    Returns
    -------
    lmfit fit_report and plot of the fitting results
    """

    # create data from broadened step
    x_full = region['data_orig'][energy_scale]
    y_full = region['data_orig']['intensity']

    # range of data to be fitted
    index_xmin = np.min(np.where(x_full > xmin))
    index_xmax = np.min(np.where(x_full > xmax))

    x = x_full[index_xmin:index_xmax]
    y = y_full[index_xmin:index_xmax]

    # model
    mod_FL = Model(FL_LDOS, independent_vars='x', prefix='FL_')
    mod_g = Model(gauss_nor_area, prefix='g_')

    # create Composite Model using the custom convolution operator
    mod = CompositeModel(mod_FL, mod_g, convolve)
    pars = mod.make_params()

    # used as an integer index, so a very poor fit variable:
    pars['FL_T'].set(value=T, vary=False)
    pars['FL_center'].set(value=center, min=-2.0, max=-1.0)
    pars['FL_m'].set(value=m)
    pars['FL_b'].set(value=b)
    pars['FL_c'].set(value=c)
    pars['FL_A'].set(value=A)
    pars['g_center'].set(value=center, expr='FL_center')
    pars['g_gw'].set(value=gw, min=0.2, max=1.5)

    # fit this model to data array y
    result = mod.fit(y, params=pars, x=x)

    region.update({'results': result})

    # generate components
    comps = result.eval_components(x=x)

    # plot results
    gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 4])
    fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                   ['left', 'lower right']],
                                  gridspec_kw=gs_kw, #figsize=(5.5, 3.5),
                                  layout="constrained")

    axd['left'].plot(region['data_orig']['BE'],region['data_orig']['intensity'])

    x0 = x[0] - 0.05*x[0]
    y0 = y[-1] - 0.2*y[-1]
    w = abs(x[0]-x[-1]) + 0.05*x[-1]
    h = abs(y[0]-y[-1]) + 0.1*y[0]
    rect = Rectangle((x0,y0),w,h, 
                        fill = False,
                        color = "purple",
                        #linewidth = 2
                        )
    axd['left'].add_patch(rect)
    # axd['left'].axvspan(x_full[index_xmin], x_full[index_xmax], y_full[index_xmin], y_full[index_xmax], alpha=0.3, color='grey')
    
    axd['upper right'].plot(x, result.residual, color='C2')
    axd['lower right'].plot(x,y, 'o')
    axd['lower right'].plot(x, result.best_fit, color='C1')
    axd['lower right'].plot(x, 10 * comps['FL_'], '--', color='b',
                 label='10 x Fermi dist comp')
    axd['lower right'].plot(x, 100 * comps['g_'], '--', color='r',
                 label='100 x Gaussian comp')
    axd['lower right'].set_xlabel(energy_scale + ' [eV]')
    axd['left'].set_xlabel(energy_scale + ' [eV]')
    axd['left'].set_ylabel('Intensity')
    axd['lower right'].legend()

    fig.suptitle('FermiEdge x linear DOS convoluted with gaussian distritution')

    print(result.fit_report())



# - - - Other special functions - - - - - - - - - - - - - - - - - - - - - - - - -

def Energy_Corr_one(region, shift):
    """
    Correction for energy scale

    Parameters
    ----------
    region: string of names for each dict of data
    shift : float
        value to shift the energy scale in eV.

    Returns
    -------
    Update the same selected dictionary with a dict containing the corrected
    energy scale as numpy.array

    """
    # shift = ref_shift['results'].best_values['FL_center']

    BE = region['data_orig']['BE'] - shift
    KE = region['data_orig']['KE'] - shift

    corr = {}
    corr.update({'BE': BE})
    corr.update({'KE': KE})
    region.update({'data_corr': corr})

    print(shift)

    plot_corr(region)

    # plt.plot(
    #     region['data_orig']['BE'],
    #     region['data_orig']['intensity'],
    #     label='Original Data',
    #     linestyle='dotted',
    #     alpha=0.7,
    #     color='C0')
    # plt.plot(
    #     region['data_corr']['BE'],
    #     region['data_orig']['intensity'],
    #     label='Corrected Data',
    #     color='C0')
    # plt.xlabel('BE [eV]')
    # plt.ylabel('Intensity [cps]')
    # plt.legend()

def Energy_Corr_one_FL(region, ref_region):
    """
    Correction for energy scale

    Parameters
    ----------
    region: string of names for each dict of data
    shift : float
        value to shift the energy scale in eV.

    Returns
    -------
    Update the same selected dictionary with a dict containing the corrected
    energy scale as numpy.array

    """
    shift = ref_region['results'].best_values['FL_center']

    BE = region['data_orig']['BE'] - shift
    KE = region['data_orig']['KE'] - shift

    corr = {}
    corr.update({'BE': BE})
    corr.update({'KE': KE})
    region.update({'data_corr': corr})

    print(shift)

    plot_corr(region)

def Energy_Corr_list_FL(list_of_regions, ref_region):
    """
    Correction for energy scale to every region in the list

    Parameters
    ----------
    region: list of dictionary with data
    ref_region : variable name
        name of fitted region (valence band) as reference

    Returns
    -------
    Update the same selected dictionary with a dict containing the corrected
    energy scale as numpy.array

    """
    shift = ref_region['results'].best_values['FL_center']

    for region in list_of_regions:
        BE = region['data_orig']['BE'] - shift
        KE = region['data_orig']['KE'] - shift
        
        corr = {}
        corr.update({'BE': BE})
        corr.update({'KE': KE})
        region.update({'data_corr': corr})

        region.keys()

    print(shift)

def plot_corr(region):
    """
    Plot the energy shift correction of the region
    """
    plt.plot(
        region['data_orig']['BE'],
        region['data_orig']['intensity'],
        label='Original Data',
        linestyle='dotted',
        alpha=0.7,
        color='C0')
    plt.plot(
        region['data_corr']['BE'],
        region['data_orig']['intensity'],
        label='Corrected Data',
        color='C0')
    plt.xlabel('BE [eV]')
    plt.ylabel('Intensity [cps]')
    plt.legend()

def plot_Energy_Corr(region):
    """
    Plot the energy shift correction for every region defined
    """
    if type(region) is dict:
        plot_corr(region)
    elif type(region) is list:
        for i in region:
            plot_corr(i)
    else :
        print("Is not possible to plot the energy shift")



def range_to_fit(region, data_set, energy_scale, xmin, xmax):
    '''
    Function to crop data to be fitted. The range of data is
    stored in "region" dictionary in the key:'data_to_fit'.

    '''

    # create data from broadened step
    if data_set == 'data_orig':
        x_full = region['data_orig'][energy_scale]
    elif data_set == 'data_corr':
        x_full = region['data_corr'][energy_scale]
    else:
        print('"data_set" is not defined. Please run "region.keys()" to known which "data_set" is present.')

    # x_full = region['data_corr'][energy_scale]
    y_full = region['data_orig']['intensity']

    # range of data to be fitted
    index_xmin = np.min(np.where(x_full > xmin))
    index_xmax = np.max(np.where(x_full < xmax))

    x = x_full[index_xmin:index_xmax]
    y = y_full[index_xmin:index_xmax]

    data_range_fit = {}
    data_range_fit.update({'x': x})
    data_range_fit.update({'y': y})

    region.update({'data_to_fit': data_range_fit})


def linear_background(region):
    """
    Create a linear background parameters and the model named bkgl
    """
    # data range to fit
    x = region['data_to_fit']['x']
    y = region['data_to_fit']['y']

    # calculate slope and intercept for linear background as initial pars
    # value from the extremes of data
    xa = np.mean(x[:5])
    xb = np.mean(x[-5:])
    ya = np.mean(y[:5])
    yb = np.mean(y[-5:])

    m = (yb - ya) / (xb - xa)
    c = ya - m * xa

    bkgl = LinearModel(independent_vars=['x'], prefix='bkgl_')

    params = Parameters()

    params.add_many(
        ('bkgl_slope', m, True),
        ('bkgl_intercept', c, True),
    )

    region.update({'params': params})

    return bkgl


# - - - Importing XPSDoniachs - - - - - - - - - - - -
#
# The same as XPSDoniachs.XOP from Igor Pro
# https://github.com/momentoscope/hextof-processor/tree/master/XPSdoniachs
# ----------------------------------------------------

# - - - Defining Doniach-Sunjic functions - - -


def ds(x, intercept, slope, lw, asym, gw, int, e):
    pi = [intercept, slope, lw, asym, gw, int, e]
    y = np.zeros_like(x)
    pp = dsg.VectorOfDouble()
    pp.extend(i for i in pi)
    try:
        for i in range(len(x)):
            y[i] = dsg.dsgn(x.values[i], pp)
        return y
    except AttributeError:
        for i in range(len(x)):
            y[i] = dsg.dsgnmEad2(x[i], pp)
        return y

# - Then doublets peaks are defined for every core level


def doublet_nf(x, sos, intercept, slope, lw, asym, gw, int, e):
    '''
    Doublet for "f" core level
    '''
    df_ds = ds(x, intercept, slope, lw, asym, gw, int, e) + 0.75 * \
        ds(x, intercept, slope, lw, asym, gw, int, e - sos)

    return df_ds


def doublet_nf_r(x, sos, intercept, slope, lw, asym, gw, int, e, r):
    '''
    Doublet for "f" core level
    '''
    df_ds_r = ds(x, intercept, slope, lw, asym, gw, int, e) + r * \
        ds(x, intercept, slope, lw, asym, gw, int, e - sos)

    return df_ds_r


def doublet_nd(x, sos, intercept, slope, lw, asym, gw, int, e):
    '''
    Doublet for "d" core level
    '''
    df_ds = ds(x, intercept, slope, lw, asym, gw, int, e) + 0.667 * \
        ds(x, intercept, slope, lw, asym, gw, int, e - sos)

    return df_ds


def doublet_np(x, sos, intercept, slope, lw, asym, gw, int, e):
    '''
    Doublet for "p" core level
    '''
    df_ds = ds(x, intercept, slope, lw, asym, gw, int, e) + 0.5 * \
        ds(x, intercept, slope, lw, asym, gw, int, e - sos)

    return df_ds
