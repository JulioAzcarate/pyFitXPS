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
import numpy as np

# --- Matplotlib --------------------------------------------------------------
import matplotlib.pyplot as plt

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
s2pi = sqrt(2*pi)
ln2 = log(2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def convolve(arr, kernel):
    """Simple convolution of two arrays."""
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts) / 2)
    return out[noff:noff+npts]

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
    """Lienear function"""
    return m * x + b


def gaussian_normalized(x, center=0.0, gw=0.5):
    """Return a 1-dimensional Gaussian function normalized in area.
    gaussian(x, center, gw) =
        (1/(s2pi*gw)) * exp(-(1.0*x-center)**2 / (2*gw**2))
    """
    return ((1/(max(tiny, s2pi*gw)))
            * exp(-(1.0*x-center)**2 / max(tiny, (2*gw**2))))


def gn(x, center=0.0, gw=0.5):
    """Return a 1-dimensional Gaussian function normalized in area.
    gaussian(x, center, gw) =
        (1/(s2pi*gw)) * exp(-(1.0*x-center)**2 / (2*gw**2))
    """
    return exp(-4 * ln2 * ((x - center)/gw)**2)


def fermi_distribution(x, center=0.0, T=300):
    """Return a thermal distribution function.

    - is the Fermi-Dirac distribution:
        thermal_distribution(x, center=0.0, T) =
            1/(exp((x - center)/kBT) + 1)
    Notes
    -----
    - `kt` should be defined in the same units as `x`. The Boltzmann
      constant is ``kB = 8.617e-5 eV/K``, the we.
    - set ``kt<0`` to implement the energy loss convention common in
      scattering research.
    For more information, see:
    http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/disfcn.html
    """
    kBT = 0.025852 / 300*T

    return real(1/(exp((x - center)/not_zero(kBT)) + 1))


def FL_LDOS(x, m, b, c, center, T):
    """Return a thermal distribution (FermiEdge) times Linear DOS

    - Fermi-Dirac distribution:
        thermal_distribution(x, center=0.0, T) =
            1/(exp((x - center)/kBT) + 1)

    - DOS (density of states) is considered as linear distribution close to
      the FermiEdge.

    Notes: 
        kBT es in Bolztmann constant (kB = 0.025852 at 300 K) times temperature
    """

    kBT = 0.025852 / 300.0*T

    FE = c + (b + m * (x - center)) * real(1/(exp((x - center)/not_zero(kBT))
                                              + 1))
    return FE


def doniach_sunjic(x, amplitude, asymmetry, lw, center):
    """
    Doniach-Sunjic lineshape

    Parameters
    ----------
    x : float
        independet variable
    asymmetry : float
        asymetric parameter.
    lw : float
        Lorentzian width (L fwhm) core hole liftime.
    center : float
        lorentzian centroind (position of the peak).

    Returns
    -------
    None.

    """
    numerator = gamma(1 - asymmetry) * cos(pi*asymmetry/2
                                           + (1 - asymmetry) * arctan(2*((x - center)/lw)))
    denominator = (max(tiny, ((x - center)**2 + 4*lw**2)**((1 - asymmetry)/2)))

    return amplitude * numerator / denominator


def dvnp(x, sos, amplitude, sigma, gamma, center):
    """
    Doublet for np core level - Voigt functions

    Parameters
    ----------
    x : array
        independet variable
    sos : float
        spin orbital spliting for the np level
    amplitude : float
        amplitude (height) for the first (3/2) peak
    sigma : float
        gaussian width
    gamma : float
        lorenztian width
    center : float
        center (position) of the first (3/2) peak

    Returns
    -------
    doublet function constructed with two voigt lineshape from lmfit-py
    """
    v1 = voigt(x, amplitude=amplitude, sigma=sigma, gamma=gamma, center=center)
    v2 = voigt(x, amplitude=0.5*amplitude, sigma=sigma,
               gamma=gamma, center=center-sos)

    return v1 + v2


def dvnd(x, sos, amplitude, sigma, gamma, center):
    """
    Doublet for np core level - Voigt functions

    Parameters
    ----------
    x : array
        independet variable
    sos : float
        spin orbital spliting for the nd level
    amplitude : float
        amplitude (height) for the first (5/2) peak
    sigma : float
        gaussian width
    gamma : float
        lorenztian width
    center : float
        center (position) of the first (5/2) peak

    Returns
    -------
    doublet function constructed with two voigt lineshape from lmfit-py
    """
    v1 = voigt(x, amplitude=amplitude, sigma=sigma, gamma=gamma, center=center)
    v2 = voigt(x, amplitude=3/2**amplitude, sigma=sigma,
               gamma=gamma, center=center-sos)

    return v1 + v2


def dvnf(x, sos, amplitude, sigma, gamma, center):
    """
    Doublet for np core level - Voigt functions

    Parameters
    ----------
    x : array
        independet variable
    sos : float
        spin orbital spliting for the nf level
    amplitude : float
        amplitude (height) for the first (7/2) peak
    sigma : float
        gaussian width
    gamma : float
        lorenztian width
    center : float
        center (position) of the first (7/2) peak

    Returns
    -------
    doublet function constructed with two voigt lineshape from lmfit-py
    """
    v1 = voigt(x, amplitude=amplitude, sigma=sigma, gamma=gamma, center=center)
    v2 = voigt(x, amplitude=0.75*amplitude, sigma=sigma,
               gamma=gamma, center=center-sos)

    return v1 + v2


# - - - Composed Function - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def FermiEdge(region, energy_scale, xmin, xmax, c, b, m, T, gw, center):
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
    mod_g = Model(gaussian_normalized, prefix='g_')

    # create Composite Model using the custom convolution operator
    mod = CompositeModel(mod_FL, mod_g, convolve)
    pars = mod.make_params()

    # used as an integer index, so a very poor fit variable:
    pars['FL_T'].set(value=T, vary=False)
    pars['FL_center'].set(value=center, min=-2.0, max=-1.0)
    pars['FL_m'].set(value=m)
    pars['FL_b'].set(value=b)
    pars['FL_c'].set(value=c)
    pars['g_center'].set(value=center)
    pars['g_gw'].set(value=gw, min=0.2, max=1.5)

    # fit this model to data array y
    result = mod.fit(y, params=pars, x=x)

    region.update({'results':result})

    # generate components
    comps = result.eval_components(x=x)

    # plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, y, 'o', color='grey')
    axes[0].plot(x, result.init_fit, 'k--', label='initial fit')
    axes[0].plot(x, result.best_fit, 'r-', label='best fit')
    axes[0].set_xlabel(energy_scale + ' [eV]')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()

    axes[1].plot(x, y, 'bo')
    axes[1].plot(x, 10*comps['FL_'], '--', color='C0',
                 label='10 x Fermi dist component')
    axes[1].plot(x, 100*comps['g_'], '-', color='C1',
                 label='100 x Gaussian component')
    axes[1].set_xlabel(energy_scale + ' [eV]')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()

    fig.suptitle(
        'FermiEde x linear DOS convoluted with gaussian distritution ')
    plt.show()

    print(result.fit_report())


def FermiEdge_gn(region, energy_scale, xmin, xmax, c, b, m, T, gw, center):
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

    mod_FL = Model(FL_LDOS, independent_vars='x', prefix='FL_')
    mod_g = Model(gn, prefix='g_')

    # create Composite Model using the custom convolution operator
    mod = CompositeModel(mod_FL, mod_g, convolve)
    pars = mod.make_params()

    # used as an integer index, so a very poor fit variable:
    pars['FL_T'].set(value=T, vary=False)
    pars['FL_center'].set(value=center, min=-2.0, max=-1.0)
    pars['FL_m'].set(value=m)
    pars['FL_b'].set(value=b)
    pars['FL_c'].set(value=c)
    pars['g_center'].set(value=center, expr='FL_center')
    pars['g_gw'].set(value=gw, min=0.2, max=1.5)

    # fit this model to data array y
    result = mod.fit(y, params=pars, x=x)

    # add result to region dict
    region.update({'results':result})

    # generate components
    comps = result.eval_components(x=x)

    result.plot(
        title='Fitted Fermi Edge',
        xlabel='{} [eV]'.format(energy_scale),
        ylabel='Intensity [cps]',
    )

    print('Fermi Level at: {}'.format(result.best_values['FL_center']))
    

    # plot results
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # axes[0].plot(x, y, 'o', color='grey')
    # axes[0].plot(x, result.init_fit, 'k--', label='initial fit')
    # axes[0].plot(x, result.best_fit, 'r-', label='best fit')
    # axes[0].set_xlabel(energy_scale + ' [eV]')
    # axes[0].set_ylabel('Intensity')
    # axes[0].legend()
     
    # axes[1].plot(x, y, 'bo')
    # axes[1].plot(x, 10*comps['FL_'], '--', color='C0',
                #  label='10 x Fermi dist component')
    # axes[1].plot(x, 100*comps['g_'], '-', color='C1',
                #  label='100 x Gaussian component')
    # axes[1].set_xlabel(energy_scale + ' [eV]')
    # axes[1].set_ylabel('Intensity')
    # axes[1].legend()
    # 
    # fig.suptitle(
        # 'FermiEde x linear DOS convoluted with gaussian distritution '
    # )
    # plt.show()
    # 
    # print(result.fit_report())


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

    plt.plot(region['data_orig']['BE'],region['data_orig']['intensity'], label='Original Data', alpha=0.3, color='C0')
    plt.plot(region['data_corr']['BE'],region['data_orig']['intensity'], label='Corrected Data', color='C0')
    plt.legend()


def range_to_fit(region, energy_scale, xmin, xmax):
    # create data from broadened step
    x_full = region['data_corr'][energy_scale]
    y_full = region['data_orig']['intensity']

    # range of data to be fitted
    index_xmin = np.min(np.where(x_full > xmin))
    index_xmax = np.max(np.where(x_full < xmax))

    x = x_full[index_xmin:index_xmax]
    y = y_full[index_xmin:index_xmax]

    data_range_fit = {}
    data_range_fit.update({'x':x})
    data_range_fit.update({'y':y})

    region.update({'data_to_fit':data_range_fit})



def define_voigt_peak_parameters(region, name, sos, amplitude, gamma, sigma, center):
    """
    Define voigt peak parameters as dictionary

    add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    """
    # pname = str(name)
    pname = Model(dvnp, independent_vars=['x'], prefix='{}_'.format(name))
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    # params = Parameters()

    # params.add_many(
    #     ('{}_sos'.format(name), sos, False),
    #     ('{}_gamma'.format(name), gamma, False),
    #     ('{}_sigma'.format(name), sigma, True, 0.1, 0.8),
    #     ('{}_amplitude'.format(name), amplitude, True),
    #     ('{}_center'.format(name), center, True, center-0.7, center+0.7),
    # )

    # pars = {}
    # pars.update({'pars':pars})

    pname.set_param_hint('{}_sos'.format(name), sos, False)
    pname.set_param_hint('{}_gamma'.format(name), gamma, False)
    pname.set_param_hint('{}_sigma'.format(name), sigma, True, 0.1, 0.8)
    pname.set_param_hint('{}_amplitude'.format(name), amplitude, True)
    pname.set_param_hint('{}_center'.format(name), center, True, center-0.7, center+0.7)

    pars = pname.make_params()

    region.update({'params':pars})

    return pname


# mod = Model(myfunc, prefix='f1_')
# params = mod.make_params()
# print('Parameters:')
# for pname, par in params.items():
#     print(pname, par)


def background_lineal(region):
    """
    Create a linear background parameters and the model named bkgl
    """
    # data range to fit
    x = region['data_to_fit']['x']
    y = region['data_to_fit']['y']

    # calculate slope and intercept for linear background as initial pars value from the extremes of data
    xa = np.mean(x[:5])
    xb = np.mean(x[-5:])
    ya = np.mean(y[:5])
    yb = np.mean(y[-5:])

    m = (yb-ya)/(xb-xa)
    c = ya - m*xa

    bkgl = LinearModel(independent_vars=['x'], prefix='bkgl_')

    params = Parameters()

    params.add_many(
        ('bkgl_slope', m, True),
        ('bkgl_intercept', c, True),
    )

    region.update({'params':params})

    return bkgl


# - - - Importing XPSDoniachs - - - - - - - - - - - -
#
# The same as XPSDoniachs.XOP from Igor Pro
# https://github.com/momentoscope/hextof-processor/tree/master/XPSdoniachs
# ----------------------------------------------------

from XPSdoniachs import XPSdoniachs_ext as dsg

# - - - Defining Doniach-Sunjic functions - - - 

def ds(x,intercept,slope,lw,asym,gw,int,e):
    pi=[intercept,slope,lw,asym,gw,int,e]
    y = np.zeros_like(x)
    pp = dsg.VectorOfDouble()
    pp.extend(i for i in pi)
    try:
        for i in range(len(x)):
            y[i] = dsg.dsgn(x.values[i],pp)
        return y
    except AttributeError:
        for i in range(len(x)):
            y[i] = dsg.dsgnmEad2(x[i],pp)
        return y

# - Then doublets peaks are defined for every core level
def doublet_nf(x,sos,intercept,slope,lw,asym,gw,int,e):
    '''
    Doublet for "f" core level
    '''
    df_ds = ds(x,intercept,slope,lw,asym,gw,int,e) + 0.75*ds(x,intercept,slope,lw,asym,gw,int,e-sos)

    return df_ds

def doublet_nd(x,sos,intercept,slope,lw,asym,gw,int,e):
    '''
    Doublet for "d" core level
    '''
    df_ds = ds(x,intercept,slope,lw,asym,gw,int,e) + 0.667*ds(x,intercept,slope,lw,asym,gw,int,e-sos)

    return df_ds

def doublet_np(x,sos,intercept,slope,lw,asym,gw,int,e):
    '''
    Doublet for "p" core level
    '''
    df_ds = ds(x,intercept,slope,lw,asym,gw,int,e) + 0.5*ds(x,intercept,slope,lw,asym,gw,int,e-sos)

    return df_ds