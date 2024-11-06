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

# --- Local imports -----------------------------------------------------------
# --- Our own modules ---------------------------------------------------------
import sys
# permatent location of functions from pyFitXPS
sys.path.insert(0, '/home/julio/Python/pyFitXPS/pyfitxps/') # Julio Laptop

import plot_config


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

def FermiEdge(spectrum, merged_label='all_average', xmin=None, xmax=None, 
              A=1.0, c=0, b=0, m=0, T=300, gw=0.5, center=0, plot=True):
    """
    Fit Fermi edge with linear DOS convoluted with a gaussian function
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum object containing the data to fit
    merged_label : str, optional
        Label of merged scan to use for fitting
    xmin, xmax : float, optional
        Range for fitting. If None, uses full range
    A, c, b, m : float, optional
        Initial parameters for linear DOS
    T : float, optional
        Temperature in Kelvin
    gw : float, optional
        Initial gaussian width
    center : float, optional
        Initial center position
    plot : bool, optional
        Whether to plot the fit results
        
    Returns
    -------
    lmfit.ModelResult
        Fit results object
    """
    # Initialize fit_results if it doesn't exist
    if not hasattr(spectrum, 'fit_results'):
        spectrum.fit_results = {}
    
    # Get data from spectrum
    if merged_label not in spectrum.working_data.merged_scans:
        raise KeyError(f"Merged scan '{merged_label}' not found")
        
    x_full = spectrum.working_data.binding_energy
    y_full = spectrum.working_data.merged_scans[merged_label]['data'][0]
    
    # Select range
    if xmin is None:
        xmin = x_full[0]
    if xmax is None:
        xmax = x_full[-1]
        
    mask = (x_full >= xmin) & (x_full <= xmax)
    x = x_full[mask]
    y = y_full[mask]
    
    # model
    mod_FL = Model(FL_LDOS, independent_vars=['x'], prefix='FL_')
    mod_g = Model(gauss_nor_area, independent_vars=['x'], prefix='g_')

    # create Composite Model using the custom convolution operator
    mod = CompositeModel(mod_FL, mod_g, convolve)
    pars = mod.make_params()

    # Set parameters
    pars['FL_T'].set(value=T, vary=False)
    pars['FL_center'].set(value=center, min=-2.0, max=1.0)
    pars['FL_m'].set(value=m)
    pars['FL_b'].set(value=b)
    pars['FL_c'].set(value=c)
    pars['FL_A'].set(value=A)
    pars['g_center'].set(value=center, expr='FL_center')
    pars['g_gw'].set(value=gw, min=0.2, max=1.5)

    # Fit model
    result = mod.fit(y, params=pars, x=x)

    # Store results with scaled components
    spectrum.fit_results['fermi_edge'] = {
        'result': result,
        'components': {
            'Fermi_dist': 10 * result.eval_components(x=x)['FL_'],
            'Gaussian': 100 * result.eval_components(x=x)['g_']
        },
        'x_range': (xmin, xmax),
        'merged_label': merged_label,
        'fit_type': 'Fermi Edge'
    }
    
    # Plot with default layout if requested
    if plot:
        fig, axes = _plot_fermi_edge(spectrum, 'fermi_edge')
        plt.show()
    
    return result



# - - - Other special functions - - - - - - - - - - - - - - - - - - - - - - - - -

def Energy_Corr(experiment, shift, regions=None):
    """
    Apply energy scale correction to selected regions or all regions
    
    Parameters
    ----------
    experiment : XPSExperiment
        Experiment containing the spectra to correct
    shift : float
        Energy shift value in eV
    regions : list of str, optional
        List of region names to correct. If None, corrects all regions
    """
    if regions is None:
        regions = experiment.list_regions()
        
    experiment.correct_energy_scale(shift, regions)
    
def plot_energy_correction(spectrum, ax=None):
    """
    Plot original and corrected energy scales
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    # Plot original data
    ax.plot(spectrum.original_data.binding_energy, 
            spectrum.original_data.intensity_scans[0],
            label='Original Data',
            linestyle='dotted',
            alpha=0.7)
            
    # Plot corrected data
    ax.plot(spectrum.working_data.binding_energy,
            spectrum.working_data.intensity_scans[0],
            label='Corrected Data')
            
    ax.set_xlabel('Binding Energy (eV)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.legend()
    
    return ax

def list_fits(spectrum):
    """
    List all available fits for a spectrum
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum object to check for fits
        
    Returns
    -------
    dict
        Dictionary of fit information if fits exist
    """
    if not hasattr(spectrum, 'fit_results') or not spectrum.fit_results:
        print("No fits available for this spectrum")
        return None
    
    print("\nAvailable fits:")
    print("-" * 40)
    
    fit_info = {}
    for fit_name, fit_data in spectrum.fit_results.items():
        print(f"\nFit type: {fit_name}")
        
        # Get basic fit information
        result = fit_data['result']
        x_range = fit_data.get('x_range', ('unknown', 'unknown'))
        merged_label = fit_data.get('merged_label', 'unknown')
        
        # Store and display fit information
        info = {
            'x_range': x_range,
            'merged_label': merged_label,
            'n_variables': len(result.var_names),
            'success': result.success,
            'r_squared': result.rsquared,
        }
        
        print(f"  Energy range: {x_range[0]:.2f} to {x_range[1]:.2f} eV")
        print(f"  Merged scan used: {merged_label}")
        print(f"  Number of variables: {info['n_variables']}")
        print(f"  Fit success: {info['success']}")
        print(f"  R-squared: {info['r_squared']:.4f}")
        
        fit_info[fit_name] = info
    
    return fit_info

def ds(x, intercept, slope, lw, asym, gw, int, e):
    """
    Doniach-Sunjic lineshape function
    
    Parameters
    ----------
    x : array-like
        Energy values
    intercept, slope : float
        Background parameters
    lw : float
        Lorentzian width
    asym : float
        Asymmetry parameter
    gw : float
        Gaussian width
    int : float
        Intensity
    e : float
        Peak position
        
    Returns
    -------
    array-like
        Calculated DS lineshape
    """
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

def fit_ds_peak(spectrum, merged_label='all_average', xmin=None, xmax=None, 
                peak_type='single', plot=True, **kwargs):
    """
    Fit XPS peak with Doniach-Sunjic lineshape
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum object containing the data to fit
    merged_label : str, optional
        Label of merged scan to use for fitting
    xmin, xmax : float, optional
        Range for fitting. If None, uses full range
    peak_type : str, optional
        Type of peak to fit: 'single', 'doublet_f', 'doublet_d', 'doublet_p', 'doublet_custom'
    plot : bool, optional
        Whether to plot the fit results
    **kwargs : dict
        Additional parameters for fitting
        
    Returns
    -------
    lmfit.ModelResult
        Fit results object
    """
    # Get data from spectrum
    if merged_label not in spectrum.working_data.merged_scans:
        raise KeyError(f"Merged scan '{merged_label}' not found")
        
    x_full = spectrum.working_data.binding_energy
    y_full = spectrum.working_data.merged_scans[merged_label]['data'][0]
    
    # Select range
    if xmin is None:
        xmin = x_full[0]
    if xmax is None:
        xmax = x_full[-1]
        
    mask = (x_full >= xmin) & (x_full <= xmax)
    x = x_full[mask]
    y = y_full[mask]
    
    # Create model based on peak type
    if peak_type == 'single':
        mod = Model(ds, prefix='ds_')
    elif peak_type == 'doublet_f':
        mod = Model(doublet_nf, prefix='ds_')
    elif peak_type == 'doublet_d':
        mod = Model(doublet_nd, prefix='ds_')
    elif peak_type == 'doublet_p':
        mod = Model(doublet_np, prefix='ds_')
    elif peak_type == 'doublet_custom':
        mod = Model(doublet_ratio, prefix='ds_')
    else:
        raise ValueError(f"Unknown peak type: {peak_type}")
    
    # Set up parameters with defaults and user overrides
    pars = mod.make_params()
    default_params = {
        'intercept': 0,
        'slope': 0,
        'lw': 0.5,
        'asym': 0.1,
        'gw': 0.5,
        'int': max(y),
        'e': x[np.argmax(y)]
    }
    
    if 'doublet' in peak_type:
        default_params.update({
            'sos': 1.0  # spin-orbit splitting
        })
        if peak_type == 'doublet_custom':
            default_params.update({
                'r': 0.5  # custom ratio
            })
    
    # Update defaults with user-provided values
    default_params.update(kwargs)
    
    # Set parameters
    for param, value in default_params.items():
        if f'ds_{param}' in pars:
            pars[f'ds_{param}'].set(value=value)
    
    # Fit model
    result = mod.fit(y, params=pars, x=x)
    
    # Store results
    fit_name = f'ds_{peak_type}'
    spectrum.fit_results[fit_name] = {
        'result': result,
        'components': result.eval_components(x=x),
        'x_range': (xmin, xmax),
        'merged_label': merged_label,
        'fit_type': 'Doniach-Sunjic',
        'peak_type': peak_type
    }
    
    # Plot if requested
    if plot:
        plot_config.plot_fit_result(spectrum, fit_name)
    
    return result

def _plot_fit_simple(spectrum, fit_name):
    """Simple plot for fitting process"""
    fit_data = spectrum.fit_results[fit_name]
    result = fit_data['result']
    xmin, xmax = fit_data['x_range']
    
    # Get data in range
    mask = (spectrum.working_data.binding_energy >= xmin) & \
           (spectrum.working_data.binding_energy <= xmax)
    x = spectrum.working_data.binding_energy[mask]
    y = spectrum.working_data.merged_scans[fit_data['merged_label']]['data'][0][mask]
    
    # Simple plot
    plt.figure()
    plt.plot(x, y, 'o', label='Data')
    plt.plot(x, result.best_fit, 'r-', label='Fit')
    plt.xlabel('Binding Energy (eV)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.show()

def _plot_fermi_edge(spectrum, fit_name):
    """Default FermiEdge plot layout"""
    fit_data = spectrum.fit_results[fit_name]
    result = fit_data['result']
    xmin, xmax = fit_data['x_range']
    
    # Get data in range
    mask = (spectrum.working_data.binding_energy >= xmin) & \
           (spectrum.working_data.binding_energy <= xmax)
    x = spectrum.working_data.binding_energy[mask]
    y = spectrum.working_data.merged_scans[fit_data['merged_label']]['data'][0][mask]
    
    # Create plot with your original layout
    gs_kw = dict(width_ratios=[1, 1.5], height_ratios=[1, 4])
    fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                  ['left', 'lower right']],
                                 gridspec_kw=gs_kw,
                                 layout="constrained")
    
    # Full spectrum plot
    axd['left'].plot(spectrum.working_data.binding_energy,
                    spectrum.working_data.intensity_scans[0])
    
    # Add rectangle for fit region
    x0 = xmin - 0.05*abs(xmin)
    y0 = min(y) - 0.2*abs(min(y))
    w = abs(xmax-xmin) + 0.05*abs(xmax)
    h = abs(max(y)-min(y)) + 0.1*abs(max(y))
    rect = Rectangle((x0,y0), w, h, 
                    fill=False,
                    color="purple")
    axd['left'].add_patch(rect)
    
    # Residuals
    axd['upper right'].plot(x, result.residual, color='C2')
    
    # Fit results
    axd['lower right'].plot(x, y, 'o')
    axd['lower right'].plot(x, result.best_fit, color='C1')
    
    # Components
    comps = fit_data['components']
    axd['lower right'].plot(x, comps['Fermi_dist'], '--', color='b',
                           label='10 x Fermi dist comp')
    axd['lower right'].plot(x, comps['Gaussian'], '--', color='r',
                           label='100 x Gaussian comp')
    
    # Labels
    axd['lower right'].set_xlabel('Binding Energy [eV]')
    axd['left'].set_xlabel('Binding Energy [eV]')
    axd['left'].set_ylabel('Intensity')
    axd['lower right'].legend()
    
    fig.suptitle('FermiEdge x linear DOS convoluted with gaussian distribution')
    
    return fig, axd
