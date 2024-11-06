"""
Global plotting configuration for pyfitxps
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import numpy as np

def set_plot_style(style='default', dpi=300):
    """Set global plotting style"""
    rcParams['figure.dpi'] = dpi
    rcParams['savefig.dpi'] = dpi
    
    if style == 'paper':
        rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
        })
    elif style == 'presentation':
        rcParams.update({
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
        })

def plot_fit_result(spectrum, fit_name, layout='standard', save_fig=False, 
                   output_path=None, show_components=True, show_residuals=True):
    """
    Plot fit results with different layouts
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum containing the fit results
    fit_name : str
        Name of the fit to plot
    layout : str, optional
        Plot layout: 'standard', 'fermi_edge', or 'simple'
        - 'standard': Regular fit plot with optional residuals
        - 'fermi_edge': Special layout for Fermi edge with full region
        - 'simple': Basic plot of data and fit
    show_components : bool, optional
        Whether to show individual fit components
    show_residuals : bool, optional
        Whether to show residuals panel
    save_fig : bool, optional
        Whether to save the figure
    output_path : str, optional
        Path for saving the figure
    """
    if fit_name not in spectrum.fit_results:
        raise KeyError(f"Fit '{fit_name}' not found")
    
    fit_data = spectrum.fit_results[fit_name]
    result = fit_data['result']
    xmin, xmax = fit_data['x_range']
    
    # Get data in range
    mask = (spectrum.working_data.binding_energy >= xmin) & \
           (spectrum.working_data.binding_energy <= xmax)
    x = spectrum.working_data.binding_energy[mask]
    y = spectrum.working_data.merged_scans[fit_data['merged_label']]['data'][0][mask]
    
    if layout == 'fermi_edge':
        # Your original FermiEdge layout
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 4])
        
        # Full range plot
        ax_full = fig.add_subplot(gs[:, 0])
        ax_full.plot(spectrum.working_data.binding_energy,
                    spectrum.working_data.intensity_scans[0])
        ax_full.axvline(xmin, color='r', linestyle='--', alpha=0.5)
        ax_full.axvline(xmax, color='r', linestyle='--', alpha=0.5)
        ax_full.set_xlabel('Binding Energy (eV)')
        ax_full.set_ylabel('Intensity')
        
        # Residuals
        ax_res = fig.add_subplot(gs[0, 1])
        ax_res.plot(x, result.residual)
        ax_res.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_res.set_ylabel('Residuals')
        
        # Fit results
        ax_fit = fig.add_subplot(gs[1, 1])
        ax_fit.plot(x, y, 'o', label='Data')
        ax_fit.plot(x, result.best_fit, 'r-', label='Best Fit')
        
        if show_components and 'components' in fit_data:
            for name, comp in fit_data['components'].items():
                ax_fit.plot(x, comp, '--', label=name)
        
        ax_fit.set_xlabel('Binding Energy (eV)')
        ax_fit.set_ylabel('Intensity')
        ax_fit.legend()
        
        axes = [ax_full, ax_res, ax_fit]
        
    elif layout == 'simple':
        # Simple plot
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='Data')
        ax.plot(x, result.best_fit, 'r-', label='Best Fit')
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend()
        axes = [ax]
        
    else:  # 'standard' layout
        # Regular fit plot with optional residuals
        if show_residuals:
            fig, (ax_res, ax_fit) = plt.subplots(2, 1, height_ratios=[1, 4])
            ax_res.plot(x, result.residual)
            ax_res.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax_res.set_ylabel('Residuals')
            axes = [ax_res, ax_fit]
        else:
            fig, ax_fit = plt.subplots()
            axes = [ax_fit]
            
        ax_fit.plot(x, y, 'o', label='Data')
        ax_fit.plot(x, result.best_fit, 'r-', label='Best Fit')
        
        if show_components and 'components' in fit_data:
            for name, comp in fit_data['components'].items():
                ax_fit.plot(x, comp, '--', label=name)
                
        ax_fit.set_xlabel('Binding Energy (eV)')
        ax_fit.set_ylabel('Intensity (a.u.)')
        ax_fit.legend()
    
    plt.tight_layout()
    
    if save_fig and output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return fig, axes

def plot_spectrum(spectrum, merged_label=None, scans=None, 
                 xmin=None, xmax=None, normalize=False, 
                 show_scans=True, show_merged=True):
    """
    Plot XPS spectrum data
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum to plot
    merged_label : str, optional
        Label of merged scan to plot
    scans : list of int, optional
        List of scan numbers to plot. If None, plots all scans
    xmin, xmax : float, optional
        X-axis range
    normalize : bool, optional
        Whether to normalize intensities
    show_scans : bool, optional
        Whether to show individual scans
    show_merged : bool, optional
        Whether to show merged scan
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    fig, ax = plt.subplots()
    
    # Plot individual scans
    if show_scans:
        if scans is None:
            scans = range(len(spectrum.working_data.intensity_scans))
        
        for i in scans:
            y = spectrum.working_data.intensity_scans[i]
            if normalize:
                y = y / np.max(y)
            ax.plot(spectrum.working_data.binding_energy, y, 
                   'o-', alpha=0.5, label=f'Scan {i+1}')
    
    # Plot merged scan
    if show_merged and merged_label:
        if merged_label in spectrum.working_data.merged_scans:
            y = spectrum.working_data.merged_scans[merged_label]['data'][0]
            if normalize:
                y = y / np.max(y)
            ax.plot(spectrum.working_data.binding_energy, y,
                   'k-', linewidth=2, label=f'Merged ({merged_label})')
    
    # Set axis ranges
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmax, xmin)  # Reversed for binding energy
    
    ax.set_xlabel('Binding Energy (eV)')
    ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ' (a.u.)'))
    ax.legend()
    
    return fig, ax

def plot_regions(experiment, regions=None, normalize=False):
    """
    Plot multiple regions from an XPS experiment
    
    Parameters
    ----------
    experiment : XPSExperiment
        Experiment containing the regions
    regions : list of str, optional
        List of region names to plot. If None, plots all regions
    normalize : bool, optional
        Whether to normalize intensities
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    if regions is None:
        regions = list(experiment.spectra.keys())
    
    n_regions = len(regions)
    fig, axes = plt.subplots(1, n_regions, figsize=(5*n_regions, 4))
    if n_regions == 1:
        axes = [axes]
    
    for ax, region_name in zip(axes, regions):
        spectrum = experiment.spectra[region_name]
        y = spectrum.working_data.intensity_scans[0]
        if normalize:
            y = y / np.max(y)
        
        ax.plot(spectrum.working_data.binding_energy, y, 'k-')
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Intensity' + (' (norm.)' if normalize else ' (a.u.)'))
        ax.set_title(region_name)
    
    plt.tight_layout()
    return fig, axes

def compare_spectra(spectra_list, labels=None, normalize=False, 
                   offset=0, xmin=None, xmax=None):
    """
    Compare multiple spectra
    
    Parameters
    ----------
    spectra_list : list of XPSSpectrum
        List of spectra to compare
    labels : list of str, optional
        Labels for each spectrum
    normalize : bool, optional
        Whether to normalize intensities
    offset : float, optional
        Vertical offset between spectra
    xmin, xmax : float, optional
        X-axis range
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    fig, ax = plt.subplots()
    
    if labels is None:
        labels = [f'Spectrum {i+1}' for i in range(len(spectra_list))]
    
    for i, (spectrum, label) in enumerate(zip(spectra_list, labels)):
        y = spectrum.working_data.intensity_scans[0]
        if normalize:
            y = y / np.max(y)
        y = y + i * offset
        
        ax.plot(spectrum.working_data.binding_energy, y, 
               '-', label=label)
    
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmax, xmin)
    
    ax.set_xlabel('Binding Energy (eV)')
    ax.set_ylabel('Intensity' + (' (normalized)' if normalize else ' (a.u.)'))
    ax.legend()
    
    return fig, ax

def compare_scans_merged(spectrum, merged_label='all_average', scans=None,
                        normalize=False, show_residuals=False, 
                        show_std=True, alpha_scans=0.3):
    """
    Compare individual scans with merged scan
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum containing the scans
    merged_label : str, optional
        Label of merged scan to compare
    scans : list of int, optional
        List of scan numbers to plot. If None, plots all scans
    normalize : bool, optional
        Whether to normalize intensities
    show_residuals : bool, optional
        Whether to show residuals panel
    show_std : bool, optional
        Whether to show standard deviation band around merged scan
    alpha_scans : float, optional
        Transparency for individual scans
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    if merged_label not in spectrum.working_data.merged_scans:
        raise KeyError(f"Merged scan '{merged_label}' not found")
    
    # Create figure
    if show_residuals:
        fig, (ax_res, ax_main) = plt.subplots(2, 1, height_ratios=[1, 4],
                                             figsize=(8, 10))
        axes = [ax_res, ax_main]
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6))
        axes = [ax_main]
        
    x = spectrum.working_data.binding_energy
    
    # Get scans to plot
    if scans is None:
        scans = range(len(spectrum.working_data.intensity_scans))
    
    # Plot individual scans
    scan_data = []
    for i in scans:
        y = spectrum.working_data.intensity_scans[i]
        if normalize:
            y = y / np.max(y)
        scan_data.append(y)
        ax_main.plot(x, y, '-', alpha=alpha_scans, color='gray',
                    label='Scans' if i == 0 else None)
    
    # Plot merged scan
    merged_y = spectrum.working_data.merged_scans[merged_label]['data'][0]
    if normalize:
        merged_y = merged_y / np.max(merged_y)
    
    ax_main.plot(x, merged_y, 'k-', linewidth=2,
                label=f'Merged ({merged_label})')
    
    # Show standard deviation band if requested
    if show_std:
        scan_array = np.array(scan_data)
        std = np.std(scan_array, axis=0)
        ax_main.fill_between(x, merged_y - std, merged_y + std,
                           alpha=0.2, color='blue',
                           label='±1σ')
    
    # Plot residuals if requested
    if show_residuals:
        for y in scan_data:
            residual = y - merged_y
            ax_res.plot(x, residual, '-', alpha=alpha_scans, color='gray')
        ax_res.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_res.set_ylabel('Residuals')
        ax_res.grid(True, alpha=0.3)
    
    # Labels and styling
    ax_main.set_xlabel('Binding Energy (eV)')
    ax_main.set_ylabel('Intensity' + (' (normalized)' if normalize else ' (a.u.)'))
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Add merge info
    merge_info = spectrum.working_data.merged_scans[merged_label]['info']
    title = f"Scans comparison - {merge_info['method']} merge\n"
    if 'group_sizes' in merge_info:
        title += f"Groups: {merge_info['group_sizes']}"
    fig.suptitle(title)
    
    plt.tight_layout()
    return fig, axes

def compare_merge_methods(spectrum, merge_labels=None, figsize=(10, 6),
                         orientation='vertical', normalize=False,
                         show_scans=True, show_std=True, alpha_scans=0.3):
    """
    Compare different merge methods (e.g., all_average vs group merges)
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum containing the merged scans
    merge_labels : list of str, optional
        Labels of merged scans to compare. If None, uses all available
    figsize : tuple, optional
        Figure size (width, height)
    orientation : str, optional
        Plot orientation: 'vertical' or 'horizontal'
    normalize : bool, optional
        Whether to normalize intensities
    show_scans : bool, optional
        Whether to show individual scans
    show_std : bool, optional
        Whether to show standard deviation band
    alpha_scans : float, optional
        Transparency for individual scans
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    # Get merge labels
    if merge_labels is None:
        merge_labels = list(spectrum.working_data.merged_scans.keys())
    n_merges = len(merge_labels)
    
    # Create figure
    if orientation == 'vertical':
        fig, axes = plt.subplots(n_merges, 1, figsize=figsize,
                                sharex=True, sharey=True)
    else:  # horizontal
        fig, axes = plt.subplots(1, n_merges, figsize=figsize,
                                sharex=True, sharey=True)
    
    if n_merges == 1:
        axes = [axes]
    
    x = spectrum.working_data.binding_energy
    
    # Plot each merge method
    for ax, label in zip(axes, merge_labels):
        # Plot individual scans if requested
        if show_scans:
            for i in range(len(spectrum.working_data.intensity_scans)):
                y = spectrum.working_data.intensity_scans[i]
                if normalize:
                    y = y / np.max(y)
                ax.plot(x, y, '-', alpha=alpha_scans, color='gray',
                       label='Scans' if i == 0 else None)
        
        # Plot merged scan
        merged_y = spectrum.working_data.merged_scans[label]['data'][0]
        if normalize:
            merged_y = merged_y / np.max(merged_y)
        
        ax.plot(x, merged_y, 'r-', linewidth=2, label=f'Merged')
        
        # Show standard deviation band if requested
        if show_std:
            scan_data = []
            for i in range(len(spectrum.working_data.intensity_scans)):
                y = spectrum.working_data.intensity_scans[i]
                if normalize:
                    y = y / np.max(y)
                scan_data.append(y)
            
            scan_array = np.array(scan_data)
            std = np.std(scan_array, axis=0)
            ax.fill_between(x, merged_y - std, merged_y + std,
                          alpha=0.2, color='blue', label='±1σ')
        
        # Add merge info
        merge_info = spectrum.working_data.merged_scans[label]['info']
        title = f"{label} - {merge_info['method']}"
        if 'group_sizes' in merge_info:
            title += f"\nGroups: {merge_info['group_sizes']}"
        ax.set_title(title)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Labels
    if orientation == 'vertical':
        axes[-1].set_xlabel('Binding Energy (eV)')
        for ax in axes:
            ax.set_ylabel('Intensity' + (' (norm.)' if normalize else ' (a.u.)'))
    else:
        axes[0].set_ylabel('Intensity' + (' (norm.)' if normalize else ' (a.u.)'))
        for ax in axes:
            ax.set_xlabel('Binding Energy (eV)')
    
    plt.tight_layout()
    return fig, axes

def compare_merged_scans(spectrum, merge_labels=None, figsize=(8, 6),
                        normalize=False, offset=0, show_std=False,
                        colors=None, linestyles=None):
    """
    Compare only merged scans from different merge methods
    
    Parameters
    ----------
    spectrum : XPSSpectrum
        Spectrum containing the merged scans
    merge_labels : list of str, optional
        Labels of merged scans to compare. If None, uses all available
    figsize : tuple, optional
        Figure size (width, height)
    normalize : bool, optional
        Whether to normalize intensities
    offset : float, optional
        Vertical offset between spectra
    show_std : bool, optional
        Whether to show standard deviation bands
    colors : list, optional
        Colors for each merged scan
    linestyles : list, optional
        Line styles for each merged scan
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    # Get merge labels
    if merge_labels is None:
        merge_labels = list(spectrum.working_data.merged_scans.keys())
    
    # Default styles
    if colors is None:
        colors = ['k', 'r', 'b', 'g', 'm']  # Add more if needed
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.']  # Add more if needed
    
    fig, ax = plt.subplots(figsize=figsize)
    x = spectrum.working_data.binding_energy
    
    # Plot each merged scan
    for i, label in enumerate(merge_labels):
        # Get merged data
        merged_y = spectrum.working_data.merged_scans[label]['data'][0]
        if normalize:
            merged_y = merged_y / np.max(merged_y)
        
        # Add offset if specified
        y_plot = merged_y + i * offset
        
        # Plot merged scan
        color = colors[i % len(colors)]
        style = linestyles[i % len(linestyles)]
        ax.plot(x, y_plot, color=color, linestyle=style, linewidth=2,
                label=f'{label}')
        
        # Add standard deviation if requested
        if show_std:
            scan_data = []
            for scan in spectrum.working_data.intensity_scans:
                y = scan
                if normalize:
                    y = y / np.max(y)
                scan_data.append(y)
            
            scan_array = np.array(scan_data)
            std = np.std(scan_array, axis=0)
            ax.fill_between(x, y_plot - std, y_plot + std,
                          alpha=0.2, color=color)
    
    # Labels and styling
    ax.set_xlabel('Binding Energy (eV)')
    ax.set_ylabel('Intensity' + 
                 (' (normalized)' if normalize else ' (a.u.)') +
                 (' + offset' if offset != 0 else ''))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add merge methods info in title
    methods = [f"{label}: {spectrum.working_data.merged_scans[label]['info']['method']}"
              for label in merge_labels]
    ax.set_title('Comparison of merge methods\n' + '\n'.join(methods))
    
    plt.tight_layout()
    return fig, ax