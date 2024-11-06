import os
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# XPS Spectrum
# Revised Data Structure
#------------------------------------------------------------------------------
class XPSMetadata:
    """Stores and handles XPS metadata with energy conversion capabilities"""
    
    REQUIRED_GLOBAL_FIELDS = [
        'Energy Axis',
        'Count Rate',
        'Separate Scan Data',
        'Transmission Function'
    ]
    
    REQUIRED_REGION_FIELDS = [
        'Region',
        'Scan Mode',
        'Curves/Scan',
        'Excitation Energy',
        'Pass Energy'
    ]
    
    def __init__(self):
        self.metadata = {}  # Dictionary storing key-value pairs
    
    def __getitem__(self, key):
        return self.metadata.get(key)
    
    def __setitem__(self, key, value):
        self.metadata[key] = value
    
    def get(self, key, default=None):
        """Get metadata value"""
        return self.metadata.get(key, default)
    
    def convert_energy(self, energy_value, to_binding=True):
        """
        Convert between Binding and Kinetic Energy
        BE = hν - KE
        """
        excitation_energy = float(self.metadata.get('Excitation Energy', 0))
        if to_binding:
            return excitation_energy - energy_value
        return excitation_energy - energy_value

class XPSOriginalData:
    """Container for original, unmodifiable data"""
    def __init__(self):
        # Energy scales (will be made read-only)
        self._binding_energy = None
        self._kinetic_energy = None
        # Intensity data for all scans (will be made read-only)
        self._intensity_scans = None
        
    @property
    def binding_energy(self):
        """Read-only binding energy values"""
        return self._binding_energy
        
    @property
    def kinetic_energy(self):
        """Read-only kinetic energy values"""
        return self._kinetic_energy
        
    @property
    def intensity_scans(self):
        """Read-only intensity values for all scans"""
        return self._intensity_scans
    
    def set_data(self, binding_energy, kinetic_energy, intensity_scans):
        """Set data once during initialization"""
        if self._binding_energy is not None:
            raise ValueError("Original data can only be set once")
        
        # Convert inputs to numpy arrays if they aren't already
        self._binding_energy = np.asarray(binding_energy, dtype=float)
        self._kinetic_energy = np.asarray(kinetic_energy, dtype=float)
        self._intensity_scans = np.asarray(intensity_scans, dtype=float)
        
        # Make arrays read-only
        self._binding_energy.flags.writeable = False
        self._kinetic_energy.flags.writeable = False
        self._intensity_scans.flags.writeable = False

class XPSWorkingData:
    """Container for working data that can be modified"""
    def __init__(self):
        self.binding_energy = None
        self.kinetic_energy = None
        self.intensity_scans = None
        self.merged_scans = {}  # Dictionary to store different merged scan sets
    
    def reset_from_original(self, original_data):
        """Reset working data from original data"""
        self.binding_energy = np.copy(original_data.binding_energy)
        self.kinetic_energy = np.copy(original_data.kinetic_energy)
        self.intensity_scans = np.copy(original_data.intensity_scans)
    
    def add_merged_scans(self, merged_scans):
        """Add merged scans to the working data"""
        self.merged_scans.update(merged_scans)

class XPSSpectrum:
    """Single spectrum/region with its metadata and data"""
    def __init__(self, metadata=None):
        """Initialize with empty data containers"""
        self.metadata = metadata if metadata is not None else XPSMetadata()
        self.original_data = XPSOriginalData()
        self.working_data = XPSWorkingData()
        self.fit_results = {}  # Add this line to initialize fit_results
    
    def set_data(self, energy_values, intensity_scans, excitation_energy):
        """
        Set data for the region
        
        Parameters
        ----------
        energy_values : array-like
            Energy values in the original scale (BE or KE according to metadata)
        intensity_scans : array-like
            2D array of intensity values (scans × energy points)
        excitation_energy : float
            Excitation energy for BE/KE conversion
        """
        # Check energy axis type from metadata
        energy_axis = self.metadata.get('Energy Axis', '')
        
        # Convert energy values to numpy array
        energy_values = np.array(energy_values, dtype=float)
        
        # Assign energy values according to metadata
        if energy_axis == 'Kinetic Energy':
            # Input values are KE, calculate BE
            kinetic_energy = energy_values
            binding_energy = excitation_energy - kinetic_energy
        else:  # 'Binding Energy' or default
            # Input values are BE, calculate KE
            binding_energy = energy_values
            kinetic_energy = excitation_energy - binding_energy
        
        # Invert binding energy for fitting
        binding_energy = -binding_energy
        
        # Set original data (becomes read-only)
        self.original_data.set_data(
            binding_energy, 
            kinetic_energy, 
            intensity_scans
        )
        
        # Initialize working data
        self.reset_data()
        
        # Automatically merge all scans
        self.merge_scans(method='average', label='all_average')
        self.merge_scans(method='sum', label='all_sum')
    
    def reset_data(self):
        """Reset working data to original values"""
        self.working_data.reset_from_original(self.original_data)
    
    @property
    def n_scans(self):
        """Number of scans in the data"""
        if self.original_data.intensity_scans is None:
            return 0
        return len(self.original_data.intensity_scans)
    
    def get_scan(self, index, energy_scale='binding'):
        """
        Get specific scan data
        
        Parameters
        ----------
        index : int
            Scan index (0-based)
        energy_scale : str
            'binding' or 'kinetic'
        
        Returns
        -------
        energy, intensity : tuple of arrays
        """
        if not 0 <= index < self.n_scans:
            raise IndexError(f"Scan index {index} out of range")
            
        energy = (self.working_data.binding_energy if energy_scale == 'binding' 
                 else self.working_data.kinetic_energy)
        return energy, self.working_data.intensity_scans[index]
    
    def merge_scans(self, method='average', label=None):
        """
        Merge all scans into a single spectrum
        
        Parameters
        ----------
        method : str, optional
            'average' or 'sum' for merging scans
        label : str, optional
            Label for the merged scan. If None, will be auto-generated
            
        Returns
        -------
        dict
            Information about the merging process
        """
        if method not in ['average', 'sum']:
            raise ValueError("Method must be 'average' or 'sum'")
        
        # Generate default label if none provided
        if label is None:
            label = f"all_{method}"
            # Ensure unique label
            base_label = label
            counter = 1
            while label in self.working_data.merged_scans:
                label = f"{base_label}_{counter}"
                counter += 1
        
        # Merge all scans
        if method == 'average':
            merged = np.mean(self.working_data.intensity_scans, axis=0)[np.newaxis, :]
        else:  # sum
            merged = np.sum(self.working_data.intensity_scans, axis=0)[np.newaxis, :]
        
        # Store merged scan with consistent info structure
        info = {
            'original_scans': self.working_data.intensity_scans.shape[0],
            'groups_created': 1,
            'group_sizes': [self.working_data.intensity_scans.shape[0]],
            'method': method
        }
        
        self.working_data.merged_scans[label] = {
            'data': merged,
            'info': info
        }
        
        return info
    
    def correct_energy_scale(self, shift):
        """
        Correct both energy scales in working data by applying a shift
        
        Parameters
        ----------
        shift : float
            Value to shift both energy scales (will be subtracted)
        """
        # Apply same shift to both scales
        self.working_data.binding_energy = self.original_data.binding_energy - shift
        self.working_data.kinetic_energy = self.original_data.kinetic_energy - shift
        
        return shift  # Return shift value for reporting
    
    def reset_energy_scale(self):
        """Reset energy scales to original values"""
        self.working_data.binding_energy = np.copy(self.original_data.binding_energy)
        self.working_data.kinetic_energy = np.copy(self.original_data.kinetic_energy)
        print("Energy scales reset to original values")
    
    def merge_scan_groups(self, group_size=None, n_groups=None, method='average', label=None):
        """
        Group and merge scans into a new array and store it
        
        Parameters
        ----------
        group_size : int, optional
            Number of scans per group. If None, will be calculated from n_groups
        n_groups : int, optional
            Number of groups desired. If None, will be calculated from group_size
        method : str, optional
            'average' or 'sum' for merging scans within each group
        label : str, optional
            Label for the merged scan set. If None, will be auto-generated
            
        Returns
        -------
        dict
            Information about the grouping process
        """
        n_scans = self.working_data.intensity_scans.shape[0]
        
        # Input validation
        if (group_size is None and n_groups is None) or \
           (group_size is not None and n_groups is not None):
            raise ValueError("Must specify either group_size or n_groups")
            
        if method not in ['average', 'sum']:
            raise ValueError("Method must be 'average' or 'sum'")
        
        # Calculate grouping parameters
        if group_size is not None:
            if group_size <= 0:
                raise ValueError("group_size must be positive")
            n_groups = n_scans // group_size
            remainder = n_scans % group_size
            default_label = f"{method}_{group_size}scans"
        else:  # n_groups is specified
            if n_groups <= 0 or n_groups > n_scans:
                raise ValueError(f"n_groups must be between 1 and {n_scans}")
            group_size = n_scans // n_groups
            remainder = n_scans % n_groups
            default_label = f"{method}_{n_groups}groups"
        
        # Use provided label or generate unique one
        if label is None:
            label = default_label
            # Ensure unique label
            base_label = label
            counter = 1
            while label in self.working_data.merged_scans:
                label = f"{base_label}_{counter}"
                counter += 1
            
        # Create new array for merged scans
        merged_scans = np.zeros((n_groups, self.working_data.intensity_scans.shape[1]))
        group_sizes = []
        start_idx = 0
        
        # Process complete groups
        for i in range(n_groups):
            current_size = group_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            
            # Extract and merge scans for this group
            group_scans = self.working_data.intensity_scans[start_idx:end_idx]
            if method == 'average':
                merged_scans[i] = np.mean(group_scans, axis=0)
            else:  # sum
                merged_scans[i] = np.sum(group_scans, axis=0)
            
            group_sizes.append(current_size)
            start_idx = end_idx
        
        # Store merged scans with label
        self.working_data.merged_scans[label] = {
            'data': merged_scans,
            'info': {
                'original_scans': n_scans,
                'groups_created': n_groups,
                'group_sizes': group_sizes,
                'method': method
            }
        }
        
        # Print summary
        print(f"\nScan grouping summary for '{label}':")
        print(f"Original scans: {n_scans}")
        print(f"Groups created: {n_groups}")
        print(f"Group sizes: {group_sizes}")
        print(f"Merge method: {method}")
        
        return self.working_data.merged_scans[label]['info']
    
    def list_merged_scans(self):
        """List all stored merged scan sets"""
        if not self.working_data.merged_scans:
            print("No merged scans available")
            return
        
        print("\nAvailable merged scan sets:")
        for label, data in self.working_data.merged_scans.items():
            info = data['info']
            print(f"\n{label}:")
            print(f"  Groups: {info['groups_created']}")
            print(f"  Method: {info['method']}")
            print(f"  Group sizes: {info['group_sizes']}")
            print(f"  Original scans: {info['original_scans']}")
    
    def get_merged_scans(self, label):
        """
        Get specific merged scan set
        
        Parameters
        ----------
        label : str
            Label of the merged scan set
            
        Returns
        -------
        numpy.ndarray
            Merged scan data
        """
        if label not in self.working_data.merged_scans:
            raise KeyError(f"Merged scan set '{label}' not found")
        return self.working_data.merged_scans[label]['data']

    def plot(self, merged_label=None, normalize=False):
        """
        Plot spectrum data
        
        Parameters
        ----------
        merged_label : str, optional
            Label of merged scan to plot. If None, plots all scans
        normalize : bool, optional
            If True, normalize intensities to [0,1] for better comparison
            Default is False to preserve original intensities
            
        Returns
        -------
        tuple
            Figure and axes
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get binding energy
        be = self.working_data.binding_energy
        
        if merged_label is not None:
            # Plot merged scan
            if merged_label not in self.working_data.merged_scans:
                raise KeyError(f"Merged scan '{merged_label}' not found")
            
            intensity = self.working_data.merged_scans[merged_label]['data'][0]
            if normalize:
                intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
            ax.plot(be, intensity, label=f'Merged ({merged_label})')
            
        else:
            # Plot all scans
            for i, scan in enumerate(self.working_data.intensity_scans):
                if normalize:
                    scan = (scan - np.min(scan)) / (np.max(scan) - np.min(scan))
                ax.plot(be, scan, alpha=0.5, label=f'Scan {i+1}')
        
        # Set labels and title
        ax.set_xlabel('Binding Energy (eV)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(self.metadata.get('Region', 'XPS Spectrum'))
        
        # Add legend
        if merged_label is None and self.working_data.intensity_scans.shape[0] > 1:
            ax.legend()
        
        plt.tight_layout()
        return fig, ax

class XPSExperiment:
    """
    Collection of XPS spectra/regions
    """
    def __init__(self):
        self.global_metadata = XPSMetadata()
        self.spectra = {}  # Dictionary of region_name: XPSSpectrum
    
    def get_region(self, name):
        """Get a specific region/spectrum by name"""
        return self.spectra.get(name)
    
    def list_regions(self):
        """List all available regions"""
        return list(self.spectra.keys())
    
    def print_summary(self):
        """Print summary of loaded data in table format"""
        # Define headers
        headers = ['Region', 'BE Range (eV)', 'Pass Energy', 'Energy Step', 
                  'Values', 'Scans', 'Comment']
        
        # Get maximum width for each column
        widths = [len(h) for h in headers]
        rows = []
        
        # Collect data for each region
        for region_name, spectrum in self.spectra.items():
            # Get BE range from working data
            be_min = min(spectrum.working_data.binding_energy)
            be_max = max(spectrum.working_data.binding_energy)
            be_range = f"{be_min:.1f} - {be_max:.1f}"
            
            # Get energy step from working data
            energy_points = spectrum.working_data.binding_energy
            energy_step = abs(energy_points[1] - energy_points[0])
            
            # Create row
            row = [
                region_name,
                be_range,
                spectrum.metadata.get('Pass Energy', ''),
                f"{energy_step:.3f}",
                str(len(energy_points)),
                str(spectrum.working_data.intensity_scans.shape[0]),
                spectrum.metadata.get('Comment', '')
            ]
            rows.append(row)
            
            # Update column widths
            widths = [max(w, len(str(item))) for w, item in zip(widths, row)]
        
        # Print table
        # Header
        print("\nSummary of XPS Data:")
        print("-" * (sum(widths) + len(widths) * 3 + 1))
        header_format = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
        print(header_format)
        print("-" * (sum(widths) + len(widths) * 3 + 1))
        
        # Data rows
        for row in rows:
            row_format = "| " + " | ".join(f"{str(item):<{w}}" for item, w in zip(row, widths)) + " |"
            print(row_format)
        
        print("-" * (sum(widths) + len(widths) * 3 + 1))
    
    def delete_region(self, region_name):
        """
        Delete a single region from the experiment
        
        Parameters
        ----------
        region_name : str
            Name of the region to delete
            
        Raises
        ------
        KeyError
            If region_name doesn't exist
        """
        if region_name in self.spectra:
            del self.spectra[region_name]
            print(f"Region '{region_name}' deleted")
        else:
            raise KeyError(f"Region '{region_name}' not found")
    
    def delete_regions(self, region_names):
        """
        Delete multiple regions from the experiment
        
        Parameters
        ----------
        region_names : list of str
            Names of regions to delete
        """
        not_found = []
        deleted = []
        
        for region in region_names:
            if region in self.spectra:
                del self.spectra[region]
                deleted.append(region)
            else:
                not_found.append(region)
        
        if deleted:
            print(f"Deleted regions: {', '.join(deleted)}")
        if not_found:
            print(f"Regions not found: {', '.join(not_found)}")
    
    def clean_regions(self, keep_regions=None):
        """
        Clean up regions, optionally keeping specified ones
        
        Parameters
        ----------
        keep_regions : list of str, optional
            List of region names to keep. If None, asks for confirmation
            to delete all regions
        """
        if keep_regions is not None:
            # Keep only specified regions
            to_delete = [reg for reg in self.spectra.keys() 
                        if reg not in keep_regions]
            self.delete_regions(to_delete)
        else:
            # Ask for confirmation before deleting all regions
            response = input("Are you sure you want to delete all regions? (y/n): ")
            if response.lower() == 'y':
                self.spectra.clear()
                print("All regions deleted")
            else:
                print("Operation cancelled")
    
    def correct_energy_scale(self, shift, regions=None):
        """
        Correct energy scales for selected regions or all regions
        
        Parameters
        ----------
        shift : float
            Value to shift energy scales (will be subtracted)
        regions : list of str, optional
            List of region names to correct. If None, corrects all regions
        """
        if regions is None:
            regions = list(self.spectra.keys())
        
        corrected = []
        not_found = []
        
        for region_name in regions:
            if region_name in self.spectra:
                self.spectra[region_name].correct_energy_scale(shift)
                corrected.append(region_name)
            else:
                not_found.append(region_name)
        
        # Print summary
        print(f"\nEnergy scale correction: -{shift:.3f} eV")
        if corrected:
            print(f"Corrected regions: {', '.join(corrected)}")
        if not_found:
            print(f"Regions not found: {', '.join(not_found)}")
    
    def plot_all_regions(self, patterns, merged_label='all_average', normalize=False):
        """
        Plot region groups in separate subplots
        
        Parameters
        ----------
        patterns : list of str
            List of patterns to search in region names (e.g., ['C', 'S', 'Au'])
        merged_label : str, optional
            Label of merged scan to plot for each region
        normalize : bool, optional
            If True, normalize intensities to [0,1] for better comparison
            Default is False to preserve original intensities
            
        Returns
        -------
        tuple
            Figure and axes array
        """
        # Create subplots
        n_patterns = len(patterns)
        n_cols = min(3, n_patterns)  # Max 3 columns
        n_rows = (n_patterns + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5*n_cols, 4*n_rows),
                                squeeze=False)
        fig.suptitle('XPS Spectra Overview', fontsize=14)
        
        # Plot each pattern group
        for i, pattern in enumerate(patterns):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Find matching regions (considering spaces in pattern)
            regions = [name for name in self.spectra.keys() 
                      if pattern in name]  # Exact pattern match
            
            if not regions:
                print(f"No regions found matching pattern: {pattern}")
                ax.set_visible(False)
                continue
            
            # Plot each matching region
            for region_name in regions:
                spectrum = self.spectra[region_name]
                
                # Get data
                be = spectrum.working_data.binding_energy
                if merged_label not in spectrum.working_data.merged_scans:
                    print(f"Warning: merged scan '{merged_label}' not found for {region_name}")
                    continue
                    
                intensity = spectrum.working_data.merged_scans[merged_label]['data'][0]
                
                # Normalize if requested
                if normalize:
                    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
                
                # Plot
                ax.plot(be, intensity, label=region_name)
            
            # Set labels and title
            ax.set_xlabel('Binding Energy (eV)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_title(f'{pattern} Regions')
            
            # Add legend if multiple regions
            if len(regions) > 1:
                ax.legend()
        
        plt.tight_layout()
        return fig, axes

#------------------------------------------------------------------------------
# File Parsing
#------------------------------------------------------------------------------

class XPSFileParser:
    """Base parser with common functionality"""
    
    def __init__(self):
        self.current_metadata = XPSMetadata()
        self.data_sections = []
    
    def parse_file(self, filepath):
        """Parse XPS data file"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Split metadata and data sections
            self._split_sections(lines)
            
            # Parse global metadata
            self._parse_global_metadata()
            
            # Determine file type and parse accordingly
            if self._is_single_spectrum():
                return self._parse_single_spectrum()
            else:
                return self._parse_multi_spectrum()
                
        except Exception as e:
            raise ValueError(f"Error parsing file {filepath}: {str(e)}")
    
    def _split_sections(self, lines):
        """Split file into metadata and data sections"""
        current_section = []
        in_data = False
        
        for line in lines:
            if line.startswith('#'):
                if 'ColumnLabels' in line:
                    in_data = True
                current_section.append(line.strip('#').strip())
            else:
                if current_section and in_data:
                    self.data_sections.append(current_section)
                    current_section = []
                if line.strip():  # Skip empty lines
                    current_section.append(line.strip())
        
        if current_section:
            self.data_sections.append(current_section)
    
    def _parse_global_metadata(self):
        """Parse global metadata section"""
        if not self.data_sections:
            raise ValueError("No data sections found")
            
        for line in self.data_sections[0]:
            if ':' in line:
                key, value = [x.strip() for x in line.split(':', 1)]
                self.current_metadata[key] = value
    
    def _is_single_spectrum(self):
        """Determine if file contains single spectrum"""
        # Look for multiple Region entries in all sections
        for section in self.data_sections:
            region_count = sum(1 for line in section 
                             if line.startswith('Region:'))
            if region_count > 1:
                return False
            
        # Check if there's a "Spectrum Group" in metadata
        for section in self.data_sections:
            for line in section:
                if 'Group:' in line and 'Spectrum Group' in line:
                    return False
                    
        return True
    
    def _parse_single_spectrum(self):
        """Parse single spectrum data"""
        spectrum = XPSSpectrum()
        spectrum.metadata = self.current_metadata
        
        # Find data section (after ColumnLabels)
        data_lines = []
        found_column_labels = False
        
        for section in self.data_sections:
            for line in section:
                if 'ColumnLabels' in line:
                    found_column_labels = True
                    continue
                if found_column_labels and not line.startswith('#'):
                    # Try to parse as numbers
                    try:
                        # Split line and try converting to float
                        values = line.split()
                        if len(values) == 2:
                            float(values[0])  # Test if convertible
                            float(values[1])  # Test if convertible
                            data_lines.append(line)
                    except ValueError:
                        continue  # Skip lines that can't be converted to float
        
        if not data_lines:
            raise ValueError("No numeric data found in file")
            
        # Parse data
        energy = []
        intensity = []
        for line in data_lines:
            e, i = map(float, line.split())
            energy.append(e)
            intensity.append(i)
                
        spectrum._original_energy = np.array(energy)
        spectrum._original_intensity = np.array(intensity)
        spectrum.reset_data()
        
        return spectrum
    
    def _parse_multi_spectrum(self):
        """Parse multi-spectrum data with support for separated scans"""
        experiment = XPSExperiment()
        experiment.global_metadata = self.current_metadata
        
        current_region = None
        current_metadata = XPSMetadata()
        current_scan_data = []
        scan_data_collection = []  # List to collect all scans for current region
        
        for section in self.data_sections:
            for line in section:
                # New region starts
                if line.startswith('Region:'):
                    # Save previous region if exists
                    if current_region and scan_data_collection:
                        self._add_region_with_scans(experiment, current_region, 
                                                  current_metadata, scan_data_collection)
                        scan_data_collection = []
                    
                    current_region = line.split(':', 1)[1].strip()
                    current_metadata = XPSMetadata()
                    current_scan_data = []
                
                # New scan starts
                elif 'Scan:' in line or 'Curve:' in line:
                    if current_scan_data:
                        scan_data_collection.append(current_scan_data)
                        current_scan_data = []
                
                # Parse metadata
                elif ':' in line and not line.startswith('ColumnLabels'):
                    key, value = [x.strip() for x in line.split(':', 1)]
                    current_metadata[key] = value
                
                # Parse data
                elif line.strip() and not line.startswith('#'):
                    try:
                        values = line.split()
                        if len(values) == 2:
                            float(values[0])  # Test if convertible
                            float(values[1])  # Test if convertible
                            current_scan_data.append(line)
                    except ValueError:
                        continue
        
        # Add last region
        if current_region and current_scan_data:
            scan_data_collection.append(current_scan_data)
            self._add_region_with_scans(experiment, current_region, 
                                      current_metadata, scan_data_collection)
        
        return experiment

    def _add_region_with_scans(self, experiment, region_name, metadata, scan_data_collection):
        """Helper method to add a region with all its scans"""
        spectrum = XPSSpectrum()
        spectrum.metadata = metadata
        
        # Process all scans to get energy and intensity arrays
        energy_values = None
        intensity_scans = []
        
        for scan_data in scan_data_collection:
            if scan_data:  # Only process if we have data
                energy = []
                intensity = []
                for line in scan_data:
                    e, i = map(float, line.split())
                    energy.append(e)
                    intensity.append(i)
                
                # Store energy values from first scan
                if energy_values is None:
                    energy_values = np.array(energy)
                
                intensity_scans.append(intensity)
        
        if energy_values is not None and intensity_scans:
            # Convert to numpy array
            intensity_scans = np.array(intensity_scans)
            
            # Get excitation energy from metadata
            excitation_energy = float(metadata.get('Excitation Energy', 0))
            
            # Set data in spectrum
            spectrum.set_data(energy_values, intensity_scans, excitation_energy)
        
        experiment.spectra[region_name] = spectrum

#------------------------------------------------------------------------------
# Simple User Interface
#------------------------------------------------------------------------------

def load_xps_data(path):
    """
    Load XPS data from file
    
    Parameters
    ----------
    path : str
        Path to XPS data file
        
    Returns
    -------
    XPSExperiment or XPSSpectrum
        Loaded data structure
    """
    if os.path.isfile(path):
        parser = XPSFileParser()
        result = parser.parse_file(path)
        
        # Print information based on type of result
        if isinstance(result, XPSExperiment):
            print(f"\nLoaded multi-region file: {os.path.basename(path)}")
            result.print_summary()
        elif isinstance(result, XPSSpectrum):
            print(f"\nLoaded single-region file: {os.path.basename(path)}")
            # Maybe add a simple summary for single spectrum?
            
        return result
    else:
        raise FileNotFoundError(f"File not found: {path}")


