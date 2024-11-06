import numpy as np
from tabulate import tabulate
from IPython.display import display, HTML

class XPSRegion:
    """
    Class to represent a single XPS region with its metadata and data
    """
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata = metadata if metadata else {}
        self.scans = []
        self._current_scan = None  # Temporary storage for scan being loaded
        self.energy = None  # Shared energy scale
        self.is_separated = metadata.get('Separate Scan Data', 'no').lower() == 'yes' if metadata else False
    
    def start_new_scan(self, metadata=None):
        """Start a new scan with temporary storage"""
        self._current_scan = {
            'energy': [],
            'counts': [],
            'metadata': metadata if metadata else {}
        }
        return len(self.scans)  # Return index of scan being added
    
    def add_data_point(self, energy, counts):
        """Add a data point to current scan"""
        if self._current_scan is None:
            self.start_new_scan()
        
        self._current_scan['energy'].append(energy)
        self._current_scan['counts'].append(counts)
    
    def finalize_current_scan(self):
        """Finalize current scan and add to scans list"""
        if self._current_scan:
            # Convert lists to numpy arrays
            energy = np.array(self._current_scan['energy'])
            counts = np.array(self._current_scan['counts'])
            
            # For first scan, set the shared energy scale
            if self.energy is None:
                self.energy = energy
            
            self.scans.append({
                'energy': energy,
                'counts': counts,
                'metadata': self._current_scan['metadata']
            })
            
            self._current_scan = None
    
    def add_scan(self, energy, counts, metadata=None):
        """Add a scan to the region"""
        if metadata and metadata.get('Separate Scan Data', 'no').lower() == 'yes':
            self.is_separated = True
            # For first scan, set the shared energy scale
            if not self.energy:
                self.energy = np.array(energy)
            
            self.scans.append({
                'energy': energy,  # Could be replaced with indices if we want to optimize
                'counts': counts,
                'metadata': metadata if metadata else {}
            })
        else:
            # For non-separated scans, store as single dataset
            self.energy = np.array(energy)
            self.counts = np.array(counts)
            # Store metadata about number of scans
            if metadata:
                self.metadata.update(metadata)
    
    def get_scan_data(self, scan_index=None):
        """Get data for a specific scan or averaged data"""
        if self.is_separated:
            if scan_index is not None and scan_index < len(self.scans):
                return self.energy, self.scans[scan_index]['counts']
            else:
                # Return average of all scans
                all_counts = np.array([scan['counts'] for scan in self.scans])
                return self.energy, np.mean(all_counts, axis=0)
        else:
            return self.energy, self.counts
    
    def get_number_of_scans(self):
        """Get the actual number of scans"""
        if self.is_separated:
            return len(self.scans)
        else:
            # For non-separated data, get from metadata or default to 1
            return int(self.metadata.get('Number of Scans', 1))

    def finalize_data(self):
        """Convert temporary lists to numpy arrays after loading"""
        if self._current_scan:
            # Convert lists to numpy arrays
            energy = np.array(self._current_scan['energy'])
            counts = np.array(self._current_scan['counts'])
            
            # For first scan, set the shared energy scale
            if not self.energy:
                self.energy = energy
            
            self.scans.append({
                'energy': energy,
                'counts': counts,
                'metadata': self._current_scan['metadata']
            })
            
            self._current_scan = None

class XPSFile:
    """
    Class to represent a single XPS file containing multiple regions
    """
    def __init__(self, filename=None):
        """Initialize XPS file object"""
        self.filename = filename
        self.regions = {}
        self.global_metadata = {}
    
    def add_region(self, region_name):
        """Add a new region to the file"""
        # Create initial metadata including energy axis info
        initial_metadata = {
            'Energy Axis': self.global_metadata.get('Energy Axis', 'Binding Energy'),
            'Separate Scan Data': self.global_metadata.get('Separate Scan Data', 'no')
        }
        
        self.regions[region_name] = XPSRegion(region_name, metadata=initial_metadata)
        return self.regions[region_name]
    
    def delete_regions(self, region_names):
        """
        Delete one or more regions from the file
        
        Parameters
        ----------
        region_names : str or list
            Single region name or list of region names to delete
        """
        if isinstance(region_names, str):
            region_names = [region_names]
            
        deleted = []
        not_found = []
        
        for name in region_names:
            if name in self.regions:
                del self.regions[name]
                deleted.append(name)
            else:
                not_found.append(name)
        
        if deleted:
            print(f"Successfully deleted regions: {', '.join(deleted)}")
        if not_found:
            print(f"Regions not found: {', '.join(not_found)}")
            
        return deleted
    
    def list_regions(self):
        """Print a list of all available regions"""
        print("\nAvailable regions:")
        for name in self.regions.keys():
            print(f"- {name}")
    
    def summary(self, return_str=False):
        """Create a formatted table summary"""
        # Get energy type
        energy_type = self.global_metadata.get('Energy Axis', 'Binding Energy')
        energy_type = ' '.join(energy_type.split())
        energy_scale = 'KE' if 'Kinetic' in energy_type else 'BE'
        
        table_data = []
        headers = ['Region', f'{energy_type} Range (eV)', 'Pass Energy (eV)', 
                  'Step (eV)', 'Points/Scan', 'Num Scans', 'Comment']
        
        for name, region in self.regions.items():
            # Get actual number of scans
            num_scans = region.get_number_of_scans()
            
            # Get energy data (now using shared energy scale)
            energy = region.energy
            
            # Calculate range and step
            if energy is None or len(energy) == 0:
                energy_range = 'N/A'
                energy_step = 'N/A'
                points = 0
            else:
                energy_range = f"{min(energy):.1f} - {max(energy):.1f}"
                if len(energy) > 1:
                    energy_step = abs(energy[1] - energy[0])
                    energy_step = f"{energy_step:.3f}"
                else:
                    energy_step = 'N/A'
                points = len(energy)
            
            # Get metadata
            pass_energy = region.metadata.get('Pass Energy', 'N/A')
            comment = region.metadata.get('Comment', '')
            
            table_data.append([
                name, energy_range, pass_energy, energy_step, points, num_scans, comment
            ])
        
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        
        if return_str:
            return table
        else:
            try:
                from IPython.display import display, HTML
                display(HTML(f"<pre>{table}</pre>"))
            except (ImportError, NameError):
                print(table)

    def _repr_html_(self):
        """
        IPython display hook for nice notebook rendering
        """
        table = self.summary(return_str=True)
        return f"<pre>{table}</pre>"

    def print_all_scans(self):
        """Print scan information for all regions"""
        print("\nScan Information for all regions:")
        for name, region in self.regions.items():
            region.print_scan_info()

class XPSDataManager:
    def __init__(self, path="./"):
        self.path = path
        self.files = {}
    
    def load_file(self, filename, show_summary=True):
        """
        Load an XY file and determine loading method based on scan separation
        
        Parameters
        ----------
        filename : str
            Path to the XY file
        show_summary : bool, optional
            Whether to print a summary after loading (default: True)
        
        Returns
        -------
        XPSFile
            Loaded XPS data file
        """
        # Determine if file uses separated scans
        separate_scans = False
        with open(filename, 'r') as f:
            for line in f:
                if 'Separate Scan Data:' in line:
                    separate_scans = 'yes' in line.lower()
                    break
        
        # Load file with appropriate method
        loaded_file = self._load_separate_scans(filename) if separate_scans else self._load_merged_scans(filename)
        
        # Show summary if requested
        if show_summary:
            print(f"\nFile: {filename}")
            print(f"Total Regions: {len(loaded_file.regions)}")
            print(f"Energy Axis: {loaded_file.global_metadata.get('Energy Axis', 'Not specified')}")
            print(f"Transmission Function: {loaded_file.global_metadata.get('Transmission Function', 'No')}")
            print("\nRegions Summary:")
            loaded_file.summary()
        
        return loaded_file
    
    def _load_merged_scans(self, filename):
        """Loading method for merged scan data"""
        current_file = XPSFile(filename)
        current_region = None
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                if line.startswith('# XY-Serializer Export Settings:'):
                    continue
                    
                if line.startswith('#   ') and ':' in line:
                    # Parse export settings
                    key = line[1:].split(':', 1)[0].strip()
                    value = line.split(':', 1)[1].strip()
                    current_file.global_metadata[key] = value
                
                elif line.startswith('# Region:'):
                    # Finalize previous region if exists
                    if current_region and current_region._current_scan:
                        current_region.finalize_current_scan()
                    
                    region_name = line.split(':')[1].strip()
                    current_region = current_file.add_region(region_name)
                    # Start a single scan for merged data
                    current_region.start_new_scan()
                
                elif line.startswith('#') and ':' in line:
                    # Parse metadata
                    key = line[1:].split(':', 1)[0].strip()
                    value = line.split(':', 1)[1].strip()
                    if current_region and current_region._current_scan:
                        current_region._current_scan['metadata'][key] = value
                    else:
                        current_file.global_metadata[key] = value
                
                elif not line.startswith('#'):
                    # Parse data line
                    parts = line.split()
                    if len(parts) >= 2 and current_region:
                        energy = float(parts[0])
                        counts = float(parts[1])
                        current_region.add_data_point(energy, counts)
        
        # Finalize last region
        if current_region and current_region._current_scan:
            current_region.finalize_current_scan()
            
        return current_file
    
    def _load_separate_scans(self, filename):
        """Loading method for separated scan data"""
        current_file = XPSFile(filename)
        current_region = None
        current_scan_metadata = {}
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                if line.startswith('# XY-Serializer Export Settings:'):
                    continue
                    
                if line.startswith('#   ') and ':' in line:
                    # Parse export settings
                    key = line[1:].split(':', 1)[0].strip()
                    value = line.split(':', 1)[1].strip()
                    current_file.global_metadata[key] = value
                
                elif line.startswith('# Region:'):
                    # Finalize previous region's current scan if exists
                    if current_region and current_region._current_scan:
                        current_region.finalize_current_scan()
                    
                    region_name = line.split(':')[1].strip()
                    current_region = current_file.add_region(region_name)
                    current_scan_metadata = {}
                
                elif line.startswith('# Cycle:') and 'Scan:' in line:
                    # Finalize previous scan if exists
                    if current_region and current_region._current_scan:
                        current_region.finalize_current_scan()
                    
                    # Start new scan with metadata
                    if current_region:
                        current_region.start_new_scan(current_scan_metadata)
                        current_scan_metadata = {}  # Reset metadata for next scan
                
                elif line.startswith('#') and ':' in line:
                    # Parse metadata
                    key = line[1:].split(':', 1)[0].strip()
                    value = line.split(':', 1)[1].strip()
                    if current_region and current_region._current_scan:
                        current_region._current_scan['metadata'][key] = value
                    elif current_region:
                        current_scan_metadata[key] = value
                    else:
                        current_file.global_metadata[key] = value
                
                elif not line.startswith('#'):
                    # Parse data line
                    parts = line.split()
                    if len(parts) >= 2 and current_region:
                        energy = float(parts[0])
                        counts = float(parts[1])
                        current_region.add_data_point(energy, counts)
        
        # Finalize last scan if exists
        if current_region and current_region._current_scan:
            current_region.finalize_current_scan()
            
        return current_file