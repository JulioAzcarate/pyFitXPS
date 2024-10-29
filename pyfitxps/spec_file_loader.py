#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:56:39 2022
Updated on Sat Aug 05 18:30:00 2023
Updated on Thu Oct 24 22:48:05 2024

@author: Julio C. Azcárate
@institution: Centro Atómico Bariloche

This module provides functionality to read and analyze XPS (X-ray Photoelectron Spectroscopy) data 
from .xy files. It includes functions to load data, handle multiple files, and visualize spectra.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

class SPECFileLoader:
    def __init__(self, path="samefolder"):
        """
        Initialize the SPECFileLoader with a path to the directory containing .xy files.

        Parameters:
        path (str): The path to the directory containing .xy files. Defaults to 'samefolder'.
        """
        self.path = os.getcwd() if path == "samefolder" else path
        self.experiment_data = {}

    # --- File Listing Functions ---
    def list_files_xy(self):
        """List all .xy files in the specified directory along with the number of regions/spectra."""
        try:
            if not os.path.isdir(self.path):
                print(f"Path does not exist: {self.path}")
                return []
            XPS_files = glob.glob(os.path.join(self.path, "*.xy"))
            print(f"Found files: {XPS_files}")  # Debug print
            return [(filename, self.count_regions_in_file(filename)) for filename in XPS_files]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def count_regions_in_file(self, filename):
        """Count the number of regions/spectra in a given .xy file."""
        try:
            with open(filename, 'r') as file:
                return sum(1 for line in file if line.startswith("# Region:"))
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
        return 0

    # --- File Loading Functions ---
    def load_regions_xy(self, filename):
        try:
            with open(filename, 'r') as file:
                self.experiment_data = {}
                current_region = None
                dict_metadata = {}
                data_buffer = []
    
                for line in file:
                    line = line.strip()
                    if line.startswith("# Region:"):
                        if current_region and data_buffer:
                            self.experiment_data[current_region] = {
                                'metadata': dict_metadata,
                                'data_orig': self.process_data(data_buffer)
                            }
                        current_region = line.split(":")[1].strip()
                        dict_metadata = {}
                        data_buffer = []
                    elif line.startswith("# "):
                        k, v = line.replace("# ", "").split(":", maxsplit=1)
                        k = k.strip()
                        v = v.strip()
                        dict_metadata[k] = v
                    else:
                        try:
                            row = [float(x) for x in line.split()]
                            data_buffer.append(row)
                        except ValueError:
                            pass
    
                if current_region and data_buffer:
                    self.experiment_data[current_region] = {
                        'metadata': dict_metadata,
                        'data_orig': self.process_data(data_buffer)
                    }
    
            if self.experiment_data:
                return self.experiment_data
    
            return None
    
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return None
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            return None
    
    def process_data(self, data_buffer):
        # Convert metadata to correct type
        par_float = ('Dwell Time', 'Excitation Energy', 'Binding Energy', 'Pass Energy',
                     'Bias Voltage', 'Detector Voltage', 'Eff. Workfunction', 'Number of Scans')
        par_int = ('Curves/Scan', 'Values/Curve', 'Number of Scans')
        for k, v in self.experiment_data[list(self.experiment_data.keys())[0]]['metadata'].items():
            if k in par_float:
                self.experiment_data[list(self.experiment_data.keys())[0]]['metadata'][k] = float(v)
            elif k in par_int:
                self.experiment_data[list(self.experiment_data.keys())[0]]['metadata'][k] = int(v)
    
        # Load data as tuple containing two arrays
        data = np.array(data_buffer)
        if len(data) > 0 and len(data[0]) > 1:
            BE = data[:, 0]
            intensity = data[:, 1]
            return {'BE': BE, 'intensity': intensity}
        else:
            return None

    def load_all_files_xy_in(self):
        """Load all .xy files in the specified directory and store them in a dictionary."""
        files = self.list_files_xy()
        for filename, _ in files:
            self.load_regions_xy(filename)

    # --- Plotting Functions ---
    def plot_region(self, region, energy_scale='BE'):
        """
        Plot all spectra of the same XPS region.

        Parameters:
        region (str): The region to plot.
        energy_scale (str): The energy scale to use ('BE' or 'KE'). Defaults to 'BE'.
        """
        for spectra in self.experiment_data.keys():
            if region in spectra:
                plt.plot(
                    self.experiment_data[spectra]['data_orig'][energy_scale],
                    self.experiment_data[spectra]['data_orig']['intensity'],
                    label=self.experiment_data[spectra]['metadata']['Region']
                )
        plt.legend()
        plt.xlabel(f'{energy_scale} [eV]')
        plt.ylabel('Intensity (cps)')
        plt.title(f'Spectra for region: {region}')
        plt.show()

    def plot_all_regions_in(self, all_region_to_plot, energy_scale='BE'):
        """
        Plot all specified regions from the experiment data.

        Parameters:
        all_region_to_plot (list): List of regions to plot.
        energy_scale (str): The energy scale to use ('BE' or 'KE'). Defaults to 'BE'.
        """
        if len(all_region_to_plot) > 1:
            cols = 2
            rows = (len(all_region_to_plot) + 1) // 2
            fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(8, 3 * rows), constrained_layout=True)
            axs = axs.flatten()
            for i, element in enumerate(all_region_to_plot):
                for spectra in sorted(self.experiment_data.keys()):
                    if element in spectra:
                        axs[i].plot(
                            self.experiment_data[spectra]['data_orig'][energy_scale],
                            self.experiment_data[spectra]['data_orig']['intensity'],
                            label=self.experiment_data[spectra]['metadata']['Region']
                        )
                        axs[i].legend()
                        axs[i].set_xlabel(f'{energy_scale} [eV]')
                        axs[i].set_ylabel('Intensity [cps]')
            plt.show()
        else:
            region = all_region_to_plot[0]
            self.plot_region(region, energy_scale)

# Example of how to use the SPECFileLoader class
if __name__ == "__main__":
    analyzer = SPECFileLoader(path="path_to_your_files")
    analyzer.load_all_files_xy_in()
    analyzer.plot_region("Survey", energy_scale='BE')