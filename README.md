# pyFitXPS

The aim of this project is to assit in fitting XPS data for secuencial spectra; e.g.: Depth profile assay, Irradiation Damage for Self-Assembled Monolayers of Molecules on Surface, and so on...

This project contain several function wich allow to load **.xy files** generated by **SpecsLab Prodigy** software from [SPECS](https://www.specs-group.com/specsgroup/support/downloads/ "download SpecsLab Prodigy"). Also have others functions to _"select the range of data to fit"_, fit the _Fermi Edge_ and _correct the energy scale_.

The data are managed by **dictionaries**. The idea to use dictionary is to store the "original data" (as tuple, to prevent modifications), the "corrected data" and fit's result in one objet which can be easily saved. The fit procedures are realised using [LMFIT](https://lmfit.github.io/lmfit-py/). 

The XPS peaks are fitted using similar aproches than is commonly used in IgorPro by **XPSDoniach** procedures. 

This project uses the most precise physical focus during the tuning process. The functions used for the adjustment are built by means of the convolution of a peak function (Lorenztian or Donian-Sunjic) with a Gaussian function of the measurement process.

## TODO

### Loading files

- [x] Read one .XY file for separated region from SpecLab Prodigy software.
- [ ] Read one .XY file for "all data in one file".
- [ ] Read .XY file for separated scans.

### Manage data

- [x] Correct energy scale manually.
- [x] Correct energy scale from reference FL previously fitted.
- [x] Correcto energy scale for several regions at once

### Fitting procedures

- [x] Functions to fit Fermi Edge

