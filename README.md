# DielectricSpectroscopyTheory

Within this directory are the tools required to reproduce the work performed in the paper titled "Simulating dielectric spectra: a demonstration of the direct electric field method and a new model for the nonlinear dielectric response".  

To cite this code, please cite our paper, listed above.

Contact information:

  - Michael Woodcox, Materials Science and Engineering Division, Material Measurement Laboratory, National Institute of Standards and Technology, michael.woodcox@nist.gov
  - Kathleen Schwarz, Materials Science and Engineering Division, Material Measurement Laboratory, National Institute of Standards and Technology, kathleen.schwarz@nist.gov
  - Ravishankar Sundararaman, Rensselaer Polytechnic Institute, sundar@rpi.edu


Folder contents:

The Matlab script titled "Main.m" can be used to calculate the coefficients of the polarization density for a sample water simulation. This sample output is located at Water\OUTPUT
The files "ReadFile.m" and "DirectFit.m" are used within this script to properly read in the data, and to successfully fir the parameters.

The directories titled "Water" and "Acetonitrile" contain the input files necessary to replicate our results. The *.lammps files contain the necessary LAMMPS input parameters, and the *.input files contain the structural information of the fluid.

The "Theory" directory contains the necessary Python input scripts (Theory.py and pulay.py) to reproduce the results of the theoretical model discussed in section 2.3 of the paper with and without the inclusion of the inertial correction. Sample outputs have also been provided
