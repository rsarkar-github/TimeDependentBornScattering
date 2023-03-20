# TimeDependentBornScattering

- Numpy, Scipy, Numba based Python codes for the Acoustic Time Dependent Born Scattering problem

- Contains all codes to reproduce figures in Part II: Chapter 1 of thesis

# How to reproduce the results? The following steps were tested on Linux (Centos & Ubuntu) only.
- After cloning this repository, install the Python packages as indicated in the file "requirements.txt".

- Run the following scripts:
  - point_scatterer_single_shot_expt.py
  - point_scatterer_multi_shot_expt.py
  - flat_reflector_single_shot_expt.py
  - flat_reflector_multi_shot_expt.py
  - gaussian_anomaly1_multi_shot_expt.py
  - gaussian_anomaly2_multi_shot_expt.py
  - sigsbee_multi_shot_expt.py
  - sigsbee1_multi_shot_expt.py
  - sigsbee_long_offset_multi_shot_expt.py
  - sigsbee1_long_offset_multi_shot_expt.py

- For example, to run **point_scatterer_single_shot_expt.py**, navigate to the directory where you 
git cloned this project, and from there execute 
**python -m TimeDependentBornScattering.Scripts.point_scatterer_single_shot_expt**. Also make sure that you
are in the right python environment with the needed packages.

- The above runs will create some files in the directory **TimeDependentBornScattering/Data/**
