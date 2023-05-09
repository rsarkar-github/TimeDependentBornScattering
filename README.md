# TimeDependentBornScattering

- Numpy, Devito based Python codes for the Acoustic Time Dependent Born Scattering problem

- Contains all codes to reproduce figures in Part II: Chapter 1 of thesis

# How to reproduce the results? The following steps were tested on Linux (Centos & Ubuntu) only.
- After cloning this repository, install the Python packages as indicated in the file "requirements.txt".

- Run the following scripts:
  - p01_point_scatterers_image_cig_illustration.py
  - p01a_point_scatterers_cig_illustration.py
  - p02_marmousi_cig.py
  - p02a_marmousi_cig_plot.py
  - p03_flat_reflector_multi_shot_expt.py
  - p03a_flat_reflector_multi_shot_expt_plots.py
  - p04_gaussian_anomaly1_multi_shot_expt.py
  - p04a_gaussian_anomaly1_multi_shot_expt_plots.py
  - p05_gaussian_anomaly2_multi_shot_expt.py
  - p05a_gaussian_anomaly2_multi_shot_expt_plots.py
  - p06_sigsbee_long_offset_multi_shot_expt.py
  - p06a_sigsbee_long_offset_multi_shot_expt_plots.py
  - p07_sigsbee1_long_offset_multi_shot_expt.py
  - p07a_sigsbee1_long_offset_multi_shot_expt_plots.py

- For example, to run **p01_point_scatterers_image_cig_illustration.py**, navigate to the directory where you 
git cloned this project, and from there execute 
**python -m TimeDependentBornScattering.Scripts.p01_point_scatterers_image_cig_illustration**. Also make sure that you
are in the right python environment with the needed packages.

- The above runs will create some files in the directory **TimeDependentBornScattering/Data/**, and all figures will be 
created in the diectory **TimeDependentBornScattering/Fig/**.
