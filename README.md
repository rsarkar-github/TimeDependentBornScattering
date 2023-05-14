# TimeDependentBornScattering

- Numpy, Devito based Python codes for the Acoustic Time Dependent Born Scattering problem

- Contains all codes to reproduce figures in Part II: Chapter 5 of thesis

# How to reproduce the results? The following steps were tested on Linux (Centos & Ubuntu) only.
- After cloning this repository, install the Python packages as indicated in the file "requirements.txt".
Steps to install the packages are indicated below in the Section **How to install the packages**.

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
  - p08_microlocal_image_pert1.py
  - p08a_microlocal_image_pert1_plots.py
  - p09_microlocal_image_pert2.py
  - p09a_microlocal_image_pert2_plots.py
  - p10_microlocal_image_pert3.py
  - p10a_microlocal_image_pert3_plots.py
  - p11_microlocal_image_pert4.py
  - p11a_microlocal_image_pert4_plots.py

- For example, to run **p01_point_scatterers_image_cig_illustration.py**, navigate to the directory where you 
git cloned this project, and from there execute 
**python -m TimeDependentBornScattering.Scripts.p01_point_scatterers_image_cig_illustration**. Also make sure that you
are in the right python environment with the needed packages.

- The above runs will create some files in the directory **TimeDependentBornScattering/Data/**, and all figures will be 
created in the diectory **TimeDependentBornScattering/Fig/**.

# How to install the packages (assumes conda is installed).

```ruby
conda create -n py39 python=3.9
conda activate py39
conda install -c anaconda numpy=1.23.5
conda install -c anaconda scipy=1.10.0
conda install -c numba numba=0.56.4
conda install matplotlib=3.7.1
pip install devito=4.8.0
```
