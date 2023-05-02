import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model, plot_image_xy, plot_images_grid_xy
from ..Utilities.Utils import ricker_time
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


if __name__ == "__main__":

    """
    This script does the following:
    - The physical velocity model is constant 2 km/s
    - Places time-invariant perturbations at different depth levels (point scatterers, but slightly smoothed)
    - Models TD Born scattering data
    - Images the modeled data
    - Experiment is repeated for single source (twice at different locations), multiple source
    - Sources placed at grid points at top of model, depth 20 m
    - Receivers placed at every grid point at top of model, depth 20 m
    """

    basepath = "TimeDependentBornScattering/"
    figdir = basepath + "Fig/"

    # -----------------------------------------------------------------------------
    # Multiple sources placed every grid point
    def multi_source_all(scale_fac=1.0):

        filestr = "p01_point_scatterers_multi_source_all"
        cig_aspect = 3.0

        # Create params dicts
        params = {
            "Nx": 300,
            "Nz": 100,
            "Nt": 100,  # this has to be updated later
            "nbl": 75,
            "Ns": 300,
            "Nr": 300,
            "so": 4,
            "to": 2
        }

        # Create velocity
        vel = create_model(shape=(params["Nx"], params["Nz"]))
        vel.vp.data[:, :] = 2.0

        vel0 = create_model(shape=(params["Nx"], params["Nz"]))
        vel0.vp.data[:, :] = 2.0 * 1.1

        # Simulation time, wavelet
        t0 = 0.
        tn = 1200.  # Simulation last 2 second (2000 ms)
        f0 = 0.015  # Source peak frequency is 15Hz (0.015 kHz)

        # Reflection acquisition geometry (sources and receivers are equally spaced in X direction)
        src_depth = 20.0  # Depth is 20m
        rec_depth = 20.0  # Depth is 20m

        src_coord = np.empty((params["Ns"], 2))
        src_coord[:, 0] = np.linspace(0, vel.domain_size[0], num=params["Ns"])
        src_coord[:, 1] = src_depth

        rec_coord = np.empty((params["Nr"], 2))
        rec_coord[:, 0] = np.linspace(0, vel.domain_size[0], num=params["Nr"])
        rec_coord[:, 1] = rec_depth

        # Create the geometry objects for background velocity models
        src_dummy = np.empty((1, 2))

        src_dummy[0, :] = src_coord[int(src_coord.shape[0] / 2), :]
        geometry = AcquisitionGeometry(vel0, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
        params["Nt"] = geometry.nt
        del src_dummy

        # Define a solver object
        solver = AcousticWaveSolver(vel0, geometry, space_order=params["so"])

        # Create point scatterers
        dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
        temp = np.zeros((params["Nx"], params["Nz"]), dtype=np.float32)
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.2)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.4)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.6)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.8)] = 1.0
        temp = sp.ndimage.gaussian_filter(input=temp, sigma=0.5, mode="nearest")
        for i in range(params["Nt"]):
            dm[i, :, :] = temp

        del temp

        # Perform Born modeling and generate data
        td_born_data_true = np.zeros((params["Ns"], params["Nt"], params["Nr"]), dtype=np.float32)
        DevitoOperators.td_born_forward(
            model_pert=dm,
            born_data=td_born_data_true,
            src_coords=src_coord,
            vel=vel,
            geometry=geometry,
            solver=solver,
            params=params
        )

        # Image the data
        dm_image = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
        dm_scale = 50

        # Create velocity
        vel1 = create_model(shape=(params["Nx"], params["Nz"]))
        vel1.vp.data[:, :] = 2.0 * scale_fac

        DevitoOperators.td_born_adjoint(
            born_data=td_born_data_true,
            model_pert=dm_image,
            src_coords=src_coord,
            vel=vel1,
            geometry=geometry,
            solver=solver,
            params=params
        )

        # Plot CIG at middle of horizontal grid
        cig = dm_image[:, int(params["Nx"] / 2), :].T
        plot_image_xy(
            cig,
            x0=t0, xn=tn,
            y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
            scale=dm_scale, clip=1.0, colorbar=False,
            ylabel="Z [km]", xlabel="Time [s]", xticklabels_fmt="{:4.2f}",
            grid="on", aspect=cig_aspect,
            fontname="STIXGeneral", fontsize=12,
            savefig_fname=figdir + filestr + "_" + "{:4.2f}".format(scale_fac) + "_cig.pdf"
        )


    print("Starting \'multi source all\' experiment 1...")
    t1 = time.time()
    multi_source_all(scale_fac=1.05)
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")

    print("Starting \'multi source all\' experiment 2...")
    t1 = time.time()
    multi_source_all(scale_fac=1.00)
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")

    print("Starting \'multi source all\' experiment 3...")
    t1 = time.time()
    multi_source_all(scale_fac=0.95)
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")
