import numpy as np
import scipy as sp
import time
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model, plot_image_xy, plot_images_grid_xy
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

    # -----------------------------------------------------------------------------
    # Single source placed at center
    def single_source_center():

        filestr = "p01_point_scatterers_single_source_center"

        # Create params dicts
        params = {
            "Nx": 300,
            "Nz": 100,
            "Nt": 100,  # this has to be updated later
            "nbl": 75,
            "Ns": 1,
            "Nr": 300,
            "so": 4,
            "to": 2
        }

        # Create velocity
        vel = create_model(shape=(params["Nx"], params["Nz"]))
        vel.vp.data[:, :] = 2.0

        # Simulation time, wavelet
        t0 = 0.
        tn = 2000.  # Simulation last 2 second (2000 ms)
        f0 = 0.015  # Source peak frequency is 15Hz (0.015 kHz)

        # Reflection acquisition geometry (sources and receivers are equally spaced in X direction)
        src_depth = 20.0  # Depth is 20m
        rec_depth = 20.0  # Depth is 20m

        src_coord = np.empty((params["Ns"], 2))
        src_coord[:, 0] = vel.domain_size[0] / 2.0
        src_coord[:, 1] = src_depth

        rec_coord = np.empty((params["Nr"], 2))
        rec_coord[:, 0] = np.linspace(0, vel.domain_size[0], num=params["Nr"])
        rec_coord[:, 1] = rec_depth

        # Create the geometry objects for background velocity models
        src_dummy = np.empty((1, 2))

        src_dummy[0, :] = src_coord[int(src_coord.shape[0] / 2), :]
        geometry = AcquisitionGeometry(vel, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
        params["Nt"] = geometry.nt
        del src_dummy

        # Define a solver object
        solver = AcousticWaveSolver(vel, geometry, space_order=params["so"])

        # Create point scatterers
        dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
        temp = np.zeros((params["Nx"], params["Nz"]), dtype=np.float32)
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.2)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.4)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.6)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.8)] = 1.0
        temp = sp.ndimage.gaussian_filter(input=temp, sigma=1, mode="nearest")
        for i in range(params["Nt"]):
            dm[i, :, :] = temp

        del temp

        # Plot input
        dm_scale = 1.0
        plot_image_xy(
            dm[100, :, :].T,
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0],
            y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
            scale=dm_scale, clip=1.0, colorbar=False,
            ylabel="Z [km]", xlabel="X [km]",
            grid="off", aspect="equal",
            fontname="STIXGeneral", fontsize=12,
            savefig_fname="/Fig/" + filestr + "_modelpert.pdf"
        )

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
        DevitoOperators.td_born_adjoint(
            born_data=td_born_data_true,
            model_pert=dm_image,
            src_coords=src_coord,
            vel=vel,
            geometry=geometry,
            solver=solver,
            params=params
        )

        # Plot imaged data (t-x sections at depth values 20%, 40%, 60%, 80%)
        image_nrows = 2
        image_ncols = 2
        image_arr = np.zeros(shape=(image_nrows, image_ncols, params["Nt"], params["Nx"]), dtype=np.float32)
        image_arr[0, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.2)]
        image_arr[0, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.4)]
        image_arr[1, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.6)]
        image_arr[1, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.8)]

        image_titles = [["(a)", "(b)"], ["(c)", "(d)"]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5,
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0], y0=t0, yn=tn, scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            xlabel="X [km]", ylabel="Time [s]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname="/Fig/" + filestr + "_adjoint_images.pdf"
        )

    print("Starting \'single source center\' experiment...")
    t1 = time.time()
    single_source_center()
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")

    # -----------------------------------------------------------------------------
    # Single source placed 1/4 away from edge
    def single_source_offset():

        filestr = "p01_point_scatterers_single_source_offset"

        # Create params dicts
        params = {
            "Nx": 300,
            "Nz": 100,
            "Nt": 100,  # this has to be updated later
            "nbl": 75,
            "Ns": 1,
            "Nr": 300,
            "so": 4,
            "to": 2
        }

        # Create velocity
        vel = create_model(shape=(params["Nx"], params["Nz"]))
        vel.vp.data[:, :] = 2.0

        # Simulation time, wavelet
        t0 = 0.
        tn = 2000.  # Simulation last 2 second (2000 ms)
        f0 = 0.015  # Source peak frequency is 15Hz (0.015 kHz)

        # Reflection acquisition geometry (sources and receivers are equally spaced in X direction)
        src_depth = 20.0  # Depth is 20m
        rec_depth = 20.0  # Depth is 20m

        src_coord = np.empty((params["Ns"], 2))
        src_coord[:, 0] = vel.domain_size[0] / 4.0
        src_coord[:, 1] = src_depth

        rec_coord = np.empty((params["Nr"], 2))
        rec_coord[:, 0] = np.linspace(0, vel.domain_size[0], num=params["Nr"])
        rec_coord[:, 1] = rec_depth

        # Create the geometry objects for background velocity models
        src_dummy = np.empty((1, 2))

        src_dummy[0, :] = src_coord[int(src_coord.shape[0] / 2), :]
        geometry = AcquisitionGeometry(vel, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
        params["Nt"] = geometry.nt
        del src_dummy

        # Define a solver object
        solver = AcousticWaveSolver(vel, geometry, space_order=params["so"])

        # Create point scatterers
        dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
        temp = np.zeros((params["Nx"], params["Nz"]), dtype=np.float32)
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.2)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.4)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.6)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.8)] = 1.0
        temp = sp.ndimage.gaussian_filter(input=temp, sigma=1, mode="nearest")
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
        dm_scale = 1.0
        DevitoOperators.td_born_adjoint(
            born_data=td_born_data_true,
            model_pert=dm_image,
            src_coords=src_coord,
            vel=vel,
            geometry=geometry,
            solver=solver,
            params=params
        )

        # Plot imaged data (t-x sections at depth values 20%, 40%, 60%, 80%)
        image_nrows = 2
        image_ncols = 2
        image_arr = np.zeros(shape=(image_nrows, image_ncols, params["Nt"], params["Nx"]), dtype=np.float32)
        image_arr[0, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.2)]
        image_arr[0, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.4)]
        image_arr[1, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.6)]
        image_arr[1, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.8)]

        image_titles = [["(a)", "(b)"], ["(c)", "(d)"]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5,
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0], y0=t0, yn=tn, scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            xlabel="X [km]", ylabel="Time [s]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname="/Fig/" + filestr + "_adjoint_images.pdf"
        )

    print("Starting \'single source offset\' experiment...")
    t1 = time.time()
    single_source_offset()
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")

    # -----------------------------------------------------------------------------
    # Multiple sources (10) placed at uniform spacing
    def multi_source_10():

        filestr = "p01_point_scatterers_multi_source_10"

        # Create params dicts
        params = {
            "Nx": 300,
            "Nz": 100,
            "Nt": 100,  # this has to be updated later
            "nbl": 75,
            "Ns": 10,
            "Nr": 300,
            "so": 4,
            "to": 2
        }

        # Create velocity
        vel = create_model(shape=(params["Nx"], params["Nz"]))
        vel.vp.data[:, :] = 2.0

        # Simulation time, wavelet
        t0 = 0.
        tn = 2000.  # Simulation last 2 second (2000 ms)
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
        geometry = AcquisitionGeometry(vel, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
        params["Nt"] = geometry.nt
        del src_dummy

        # Define a solver object
        solver = AcousticWaveSolver(vel, geometry, space_order=params["so"])

        # Create point scatterers
        dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
        temp = np.zeros((params["Nx"], params["Nz"]), dtype=np.float32)
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.2)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.4)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.6)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.8)] = 1.0
        temp = sp.ndimage.gaussian_filter(input=temp, sigma=1, mode="nearest")
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
        dm_scale = 2.0
        DevitoOperators.td_born_adjoint(
            born_data=td_born_data_true,
            model_pert=dm_image,
            src_coords=src_coord,
            vel=vel,
            geometry=geometry,
            solver=solver,
            params=params
        )

        # Plot imaged data (t-x sections at depth values 20%, 40%, 60%, 80%)
        image_nrows = 2
        image_ncols = 2
        image_arr = np.zeros(shape=(image_nrows, image_ncols, params["Nt"], params["Nx"]), dtype=np.float32)
        image_arr[0, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.2)]
        image_arr[0, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.4)]
        image_arr[1, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.6)]
        image_arr[1, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.8)]

        image_titles = [["(a)", "(b)"], ["(c)", "(d)"]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5,
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0], y0=t0, yn=tn, scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            xlabel="X [km]", ylabel="Time [s]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname="/Fig/" + filestr + "_adjoint_images.pdf"
        )

    print("Starting \'multi source 10\' experiment...")
    t1 = time.time()
    multi_source_10()
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")

    # -----------------------------------------------------------------------------
    # Multiple sources placed every grid point
    def multi_source_all():

        filestr = "p01_point_scatterers_multi_source_all"

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

        # Simulation time, wavelet
        t0 = 0.
        tn = 2000.  # Simulation last 2 second (2000 ms)
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
        geometry = AcquisitionGeometry(vel, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
        params["Nt"] = geometry.nt
        del src_dummy

        # Define a solver object
        solver = AcousticWaveSolver(vel, geometry, space_order=params["so"])

        # Create point scatterers
        dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
        temp = np.zeros((params["Nx"], params["Nz"]), dtype=np.float32)
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.2)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.4)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.6)] = 1.0
        temp[int(params["Nx"] / 2), int(params["Nz"] * 0.8)] = 1.0
        temp = sp.ndimage.gaussian_filter(input=temp, sigma=1, mode="nearest")
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
        DevitoOperators.td_born_adjoint(
            born_data=td_born_data_true,
            model_pert=dm_image,
            src_coords=src_coord,
            vel=vel,
            geometry=geometry,
            solver=solver,
            params=params
        )

        # Plot imaged data (t-x sections at depth values 20%, 40%, 60%, 80%)
        image_nrows = 2
        image_ncols = 2
        image_arr = np.zeros(shape=(image_nrows, image_ncols, params["Nt"], params["Nx"]), dtype=np.float32)
        image_arr[0, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.2)]
        image_arr[0, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.4)]
        image_arr[1, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.6)]
        image_arr[1, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.8)]

        image_titles = [["(a)", "(b)"], ["(c)", "(d)"]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5,
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0], y0=t0, yn=tn, scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            xlabel="X [km]", ylabel="Time [s]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname="/Fig/" + filestr + "_adjoint_images.pdf"
        )

    print("Starting \'multi source all\' experiment...")
    t1 = time.time()
    multi_source_all()
    t2 = time.time()
    print("Time taken = ", "{:4.2f}".format((t2 - t1)), " s.\n")