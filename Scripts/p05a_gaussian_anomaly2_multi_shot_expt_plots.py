import numpy as np
from scipy import ndimage
import time
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model, plot_image_xy, plot_images_grid_xy
import matplotlib.pyplot as plt
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


if __name__ == "__main__":

    basepath = "TimeDependentBornScattering/"
    figdir = basepath + "Fig/"
    datadir = basepath + "Data/"
    filestr = "p05_gaussian_anomaly2_multi_shot"

    # ----------------------------------------------------------------------------
    # Geometry and defining the problem

    # Create params dicts
    params1 = {
        "Nx": 500,
        "Nz": 200,
        "Nt": 100,   # this has to be updated later
        "nbl": 75,
        "Ns": 10,
        "Nr": 200,
        "so": 4,
        "to": 2
    }

    ######################################################
    # This part of the code creates the models
    ######################################################
    # Create models
    v1 = create_model(shape=(params1["Nx"], params1["Nz"]))
    v1.vp.data[:, :] = 2.5

    # Initialize based on params1
    dv = np.zeros(shape=(params1["Nx"] + 2 * params1["nbl"], params1["Nz"] + 2 * params1["nbl"]), dtype=np.float32)
    n0 = dv.shape[0]
    n1 = dv.shape[1]

    # We will put 3 wide Gaussians in center, and t narrower Gaussians on top and bottom of it

    t = 10
    sigma_big = 20
    sigma_small = 1
    amplitude_big = 1500.0
    amplitude_small = 3.0

    big_gaussian = dv * 0
    big_gaussian[int(n0 * 0.5), int(n1 * 0.5)] = 1
    big_gaussian = ndimage.gaussian_filter(input=big_gaussian, sigma=sigma_big)
    dv += amplitude_big * big_gaussian

    big_gaussian = dv * 0
    big_gaussian[int(n0 * 0.25), int(n1 * 0.5)] = 1
    big_gaussian = ndimage.gaussian_filter(input=big_gaussian, sigma=sigma_big)
    dv += amplitude_big * big_gaussian

    big_gaussian = dv * 0
    big_gaussian[int(n0 * 0.75), int(n1 * 0.5)] = 1
    big_gaussian = ndimage.gaussian_filter(input=big_gaussian, sigma=sigma_big)
    dv += amplitude_big * big_gaussian

    step = int(params1["Nx"] / (t + 1))
    for i in range(t):
        small_gaussian = dv * 0
        small_gaussian[params1["nbl"] + (i + 1) * step, params1["nbl"] + int(params1["Nz"] * 0.25)] = 1
        small_gaussian = ndimage.gaussian_filter(input=small_gaussian, sigma=sigma_small)
        dv += amplitude_small * small_gaussian
    for i in range(t):
        small_gaussian = dv * 0
        small_gaussian[params1["nbl"] + (i + 1) * step, params1["nbl"] + int(params1["Nz"] * 0.75)] = 1
        small_gaussian = ndimage.gaussian_filter(input=small_gaussian, sigma=sigma_small)
        dv += amplitude_small * small_gaussian

    del n0, n1, t, sigma_big, sigma_small, amplitude_big, amplitude_small, big_gaussian, step

    # Create models
    v1_prime = create_model(shape=(params1["Nx"], params1["Nz"]))
    v1_prime.vp.data[:, :] = v1.vp.data - dv

    # Plot velocity
    plot_image_xy(
        v1_prime.vp.data[params1["nbl"]:params1["nbl"]+params1["Nx"], params1["nbl"]:params1["nbl"] + params1["Nz"]].T,
        x0=v1.origin[0], xn=v1.origin[0] + v1.domain_size[0],
        y0=v1.origin[1], yn=v1.origin[1] + v1.domain_size[1],
        vmin=1.8, vmax=2.7, colorbar=True, cmap="jet",
        ylabel="Z [km]", xlabel="X [km]",
        grid="on", aspect="equal",
        fontname="STIXGeneral", fontsize=12,
        savefig_fname=figdir + filestr + "_vel.pdf"
    )

    ######################################################################
    # This part of the code creates the acquisition geometry, solvers
    ######################################################################

    # Simulation time, wavelet
    t0 = 0.
    tn = 4000.          # Simulation last 4 second (4000 ms)
    f0 = 0.010          # Source peak frequency is 10Hz (0.010 kHz)

    # Reflection acquisition geometry (sources and receivers are equally spaced in X direction)
    src_depth = 20.0                        # Depth is 20m
    rec_depth = 20.0                        # Depth is 20m

    src_coord = np.empty((params1["Ns"], 2))
    if params1["Ns"] == 1:
        src_coord[:, 0] = 0.5 * v1.domain_size[0]
        src_coord[:, 1] = src_depth
    else:
        src_coord[:, 0] = np.linspace(0, v1.domain_size[0], num=params1["Ns"])
        src_coord[:, 1] = src_depth

    rec_coord = np.empty((params1["Nr"], 2))
    rec_coord[:, 0] = np.linspace(0, v1.domain_size[0], num=params1["Nr"])
    rec_coord[:, 1] = rec_depth

    # Create the geometry objects for background velocity models
    src_dummy = np.empty((1, 2))

    src_dummy[0, :] = src_coord[int(src_coord.shape[0] / 2), :]
    geometry = AcquisitionGeometry(v1, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
    params1["Nt"] = geometry.nt
    del src_dummy

    # Define a solver object
    solver = AcousticWaveSolver(v1, geometry, space_order=params1["so"])

    ##################################################################################################
    # This part of the code generates the forward data using the two models and computes the residual
    ##################################################################################################

    dt = v1.critical_dt

    # Allocate numpy arrays to store data
    data = np.zeros(shape=(params1["Ns"], params1["Nt"], params1["Nr"]), dtype=np.float32)
    data_prime = data * 0

    # Call wave_propagator_forward with appropriate arguments
    t_start = time.time()
    DevitoOperators.wave_propagator_forward(
        data=data,
        src_coords=src_coord,
        vel=v1,
        geometry=geometry,
        solver=solver,
        params=params1
    )
    t_end = time.time()
    print("\n Time to model shots for v1 took ", t_end - t_start, " sec.")

    t_start = time.time()
    DevitoOperators.wave_propagator_forward(
        data=data_prime,
        src_coords=src_coord,
        vel=v1_prime,
        geometry=geometry,
        solver=solver,
        params=params1
    )
    t_end = time.time()
    print("\n Time to model shots for v1_prime took ", t_end - t_start, " sec.")

    # Calculate residuals
    res = data - data_prime

    # Create rhs
    dm_adjoint_image = np.zeros((params1["Nt"], params1["Nx"], params1["Nz"]), dtype=np.float32)
    t_start = time.time()
    DevitoOperators.td_born_adjoint(
        born_data=res,
        model_pert=dm_adjoint_image,
        src_coords=src_coord,
        vel=v1,
        geometry=geometry,
        solver=solver,
        params=params1,
        dt=dt
    )
    t_end = time.time()
    print("\nCreate adjoint image took ", t_end - t_start, " sec")

    # ---------------------------------------------------------------------------------
    # Load inverted model
    dm_invert_multi_shot = np.load(datadir + filestr + ".npz")["arr_0"]

    # Plot stack, depth slices, and CIGs through inverted stack
    dm_scale = 1e-3
    cig_aspect = 2
    shotnum_list = [1, 3, 5, 7, 9]

    def plot_stack_slices_cigs():

        # Locations for CIGs
        locs = [0.3, 0.4, 0.5, 0.6, 0.7]

        # Stack plot
        draw_line_coords = []
        for item in locs:
            draw_line_coords.append(
                [
                    [1e-3 * v1.domain_size[0] * item, 1e-3 * v1.domain_size[0] * item],
                    [1e-3 * v1.origin[1], 1e-3 * (v1.origin[1] + v1.domain_size[1])]
                ]
            )

        locs_tx = [0.25, 0.5, 0.6, 0.75]
        draw_line_coords_grp1 = []
        for item in locs_tx:
            draw_line_coords_grp1.append(
                [
                    [1e-3 * v1.origin[0], 1e-3 * (v1.origin[0] + v1.domain_size[0])],
                    [1e-3 * v1.domain_size[1] * item, 1e-3 * v1.domain_size[1] * item]
                ]
            )

        marker_coords = []
        for item in shotnum_list:
            marker_coords.append(
                [
                    [1e-3 * src_coord[item, 0]],
                    [1e-3 * src_coord[item, 1]]
                ]
            )

        plot_image_xy(
            np.sum(dm_invert_multi_shot, axis=0).T,
            x0=v1.origin[0], xn=v1.origin[0] + v1.domain_size[0],
            y0=v1.origin[1], yn=v1.origin[1] + v1.domain_size[1],
            scale=None, sfac=0.5, clip=1.0, colorbar=False,
            ylabel="Z [km]", xlabel="X [km]",
            grid="on", aspect="equal",
            draw_line_coords=draw_line_coords, linewidth=1.0, linestyle="-", linecolor="red",
            draw_line_coords_grp1=draw_line_coords_grp1, linewidth_grp1=0.75, linestyle_grp1="--", linecolor_grp1="g",
            marker_coords=marker_coords, markersize=6, markerstyle="x", markercolor="b",
            fontname="STIXGeneral", fontsize=12,
            savefig_fname=figdir + filestr + "_invert_stack.pdf"
        )

        # Plot imaged data (t-x sections at depth values 20%, 40%, 60%, 80%)
        image_nrows = 2
        image_ncols = 2
        image_arr = np.zeros(
            shape=(image_nrows, image_ncols, params1["Nt"], params1["Nx"]),
            dtype=np.float32
        )
        image_arr[0, 0, :, :] = dm_invert_multi_shot[:, :, int(params1["Nz"] * 0.25)]
        image_arr[0, 1, :, :] = dm_invert_multi_shot[:, :, int(params1["Nz"] * 0.5)]
        image_arr[1, 0, :, :] = dm_invert_multi_shot[:, :, int(params1["Nz"] * 0.6)]
        image_arr[1, 1, :, :] = dm_invert_multi_shot[:, :, int(params1["Nz"] * 0.75)]

        image_titles = [["Z = 0.5 km", "Z = 1.0 km"], ["Z = 1.2 km", "Z = 1.5 km"]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5, figsize=(20, 10),
            x0=v1.origin[0], xn=v1.origin[0] + v1.domain_size[0], y0=t0, yn=tn,
            scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            xlabel="X [km]", ylabel="Time [s]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname=figdir + filestr + "_invert_tx_images.pdf"
        )

        # Plot CIGs
        image_nrows = 1
        image_ncols = len(locs)
        image_arr = np.zeros(
            shape=(image_nrows, image_ncols, params1["Nz"], int(params1["Nt"] / 2)),
            dtype=np.float32
        )

        for ii, item in enumerate(locs):
            image_arr[0, ii, :, :] = dm_invert_multi_shot[
                                    0:int(dm_invert_multi_shot.shape[0] / 2), int(params1["Nx"] * item), :
                                    ].T

        image_titles = [["X = " + "{:4.1f}".format(item * v1.domain_size[0] * 1e-3) + " km" for item in locs]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5,
            x0=t0, xn=tn / 2, y0=v1.origin[1], yn=v1.origin[1] + v1.domain_size[1],
            scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect=cig_aspect, cmap="Greys", colorbar=True, clip=1.0,
            xlabel="Time [s]", ylabel="Z [km]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname=figdir + filestr + "_invert_cigs.pdf"
        )

    plot_stack_slices_cigs()

    # ---------------------------------------------------------------------------------
    # Shot record comparisons

    td_born_data_adjoint_image_multi_shot = np.zeros((params1["Ns"], params1["Nt"], params1["Nr"]), dtype=np.float32)
    td_born_data_inverted_model_multi_shot = np.zeros((params1["Ns"], params1["Nt"], params1["Nr"]), dtype=np.float32)

    # Data modeled using inverted model
    DevitoOperators.td_born_forward(
        model_pert=dm_invert_multi_shot,
        born_data=td_born_data_inverted_model_multi_shot,
        src_coords=src_coord,
        vel=v1,
        geometry=geometry,
        solver=solver,
        params=params1,
        dt=dt
    )

    # Data modeled using adjoint + normalization
    DevitoOperators.td_born_forward(
        model_pert=dm_adjoint_image,
        born_data=td_born_data_adjoint_image_multi_shot,
        src_coords=src_coord,
        vel=v1,
        geometry=geometry,
        solver=solver,
        params=params1,
        dt=dt
    )
    td_born_data_adjoint_image_multi_shot *= \
        np.linalg.norm(res) / np.linalg.norm(td_born_data_adjoint_image_multi_shot)

    shot_scale = 0.02

    def plot_shot_comparison():

        # Plot imaged data (t-x sections at depth values 20%, 40%, 60%, 80%)
        image_nrows = len(shotnum_list)
        image_ncols = 3
        image_arr = np.zeros(
            shape=(image_nrows, image_ncols, params1["Nt"], params1["Nr"]),
            dtype=np.float32
        )

        image_titles = []

        for ii, item in enumerate(shotnum_list):
            image_arr[ii, 0, :, :] = res[item, :, :]
            image_arr[ii, 1, :, :] = td_born_data_inverted_model_multi_shot[item, :, :]
            image_arr[ii, 2, :, :] = td_born_data_adjoint_image_multi_shot[item, :, :]

            image_titles.append(["X = " + "{:4.2f}".format(1e-3 * src_coord[item, 0]) + " km", "", ""])

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5, figsize=(20, 20),
            x0=v1.origin[0], xn=v1.origin[0] + v1.domain_size[0], y0=t0, yn=tn,
            scale=shot_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            ylabel="Time [s]", xlabel="X [km]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname=figdir + filestr + "_shots.pdf"
        )

    plot_shot_comparison()

    # ---------------------------------------------------------------------------------
    # Plot inversion residual

    def plot_residual():

        residual = np.load(datadir + filestr + ".npz")["arr_2"]
        f = np.linalg.norm(res) ** 2.0
        residual = 1 + (residual / f)
        residual_max = np.max(residual)
        niter = residual.shape[0]

        plt.rc('text', usetex=True)
        fig = plt.figure(figsize=(30, 10))
        plt.plot([ii for ii in range(niter)], residual, "-k", linewidth=1)
        plt.grid("on")
        plt.xlim(0, niter)
        plt.ylim(0, np.max(residual))

        nxticks = 5
        nyticks = 5
        xticks = np.arange(0, niter, niter / nxticks)
        xticklabels = ["{:4.0f}".format(item) for item in xticks]
        yticks = np.arange(0, residual_max, residual_max / nyticks)
        yticklabels = ["{:4.2f}".format(item) for item in yticks]

        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontname="STIXGeneral", fontsize=40)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontname="STIXGeneral", fontsize=40)

        plt.xlabel("Iterations", fontname="STIXGeneral", fontsize=40)
        plt.ylabel(r"$||v - \bar{v}||_2 \;/\; ||v||_2$", fontname="STIXGeneral", fontsize=40)

        fig.savefig(figdir + filestr + "_obj.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01)

    plot_residual()
