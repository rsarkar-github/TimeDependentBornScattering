import numpy as np
import scipy as sp
import multiprocessing as mp
from ..Utilities.DevitoUtils import create_model, plot_image_xy


def marmousi_cig_plot(scale_fac, figdir, datadir, nx, nz, cig_aspect, thread_num):

    print("Task starting on thread ", thread_num)

    filestr = "p02_marmousi_scalefac_" + "{:4.2f}".format(scale_fac)

    # Create params dicts
    params = {
        "Nx": nx,
        "Nz": nz,
        "Nt": 100,  # this has to be updated later
        "nbl": 75,
        "Ns": nx,
        "Nr": nz,
        "so": 4,
        "to": 2
    }

    vel = create_model(shape=(params["Nx"], params["Nz"]))
    vel.vp.data[:, :] = 2.0

    # Simulation time
    t0 = 0.
    tn = 4000.  # Simulation last 2 second (2000 ms)

    # Image the data
    dm_image = np.load(datadir + filestr + "_cig.npz")["arr_0"]
    dm_scale = 5.0

    # Form stacked image
    if scale_fac == 1.0:
        dm_image_stack = np.sum(dm_image, axis=0)
        dm_image_stack = sp.ndimage.laplace(dm_image_stack, mode="nearest")

        plot_image_xy(
            dm_image_stack.T,
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0],
            y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
            scale=None, sfac=0.3, clip=1.0, colorbar=False,
            ylabel="Z [km]", xlabel="X [km]",
            grid="on", aspect="equal",
            fontname="STIXGeneral", fontsize=12,
            savefig_fname=figdir + filestr + "_stacked_image.pdf"
        )

    # Plot CIG at middle of horizontal grid
    locs = [0.3, 0.4, 0.5, 0.6, 0.7]

    for item in locs:

        cig = dm_image[0:int(dm_image.shape[0] / 2), int(params["Nx"] * item), :].T

        # Apply WB mute (sample 22)
        cig[0:18, :] = 0

        plot_image_xy(
            cig,
            x0=t0, xn=tn/2,
            y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
            scale=dm_scale, clip=1.0, colorbar=False,
            ylabel="Z [km]", xlabel="Time [s]", xticklabels_fmt="{:4.2f}",
            grid="on", aspect=cig_aspect,
            fontname="STIXGeneral", fontsize=12,
            savefig_fname=figdir + filestr + "_loc_" + "{:4.2f}".format(item) + "_cig.pdf"
        )
        plot_image_xy(
            sp.ndimage.laplace(cig, mode="nearest"),
            x0=t0, xn=tn/2,
            y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
            scale=dm_scale / 5, clip=1.0, colorbar=False,
            ylabel="Z [km]", xlabel="Time [s]", xticklabels_fmt="{:4.2f}",
            grid="on", aspect=cig_aspect,
            fontname="STIXGeneral", fontsize=12,
            savefig_fname=figdir + filestr + "_loc_" + "{:4.2f}".format(item) + "_cig_filt.pdf"
        )
    print("Task finished on thread ", thread_num)


if __name__ == "__main__":

    basepath = "TimeDependentBornScattering/"
    figdir = basepath + "Fig/"
    datadir = basepath + "Data/"

    # Load Marmousi model
    nx = 500
    nz = 174
    cig_aspect = 2.0

    nthreads = 3
    arglist = [
        (1.0, figdir, datadir, nx, nz, cig_aspect, 0),
        (0.97, figdir, datadir, nx, nz, cig_aspect, 1),
        (1.03, figdir, datadir, nx, nz, cig_aspect, 2)
    ]

    pool = mp.Pool(min(nthreads, mp.cpu_count()))
    pool.starmap(marmousi_cig_plot, arglist)

    # -----------------------------------------------------------
    # Plot Marmousi and perturbation
    params = {
        "Nx": nx,
        "Nz": nz,
        "Nt": 100,  # this has to be updated later
        "nbl": 75,
        "Ns": nx,
        "Nr": nz,
        "so": 4,
        "to": 2
    }

    with np.load(datadir + "marmousi-vp.npz") as data:
        vp = data["arr_0"] / 1000.0
    vel = create_model(shape=(params["Nx"], params["Nz"]))
    vel.vp.data[:, :] = 2.0

    smooth_filt = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0
    vp_smooth = sp.signal.convolve2d(in1=vp, in2=smooth_filt, mode="same", boundary="symm")
    dm = vp - vp_smooth

    plot_image_xy(
        vp_smooth.T,
        x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0],
        y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
        scale=None, vmin=1.5, vmax=4.5, clip=1.0, colorbar=True, cmap="jet",
        ylabel="Z [km]", xlabel="X [km]",
        grid="on", aspect="equal",
        fontname="STIXGeneral", fontsize=12,
        savefig_fname=figdir + "p02_marmousi_sound_speed.pdf"
    )

    plot_image_xy(
        dm.T,
        x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0],
        y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
        scale=0.2, clip=1.0, colorbar=True, cmap="Greys",
        ylabel="Z [km]", xlabel="X [km]",
        grid="on", aspect="equal",
        fontname="STIXGeneral", fontsize=12,
        savefig_fname=figdir + "p02_marmousi_model_pert.pdf"
    )
