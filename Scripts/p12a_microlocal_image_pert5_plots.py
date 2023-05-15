import numpy as np
from ..Utilities.DevitoUtils import create_model, plot_images_grid_xy
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


if __name__ == "__main__":

    basepath = "TimeDependentBornScattering/"
    datadir = basepath + "Data/"
    figdir = basepath + "Fig/"
    filestr = "p12_microlocal_image_pert5"

    # Create params dicts
    params = {
        "Nx": 500,
        "Nz": 100,
        "Nt": 100,  # this has to be updated later
        "nbl": 75,
        "Ns": 500,
        "Nr": 500,
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

    # Load dm_image
    dm_image = np.load(datadir + filestr + ".npz")["arr_0"]
    cig_aspect = 4
    dm_scale = 50.0

    def plot_slices_cigs():

        # Locations for CIGs
        locs = [0.46, 0.48, 0.5, 0.52, 0.54]

        # Plot imaged data (t-x sections at depth values 50%)
        image_nrows = 2
        image_ncols = 2
        image_arr = np.zeros(
            shape=(image_nrows, image_ncols, params["Nt"], params["Nx"]),
            dtype=np.float32
        )
        image_arr[0, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.3)]
        image_arr[0, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.4)]
        image_arr[1, 0, :, :] = dm_image[:, :, int(params["Nz"] * 0.5)]
        image_arr[1, 1, :, :] = dm_image[:, :, int(params["Nz"] * 0.6)]

        image_titles = [["Z = 0.3 km", "Z = 0.4 km"], ["Z = 0.5 km", "Z = 0.6 km"]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, axes_pad=0.5, figsize=(20, 10),
            x0=vel.origin[0], xn=vel.origin[0] + vel.domain_size[0], y0=t0, yn=tn,
            scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect="auto", cmap="Greys", colorbar=True, clip=1.0,
            xlabel="X [km]", ylabel="Time [s]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname=figdir + filestr + "_tx_images.pdf"
        )

        # Plot CIGs
        image_nrows = 1
        image_ncols = len(locs)
        image_arr = np.zeros(
            shape=(image_nrows, image_ncols, params["Nz"], int(params["Nt"] * 0.66)),
            dtype=np.float32
        )

        for i, item in enumerate(locs):
            image_arr[0, i, :, :] = dm_image[
                                    0:int(dm_image.shape[0] * 0.66), int(params["Nx"] * item), :
                                    ].T

        image_titles = [["X = " + "{:4.1f}".format(item * vel.domain_size[0] * 1e-3) + " km" for item in locs]]

        plot_images_grid_xy(
            image_grid=image_arr, image_titles=image_titles, figsize=(20, 10), axes_pad=0.5,
            x0=t0, xn=tn * 0.66, y0=vel.origin[1], yn=vel.origin[1] + vel.domain_size[1],
            scale=dm_scale, vmin=None, vmax=None,
            grid="on", aspect=cig_aspect, cmap="Greys", colorbar=True, clip=1.0,
            xlabel="Time [s]", ylabel="Z [km]",
            fontname="STIXGeneral", fontsize=20,
            nxticks=5, nyticks=5,
            savefig_fname=figdir + filestr + "_cigs.pdf"
        )

    plot_slices_cigs()
