import numpy as np
from scipy import ndimage
import time
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


if __name__ == "__main__":

    basepath = "TimeDependentBornScattering/"
    datadir = basepath + "Data/"
    filestr = "p11_microlocal_image_pert4"

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

    # Define a solver object
    solver = AcousticWaveSolver(vel, geometry, space_order=params["so"])
    print("dt = ", vel.critical_dt, " s\n")

    # Allocate space for dm, dm_image
    dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
    dm_image = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)

    # Create dm
    # Note: params["Nt"] = 655
    # Note: vel.critical_dt = 3.062 ms
    dt = vel.critical_dt
    for i in range(params["Nt"]):
        # start and end x indices
        x_start = 2.49 + 0.5 * i * dt / 1000
        x_end = 2.51 + 0.5 * i * dt / 1000
        x_start_index = int(x_start * 1000 / vel.spacing[0])
        x_end_index = int(x_end * 1000 / vel.spacing[0])

        dm[i, x_start_index:x_end_index, int(params["Nz"] / 2)] = 1.0

    temp = dm[:, :, int(params["Nz"] / 2)]
    ndimage.gaussian_filter(
        input=temp,
        sigma=3.0,
        mode='nearest',
        output=temp
    )
    dm[:, :, int(params["Nz"] / 2)] = temp

    # Time dependent Born propagator Hessian
    dm_image *= 0
    t_start = time.time()
    DevitoOperators.td_born_hessian(
        model_pert_in=dm,
        model_pert_out=dm_image,
        src_coords=src_coord,
        vel=vel,
        geometry=geometry,
        solver=solver,
        params=params
    )
    t_end = time.time()
    print("\nHessian application took ", t_end - t_start, " sec")
    np.savez(datadir + filestr + ".npz", dm_image)
