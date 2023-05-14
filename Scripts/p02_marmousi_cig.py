import numpy as np
import scipy as sp
import time
import multiprocessing as mp
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model
from ..Utilities.Utils import extrapolate_same
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


def marmousi_cig(scale_fac, figdir, datadir, nx, nz, vp, cig_aspect, thread_num):

    t1 = time.time()
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

    # Create velocity
    vp_pad = np.zeros(shape=(2 * params["nbl"] + params["Nx"], 2 * params["nbl"] + params["Nz"]), dtype=np.float32)
    vp_pad[params["nbl"]: params["nbl"] + params["Nx"], params["nbl"]: params["nbl"] + params["Nz"]] = vp / 1000.0
    extrapolate_same(array2d=vp_pad, ncells_pad_x=params["nbl"], ncells_pad_z=params["nbl"], create_new=False)

    smooth_filt = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0
    vp_pad_smooth = sp.signal.convolve2d(in1=vp_pad, in2=smooth_filt, mode="same", boundary="symm")

    vel = create_model(shape=(params["Nx"], params["Nz"]))
    vel.vp.data[:, :] = vp_pad_smooth

    # Simulation time, wavelet
    t0 = 0.
    tn = 4000.  # Simulation last 2 second (2000 ms)
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

    # Create perturbation
    dm = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
    for i in range(params["Nt"]):
        dm[i, :, :] = vp_pad[
                      params["nbl"]: params["nbl"] + params["Nx"],
                      params["nbl"]: params["nbl"] + params["Nz"]
                      ] - vp_pad_smooth[
                          params["nbl"]: params["nbl"] + params["Nx"],
                          params["nbl"]: params["nbl"] + params["Nz"]
                          ]

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

    # Create scaled models
    vel1 = create_model(shape=(params["Nx"], params["Nz"]))
    vel1.vp.data[:, :] = vp_pad_smooth * scale_fac

    # Image the data
    dm_image = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
    dm_scale = 50.0
    DevitoOperators.td_born_adjoint(
        born_data=td_born_data_true,
        model_pert=dm_image,
        src_coords=src_coord,
        vel=vel1,
        geometry=geometry,
        solver=solver,
        params=params
    )
    np.savez(datadir + filestr + "_cig.npz", dm_image)

    t2 = time.time()
    print("Task finished on thread ", thread_num, ". Total time elapsed = ", "{:4.2f}".format(t2 - t1), " s.")


if __name__ == "__main__":

    _basepath = "TimeDependentBornScattering/"
    _figdir = _basepath + "Fig/"
    _datadir = _basepath + "Data/"

    # Load Marmousi model
    _nx = 500
    _nz = 174
    with np.load(_datadir + "marmousi-vp.npz") as data:
        _vp = data["arr_0"]

    _cig_aspect = 1.0

    nthreads = 3
    arglist = [
        (1.0, _figdir, _datadir, _nx, _nz, _vp, _cig_aspect, 0),
        (0.97, _figdir, _datadir, _nx, _nz, _vp, _cig_aspect, 1),
        (1.03, _figdir, _datadir, _nx, _nz, _vp, _cig_aspect, 2)
    ]

    pool = mp.Pool(min(nthreads, mp.cpu_count()))
    pool.starmap(marmousi_cig, arglist)
