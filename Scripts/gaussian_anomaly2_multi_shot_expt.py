import numpy as np
from scipy import ndimage
import time
import os
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model, conjugate_gradient
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


if __name__ == "__main__":

    filestr = "gaussian_anomaly2_multi_shot"

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

    ##################################################################################################
    # This part of the code performs the inversion
    ##################################################################################################

    # Create wrapper for time dependent Born Hessian
    def hessian_wrap(model_pert_in, model_pert_out):
        """
        @Params
        model_pert_in: input numpy array
        model_pert_out: output numpy array
        """
        model_pert_out *= 0.

        DevitoOperators.td_born_hessian(
            model_pert_in=model_pert_in,
            model_pert_out=model_pert_out,
            src_coords=src_coord,
            vel=v1,
            geometry=geometry,
            solver=solver,
            params=params1,
            dt=dt
        )

    # Create rhs for inversion
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

    # Run the inversion
    niter = 100

    if os.path.exists("Data/" + filestr + ".npz"):
        x0 = np.load("Data/" + filestr + ".npz")["arr_0"]
    else:
        x0 = np.zeros((params1["Nt"], params1["Nx"], params1["Nz"]), dtype=np.float32)

    dm_invert, resid = conjugate_gradient(
        hessian_wrap,
        rhs=dm_adjoint_image,
        x0=x0,
        niter=niter,
        printobj=False
    )

    # Save results
    np.savez("TimeDependentBornScattering/Data/" + filestr + ".npz", dm_invert, resid)
