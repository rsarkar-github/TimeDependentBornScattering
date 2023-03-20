import numpy as np
import time
import os
from ..Operators import DevitoOperators
from ..Utilities.DevitoUtils import create_model, conjugate_gradient
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry
from devito import configuration
configuration['log-level'] = 'WARNING'


if __name__ == "__main__":

    filestr = "sigsbee_lo_multi_shot"

    ######################################################
    # This part of the code loads the Sigsbee model
    ######################################################

    # Create params dicts
    params = {
        "Nx": 600,
        "Nz": 200,
        "Nt": 100,   # this has to be updated later
        "nbl": 75,
        "Ns": 5,
        "Nr": 200,
        "so": 4,
        "to": 2
    }

    # Load full Sigsbee model
    vel_sigsbee = np.load("TimeDependentBornScattering/Data/sigsbee.npz")["arr_0"]
    Nx = vel_sigsbee.shape[0]
    Nz = vel_sigsbee.shape[1]

    # Create model
    sigsbee = create_model(shape=(Nx - 2 * params["nbl"], Nz - 2 * params["nbl"]))
    sigsbee.vp.data[:, :] = vel_sigsbee * 0.3048 * 0.001

    del Nx, Nz, vel_sigsbee

    ######################################################
    # This part of the code creates the models
    ######################################################

    # Specify start index for cropping
    start = (200, 50)

    # Create cropped model (true)
    v = create_model(shape=(params["Nx"], params["Nz"]))
    v.vp.data[:, :] = sigsbee.vp.data[start[0]: start[0] + params["Nx"] + 2 * params["nbl"],
                                      start[1]: start[1] + params["Nz"] + 2 * params["nbl"]]

    # Create background model
    v1 = create_model(shape=(params["Nx"], params["Nz"]))
    v1.vp.data[:, :] = 1.5

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

    src_coord = np.empty((params["Ns"], 2))
    if params["Ns"] == 1:
        src_coord[:, 0] = 0.5 * v.domain_size[0]
        src_coord[:, 1] = src_depth
    else:
        src_coord[:, 0] = np.linspace(0, v.domain_size[0], num=params["Ns"])
        src_coord[:, 1] = src_depth

    rec_coord = np.empty((params["Nr"], 2))
    rec_coord[:, 0] = np.linspace(0, v.domain_size[0], num=params["Nr"])
    rec_coord[:, 1] = rec_depth

    # Create the geometry objects for background velocity models
    src_dummy = np.empty((1, 2))

    src_dummy[0, :] = src_coord[int(src_coord.shape[0] / 2), :]
    geometry = AcquisitionGeometry(v, rec_coord, src_dummy, t0, tn, f0=f0, src_type='Ricker')
    params["Nt"] = geometry.nt
    del src_dummy

    # Define a solver object
    solver = AcousticWaveSolver(v, geometry, space_order=params["so"])

    ##################################################################################################
    # This part of the code generates the forward data using the two models and computes the residual
    ##################################################################################################

    dt = v.critical_dt

    # Allocate numpy arrays to store data
    data = np.zeros(shape=(params["Ns"], params["Nt"], params["Nr"]), dtype=np.float32)
    data1 = data * 0

    # Call wave_propagator_forward with appropriate arguments
    t_start = time.time()
    DevitoOperators.wave_propagator_forward(
        data=data,
        src_coords=src_coord,
        vel=v,
        geometry=geometry,
        solver=solver,
        params=params
    )
    t_end = time.time()
    print("\n Time to model shots for v took ", t_end - t_start, " sec.")

    t_start = time.time()
    DevitoOperators.wave_propagator_forward(
        data=data1,
        src_coords=src_coord,
        vel=v1,
        geometry=geometry,
        solver=solver,
        params=params
    )
    t_end = time.time()
    print("\n Time to model shots for v1 took ", t_end - t_start, " sec.")

    # Calculate residuals
    res = data - data1

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
            params=params,
            dt=dt
        )

    # Create rhs for inversion
    dm_adjoint_image = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)
    t_start = time.time()
    DevitoOperators.td_born_adjoint(
        born_data=res,
        model_pert=dm_adjoint_image,
        src_coords=src_coord,
        vel=v1,
        geometry=geometry,
        solver=solver,
        params=params,
        dt=dt
    )
    t_end = time.time()
    print("\nCreate adjoint image took ", t_end - t_start, " sec")

    # Run the inversion
    niter = 100

    if os.path.exists("Data/" + filestr + ".npz"):
        x0 = np.load("Data/" + filestr + ".npz")["arr_0"]
    else:
        x0 = np.zeros((params["Nt"], params["Nx"], params["Nz"]), dtype=np.float32)

    dm_invert, resid = conjugate_gradient(
        hessian_wrap,
        rhs=dm_adjoint_image,
        x0=x0,
        niter=niter,
        printobj=False
    )

    # Save results
    np.savez("TimeDependentBornScattering/Data/" + filestr + ".npz", dm_invert, resid)
