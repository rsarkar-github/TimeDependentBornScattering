import numpy as np
from scipy import ndimage as spim

from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource


def td_born_forward_op(model, geometry, time_order, space_order):

    nt = geometry.nt

    # Define the wavefields with the size of the model and the time dimension
    u0 = TimeFunction(
        name='u0',
        grid=model.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )
    u = TimeFunction(
        name='u',
        grid=model.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )
    dm = TimeFunction(
        name='dm',
        grid=model.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )

    # Define the wave equation
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt - dm * u0

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(u.forward, solve(pde, u.forward), subdomain=model.grid.subdomains['physdomain'])

    # Sample at receivers
    born_data_rec = PointSource(
        name='born_data_rec',
        grid=model.grid,
        time_range=geometry.time_axis,
        coordinates=geometry.rec_positions
    )
    rec_term = born_data_rec.interpolate(expr=u)

    return Operator([stencil] + rec_term, subs=model.spacing_map)


def td_born_forward(model_pert, born_data, src_coords, vel, geometry, solver, params, dt=None):
    """
    @Params
    model_pert: float32 numpy array of size (Nt, Nx, Nz)
    born_data: float32 numpy array of size (Ns, Nt, Nr). This will be the output. Assumed to be zeros.
    src_coords: float32 numpy array of size (Ns, 2) with the source coordinates.
    vel: background velocity model object (The object should be the same type as returned by create_model() function)
    geometry: geometry object
    solver: acoustic solver object
    params: python dict of parameters

    Note: The receiver coordinates are not needed since it is assumed that they remain fixed for all sources.
    So the receiver information should already be available to the solver object.
    """

    # Get params
    nbl = params["nbl"]
    nx = params["Nx"]
    nz = params["Nz"]
    ns = params["Ns"]
    nt = params["Nt"]
    space_order = params["so"]
    time_order = params["to"]

    offset = nbl + space_order
    if dt is None:
        dt = vel.critical_dt

    # Create padded model perturbation with halo
    model_pert_padded = np.zeros((nt, nx + 2 * offset, nz + 2 * offset), dtype=np.float32)

    # Copy model_pert into model_pert_padded appropriately
    model_pert_padded[:, offset:nx + offset, offset:nz + offset] = model_pert

    # Create time dependent Born modeling operator
    op = td_born_forward_op(model=vel, geometry=geometry, time_order=time_order, space_order=space_order)

    # Second derivative filter stencil
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)

    u = TimeFunction(
        name='u',
        grid=vel.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )

    # Perform Born modeling
    for i in range(ns):
        # Update source location
        geometry.src_positions[0, :] = src_coords[i, :]

        # Get Born modeled data for the current shot and update born_data array appropriately
        _, u0, _ = solver.forward(vp=vel.vp, save=True)
        spim.convolve1d(
            input=u0.data_with_halo,
            weights=laplacian_filter,
            axis=0,
            output=u0.data_with_halo[:, :, :],
            mode='nearest'
        )
        op.apply(u0=u0, u=u, dm=model_pert_padded, born_data_rec=born_data[i, :, :], dt=dt)


def td_born_adjoint_op(model, geometry, time_order, space_order, dt):

    nt = geometry.nt

    # Define the wavefields with the size of the model and the time dimension
    u = TimeFunction(
        name='u',
        grid=model.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )

    # Define the wave equation
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt.T

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(u.backward, solve(pde, u.backward), subdomain=model.grid.subdomains['physdomain'])

    # Inject at receivers
    born_data_rec = PointSource(
        name='born_data_rec',
        grid=model.grid,
        time_range=geometry.time_axis,
        coordinates=geometry.rec_positions
    )
    rec_term = born_data_rec.inject(field=u.backward, expr=born_data_rec * (dt ** 2) / model.m)

    return Operator([stencil] + rec_term, subs=model.spacing_map)


def td_born_adjoint(born_data, model_pert, src_coords, vel, geometry, solver, params, dt=None):
    """
    @Params
    born_data: float32 numpy array of size (Ns, Nt, Nr).
    model_pert: float32 numpy array of size (Nt, Nx, Nz). This will be the output. Assumed to be zeros.
    src_coords: float32 numpy array of size (Ns, 2) with the source coordinates.
    vel: background velocity model object (The object should be the same type as returned by create_model() function)
    geometry: geometry object
    solver: acoustic solver object
    params: python dict of parameters

    Note: The receiver coordinates are not needed since it is assumed that they remain fixed for all sources.
    So the receiver information should already be available to the solver object.
    """

    # Get params
    nbl = params["nbl"]
    nx = params["Nx"]
    nz = params["Nz"]
    ns = params["Ns"]
    nt = params["Nt"]
    space_order = params["so"]
    time_order = params["to"]

    offset = nbl + space_order
    if dt is None:
        dt = vel.critical_dt

    # Allocate time function to store adjoint wavefield
    u = TimeFunction(
        name='u',
        grid=vel.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )

    # Create time dependent Born modeling operator
    op = td_born_adjoint_op(model=vel, geometry=geometry, time_order=time_order, space_order=space_order, dt=dt)

    # Second derivative filter stencil
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)

    # Perform adjoint Born modeling
    for i in range(ns):
        # Update source location
        geometry.src_positions[0, :] = src_coords[i, :]

        # Get Born modeled data for the current shot and update born_data array appropriately
        _, u0, _ = solver.forward(vp=vel.vp, save=True)
        spim.convolve1d(
            input=u0.data_with_halo,
            weights=laplacian_filter,
            axis=0,
            output=u0.data_with_halo[:, :, :],
            mode='nearest'
        )
        op.apply(u=u, born_data_rec=born_data[i, :, :], dt=dt)

        # Add to model_pert
        model_pert += \
            u.data_with_halo[:, offset:nx + offset, offset:nz + offset] * \
            u0.data_with_halo[:, offset:nx + offset, offset:nz + offset]


def td_born_hessian(model_pert_in, model_pert_out, src_coords, vel, geometry, solver, params, dt=None):
    """
    @Params
    model_pert_in: float32 numpy array of size (Nx, Nz).
    model_pert_out: float32 numpy array of size (Nx, Nz). This will be the output. Assumed to be zeros.
    src_coords: float32 numpy array of size (Ns, 2) with the source coordinates.
    vel: background velocity model object (The object should be the same type as returned by create_model() function)
    geometry: geometry object
    solver: solver object
    params: python dict of parameters

    Note: The receiver coordinates are not needed since it is assumed that they remain fixed for all sources.
    So the receiver information should already be available to the solver object.
    """

    # Get params
    nbl = params["nbl"]
    nx = params["Nx"]
    nz = params["Nz"]
    ns = params["Ns"]
    nr = params["Nr"]
    nt = params["Nt"]
    space_order = params["so"]
    time_order = params["to"]

    offset = nbl + space_order
    if dt is None:
        dt = vel.critical_dt

    # Create padded model perturbation with halo and copy model_pert_in into model_pert_padded
    model_pert_padded = np.zeros((nt, nx + 2 * offset, nz + 2 * offset), dtype=np.float32)
    model_pert_padded[:, offset:nx + offset, offset:nz + offset] = model_pert_in

    # 1. Allocate space for Born modeled data for 1 source
    # 2. Allocate time function to store intermediate wavefields
    born_data = np.zeros((1, nt, nr), dtype=np.float32)
    u1 = TimeFunction(
        name='u',
        grid=vel.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )
    u2 = TimeFunction(
        name='u',
        grid=vel.grid,
        time_order=time_order,
        space_order=space_order,
        save=nt
    )

    # Second derivative filter stencil
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)

    # Initialize forward and adjoint operators
    op_fwd = td_born_forward_op(
        model=vel,
        geometry=geometry,
        time_order=time_order,
        space_order=space_order
    )
    op_adjoint = td_born_adjoint_op(
        model=vel,
        geometry=geometry,
        time_order=time_order,
        space_order=space_order,
        dt=dt
    )

    for i in range(ns):
        # Update source location
        geometry.src_positions[0, :] = src_coords[i, :]

        # Get Born modeled data for the current shot
        _, u0, _ = solver.forward(vp=vel.vp, save=True)
        spim.convolve1d(
            input=u0.data_with_halo,
            weights=laplacian_filter,
            axis=0,
            output=u0.data_with_halo[:, :, :],
            mode='nearest'
        )
        op_fwd.apply(u0=u0, u=u1, dm=model_pert_padded, born_data_rec=born_data[0, :, :], dt=dt)

        # Create adjoint image
        op_adjoint.apply(u=u2, born_data_rec=born_data[0, :, :], dt=dt)

        # Add to model_pert_out
        model_pert_out += \
            u2.data_with_halo[:, offset:nx + offset, offset:nz + offset] * \
            u0.data_with_halo[:, offset:nx + offset, offset:nz + offset]
