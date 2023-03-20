import numpy as np
import numba
import scipy.ndimage as spim
import matplotlib.pyplot as plt
from .Utils import ricker_time, cosine_taper_2d, boxcar_taper_2d, laplacian


def acoustic_propagator(
        vel2d, dx: float, dz: float, dt: float, fmax: float,
        source_wavefield,
        propagated_wavefield,
        ncells_pad_x: int=0,
        ncells_pad_z: int=0,
        check_params=True
):

    if check_params:
        # Check dimensions
        if source_wavefield.shape != propagated_wavefield.shape:
            raise ValueError
        if source_wavefield[0, :, :].shape != vel2d.shape:
            raise ValueError

    # Get grid information
    grid_points_z, grid_points_x = vel2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if check_params:
        if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
            raise ValueError

    time_steps = source_wavefield.shape[0]

    if check_params:
        # Check CFL conditions
        vmin = np.min(vel2d)
        vmax = np.max(vel2d)
        f = vmax * dt * np.sqrt(2.0) / np.min([dx, dz])
        if f >= 1:
            raise ValueError("CFL conditions violated for 2d propagation")

        # Check numerical dispersion condition
        f = vmin / (fmax * np.max([dx, dz]))
        if f <= 8.0:
            raise ValueError("Numerical dispersion conditions violated")

    # Get taper filter mask
    taper_mask = cosine_taper_2d(
        array2d=source_wavefield[0, :, :],
        ncells_pad_x=ncells_pad_x,
        ncells_pad_z=ncells_pad_z,
        get_mask_only=True
    )

    # Start propagation
    propagated_wavefield *= 0
    f1 = (vel2d * dt) ** 2.0

    if time_steps >= 2:
        u_next = propagated_wavefield[1, :, :]
        u_next += source_wavefield[0, :, :]
        u_next *= f1
        u_next *= taper_mask

    for i in range(2, time_steps):

        u_prev = propagated_wavefield[i - 2, :, :]
        u_curr = propagated_wavefield[i - 1, :, :]
        u_next = propagated_wavefield[i, :, :]

        laplacian(
            array2d_in=u_curr,
            array2d_out=u_next,
            dx=dx,
            dz=dz,
            order=10
        )
        u_next += source_wavefield[i - 1, :, :]
        u_next *= f1
        u_next += 2 * u_curr - u_prev
        u_next *= taper_mask
        u_curr *= taper_mask


def born_time_dependent_pert_propagator(
        vel2d, dx: float, dz: float, dt: float, fmax: float,
        vel_pert2d,
        source_wavefield,
        born_scattered_wavefield,
        ncells_pad_x: int=0,
        ncells_pad_z: int=0,
        check_params=True,
        adjoint_mode=False
):
    if check_params:
        # Check dimensions
        if source_wavefield.shape != born_scattered_wavefield.shape or source_wavefield.shape != vel_pert2d.shape:
            raise ValueError
        if source_wavefield[0, :, :].shape != vel2d.shape:
            raise ValueError

    # Get grid information
    grid_points_z, grid_points_x = vel2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if check_params:
        if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
            raise ValueError

    time_steps = source_wavefield.shape[0]

    if check_params:
        # Check CFL conditions
        vmin = np.min(vel2d)
        vmax = np.max(vel2d)
        f = vmax * dt * np.sqrt(2.0) / np.min([dx, dz])
        if f >= 1:
            raise ValueError("CFL conditions violated for 2d propagation")

        # Check numerical dispersion condition
        f = vmin / (fmax * np.max([dx, dz]))
        if f <= 8.0:
            raise ValueError("Numerical dispersion conditions violated")

    # Get boxcar filter mask
    boxcar_mask = boxcar_taper_2d(
        array2d=source_wavefield[0, :, :],
        ncells_pad_x=ncells_pad_x,
        ncells_pad_z=ncells_pad_z,
        get_mask_only=True
    )

    # Apply boxcar mask to velocity pertubation, source wavefield, and born scattered wavefield
    vel_pert2d *= boxcar_mask
    source_wavefield *= boxcar_mask
    born_scattered_wavefield *= boxcar_mask

    # Compute primary wavefield
    primary_wavefield = np.zeros((time_steps, grid_points_z, grid_points_x), dtype=np.float32)

    acoustic_propagator(
        vel2d=vel2d,
        dx=dx, dz=dz, dt=dt, fmax=fmax,
        source_wavefield=source_wavefield,
        propagated_wavefield=primary_wavefield,
        ncells_pad_z=ncells_pad_z,
        ncells_pad_x=ncells_pad_x,
        check_params=False
    )

    # Compute 2nd derivative
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)
    primary_wavefield = spim.convolve1d(
        input=primary_wavefield,
        weights=laplacian_filter,
        axis=0,
        mode='constant',
        cval=0.0
    )

    # Compute vel cubed inverse
    velcube_inv = vel2d ** (-3.0)

    if not adjoint_mode:
        # Compute secondary source
        primary_wavefield *= np.reshape(2 * velcube_inv, newshape=(1, grid_points_z, grid_points_x)) * vel_pert2d

        # Propagate secondary source
        acoustic_propagator(
            vel2d=vel2d,
            dx=dx, dz=dz, dt=dt, fmax=fmax,
            source_wavefield=primary_wavefield,
            propagated_wavefield=born_scattered_wavefield,
            ncells_pad_z=ncells_pad_z,
            ncells_pad_x=ncells_pad_x,
            check_params=False
        )

    else:
        # Compute diagonal operator
        primary_wavefield *= np.reshape(2 * velcube_inv * boxcar_mask, newshape=(1, grid_points_z, grid_points_x))

        # Propagate born scattered wavefield as source
        acoustic_propagator(
            vel2d=vel2d,
            dx=dx, dz=dz, dt=dt, fmax=fmax,
            source_wavefield=np.flip(born_scattered_wavefield, axis=0),
            propagated_wavefield=np.flip(vel_pert2d, axis=0),
            ncells_pad_z=ncells_pad_z,
            ncells_pad_x=ncells_pad_x,
            check_params=False
        )

        # Multiply with diagonal operator
        vel_pert2d *= primary_wavefield


def born_time_dependent_pert_normal_op(
        vel2d, dx: float, dz: float, dt: float, fmax: float,
        vel_pert2d,
        output,
        source_wavefield,
        restriction_mask,
        ncells_pad_x: int=0,
        ncells_pad_z: int=0,
        check_params=True,
        precomputed_primary_wavefield=False
):
    if check_params:
        # Check dimensions
        if source_wavefield.shape != vel_pert2d.shape or output.shape != vel_pert2d.shape:
            raise ValueError
        if source_wavefield[0, :, :].shape != vel2d.shape or vel2d.shape != restriction_mask.shape:
            raise ValueError

    # Get grid information
    grid_points_z, grid_points_x = vel2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if check_params:
        if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
            raise ValueError

    time_steps = source_wavefield.shape[0]

    if check_params:
        # Check CFL conditions
        vmin = np.min(vel2d)
        vmax = np.max(vel2d)
        f = vmax * dt * np.sqrt(2.0) / np.min([dx, dz])
        if f >= 1:
            raise ValueError("CFL conditions violated for 2d propagation")

        # Check numerical dispersion condition
        f = vmin / (fmax * np.max([dx, dz]))
        if f <= 8.0:
            raise ValueError("Numerical dispersion conditions violated")

    # Get boxcar filter mask
    boxcar_mask = boxcar_taper_2d(
        array2d=source_wavefield[0, :, :],
        ncells_pad_x=ncells_pad_x,
        ncells_pad_z=ncells_pad_z,
        get_mask_only=True
    )

    # Apply boxcar mask to velocity perturbation, source wavefield
    vel_pert2d *= boxcar_mask
    source_wavefield *= boxcar_mask

    # Allocate memory
    primary_wavefield = np.zeros((time_steps, grid_points_z, grid_points_x), dtype=np.float32)
    born_scattered_wavefield = np.zeros((time_steps, grid_points_z, grid_points_x), dtype=np.float32)

    if not precomputed_primary_wavefield:
        # Compute primary wavefield
        acoustic_propagator(
            vel2d=vel2d,
            dx=dx, dz=dz, dt=dt, fmax=fmax,
            source_wavefield=source_wavefield,
            propagated_wavefield=primary_wavefield,
            ncells_pad_z=ncells_pad_z,
            ncells_pad_x=ncells_pad_x,
            check_params=False
        )

    else:
        primary_wavefield += source_wavefield

    # Compute 2nd derivative
    laplacian_filter = np.asarray([1, -2, 1], dtype=np.float32) / (dt ** 2.0)
    primary_wavefield = spim.convolve1d(
        input=primary_wavefield,
        weights=laplacian_filter,
        axis=0,
        mode='constant',
        cval=0.0
    )

    # Compute vel cubed inverse
    velcube_inv = vel2d ** (-3.0)

    # Compute diagonal operator
    primary_wavefield *= np.reshape(2 * velcube_inv * boxcar_mask, newshape=(1, grid_points_z, grid_points_x))

    # Forward
    acoustic_propagator(
        vel2d=vel2d,
        dx=dx, dz=dz, dt=dt, fmax=fmax,
        source_wavefield=primary_wavefield * vel_pert2d,
        propagated_wavefield=born_scattered_wavefield,
        ncells_pad_z=ncells_pad_z,
        ncells_pad_x=ncells_pad_x,
        check_params=False
    )
    born_scattered_wavefield *= boxcar_mask * restriction_mask

    # Adjoint
    acoustic_propagator(
        vel2d=vel2d,
        dx=dx, dz=dz, dt=dt, fmax=fmax,
        source_wavefield=np.flip(born_scattered_wavefield, axis=0),
        propagated_wavefield=np.flip(output, axis=0),
        ncells_pad_z=ncells_pad_z,
        ncells_pad_x=ncells_pad_x,
        check_params=False
    )
    output *= primary_wavefield


if __name__ == "__main__":


    _arr = np.zeros(shape=(100, 100), dtype=np.float32) + 1.0
    print(_arr.shape)
    plt.imshow(_arr)
    plt.colorbar()
    plt.show()