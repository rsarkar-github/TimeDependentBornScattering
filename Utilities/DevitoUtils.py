from examples.seismic import demo_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import copy


def create_model(shape=(200, 200)):
    """
    @Params
    shape: 2d numpy array with shape of model (without padding)

    @Returns
    A velocity model object
    """
    return demo_model(
        'layers-isotropic',
        origin=(0., 0.),
        shape=shape,
        spacing=(10., 10.),
        nbl=75,
        grid=None,
        nlayers=1,
    )


def plot_image(model, source=None, receiver=None, colorbar=True, colormap='jet', clip=1.0):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.
    Parameters
    ----------
    model: Velocity object that holds the image.
    source: Coordinates of the source point.
    receiver: Coordinates of the receiver points.
    colorbar: Option to plot the colorbar.
    colormap: Colormap
    clip: Controls min / max of color bar (1.0 means full range)
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices]
    plot = plt.imshow(np.transpose(field), animated=True, cmap=colormap,
                      vmin=clip * np.min(field), vmax=clip * np.max(field),
                      extent=extent)
    plt.xlabel('X position (km)', fontname="Times New Roman", fontsize=15)
    plt.ylabel('Depth (km)', fontname="Times New Roman", fontsize=15)
    plt.axis("equal")

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3 * receiver[:, 0], 1e-3 * receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3 * source[:, 0], 1e-3 * source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Field')
    plt.show()


def plot_image_tx(image, x0, xn, t0, tn, scale=None, colorbar=True,
                  clip=1.0, tlabel=None, xlabel=None, grid="off", aspect="auto"):

    if scale is None:
        scale = np.max(np.abs(image))
    extent = [1e-3 * x0, 1e-3 * xn, 1e-3 * tn, 1e-3 * t0]

    plot = plt.imshow(image, aspect=aspect, vmin=-clip * scale, vmax=clip * scale, cmap="Greys", extent=extent)
    plt.axis("equal")
    
    if grid == "on":
        plt.grid()

    if xlabel is None:
        plt.xlabel('X position (km)', fontname="Times New Roman", fontsize=15)
    else:
        plt.xlabel(xlabel, fontname="Times New Roman", fontsize=15)

    if tlabel is None:
        plt.ylabel('Time (s)', fontname="Times New Roman", fontsize=15)
    else:
        plt.ylabel(tlabel, fontname="Times New Roman", fontsize=15)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def plot_shotrecord(rec, model, t0, tn, colorbar=True, clip=1.0):
    """
    Plot a shot record (receiver values over time).
    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    colorbar: bool
        Whether to add colorbar to plot or not
    clip: float
        Clip value
    """
    scale = np.max(np.abs(rec))
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-clip*scale, vmax=clip*scale, cmap="Greys", extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def conjugate_gradient(linear_operator, rhs, x0=None, niter=5, c=0, printobj=False):
    """
    This function runs the conjugate gradient solver for solving the
    linear s.p.d. system Ax = b
    The objective function is x^T A X - 2 x^T b + c

    @Params
    linear_operator: This is a function that has the signature (np.ndarray, np.ndarray) -> void
                     When called as linear_operator(x, y), it should evaluate y=Ax.
                     It should leave x unchanged.
    rhs: The right hand side of Ax = b, i.e. b.
    x0: Starting solution. Default is None (in this case x0 = 0).
    niter: Number of iterations to run. Default is 5.
    printobj: A boolean flag that controls the printout level of this function. Default to False.

    @Returns
    x: Solution after niter CG iterations
    residual: An array with the normalized residuals (w.r.t initial) computed over each iteration
    """

    # Get rhs norm
    fac = np.linalg.norm(x=rhs)
    print("Norm of rhs = ", fac)
    if fac < 1e-15:
        raise ValueError("Norm of rhs < 1e-15. Trivial solution of zero. Scale up the problem.")

    # Handle if x0 is provided
    if x0 is None:
        x0 = rhs * 0

    # Scale rhs
    rhs_new = rhs / fac
    x = x0 / fac

    # Define temporary variables
    y = x * 0
    matrix_times_p = x * 0

    # Calculate initial residual, and residual norm
    linear_operator(x, y)
    r = rhs_new - y
    r_norm = np.linalg.norm(x=r)
    if r_norm < 1e-12:
        return x0, [r_norm]
    r_norm_sq = r_norm ** 2

    # Initialize p
    p = copy.deepcopy(r)

    # Initialize residual array, iteration array
    residual = [r_norm]
    if printobj:
        linear_operator(x, y)
        objective = [np.real(0.5 * np.vdot(x, y) - np.vdot(x, rhs_new))]

    # Run CG iterations
    for num_iter in range(niter):

        t1 = time.time()

        if printobj:
            print(
                "Beginning iteration : ", num_iter,
                " , Residual : ", residual[num_iter],
                " , Objective : ", objective[num_iter]
            )
        else:
            print(
                "Beginning iteration : ", num_iter,
                " , Residual : ", residual[num_iter]
            )

        # Compute A*p and alpha
        linear_operator(p, matrix_times_p)
        alpha = r_norm_sq / np.vdot(p, matrix_times_p)

        # Update x0, residual
        x += alpha * p
        r -= alpha * matrix_times_p

        # Calculate beta
        r_norm_new = np.linalg.norm(x=r)
        r_norm_new_sq = r_norm_new ** 2
        beta = r_norm_new_sq / r_norm_sq

        # Check convergence
        if r_norm_new < 1e-12:
            break

        # Update p, residual norm
        p = r + beta * p
        r_norm_sq = r_norm_new_sq

        # Update residual array, iteration array
        residual.append(r_norm_new)
        if printobj:
            linear_operator(x, y)
            objective.append(np.real(0.5 * np.vdot(x, y) - np.vdot(x, rhs_new)))

        t2 = time.time()
        print("Iteration took ", t2 - t1, " s\n")

    # Remove the effect of the scaling
    x = x * fac

    return x, residual
