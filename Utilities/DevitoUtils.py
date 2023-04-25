from examples.seismic import demo_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
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


def plot_image(model, source=None, receiver=None, colorbar=True, colormap='jet',
               clip=1.0, fontname="Times New Roman", fontsize=15):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.
    Parameters
    ----------
    model:
        Velocity object that holds the image.
    source:
        Coordinates of the source point.
    receiver:
        Coordinates of the receiver points.
    colorbar:
        Option to plot the colorbar.
    colormap:
        Colormap
    clip:
        Controls min / max of color bar (1.0 means full range)
    fontname:
        Fontname to use for plots
    fontsize:
        Fontsize to use for plots
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    field = (getattr(model, 'vp', None) or getattr(model, 'lam')).data[slices]
    plot = plt.imshow(np.transpose(field), animated=True, cmap=colormap,
                      vmin=clip * np.min(field), vmax=clip * np.max(field),
                      extent=extent)
    plt.xlabel('X position (km)', fontname=fontname, fontsize=fontsize)
    plt.ylabel('Depth (km)', fontname=fontname, fontsize=fontsize)
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


def plot_image_xy(
        image, x0, xn, y0, yn, scale=None, vmin=None, vmax=None, sfac=1.0, clip=1.0,
        grid="off", aspect="auto", cmap="Greys", colorbar=True, cbar_size="1%", cbar_pad=0.15,
        xlabel=None, ylabel=None, fontname="Times New Roman", fontsize=15, nxticks=5, nyticks=5,
        savefig_fname=None
):

    extent = [1e-3 * x0, 1e-3 * xn, 1e-3 * yn, 1e-3 * y0]
    xticks = np.arange(1e-3 * x0, 1e-3 * xn, 1e-3 * (xn - x0) / nxticks)
    xticklabels = ["{:4.1f}".format(item) for item in xticks]
    yticks = np.arange(1e-3 * y0, 1e-3 * yn, 1e-3 * (yn - y0) / nyticks)
    yticklabels = ["{:4.1f}".format(item) for item in yticks]

    if scale is None:
        scale = np.max(np.abs(image)) * sfac
    if vmin is None:
        vmin = -scale
    if vmax is None:
        vmax = scale

    plot = plt.imshow(image, aspect=aspect, vmin=clip * vmin, vmax=clip * vmax, cmap=cmap, extent=extent)

    if grid == "on":
        plt.grid()

    if xlabel is None:
        plt.xlabel('X position (km)', fontname=fontname, fontsize=fontsize)
    else:
        plt.xlabel(xlabel, fontname=fontname, fontsize=fontsize)

    if ylabel is None:
        plt.ylabel('Time (s)', fontname=fontname, fontsize=fontsize)
    else:
        plt.ylabel(ylabel, fontname=fontname, fontsize=fontsize)

    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname=fontname, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=fontname, fontsize=fontsize)

    # Create aligned colorbar on the right
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        plt.colorbar(plot, cax=cax)
        for i in cax.yaxis.get_ticklabels():
            i.set_family(fontname)
            i.set_size(fontsize)

    # Save the figure
    if savefig_fname is not None:
        plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()
    if mpl.get_backend() in ["QtAgg", "Qt4Agg"]:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()


def plot_images_grid_xy(
        image_grid, image_titles, x0, xn, y0, yn, axes_pad=0.35, scale=None, vmin=None, vmax=None, sfac=1.0, clip=1.0,
        grid="off", aspect="auto", cmap="Greys", colorbar=True, cbar_size="1%", cbar_pad=0.15,
        xlabel=None, ylabel=None, fontname="Times New Roman", fontsize=15, nxticks=5, nyticks=5,
        savefig_fname=None
):
    # Get number of rows & cols
    nrows, ncols, _, _ = image_grid.shape

    # Get axis limits and labels
    extent = [1e-3 * x0, 1e-3 * xn, 1e-3 * yn, 1e-3 * y0]
    xticks = np.arange(1e-3 * x0, 1e-3 * xn, 1e-3 * (xn - x0) / nxticks)
    xticklabels = ["{:4.1f}".format(item) for item in xticks]
    yticks = np.arange(1e-3 * y0, 1e-3 * yn, 1e-3 * (yn - y0) / nyticks)
    yticklabels = ["{:4.1f}".format(item) for item in yticks]

    if scale is None:
        scale = np.max(np.abs(image_grid[0, 0, :, :])) * sfac
    if vmin is None:
        vmin = -scale
    if vmax is None:
        vmax = scale

    if xlabel is None:
        xlabel = "X position [km]"
    if ylabel is None:
        ylabel = "Z position [km]"

    # Create figure and image grid
    fig = plt.figure(figsize=(30, 10))
    img_grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=axes_pad,
        share_all=True,
        aspect=False,
        cbar_location="right",
        cbar_mode="single",
        cbar_size=cbar_size,
        cbar_pad=cbar_pad,
    )

    # Plot images
    for i in range(nrows):
        for j in range(ncols):

            idx = i * ncols + j
            ax = img_grid[idx]
            im = ax.imshow(
                np.squeeze(image_grid[i, j, :, :]),
                aspect=aspect,
                cmap=cmap,
                vmin=clip * vmin,
                vmax=clip * vmax,
                extent=extent
            )

            if grid == "on":
                ax.grid(True, color="white", linestyle="-", linewidth=0.5)

            ax.set_xticks(xticks)
            if i == nrows - 1:
                ax.set_xticklabels(xticklabels, fontname=fontname, fontsize=fontsize)
                ax.set_xlabel(xlabel, fontname=fontname, fontsize=fontsize)

            ax.set_yticks(yticks)
            if j == 0:
                ax.set_yticklabels(yticklabels, fontname=fontname, fontsize=fontsize)
                ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)

            ax.set_aspect(aspect)
            ax.set_title(image_titles[i][j], fontname=fontname, fontsize=fontsize)

            # Create aligned colorbar on the right
            if colorbar and idx == nrows * ncols - 1:
                ax.cax.colorbar(im)
                for i1 in ax.cax.yaxis.get_ticklabels():
                    i1.set_family(fontname)
                    i1.set_size(fontsize)

    # Save the figure
    if savefig_fname is not None:
        fig.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()
    if mpl.get_backend() in ["QtAgg", "Qt4Agg"]:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()


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
