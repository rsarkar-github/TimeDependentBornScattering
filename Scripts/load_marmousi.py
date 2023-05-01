import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # --------------------------
    # Import Marmousi-2 Vp model
    # --------------------------

    # DEFINE MODEL DISCRETIZATION HERE!
    nx = 500  # number of grid points in x-direction
    nz = 174  # number of grid points in z-direction
    dx = 20   # spatial grid point distance in x-direction (m)
    dz = dx   # spatial grid point distance in z-direction (m)

    # Define model filename
    name_vp = "TimeDependentBornScattering/Data/marmousi_II_marine.vp"

    # Open file and write binary data to vp
    f = open(name_vp)
    data_type = np.dtype('float32').newbyteorder('<')
    vp = np.fromfile(f, dtype=data_type)

    # Reshape (1 x nx*nz) vector to (nx x nz) matrix
    vp = vp.reshape(nx, nz)

    # Save Marmousi
    np.savez("TimeDependentBornScattering/Data/marmousi-vp.npz", vp)
    with np.load("TimeDependentBornScattering/Data/marmousi-vp.npz") as data:
        vp = data["arr_0"]

    # Plot Marmousi-2 vp-model
    # ------------------------

    # Define xmax, zmax and model extension
    xmax = nx * dx
    zmax = nz * dz
    extent = [0, xmax, zmax, 0]

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow((vp.T) / 1000, cmap="jet", interpolation='nearest', extent=extent)

    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Marmousi-2 model')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()
