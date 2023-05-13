


import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigs


def guided_modes_1DTE(prm, k0, h):
    """Computes the effective permittivity of a TE polarized guided eigenmode.
    All dimensions are in µm.
    Note that modes are filtered to match the requirement that
    their effective permittivity is larger than the substrate (cladding).

    Parameters
    ----------
    prm : 1d-array
        Dielectric permittivity in the x-direction
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization

    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated modes
    guided : 2d-array
        Field distributions of the guided eigenmodes
    """

    # set up operator matrix
    main = -2.0 * np.ones(len(prm)) / (h**2 * k0**2) + prm
    sec = np.ones(len(prm) - 1) / (h**2 * k0**2)
    L = np.diag(sec, -1) + np.diag(main, 0) + np.diag(sec, 1)

    # solve eigenvalue problem
    eff_eps, guided = np.linalg.eig(L)

    # pick only guided modes
    idx = (eff_eps < np.max(prm)) & (eff_eps > np.min(prm))
    eff_eps = eff_eps[idx]
    guided = guided[:, idx]

    # sort modes from highest to lowest effective permittivity
    # (the fundamental mode has the highest effective permittivity)
    idx = np.argsort(-eff_eps)
    eff_eps = eff_eps[idx]
    guided = guided[:, idx]
    return eff_eps, guided


def guided_modes_2D(prm, k0, h, numb):
    """Computes the effective permittivity of a quasi-TE polarized guided
    eigenmode. All dimensions are in µm.

    Parameters
    ----------
    prm  : 2d-array
        Dielectric permittivity in the xy-plane
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    numb : int
        Number of eigenmodes to be calculated

    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated eigenmodes
    guided : 3d-array
        Field distributions of the guided eigenmodes
    """

    NX, NY = prm.shape
    N = NX * NY
    prmk = prm * k0**2
    ihx2 = 1 / h**2
    ihy2 = 1 / h**2

# ‘F’ means to index the elements in column-major, 
# with the first index changing fastest, and the last index changing slowest.
    md = -2.0 * (ihx2 + ihy2) * np.ones(N) + prmk.ravel(order='F')
    xd = ihx2 * np.ones(N)
    yd = ihy2 * np.ones(N)
    H = sps.spdiags([yd, xd, md, xd, yd], [-NX, -1, 0, 1, NX], N, N, format='csc')

    # remove the '1' when moving to a new line
    # -> look into script pg. 31, upper and lower blue line in first figure
    for i in range(1, NY):
        n = i * NX
        H[n-1, n] = 0
        H[n, n-1] = 0

    # solve eigenvalue problem, #LR: Largest Real Part
    eigvals, eigvecs = eigs(H, k=numb, which='LM')
    eff_eps = eigvals / (k0**2)

    print(eff_eps)
    # pick only guided modes
    ind = (eff_eps > np.min(prm)) & (eff_eps < np.max(prm))
    eff_eps = eff_eps[ind]
    eigvecs = eigvecs[:, ind]

    # sort modes from highest to lowest effective permittivity
    # (the fundamental mode has the highest effective permittivity)
    idx = np.argsort(-eff_eps)
    eff_eps = eff_eps[idx]
    eigvecs = eigvecs[:, idx]

    # reshape eigenvectors to a 2D matrix and store them in a 3D array
    guided = np.zeros((len(eff_eps), NX, NY), dtype=eigvecs.dtype)
    for i in range(len(eff_eps)):
        guided[i, :, :] = np.reshape(eigvecs[:, i], (NX, NY), order='F')

    return eff_eps, guided


def mode_operator_2D(field, prm, k0, h):
    """ Calculates the scalar finite differnece mode operator in 2D.

    Parameters
    ----------
    field : 1d-array
        Array containing the unwrapped field.
    prm : 2d-array
        Dielectric permittivity in the xy-plane.
    h : float
        Spatial discretization
    """
    field = field.reshape(prm.shape, order='F')
    res = -4 * field
    res[:-1, :] += field[1:, :]
    res[1:, :] += field[:-1, :]
    res[:, :-1] += field[:, 1:]
    res[:, 1:] += field[:, :-1]
    res /= h * h
    res += prm * k0 * k0 * field
    return np.ravel(res, order='F')


def guided_modes_2D_direct(prm, k0, h, numb):
    """Computes the effective permittivity of a quasi-TE polarized guided
    eigenmode. All dimensions are in µm.

    Parameters
    ----------
    prm  : 2d-array
        Dielectric permittivity in the xy-plane
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    numb : int
        Number of eigenmodes to be calculated

    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated eigenmodes
    guided : 3d-array
        Field distributions of the guided eigenmodes
    """

    NX, NY = prm.shape
    N = NX * NY

    matvec = lambda x: mode_operator_2D(x, prm, k0, h)
    print(matvec)
    H = sps.linalg.LinearOperator(dtype=np.float64, shape=(N, N),
                                  matvec=matvec)

    # solve eigenvalue problem
    eigvals, eigvecs = eigs(H, k=numb, which='LR')
    eff_eps = eigvals / (k0**2)

    # pick only guided modes
    ind = (eff_eps > np.min(prm)) & (eff_eps < np.max(prm))
    eff_eps = eff_eps[ind]
    eigvecs = eigvecs[:, ind]

    # sort modes from highest to lowest effective permittivity
    # (the fundamental mode has the highest effective permittivity)
    idx = np.argsort(-eff_eps)
    eff_eps = eff_eps[idx]
    eigvecs = eigvecs[:, idx]

    # reshape eigenvectors to a 2D matrix and store them in a 3D array
    guided = np.zeros((len(eff_eps), NX, NY), dtype=eigvecs.dtype)
    for i in range(len(eff_eps)):
        guided[i, :, :] = np.reshape(eigvecs[:, i], (NX, NY), order='F')

    return eff_eps, guided