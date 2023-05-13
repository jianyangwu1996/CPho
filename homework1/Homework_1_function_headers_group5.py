'''Homework 1, Computational Photonics, SS 2020:  FD mode solver.
'''
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

    diag = -2 / h ** 2 + k0 ** 2 * prm
    M = np.diag(diag)
    idx = np.linspace(0, len(prm) - 1, len(prm)).astype('int')
    idx_right = (idx[:-1], idx[:-1] + 1)
    idx_left = (idx[1:], idx[1:] - 1)
    M[idx_left] = 1 / h ** 2
    M[idx_right] = 1 / h ** 2
    M *= 1 / k0**2

    eff_eps, guided = np.linalg.eig(M)

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

    (m, n) = prm.shape
    count = np.sum(np.ones_like(prm)).astype('int')
    prm_flat = np.reshape(prm, (1, count))

    M_laplace = np.diag(-4 * np.ones(count))
    idx = np.linspace(0, count-1, count).astype('int')
    idx_left = (idx[1:], idx[1:] - 1)
    idx_right = (idx[:-1], idx[:-1] + 1)
    idx_up = (idx[: -n], idx[:-n] + n)
    idx_down = (idx[n:], idx[n:] - n)
    M_laplace[idx_left] = 1
    M_laplace[idx_right] = 1
    M_laplace[idx_up] = 1
    M_laplace[idx_down] = 1
    M_laplace /= 1 / h**2
    M = M_laplace + np.diag(k0**2 * prm_flat)

    eff_eps, guided = eigs(M, numb)

    return eff_eps, guided
