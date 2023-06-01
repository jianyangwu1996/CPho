'''
Homework 2, Computational Photonics, SS 2020:  Beam propagation method.
'''

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve


def waveguide(xa, xb, Nx, n_cladding, n_core):
    '''Generates the refractive index distribution of a slab waveguide
    with step profile centered around the origin of the coordinate
    system with a refractive index of n_core in the waveguide region
    and n_cladding in the surrounding cladding area.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        xb : float
            Width of waveguide
        Nx : int
            Number of grid points
        n_cladding : float
            Refractive index of cladding
        n_core : float
            Refractive index of core

    Returns
    -------
        n : 1d-array
            Generated refractive index distribution
        x : 1d-array
            Generated coordinate vector
    '''

    x = np.linspace(-xa // 2, xa // 2, Nx)
    n = np.ones(len(x)) * n_cladding
    idx_core = ((x <= xb // 2) & (x >= -xb // 2))
    n[idx_core] = n_core

    return n, x


def gauss(xa, Nx, w):
    '''Generates a Gaussian field distribution v = exp(-x^2/w^2) centered
    around the origin of the coordinate system and having a width of w.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        Nx : int
            Number of grid points
        w  : float
            Width of Gaussian field

    Returns
    -------
        v : 1d-array
            Generated field distribution
        x : 1d-array
            Generated coordinate vector
    '''

    x = np.linspace(-xa // 2, xa // 2, Nx)
    v = np.exp(-x ** 2 / w ** 2)

    return v, x


def beamprop_CN(v_in, lam, dx, n, nd, z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the explicit-implicit
    Crank-Nicolson scheme. All lengths have to be specified in µm.

    Parameters
    ----------
        v_in : 1d-array
            Initial field
        lam : float
            Wavelength
        dx : float
            Transverse step size
        n : 1d-array
            Refractive index distribution
        nd : float
            Reference refractive index
        z_end : float
            Propagation distance
        dz : float
            Step size in propagation direction
        output_step : int
            Number of steps between field outputs

    Returns
    -------
        v_out : 2d-array
            Propagated field
        z : 1d-array
            z-coordinates of field output
    '''

    dz = dz * output_step
    z = np.arange(0, z_end + 1e-6, dz)

    N = len(v_in)
    k = 2 * np.pi / lam * n
    k_mean = 2 * np.pi / lam * nd
    W = (k ** 2 - k_mean ** 2) / (2 * k_mean)
    sec = 1j / (2 * k_mean * dx ** 2) * np.ones(N)
    main = -2 * sec + 1j * W
    data = np.array([sec, main, sec])
    offsets = np.array([-1, 0, 1])
    L = sps.dia_array((data, offsets), shape=(N, N)).tocsc()
    I = sps.eye(N)
    A = I - 0.5 * dz * L
    B = I + 0.5 * dz * L

    v = np.zeros((N, len(z)+1)).astype('complex')
    v[:, 0] = v_in
    for i in range(1, len(z) + 1):
        v[:, i] = spsolve(A, B @ v[:, i - 1])

    return v, z
