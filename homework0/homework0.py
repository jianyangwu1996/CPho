''' Python module containing functions to use the transfer matrix method.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def transfermatrix(thickness, epsilon, polarisation, wavelength, kz):
    '''Computes the transfer matrix for a given stratified medium.
    
    Parameters
    ----------
    thickness : 1d-array
        Thicknesses of the layers in µm.
    epsilon : 1d-array
        Relative dielectric permittivity of the layers.
    polarisation : str
        Polarisation of the computed field, either 'TE' or 'TM'.
    wavelength : float
        The wavelength of the incident light in µm.
    kz : float
        Transverse wavevector in 1/µm.
        
    Returns
    -------
    M : 2d-array
        The transfer matrix of the medium.
    '''
    
    M = np.eye(2, 2, dtype='complex')
    
    if polarisation == 'TE':
        q = np.ones(len(thickness), dtype='complex')
    elif polarisation == 'TM':
        q = 1 / epsilon
    
    k0 = 2 * np.pi / wavelength
    kx = np.sqrt(k0**2 * epsilon - kz**2).astype('complex')
    cos = np.cos(kx * thickness)
    sin = np.sin(kx * thickness)
    kq = q * kx
    
    for kqi, cosi, sini in zip(kq, cos, sin):
        mi = np.array([[cosi, sini/kqi],
                       [-kqi * sini, cosi]])
        M = mi @ M
    
    return M


def spectrum(thickness, epsilon, polarisation, wavelength, angle_inc, n_in, n_out):

    '''Computes the reflection and transmission of a stratified medium.
    
    Parameters
    ----------
    thickness : 1d-array
        Thicknesses of the layers in µm.
    epsilon : 1d-array
        Relative dielectric permittivity of the layers.
    polarisation : str
        Polarisation of the computed field, either 'TE' or 'TM'.
    wavelength : 1d-array
        The wavelength of the incident light in µm.
    angle_inc : float
        The angle of incidence in degree (not radian!).
    n_in, n_out : float
        The refractive indices of the input and output layers.
        
	Returns
    -------
    t : 1d-array
        Transmitted amplitude
    r : 1d-array
        Reflected amplitude
    T : 1d-array
        Transmitted energy
    R : 1d-array
        Reflected energy
    '''
    
    epsilon_in = n_in**2
    epsilon_out = n_out**2
    
    if polarisation == 'TE':
        q_in, q_out = 1, 1
    elif polarisation == 'TM':
        q_in, q_out = 1/epsilon_in, 1/epsilon_out
    
    k0 = 2*np.pi/wavelength
    kz = k0 * n_in * np.sin(np.deg2rad(angle_inc))
    k_in = np.sqrt(k0**2 * epsilon_in - kz**2)
    k_out = np.sqrt(k0**2 * epsilon_out - kz**2)
    kq_in = q_in * k_in
    kq_out = q_out * k_out

    t = np.ones(len(wavelength)).astype('complex')
    r = np.ones(len(wavelength)).astype('complex')

    for i, (wi, kqi_in, kqi_out) in enumerate(zip(wavelength, kq_in, kq_out)):
        M = transfermatrix(thickness, epsilon, polarisation, wavelength[i], kz[i])
        nume = kqi_in * M[1,1] - kqi_out * M[0,0] - 1j * (M[1,0] + kqi_in * kqi_out * M[0,1])
        N = kqi_in * M[1,1] + kqi_out * M[0,0] + 1j * (M[1,0] - kqi_in * kqi_out * M[0,1])
        t[i] = 2 * kqi_in / N
        r[i] = nume / N
    T = (q_out * k_out).real / (q_in * k_in).real * np.abs(t)**2
    R = np.abs(r)**2
    
    return t, r, T, R


def field(thickness, epsilon, polarisation, wavelength, kz, n_in, n_out, Nx, l_in, l_out):
    '''Computes the field inside a stratified medium.

    The medium starts at x = 0 on the entrance side. The transmitted field
    has a magnitude of unity.
    
    Parameters
    ----------
    thickness : 1d-array
        Thicknesses of the layers in µm.
    epsilon : 1d-array
        Relative dielectric permittivity of the layers.
    polarisation : str
        Polarisation of the computed field, either 'TE' or 'TM'.
    wavelength : float
        The wavelength of the incident light in µm.
    kz : float
        Transverse wavevector in 1/µm.            
	n_in, n_out : float
        The refractive indices of the input and output layers.
    Nx : int
        Number of points where the field will be computed.
    l_in, l_out : float
        Additional thickness of the input and output layers where the field will be computed.
        
    Returns
    -------
    f : 1d-array
        Field structure
    index : 1d-array
        Refractive index distribution
    x : 1d-array
        Spatial coordinates
    '''

    epsilon_in = n_in ** 2
    epsilon_out = n_out ** 2

    epsilon = np.concatenate(([epsilon_in], epsilon, [epsilon_out]))
    thickness = np.concatenate(([l_in], thickness, [l_out]))
    epsilon = epsilon[::-1]
    thickness = thickness[::-1]

    if polarisation == 'TE':
        q = np.ones(len(epsilon)).astype('complex')
    elif polarisation == 'TM':
        q = 1 / epsilon

    k0 = 2 * np.pi / wavelength
    kx = np.sqrt(k0**2 * epsilon - kz**2)
    kq = kx * q
    kq_out = kq[0]
    f_vector = np.array([[1], [1j * kq_out]])
    M = np.eye(2, dtype='complex')

    x = np.linspace(0, np.sum(thickness), Nx)
    index = np.ones(Nx)
    f = np.ones(Nx).astype('complex')
    layer = 0
    layer_below = 0.0
    low = 0.0

    for i in range(len(x)):
        if x[i] - layer_below > thickness[layer]:
            layer_upper = layer_below + thickness[layer]
            cos = np.cos(kx[layer] * (layer_upper - low))
            sin = np.sin(kx[layer] * (layer_upper - low))
            m = np.array([[cos, -sin / kq[layer]],
                          [kq[layer] * sin, cos]])
            M = m @ M
            low = layer_upper
            layer_below += thickness[layer]
            layer += 1

        cos = np.cos(kx[layer] * (x[i] - low))
        sin = np.sin(kx[layer] * (x[i] - low))
        m = np.array([[cos, -sin / kq[layer]],
                      [kq[layer] * sin, cos]])
        M = m @ M
        f[i] = (M @ f_vector)[0]
        index[i] = np.sqrt(epsilon[layer])
        low = x[i]

    f = f[::-1]
    index = index[::-1]
    return f, index, x


def bragg(n1, n2, d1, d2, N):
    '''Generates the stack parameters of a Bragg mirror
    The Bragg mirror starts at the incidence side with layer 1 and is
    terminated by layer 2

    Parameters
    ----------
    n1, n2 : float or complex
        Refractive indices of the layers of one period
    d1, d2 : float
        Thicknesses of layers of one period
    N  : in
        Number of periods

    Returns
    -------
    epsilon : 1d-array
        Vector containing the permittivities
    thickness : 1d-array
        Vector containing the thicknesses
    '''

    epsilon = np.zeros(2*N).astype(type(n1))
    thickness = np.zeros(2*N).astype(type(d1))

    epsilon[::2] = n1**2
    epsilon[1::2] = n2**2
    thickness[::2] = d1
    thickness[1::2] = d2

    return epsilon, thickness

def timeanimation(x, f, index, steps, periods):
    ''' Animation of a quasi-stationary field.
    
    Parameters
    ----------
    x : 1d-array
        Spatial coordinates
    f : 1d-array
        Field
    index : 1d-array
        Refractive index
    steps : int
        Total number of time points
    periods : int
        Number of the oscillation periods.
    '''
    freq = periods / (steps - 1)

    fig = plt.figure()
    line_f, = plt.plot(x, f / f.max() * index.max(), label='EM field')
    line_index = plt.plot(x, index, label='refr. index')
    plt.xlim(0,x[-1])
    plt.xlabel('x [µm]')
    plt.ylabel('normalized field (real part)')
    plt.legend()

    def update(n):
        line_f.set_ydata(f / f.max() * index.max() * np.exp(-2.0j * np.pi * freq * n))
        return line_f,

    ani = animation.FuncAnimation(fig, update, frames=steps)

    return ani

    