'''
Solution to the Voluntary homework 0 of Computational Photonics - Transfer Matrix Method
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


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

    # find suitable type for epsilon
    dt = np.common_type(np.array([n1]), np.array([n2]))
    epsilon = np.zeros((2*N,), dtype = dt)
    epsilon[::2] = n1**2
    epsilon[1::2] = n2**2
    thickness = np.zeros((2*N,))
    thickness[::2] = d1
    thickness[1::2] = d2
    return epsilon, thickness


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
    # initialize system transfer matrix to identity matrix
    M = np.eye(2, dtype=np.complex128)

    # determine alpha, depending on the polarization
    if polarisation == 'TE':
        alpha = np.ones_like(epsilon)
    elif polarisation == 'TM':
        alpha = 1.0/epsilon
    else:
        raise ValueError('Invalid input: '
                         'polarisation must either be "TE" or "TM"')

    # vacuum wavenumber
    k0 = 2.0*np.pi/wavelength
    # wave vector component normal to layer stack
    kx = np.sqrt(k0**2*epsilon - kz**2, dtype=np.complex128)

    # iterate over layers
    for kxi, di, alphai in zip(kx, thickness, alpha):
        c = np.cos(kxi*di)
        s = np.sin(kxi*di)
        ka = kxi*alphai
        m = np.array([[    c, s/ka],
                      [-ka*s,    c]])
        # prepend new layer transfer matrix to system transfer matrix
        M = m@M

    return M


def spectrum(thickness, epsilon, polarisation, wavelength, angle_inc,
             n_in, n_out):
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

    # Definition of the paramters of the input and output layers
    epsilon_in = n_in**2
    epsilon_out = n_out**2

    if polarisation == 'TE':
        alpha_in = 1
        alpha_out = 1
    elif polarisation == 'TM':
        alpha_in = 1/epsilon_in
        alpha_out = 1/epsilon_out
    else:
        raise ValueError('Invalid input: '
                          'polarisation must either be "TE" or "TM"')

    k0 = 2.0*np.pi/wavelength # vacuum wavenumber
    kz = k0*n_in*np.sin(np.deg2rad(angle_inc))
    kx_in = np.sqrt(epsilon_in*k0**2 - kz**2, dtype=np.complex128)
    kx_out = np.sqrt(epsilon_out*k0**2 - kz**2, dtype=np.complex128)

    r = np.zeros(wavelength.shape, dtype=np.complex128)
    N = np.zeros(wavelength.shape, dtype=np.complex128)

    # iterate over wavelengths
    for i, (lami, kzi, kx_outi, kx_ini) in enumerate(zip(wavelength, kz, kx_out, kx_in)):
        M = transfermatrix(thickness, epsilon, polarisation, lami, kzi)
        N[i] = (alpha_in*kx_ini*M[1,1] + alpha_out*kx_outi*M[0,0]
              + 1j*(M[1,0] - alpha_in*kx_ini*alpha_out*kx_outi*M[0,1]))
        r[i] = (alpha_in*kx_ini*M[1,1] - alpha_out*kx_outi*M[0,0]
              - 1j*(M[1,0] + alpha_in*kx_ini*alpha_out*kx_outi*M[0,1]))
    r /= N

    # calculate remaining coefficients
    t = 2.0*alpha_in*kx_in/N
    R = np.real(r*np.conj(r))
    T = np.real(alpha_out*kx_out)/np.real(alpha_in*kx_in)*np.real(t*np.conj(t))

    return t, r, T, R




def field(thickness, epsilon, polarisation, wavelength, kz, n_in, n_out,
          Nx, l_in, l_out):
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
        Additional thickness of the input and output layers where the field
        will be computed.

    Returns
    -------
    f : 1d-array
        Field structure
    index : 1d-array
        Refractive index distribution
    x : 1d-array
        Spatial coordinates
    '''
    # Input layer for x < 0; and output layer for x > 0; illumination from the input side
    epsilon_in = n_in**2
    epsilon_out = n_out**2

    # extension of the vectors epsilon and thickness to take the input
    #  and output layers into account
    thickness = np.concatenate(([l_in], thickness, [l_out]))
    epsilon = np.concatenate(([epsilon_in], epsilon, [epsilon_out]))

    # flip layers (calculation proceeds backwars from the transmitted field)
    epsilon = epsilon[::-1]
    thickness = thickness[::-1]

    # determine alpha, depending on the polarization
    if polarisation == 'TE':
        alpha = np.ones_like(epsilon)
    elif polarisation == 'TM':
        alpha = 1.0/epsilon
    else:
        raise ValueError('Invalid input: '
                         'polarisation must either be "TE" or "TM"')

    # vacuum wavenumber
    k0 = 2.0*np.pi/wavelength
    # wave vector component normal to layer stack
    kx = np.sqrt(epsilon*k0**2 - kz**2, dtype=np.complex128)

    # output layer parameters
    alpha_out = alpha[0]
    kx_out = kx[0]

    # further computation starts from the transmitted field because the fields
    # are calculated from the back
    incident_vec = np.array([[1.0], [1.0j*alpha_out*kx_out]])

    # definition of output positions
    x = np.linspace(0, np.sum(thickness), Nx)

    curr_layer = 0
    pos_in_layer = 0.0
    thickness_below = 0.0
    M = np.eye(2, dtype=np.complex128)
    f = np.zeros(x.shape, dtype=np.complex128)
    index = np.zeros(x.shape, dtype=epsilon.dtype)

    for i, xi in enumerate(x):
        # get postion within current layer
        pos_in_layer = xi - thickness_below

        # check if a layer interface has been crossed
        if pos_in_layer > thickness[curr_layer]:
            # propagate until layer interface
            c = np.cos(kx[curr_layer]*thickness[curr_layer])
            s = np.sin(kx[curr_layer]*thickness[curr_layer])
            ka = kx[curr_layer]*alpha[curr_layer]
            m = np.array([[    c, -s/ka],
                          [ka*s,    c]])
            # update transfer matrix
            M = m@M
            # update state variables
            pos_in_layer = pos_in_layer - thickness[curr_layer]
            thickness_below = thickness_below + thickness[curr_layer]
            curr_layer += 1

        # propagate within layer to current position
        c = np.cos(kx[curr_layer]*pos_in_layer)
        s = np.sin(kx[curr_layer]*pos_in_layer)
        ka = kx[curr_layer]*alpha[curr_layer]
        m = np.array([[    c, -s/ka],
                      [ka*s,    c]])
        out_vector = m@M@incident_vec
        f[i] = out_vector[0,0]
        index[i] = np.sqrt(epsilon[curr_layer])

    # at the end, the fields have to be flipped
    f = f[::-1]
    index = index[::-1]

    return f, index, x


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

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The time animation of the field
    '''
    # based on https://matplotlib.org/gallery/animation/simple_anim.html

    freq = periods/(steps - 1)
    max_f = np.abs(f).max()
    max_index = index.real.max()

    # helper function to calculate field at step n
    def field_at_step(n):
        return np.real(f*np.exp(-2.0j*np.pi*freq*n))/max_f*max_index

    # set up initial plot
    fig = plt.figure()
    line_f, line_eps = plt.plot(x, field_at_step(0),
                                x, index.real)
    plt.xlabel('x [µm]')
    plt.ylabel('normalized field (real part)')
    plt.legend(['EM field', 'refr. index'], loc='lower right', frameon=False)
    plt.xlim(x[[0,-1]])
    plt.ylim(np.array([-1.1, 1.1])*max_index)

    # function that updates plot data during animation
    def animate(i):
        line_f.set_ydata(field_at_step(i))
        return (line_f,)

    # function that inits plot data for animation (clean state for blitting)
#    def init():
#        line_f.set_ydata(x*np.nan)
#        return (line_f,)

    ani = animation.FuncAnimation(fig, animate, #init_func=init,  blit=True,
                                  save_count=steps, interval=100)

    plt.show()
    return ani

