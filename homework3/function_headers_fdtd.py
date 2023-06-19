'''
Homework 3, Computational Photonics, SS 2023:  FDTD method.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import time


def fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position,
            source_pulse_length):
    """Computes the temporal evolution of a pulsed excitation using the
    1D FDTD method. The temporal center of the pulse is placed at a
    simulation time of 3*source_pulse_length. The origin x=0 is in the
    center of the computational domain. All quantities have to be
    specified in SI units.

    Arguments
    ---------
        eps_rel : 1d-array
            Rel. permittivity distribution within the computational domain.
        dx : float
            Spacing of the simulation grid (please ensure dx <= lambda/20).
        time_span : float
            Time span of simulation.
        source_frequency : float
            Frequency of current source.
        source_position : float
            Spatial position of current source.
        source_pulse_length :
            Temporal width of Gaussian envelope of the source.

    Returns
    -------
        Ez : 2d-array
            Z-component of E(x,t) (each row corresponds to one time step)
        Hy : 2d-array
            Y-component of H(x,t) (each row corresponds to one time step)
        x  : 1d-array
            Spatial coordinates of the field output
        t  : 1d-array
            Time of the field output
    """
    c = 2.99792458e8
    mu0 = 4 * np.pi * 1e-7
    eps0 = 1 / (mu0 * c ** 2)

    lam = c/source_frequency
    if dx > lam/20:
        dx = lam/20
    else:
        pass

    Nx = len(eps_rel)
    x = np.linspace(-(Nx-1)*dx/2, (Nx-1)*dx/2, Nx)
    dt = dx / 2 / c
    t = np.arange(0, time_span, dt)
    Nt = len(t)

    t0 = 3 * source_pulse_length
    j0 = 1
    carrier = np.exp(-2j * np.pi * source_frequency * (t-t0+0.5*dt))
    A = np.exp(-(t+0.5*dt - t0)**2 / source_pulse_length**2)
    jz = A * carrier * j0

    Ez = np.zeros((Nt, Nx)).astype('complex')
    Hy = np.zeros((Nt, Nx-1)).astype('complex')
    ind = int(round((source_position - x[0]) / dx))

    for n in range(1, Nt):
        Ez[n, 1:-1] = Ez[n-1, 1:-1] + 1/eps0/eps_rel[1:-1] * dt/dx * (Hy[n-1, 1:] - Hy[n-1, :-1])
        Ez[n, ind] -= dt/eps0/eps_rel[ind] * jz[n]

        Hy[n, :] = Hy[n-1, :] + 1/mu0 * dt/dx * (Ez[n, 1:] - Ez[n, :-1])

    # interpolate Hy
    Hy[1:, :] = 0.5 * (Hy[:-1, :] + Hy[1:, :])
    Hy = np.pad(Hy, ((0,0),(1,1)), 'edge')
    Hy = 0.5 * (Hy[:, 1:] + Hy[:, :-1])

    return Ez, Hy, x, t


def fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz,
            field_component, z_ind, output_step):
    '''Computes the temporal evolution of a pulsed spatially extended current
    source using the 3D FDTD method. Returns z-slices of the selected
    field at the given z-position every output_step time steps. The pulse
    is centered at a simulation time of 3*tau. All quantities have to be
    specified in SI units.

    Arguments
    ---------
        eps_rel: 3d-array
            Rel. permittivity distribution within the computational domain.
        dr: float
            Grid spacing (please ensure dr<=lambda/20).
        time_span: float
            Time span of simulation.
        freq: float
            Center frequency of the current source.
        tau: float
            Temporal width of Gaussian envelope of the source.
        jx, jy, jz: 3d-array
            Spatial density profile of the current source.
        field_component : str
            Field component which is stored (one of ‘ex’,’ey’,’ez’,
            ’hx’,’hy’,’hz’).
        z_index: int
            Z-position of the field output.
        output_step: int
            Number of time steps between field outputs.

    Returns
    -------
        F: 3d-array
            Z-slices of the selected field component at the
            z-position specified by z_ind stored every output_step
            time steps (time varies along the first axis).
        t: 1d-array
            Time of the field output.
    '''

    c = 2.99792458e8
    mu0 = 4 * np.pi * 1e-7
    eps0 = 1 / (mu0 * c ** 2)

    lam = c / freq
    if dr > lam / 20:
        dr = lam / 20
    else:
        pass

    dt = dr / (2*c)
    Niter = int(time_span//dt)
    t = np.arange(0, time_span, dt*output_step)
    Nt = len(t)

    Nx, Ny, Nz = eps_rel.shape
    Ex = np.zeros((Nx-1, Ny, Nz)).astype('complex64')
    Ey = np.zeros((Nx, Ny-1, Nz)).astype('complex64')
    Ez = np.zeros((Nx, Ny, Nz-1)).astype('complex64')
    Hx = np.zeros((Nx, Ny-1, Nz-1)).astype('complex64')
    Hy = np.zeros((Nx-1, Ny, Nz-1)).astype('complex64')
    Hz = np.zeros((Nx-1, Ny-1, Nz)).astype('complex64')

    epsx_rec = (1/eps_rel[:-1, :, :] + 1/eps_rel[1:, :, :])/2
    epsx_rec = epsx_rec[0:Nx-1, 1:Ny-1, 1:Nz-1].astype('float32')
    epsy_rec = (1/eps_rel[:, :-1, :] + 1/eps_rel[:, 1:, :])/2
    epsy_rec = epsy_rec[1:Nx-1, 0:Ny-1, 1:Nz-1].astype('float32')
    epsz_rec = (1/eps_rel[:, :, :-1] + 1/eps_rel[:, :, 1:])/2
    epsz_rec = epsz_rec[1:Nx-1, 1:Ny-1, 0:Nz-1].astype('float32')

    jx = ((jx[:-1, :, :] + jx[1:, :, :])/2).astype('complex64')
    jy = ((jy[:, :-1, :] + jy[:, 1:, :])/2).astype('complex64')
    jz = ((jz[:, :, :-1] + jz[:, :, 1:])/2).astype('complex64')

    F = np.zeros((Nt,Nx,Ny)).astype('complex64')
    count = 0
    for n in range(Niter):
        t_source = dt*(n + 1/2) - 3*tau
        jx_n = jx * np.exp(-2j * np.pi * freq * t_source) * np.exp(-(t_source/tau)**2)
        jy_n = jy * np.exp(-2j * np.pi * freq * t_source) * np.exp(-(t_source/tau)**2)
        jz_n = jz * np.exp(-2j * np.pi * freq * t_source) * np.exp(-(t_source/tau)**2)

        Ex[0:Nx-1, 1:Ny-1, 1:Nz-1] += (dt/(eps0*epsx_rec) *
                                       ((Hz[0:Nx-1, 1:Ny-1, 1:Nz-1] - Hz[0:Nx-1, 0:Ny-2, 1:Nz-1])/dr -
                                        (Hy[0:Nx-1, 1:Ny-1, 1:Nz-1] - Hy[0:Nx-1, 1:Ny-1, 0:Nz-2])/dr -
                                        jx_n[0:Nx-1, 1:Ny-1, 1:Nz-1]))
        Ey[1:Nx-1, 0:Ny-1, 1:Nz-1] += (dt/(eps0*epsy_rec) *
                                       (Hx[1:Nx-1, 0:Ny-1, 1:Nz-1] - Hx[1:Nx-1, 0:Ny-1, 0:Nz-2])/dr -
                                       (Hz[1:Nx-1, 0:Ny-1, 1:Nz-1] - Hz[0:Nx-2, 0:Ny-1, 1:Nz-1])/dr -
                                       jy_n[1:Nx-1, 0:Ny-1, 1:Nz-1])
        Ez[1:Nx-1, 1:Ny-1, 0:Nz-1] += (dt/(eps0 * epsz_rec) *
                                       (Hy[1:Nx-1, 1:Ny-1, 0:Nz-1] - Hy[0:Nx-2, 1:Ny-1, 0:Nz-1])/dr -
                                       (Hx[1:Nx-1, 1:Ny-1, 0:Nz-1] - Hx[1:Nx-1, 0:Ny-2, 0:Nz-1])/dr -
                                       jz_n[1:Nx-1, 1:Ny-1, 0:Nz-1])

        if field_component == 'hx':
            temp = Hx[1:Nx-1, 0:Ny-1, 0:Nz-1]
        elif field_component == 'hy':
            temp = Hy[0:Nx-1, 1:Ny-1, 0:Nz-1]
        elif field_component == 'hz':
            temp = Hz[0:Nx-1, 0:Ny-1, 1:Nz-1]

        Hx[1:Nx-1, 0:Ny-1, 0:Nz-1] += (dt/mu0 * (Ey[1:Nx-1, 0:Ny-1, 1:Nz] - Ey[1:Nx-1, 0:Ny-1, 0:Nz-1])/dr -
                                       (Ez[1:Nx-1, 1:Ny, 0:Nz-1] - Ez[1:Nx-1, 0:Ny-1, 0:Nz-1])/dr)
        Hy[0:Nx-1, 1:Ny-1, 0:Nz-1] += (dt/mu0 * (Ez[1:Nx, 1:Ny-1, 0:Nz-1] - Ez[0:Nx-1, 1:Ny-1, 0:Nz-1])/dr -
                                       (Ex[0:Nx-1, 1:Ny-1, 1:Nz] - Ex[0:Nx-1, 1:Ny-1, 0:Nz-1])/dr)
        Hz[0:Nx-1, 0:Ny-1, 1:Nz-1] += (dt/mu0 * (Ex[0:Nx-1, 1:Ny, 1:Nz-1] - Ex[0:Nx-1, 0:Ny-1, 1:Nz-1])/dr -
                                       (Ey[1:Nx, 0:Ny-1, 1:Nz-1] - Ey[0:Nx-1, 0:Ny-1, 1:Nz-1])/dr)

        if (n+1)%output_step == 0:
            count += 1
            if field_component == 'ex':
                res = Ex[0:Nx-1, 1:Ny-1, 1:Nz-1]
                res = np.pad(res, ((0,0), (1,1), (1,1)))
                res = np.pad(res, ((1,1), (0,0), (0,0)), 'edge')
                res = (res[:-1,...] + res[1:,...]) * 0.5
            elif field_component == 'ey':
                res = Ey[1:Nx-1, 0:Ny-1, 1:Nz-1]
                res = np.pad(res, ((1, 1), (0, 0), (1, 1)))
                res = np.pad(res, ((0, 0), (1, 1), (0, 0)), 'edge')
                res = (res[:, :-1, :] + res[:, 1:, :]) * 0.5
            elif field_component == 'ez':
                res = Ez[1:Nx-1, 1:Ny-1, 0:Nz-1]
                res = np.pad(res, ((1, 1), (1, 1), (0, 0)))
                res = np.pad(res, ((0, 0), (0, 0), (1, 1)), 'edge')
                res = (res[..., :-1] + res[..., 1:]) * 0.5
            elif field_component == 'hx':
                res = (Hx[1:Nx-1, 0:Ny-1, 0:Nz-1] + temp) * 0.5
                res = np.pad(res, ((1, 1), (0, 0), (0, 0)))
                res = np.pad(res, ((0, 0), (1, 1), (1, 1)), 'edge')
                res = (res[:, :-1, :] + res[:, 1:, :]) * 0.5
                res = (res[..., :-1] + res[..., 1:]) * 0.5
            elif field_component == 'hy':
                res = (Hy[0:Nx-1, 1:Ny-1, 0:Nz-1] + temp) * 0.5
                res = np.pad(res, ((0, 0), (1, 1), (0, 0)))
                res = np.pad(res, ((1, 1), (0, 0), (1, 1)), 'edge')
                res = (res[:-1, ...] + res[1:, ...]) * 0.5
                res = (res[..., :-1] + res[..., 1:]) * 0.5
            elif field_component == 'hz':
                res = (Hz[0:Nx-1, 0:Ny-1, 1:Nz-1] + temp) * 0.5
                res = np.pad(res, ((0, 0), (0, 0), (1, 1)))
                res = np.pad(res, ((1, 1), (1, 1), (0, 0)), 'edge')
                res = (res[:-1, ...] + res[1:, ...]) * 0.5
                res = (res[:, :-1, :] + res[:, 1:, :]) * 0.5

            F[count, ...] = res[..., z_ind]

    return F, t

class Fdtd1DAnimation(animation.TimedAnimation):
    '''Animation of the 1D FDTD fields.

    Based on https://matplotlib.org/examples/animation/subplots.html

    Arguments
    ---------
    x : 1d-array
        Spatial coordinates
    t : 1d-array
        Time
    x_interface : float
        Position of the interface (default: None)
    step : float
        Time step between frames (default: 2e-15/25)
    fps : int
        Frames per second (default: 25)
    Ez: 2d-array
        Ez field to animate (each row corresponds to one time step)
    Hy: 2d-array
        Hy field to animate (each row corresponds to one time step)
    '''

    def __init__(self, x, t, Ez, Hy, x_interface=None, step=2e-15/25, fps=25):
        # constants
        c = 2.99792458e8 # speed of light [m/s]
        mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
        eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
        Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]
        self.Ez = Ez
        self.Z0Hy = Z0*Hy
        self.x = x
        self.ct = c*t

        # index step between consecutive frames
        self.frame_step = int(round(step/(t[1] - t[0])))

        # set up initial plot
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        vmax = max(np.max(np.abs(Ez)),np.max(np.abs(Hy))*Z0)*1e6
        fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0.4})
        self.line_E, = ax[0].plot(x*1e6, self.E_at_step(0),
                         color=colors[0], label='$\\Re\\{E_z\\}$')
        self.line_H, = ax[1].plot(x*1e6, self.H_at_step(0),
                         color=colors[1], label='$Z_0\\Re\\{H_y\\}$')
        if x_interface is not None:
            for a in ax:
                a.axvline(x_interface*1e6, ls='--', color='k')
        for a in ax:
            a.set_xlim(x[[0,-1]]*1e6)
            a.set_ylim(np.array([-1.1, 1.1])*vmax)
        ax[0].set_ylabel('$\\Re\\{E_z\\}$ [µV/m]')
        ax[1].set_ylabel('$Z_0\\Re\\{H_y\\}$ [µV/m]')
        self.text_E = ax[0].set_title('')
        self.text_H = ax[1].set_title('')
        ax[1].set_xlabel('$x$ [µm]')
        super().__init__(fig, interval=1000/fps, blit=False)

    def E_at_step(self, n):
        return self.Ez[n,:].real*1e6

    def H_at_step(self, n):
        return self.Z0Hy[n,:].real*1e6

    def new_frame_seq(self):
        return iter(range(0, self.ct.size, self.frame_step))

    def _init_draw(self):
        self.line_E.set_ydata(self.x*np.nan)
        self.line_H.set_ydata(self.x*np.nan)
        self.text_E.set_text('')
        self.text_E.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.line_E.set_ydata(self.E_at_step(i))
        self.line_H.set_ydata(self.H_at_step(i))
        self.text_E.set_text(
                'Electric field, $ct = {0:1.2f}$µm'.format(self.ct[i]*1e6))
        self.text_H.set_text(
                'Magnetic field, $ct = {0:1.2f}$µm'.format(self.ct[i]*1e6))
        self._drawn_artists = [self.line_E, self.line_H,
                               self.text_E, self.text_H]


class Fdtd3DAnimation(animation.TimedAnimation):
    '''Animation of a 3D FDTD field.

    Based on https://matplotlib.org/examples/animation/subplots.html

    Arguments
    ---------
    x, y : 1d-array
        Coordinate axes.
    t : 1d-array
        Time
    field: 3d-array
        Slices of the field to animate (the time axis is assumed to be be
        the first axis of the array)
    titlestr : str
        Plot title.
    cb_label : str
        Colrbar label.
    rel_color_range: float
        Range of the colormap relative to the full scale of the field magnitude.
    fps : int
        Frames per second (default: 25)
    '''

    def __init__(self, x, y, t, field, titlestr, cb_label, rel_color_range, fps=25):
        # constants
        c = 2.99792458e8 # speed of light [m/s]
        self.ct = c*t

        self.fig = plt.figure()
        self.F = field
        color_range = rel_color_range*np.max(np.abs(field))
        phw = 0.5*(x[1] - x[0]) # pixel half-width
        extent = ((x[0] - phw)*1e6, (x[-1] + phw)*1e6,
                  (y[-1] + phw)*1e6, (y[0] - phw)*1e6)
        self.mapable = plt.imshow(self.F[0,:,:].real.T,
                                  vmin=-color_range, vmax=color_range,
                                  extent=extent)
        cb = plt.colorbar(self.mapable)
        plt.gca().invert_yaxis()
        self.titlestr = titlestr
        self.text = plt.title('')
        plt.xlabel('x position [µm]')
        plt.ylabel('y position [µm]')
        cb.set_label(cb_label)
        super().__init__(self.fig, interval=1000/fps, blit=False)

    def new_frame_seq(self):
        return iter(range(self.ct.size))

    def _init_draw(self):
        self.mapable.set_array(np.nan*self.F[0, :, :].real.T)
        self.text.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.mapable.set_array(self.F[i, :, :].real.T)
        self.text.set_text(self.titlestr
                           + ', $ct$ = {0:1.2f}µm'.format(self.ct[i]*1e6))
        self._drawn_artists = [self.mapable, self.text]


class Timer(object):
    '''Tic-toc timer.
    '''
    def __init__(self):
        '''Initializer.
        Stores the current time.
        '''
        self._tic = time.time()

    def tic(self):
        '''Stores the current time.
        '''
        self._tic = time.time()

    def toc(self):
        '''Returns the time in seconds that has elapsed since the last call
        to tic().
        '''
        return time.time() - self._tic

