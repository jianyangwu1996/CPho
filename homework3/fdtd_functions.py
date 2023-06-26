# -*- coding: utf-8 -*-


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
    # constants
    c = 2.99792458e8  # m/s
    mu0 = 4 * np.pi * 1e-7  # Vs/(Am)
    eps0 = 1 / (mu0 * c ** 2)  # As/(Vm)

    # calculate time step and prefactors
    Delta_t = dx / (2 * c)
    e_factor = Delta_t / eps0
    h_factor = Delta_t / mu0

    # create position and time vectors
    Nx = eps_rel.size
    x = np.arange(Nx) * dx - (Nx - 1) / 2.0 * dx
    Niter = int(round(time_span / Delta_t))
    t = np.arange(Niter + 1) * Delta_t

    # allocate field arrays
    Ez = np.zeros((Niter + 1, Nx), dtype=complex)
    Hy = np.zeros((Niter + 1, Nx - 1), dtype=complex)

    # source properties
    # angular frequency (avoids multiplication by 2*pi every iteration)
    source_angular_frequency = 2 * np.pi * source_frequency
    # time offset of pulse center
    t0 = 3 * source_pulse_length
    # x-grid index of delta-source (rounded to nearest grid point)
    source_ind = int(round((source_position - x[0]) / dx))
    if (source_ind < 1) or (source_ind > Nx - 2):
        raise ValueError('Source position out of range')

    for n in range(0, Niter):
        # calculate E at time n + 1, the values at the spatial indices
        # 0 and Nx -1 are determined by the PEC boundary conditions
        # and do not have to be updated
        Ez[n + 1, 1:-1] = (Ez[n, 1:-1]
                + e_factor / dx * (Hy[n, 1:] - Hy[n, :-1]) / eps_rel[1:-1])

        # add source term to Ez
        # source current has to  be taken at n + 1/2
        t_source = (n + 0.5) * Delta_t - t0
        j_source = (np.exp(-1j * source_angular_frequency * t_source)  # carrier
                 * np.exp(-(t_source / source_pulse_length) ** 2))  # envelope
        Ez[n + 1, source_ind] -= e_factor / eps_rel[source_ind] * j_source

        # calculate H at time n + 3/2
        Hy[n + 1, :] = Hy[n, :] + h_factor / dx * (Ez[n + 1, 1:] - Ez[n + 1, :-1])

    # The fields are returned on the same x-grid as eps_rel whereby
    # both Ez and Hy are returned at the same points in space and time
    # (the user should not need to care about the peculiarities of
    # the Yee grid and the leap frog algorithm).

    # interpolate Hy to same t and x grid as Ez
    Hy[1:, :] = 0.5 * (Hy[:-1, :] + Hy[1:, :])
    Hy = average_axes(replicate_boundary_values(Hy, [1]), [1])
    return Ez, Hy, x, t


def fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz,
            field_component, z_ind, output_step):
    """Computes the temporal evolution of a pulsed spatially extended current
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
        z_ind: int
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
    """
    # convert inputs to single precision
    eps_rel = eps_rel.astype(np.float32)
    jx = jx.astype(np.complex64)
    jy = jy.astype(np.complex64)
    jz = jz.astype(np.complex64)

    # constants
    c = 2.99792458e8  # m/s
    mu0 = 4 * np.pi * 1e-7  # Vs/(Am)
    eps0 = 1 / (mu0 * c ** 2)  # As/(Vm)

    # calculate time step and prefactors
    Delta_t = np.float32(dr / (2.0 * c))
    e_factor = np.float32(Delta_t / eps0)
    h_factor = np.float32(Delta_t / mu0)
    dr = np.float32(dr)

    # get input grid size
    Nx, Ny, Nz = eps_rel.shape

    # calculate number of iterations and set up return values
    # round to nearest integer number of outputs
    Niter = int(round(time_span / Delta_t / output_step) * output_step)
    t = np.arange(0, Niter + 1, output_step) * Delta_t
    #slice which we return
    F = np.zeros((t.size, Nx, Ny), dtype=np.complex64)

    # interpolate inverse permittivity to grid of electric field
    eps_rel = np.float32(1.0) / eps_rel
    iepsx = average_axes(eps_rel, [0])
    iepsy = average_axes(eps_rel, [1])
    iepsz = average_axes(eps_rel, [2])
    del eps_rel

    # interpolate currents to grid of electric field
    jx = average_axes(jx, [0])
    jy = average_axes(jy, [1])
    jz = average_axes(jz, [2])

    # allocate field arrays
    Ex = np.zeros((Nx - 1, Ny, Nz), dtype=np.complex64)
    Ey = np.zeros((Nx, Ny - 1, Nz), dtype=np.complex64)
    Ez = np.zeros((Nx, Ny, Nz - 1), dtype=np.complex64)
    Hx = np.zeros((Nx, Ny - 1, Nz - 1), dtype=np.complex64)
    Hy = np.zeros((Nx - 1, Ny, Nz - 1), dtype=np.complex64)
    Hz = np.zeros((Nx - 1, Ny - 1, Nz), dtype=np.complex64)

    # valid indices for unshifted axes (without extra boundary values)
    iux = slice(1, Nx - 1)
    iuy = slice(1, Ny - 1)
    iuz = slice(1, Nz - 1)

    # indices for derivatives in E step
    iuxm1 = slice(0, Nx - 2)
    iuym1 = slice(0, Ny - 2)
    iuzm1 = slice(0, Nz - 2)

    # valid indices for shifted axes
    isx = slice(0, Nx - 1)
    isy = slice(0, Ny - 1)
    isz = slice(0, Nz - 1)

    # indices for derivatives in H step
    isxp1 = slice(1, Nx)
    isyp1 = slice(1, Ny)
    iszp1 = slice(1, Nz)

    next_out = 1  # next slice index in output matrix F
    report_inc = 4.0  # progress report interval in seconds
    next_report = report_inc  # time of next progress report
    f = np.float32(0.5)  # prefactor for calculation of averages
    timer = Timer()  # tic-toc timer (defined further down in this module)

    for n in range(Niter):
        # %% calculate source amplitude %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # E(t=(n + 1)*dt) is updated with j(t=(n + 0.5)*dt)!
        t_source = (n + 0.5) * Delta_t - 3.0 * tau
        source_factor = np.complex64(e_factor * np.exp(-2j * np.pi * freq * t_source)
                                     * np.exp(-(t_source / tau) ** 2))

        # %% update Ex(t=(n+1)*dt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Delta_t/eps0*dHz/dy
        U = e_factor / dr * (Hz[isx, iuy, iuz] - Hz[isx, iuym1, iuz])
        # - Delta_t/eps0*dHy/dz
        U -= e_factor / dr * (Hy[isx, iuy, iuz] - Hy[isx, iuy, iuzm1])
        # - Delta_t/eps0*jx (interpolated to Ex grid)
        U -= source_factor * jx[isx, iuy, iuz]
        # divide by eps_rel (interpolated to Ex grid)
        U *= iepsx[isx, iuy, iuz]
        # + Ex(t=n*dt)
        U += Ex[isx, iuy, iuz]
        if ((n + 1) % output_step == 0) and (field_component == 'ex'):
            F[next_out, :, :] = fdtd_3d_interpolate_field(U, z_ind,
                                                          field_component)
            next_out += 1
        Ex[isx, iuy, iuz] = U  # store update wihtout changing boundary values

        # %% update Ey(t=(n+1)*dt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Delta_t/eps0*dHx/dz
        U = e_factor / dr * (Hx[iux, isy, iuz] - Hx[iux, isy, iuzm1])
        # - Delta_t/eps0*dHz/dx
        U -= e_factor / dr * (Hz[iux, isy, iuz] - Hz[iuxm1, isy, iuz])
        # - Delta_t/eps0*jy (interpolated to Ey grid)
        U -= source_factor * jy[iux, isy, iuz]
        # divide by eps_rel (interpolated to Ey grid)
        U *= iepsy[iux, isy, iuz]
        # + Ey(t=n*dt)
        U += Ey[iux, isy, iuz]
        if ((n + 1) % output_step == 0) and (field_component == 'ey'):
            F[next_out, :, :] = fdtd_3d_interpolate_field(U, z_ind,
                                                          field_component)
            next_out += 1
        Ey[iux, isy, iuz] = U  # store update wihtout changing boundary values

        # %% update Ez(t=(n+1)*dt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Delta_t/eps0*dHy/dx
        U = e_factor / dr * (Hy[iux, iuy, isz] - Hy[iuxm1, iuy, isz])
        # - Delta_t/eps0*dHx/dy
        U -= e_factor / dr * (Hx[iux, iuy, isz] - Hx[iux, iuym1, isz])
        # - Delta_t/eps0*jz (interpolated to Ez grid)
        U -= source_factor * jz[iux, iuy, isz]
        # divide by eps_rel (interpolated to Ez grid)
        U *= iepsz[iux, iuy, isz]
        # + Ez(t=n*dt)
        U += Ez[iux, iuy, isz]
        if ((n + 1) % output_step == 0) and (field_component == 'ez'):
            F[next_out, :, :] = fdtd_3d_interpolate_field(U, z_ind,
                                                          field_component)
            next_out += 1
        Ez[iux, iuy, isz] = U  # store update wihtout changing boundary values

        # %% update Hx(t=(n+1.5)*dt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = Hx[iux, isy, isz]  # old value at t=(n+0.5)*dt
        # - Delta_t/mu_0*dEz/dy, only valid values of Ez along x
        U -= h_factor / dr * (Ez[iux, isyp1, isz] - Ez[iux, isy, isz])
        # + Delta_t/mu_0*dEy/dz, only valid values of Ey along x
        U += h_factor / dr * (Ey[iux, isy, iszp1] - Ey[iux, isy, isz])
        # interpolation to t=n*dt: average of old and new field
        if ((n + 1) % output_step == 0) and (field_component == 'hx'):
            F[next_out, :, :] = fdtd_3d_interpolate_field(
                f * (U + Hx[iux, isy, isz]), z_ind,
                field_component)
            next_out += 1
        Hx[iux, isy, isz] = U

        # %% update Hy(t=(n+1.5)*dt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = Hy[isx, iuy, isz]  # old value at t=(n+0.5)*dt
        # - Delta_t/mu_0*dEx/dz, only valid values of Ex along y
        U -= h_factor / dr * (Ex[isx, iuy, iszp1] - Ex[isx, iuy, isz])
        # + Delta_t/mu_0*dEz/dx, only valid values of Ez along y
        U += h_factor / dr * (Ez[isxp1, iuy, isz] - Ez[isx, iuy, isz])
        if ((n + 1) % output_step == 0) and (field_component == 'hy'):
            F[next_out, :, :] = fdtd_3d_interpolate_field(
                f * (U + Hy[isx, iuy, isz]), z_ind,
                field_component)
            next_out += 1
        Hy[isx, iuy, isz] = U

        # %% update Hz(t=(n+1.5)*dt) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = Hz[isx, isy, iuz]  # old value at t=(n+0.5)*dt
        # - Delta_t/mu_0*dEy/dx, only valid values of Ey along z
        U -= h_factor / dr * (Ey[isxp1, isy, iuz] - Ey[isx, isy, iuz])
        # + Delta_t/mu_0*dEx/dy, only valid values of Ex along z
        U += h_factor / dr * (Ex[isy, isyp1, iuz] - Ex[isx, isy, iuz])
        if ((n + 1) % output_step == 0) and (field_component == 'hz'):
            F[next_out, :, :] = fdtd_3d_interpolate_field(
                f * (U + Hz[isx, isy, iuz]), z_ind,
                field_component)
            next_out += 1
        Hz[isx, isy, iuz] = U

        # %% report progress %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        progress = (n + 1) / Niter
        elapsed = timer.toc()
        if (elapsed >= next_report) or (progress == 1.0):
            elapsed = timer.toc()
            next_report += report_inc
            print('Iteration {0} of {1}, elapsed {2:g}s, '
                  'remaining {3:g}s, {4:g} iter/s'.format(n + 1, Niter,
                                                          elapsed, elapsed * (1.0 / progress - 1.0), n / elapsed))

    return F, t


def replicate_boundary_values(field, axes):
    """Replicates the roundary values of the given field along the
    specified axes

    Arguments
    ---------
        field : nd-array
            Field array to be padded.
        axes : sequence
            Sequence of axes that shall be padded.

    Returns
    -------
        res: nd-array
            Field padded along the specified axes.
    """
    res = field
    for ax in axes:
        a = tuple((0 if i == ax else slice(None)
                   for i in range(field.ndim)))
        b = tuple((-1 if i == ax else slice(None)
                   for i in range(field.ndim)))
        na = tuple((np.newaxis if i == ax else slice(None)
                    for i in range(field.ndim)))
        res = np.concatenate((res[a][na], res, res[b][na]), axis=ax)
    return res


def pad_boundary_values(field, axes):
    """Replicates the roundary values of the given field along the
    specified axes

    Arguments
    ---------
        field : nd-array
            Field array to be padded.
        axes : sequence
            Sequence of axes that shall be padded.

    Returns
    -------
        res: nd-array
            Field padded along the specified axes.
    """
    res = field
    for ax in axes:
        shape = tuple((1 if i == ax else n for i, n in enumerate(res.shape)))
        pad = np.zeros(shape, dtype=res.dtype)
        res = np.concatenate((pad, res, pad), axis=ax)
    return res


def average_axes(field, axes):
    """Averages neighboring values of the given field along the specified axes

    Arguments
    ---------
        field : nd-array
            Field array to be averaged.
        axes : sequence
            Sequence of axes that shall be averaged.

    Returns
    -------
        res: nd-array
            Field averaged along the specified axes.
    """

    # prefactor for calculating avergaes with the same numerical preciosion
    # as the input field.
    f = np.array(0.5, dtype=field.dtype)

    c = slice(0, None)  # full
    l = slice(0, -1)  # left part of average
    r = slice(1, None)  # right part of average
    res = field
    for ax in axes:
        a = tuple((l if i == ax else c for i in range(res.ndim)))
        b = tuple((r if i == ax else c for i in range(res.ndim)))
        res = f * (res[a] + res[b])
    return res


def fdtd_3d_interpolate_field(field, z_ind, component):
    """Interpolates the given field to the
    centered grid (i,j,k) and selects a z-slice at the z-position specified
    by z_ind.

    Arguments
    ---------
        field: 3d-array
            Field array to be interpolated.
        z_ind: integer
            Z-index of the output z-slice.
        component: str
            Field component stored in field
            (one of 'ex', 'ey', 'ez', 'hx', 'hy', 'hz').

    Returns
    -------
        interp : 2d-array
            Interpolated z-slice of the field at z-index z_ind.
    """
    component = component.lower()
    if component == 'ex':
        # Ex needs interpolation along x
        rep_axes = [0]
        pad_axes = [1, 2]
    elif component == 'ey':
        # Ey needs interpolation along y
        rep_axes = [1]
        pad_axes = [0, 2]
    elif component == 'ez':
        # Ez needs interpolation along z
        rep_axes = [2]
        pad_axes = [0, 1]
    elif component == 'hx':
        # Hx needs interpolation along y, z
        rep_axes = [1, 2]
        pad_axes = [0]
    elif component == 'hy':
        # Hy needs interpolation along x, z
        rep_axes = [0, 2]
        pad_axes = [1]
    elif component == 'hz':
        # Hz needs interpolation along x, y
        rep_axes = [0, 1]
        pad_axes = [2]
    else:
        raise ValueError('Invalid field component')
    res = average_axes(
        replicate_boundary_values(
            pad_boundary_values(field, pad_axes), rep_axes),
        rep_axes)
    return res[:, :, z_ind]


class Fdtd1DAnimation(animation.TimedAnimation):
    """Animation of the 1D FDTD fields.

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
    """

    def __init__(self, x, t, Ez, Hy, x_interface=None, step=2e-15 / 25, fps=25):
        # constants
        c = 2.99792458e8  # speed of light [m/s]
        mu0 = 4 * np.pi * 1e-7  # vacuum permeability [Vs/(Am)]
        eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [As/(Vm)]
        Z0 = np.sqrt(mu0 / eps0)  # vacuum impedance [Ohm]
        self.Ez = Ez
        self.Z0Hy = Z0 * Hy
        self.x = x
        self.ct = c * t

        # index step between consecutive frames
        self.frame_step = int(round(step / (t[1] - t[0])))

        # set up initial plot
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        vmax = max(np.max(np.abs(Ez)), np.max(np.abs(Hy)) * Z0) * 1e6
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
        self.line_E, = ax[0].plot(x * 1e6, self.E_at_step(0),
                                  color=colors[0], label='$\\Re\\{E_z\\}$')
        self.line_H, = ax[1].plot(x * 1e6, self.H_at_step(0),
                                  color=colors[1], label='$Z_0\\Re\\{H_y\\}$')
        if x_interface is not None:
            for a in ax:
                a.axvline(x_interface * 1e6, ls='--', color='k')
        for a in ax:
            a.set_xlim(x[[0, -1]] * 1e6)
            a.set_ylim(np.array([-1.1, 1.1]) * vmax)
        ax[0].set_ylabel('$\\Re\\{E_z\\}$ [µV/m]')
        ax[1].set_ylabel('$Z_0\\Re\\{H_y\\}$ [µV/m]')
        self.text_E = ax[0].set_title('')
        self.text_H = ax[1].set_title('')
        ax[1].set_xlabel('$x$ [µm]')
        super().__init__(fig, interval=1000 / fps, blit=False)

    def E_at_step(self, n):
        return self.Ez[n, :].real * 1e6

    def H_at_step(self, n):
        return self.Z0Hy[n, :].real * 1e6

    def new_frame_seq(self):
        return iter(range(0, self.ct.size, self.frame_step))

    def _init_draw(self):
        self.line_E.set_ydata(self.x * np.nan)
        self.line_H.set_ydata(self.x * np.nan)
        self.text_E.set_text('')
        self.text_E.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.line_E.set_ydata(self.E_at_step(i))
        self.line_H.set_ydata(self.H_at_step(i))
        self.text_E.set_text(
            'Electric field, $ct = {0:1.2f}$µm'.format(self.ct[i] * 1e6))
        self.text_H.set_text(
            'Magnetic field, $ct = {0:1.2f}$µm'.format(self.ct[i] * 1e6))
        self._drawn_artists = [self.line_E, self.line_H,
                               self.text_E, self.text_H]


class Fdtd3DAnimation(animation.TimedAnimation):
    """Animation of a 3D FDTD field.

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
    """

    def __init__(self, x, y, t, field, titlestr, cb_label, rel_color_range, fps=25):
        # constants
        c = 2.99792458e8  # speed of light [m/s]
        self.ct = c * t

        self.fig = plt.figure()
        self.F = field
        color_range = rel_color_range * np.max(np.abs(field))
        phw = 0.5 * (x[1] - x[0])  # pixel half-width
        extent = ((x[0] - phw) * 1e6, (x[-1] + phw) * 1e6,
                  (y[-1] + phw) * 1e6, (y[0] - phw) * 1e6)
        self.mapable = plt.imshow(self.F[0, :, :].real.T,
                                  vmin=-color_range, vmax=color_range,
                                  extent=extent)
        cb = plt.colorbar(self.mapable)
        plt.gca().invert_yaxis()
        self.titlestr = titlestr
        self.text = plt.title('')
        plt.xlabel('x position [µm]')
        plt.ylabel('y position [µm]')
        cb.set_label(cb_label)
        super().__init__(self.fig, interval=1000 / fps, blit=False)

    def new_frame_seq(self):
        return iter(range(self.ct.size))

    def _init_draw(self):
        self.mapable.set_array(np.nan * self.F[0, :, :].real.T)
        self.text.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.mapable.set_array(self.F[i, :, :].real.T)
        self.text.set_text(self.titlestr
                           + ', $ct$ = {0:1.2f}µm'.format(self.ct[i] * 1e6))
        self._drawn_artists = [self.mapable, self.text]


class Timer(object):
    """Tic-toc timer.
    """

    def __init__(self):
        """Initializer.
        Stores the current time.
        """
        self._tic = time.time()

    def tic(self):
        """Stores the current time.
        """
        self._tic = time.time()

    def toc(self):
        """Returns the time in seconds that has elapsed since the last call
        to tic().
        """
        return time.time() - self._tic