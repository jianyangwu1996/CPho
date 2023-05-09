'''
Test script for the solution of Voluntary homework 0 of Computational Photonics
 - Transfer Matrix Method.
'''

import numpy as np
from matplotlib import pyplot as plt
from Homework_0_solution import bragg, transfermatrix, spectrum, field, timeanimation

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.15,
        'figure.subplot.right': 0.95,
        'figure.subplot.top': 0.9,
        'axes.grid': True,
})
plt.close('all')

save_figures = True

# %% task 1: transfer matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = np.sqrt(2.25)
n2 = np.sqrt(15.21)
d1 = 0.13
d2 = 0.05
N  = 5
epsilon, thickness = bragg(n1, n2,d1, d2, N)

wavelength = 0.78
kz = 0.0
polarisation = 'TE'
M = transfermatrix(thickness, epsilon, polarisation, wavelength, kz)
print('M = {0}'.format(M))
print('det(M) = {0}'.format(np.linalg.det(M)))
print('eig(M) = {0}'.format(np.linalg.eig(M)[0]))

wavelength = 1.2
kz = 0.0
polarisation = 'TE'
M = transfermatrix(thickness, epsilon, polarisation, wavelength, kz)
print('M = {0}'.format(M))
print('det(M) = {0}'.format(np.linalg.det(M)))
print('eig(M) = {0}'.format(np.linalg.eig(M)[0]))


# %% task 2: reflection and transmission spectrum %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

case_name = ['Bragg_mirror','Fabry-Perot']

for i in range(2): # Two cases: Bragg mirror & Fabry-Perot
    
    if i==0: # Bragg mirror with 5 periods
        epsilon, thickness = bragg(n1, n2, d1, d2, N)
    if i==1: # Fabry-Perot resonator
        epsilon, thickness = bragg(n1, n2, d1, d2, N*2)
        thickness[N*2] = 2*thickness[N*2]
    
    n_in = 1
    n_out = 1.5
    angle_inc = 0
    wavelength_vector = np.linspace(0.5, 1.5, 1001)
    T, R, tau, rho = spectrum(thickness, epsilon, polarisation,
                              wavelength_vector, angle_inc, n_in, n_out)
    
    ## plot reflectance and transmittance
    plt.figure()
    plt.plot(wavelength_vector, tau,
             wavelength_vector, rho)
    plt.xlim(wavelength_vector[[0,-1]])
    plt.ylim([0, 1])
    plt.xlabel('wavelength [µm]')
    plt.ylabel('reflectance, transmittance')
    plt.legend(['transmittance', 'reflectance'], loc='center', frameon=False)
    if save_figures:
        plt.savefig('{}_spectrum.pdf'.format(case_name[i]), dpi=300)
    
    ## plot transmittance on log scale
    plt.figure()
    plt.semilogy(wavelength_vector, tau)
    plt.xlim(wavelength_vector[[0,-1]])
    plt.xlabel('wavelength [µm]')
    plt.ylabel('transmittance')
    plt.legend(['transmittance'], loc='lower right', frameon=False)
    plt.show()
    if save_figures:
        plt.savefig('{}_spectrum_log.pdf'.format(case_name[i]), dpi=300)
    
    print('Maximum absorption: {0}'.format(np.abs(1 - tau - rho).max()))


# %% task 3: field distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i in range(2): # Two cases: Bragg mirror & Fabry-Perot
    
    if i==0: # Bragg mirror with 5 periods
        epsilon, thickness = bragg(n1, n2, d1, d2, N)
    if i==1: # Fabry-Perot resonator
        epsilon, thickness = bragg(n1, n2, d1, d2, N*2)
        thickness[N*2] = 2*thickness[N*2]
    
    l_in = 1.0
    l_out = 1.0
    Nx = 1000
    wavelength = 0.78
    kz = 0
    f, index, x = field(thickness, epsilon, polarisation, wavelength, kz,
                        n_in ,n_out, Nx, l_in, l_out)
    
    ## plot magnitude of field
    plt.figure()
    plt.plot(x, np.abs(f)/np.abs(f).max()*index.real.max(),
             x, index.real)
    plt.xlim(x[[0,-1]])
    plt.ylim([0, 1.1*index.real.max()])
    plt.xlabel('x [µm]')
    plt.ylabel('normalized field (magnitude)')
    plt.legend(['EM field', 'refr. index'], loc='lower right', frameon=False)
    if save_figures:
        plt.savefig('{}_field_magnitude.pdf'.format(case_name[i]), dpi=300)

    #
    ## plot real part of field
    plt.figure()
    plt.plot(x, f.real/np.abs(f).max()*index.real.max(),
             x, index.real)
    plt.xlim(x[[0,-1]])
    plt.ylim(np.array([-1.1, 1.1])*index.real.max())
    plt.xlabel('x [µm]')
    plt.ylabel('normalized field (real part)')
    plt.legend(['EM field', 'refr. index'], loc='lower right', frameon=False)
    plt.show()
    if save_figures:
        plt.savefig('{}_field_real_part.pdf'.format(case_name[i]), dpi=300)



# %% task 4: time animation of field %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
steps = 200
periods = 10
ani = timeanimation(x, f, index, steps, periods)

# requires Ffmpeg
if save_figures:
    ani.save("{}_field_animation.mp4".format(case_name[i]))
