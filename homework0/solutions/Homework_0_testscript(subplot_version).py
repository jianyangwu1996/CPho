'''
Test script for the solution of Voluntary homework 0 of Computational Photonics
 - Transfer Matrix Method.
'''

import numpy as np
from matplotlib import pyplot as plt
from Homework_0_solution import bragg, transfermatrix, spectrum, field, timeanimation

plt.rcParams.update({
        'figure.figsize': (12*2/2.54, 9*2/2.54),
        'axes.grid': True
})
plt.close('all')

save_figures = True

# %% task 1: transfer matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = np.sqrt(2.25)
n2 = np.sqrt(15.21)
d1 = 0.13
d2 = 0.05
N = 5
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
fig = [None]*len(case_name)
ax = [[0]*4]*len(case_name)

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
    t, r, T, R = spectrum(thickness, epsilon, polarisation,
                              wavelength_vector, angle_inc, n_in, n_out)
    
    ## Create subplot for all results
    fig[i] = plt.figure(tight_layout=True)
    
    ## plot reflectance and transmittance
    ax[i][0]=fig[i].add_subplot(221,xlabel='wavelength [µm]',ylabel='reflectance, transmittance')
    ax[i][0].plot(wavelength_vector, T, label='transmittance')
    ax[i][0].plot(wavelength_vector, R, label='reflectance')
    ax[i][0].set_xlim(wavelength_vector[[0,-1]])
    ax[i][0].set_ylim([0, 1])
    ax[i][0].legend(loc='center', frameon=False)

    ## plot transmittance on log scale
    ax[i][1]=fig[i].add_subplot(222,xlabel='wavelength [µm]',ylabel='transmittance')
    ax[i][1].semilogy(wavelength_vector, T, label='transmittance')
    ax[i][1].legend(loc='lower right', frameon=False)

    print('Maximum absorption: {0}'.format(np.abs(1 - T - R).max()))

    # %% task 3: field distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    l_in = 1.0
    l_out = 1.0
    Nx = 1000
    wavelength = 0.78
    kz = 0
    f, index, x = field(thickness, epsilon, polarisation, wavelength, kz,
                        n_in ,n_out, Nx, l_in, l_out)
    
    ## plot magnitude of field
    ax[i][2]=fig[i].add_subplot(223,xlabel='x [µm]',ylabel='normalized field (magnitude)')
    ax[i][2].plot(x, np.abs(f)/np.abs(f).max()*index.real.max(), label='EM field')
    ax[i][2].plot(x, index.real, label='refr. index')
    ax[i][2].set_xlim(x[[0,-1]])
    ax[i][2].set_ylim([0, 1.1*index.real.max()])
    ax[i][2].legend(loc='lower right', frameon=False)

    ## plot real part of field
    ax[i][3]=fig[i].add_subplot(224,xlabel='x [µm]',ylabel='normalized field (real part)')
    ax[i][3].plot(x, f.real/np.abs(f).max()*index.real.max(), label='EM field')
    ax[i][3].plot(x, index.real, label='refr. index')
    ax[i][3].set_xlim(x[[0,-1]])
    ax[i][3].set_ylim(np.array([-1.1, 1.1])*index.real.max())
    ax[i][3].legend(loc='lower right', frameon=False)
    plt.show()
    
    # if save_figures:
    #     plt.savefig('{}_plots.pdf'.format(case_name[i]), dpi=300)

# %% task 4: time animation of field %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    steps = 200
    periods = 10
    ani = timeanimation(x, f, index, steps, periods)

    # requires Ffmpeg
    if save_figures:
        ani.save("{}_field_animation.gif".format(case_name[i]))
