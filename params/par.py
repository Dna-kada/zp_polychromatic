# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 00:26:13 2026

@author: Dunka
"""

import numpy as np

mm = 1e-3
um = 1e-6 
nm = 1e-9 

lamda = 2.73e-9 # wavelength (not lambda for obvious reasons)
n_zones = 200 # number of zones for the zone plate
#the res factor could be higher but it's more than adequate at this number.
resolution_factor = 30 # this is simply a rule of thumb
N = n_zones * resolution_factor # number of samples
mag = 1050

#the quantity being invsetigated
Y_values = np.array([0, 0.8, 1., 1.25, 1.5, 1.875, 2., 2.5, 3., 3.5, 4., 4.5, 5. ])  
mod_val = [0.05, 0.1, 0.15] # modulation cut-offs to measure




#fraction of field taken up by zone plate
# 1 - field_fraction gives fraction of array that is zero padding
field_fraction = 0.3 
scale = 20 # scale down the physical space for producing PSF


zp_radius = 126e-6 #/ 2 # radius of zone plate
zp_focal_length = (zp_radius**2 - (n_zones * lamda / 2)**2) / (n_zones * lamda)
NA = np.sin( np.arctan(zp_radius / zp_focal_length))
cut_off_freq = 2 * NA / lamda
L = zp_radius / field_fraction # total length of field
r = np.linspace(0, L, N) # real space array
dr = L/N # increment in real space
fr = np.linspace(0, 1 / (2 * dr), N) # frequency space array
kr = 2 * np.pi * fr # angular frequency space

SS_N = 2**10
no_spokes = 43
px_size = (13.5 * um / mag) / 1.9
field_length = px_size * SS_N
freq_val = 1 / ( 2 * px_size)
fr_SS = np.linspace(0, freq_val, SS_N) # frequency space array
kr_SS = 2 * np.pi * fr_SS # angular frequency space
x_SS = np.linspace(-field_length / 2, field_length / 2, SS_N) * px_size

reduction_factor = int(3) # reduces the no. of samples in computed array
# this reduces the number of computed points in the resulting field. 

mult = 4 # multiple of 'DOF_mono's to measure over
DOF_points = 17 # samples for DOF