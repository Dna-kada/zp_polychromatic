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
zeta = 80
FWHM = lamda / zeta
focal_length = 1.4e-3

#the quantity being invsetigated
Y_values = np.array([ 0.8, 1., 1.5, 1.875, 2. , 2.5, 3, 3.3, 4, 5])
# Y = 0 is treated as monochromatic, but in theory undefined
resolution_factor = 30 # this is simply a rule of thumb
N = int(Y_values[-1] * zeta) * resolution_factor # number of samples

#fraction of field taken up by zone plate
# 1 - field_fraction gives fraction of array that is zero padding
field_fraction = 0.3 
scale = 20 # scale down the physical space for producing PSF
reduction_factor = int(3) # reduces the no. of samples in computed array
# this reduces the number of computed points in the resulting field. 

zp_radius = np.sqrt(focal_length * int(Y_values[-1] * zeta) * lamda)
L = zp_radius / field_fraction
dr = L/N
NA = np.sin( np.arctan(zp_radius / focal_length))
cut_off_freq_max = 2*NA / lamda
