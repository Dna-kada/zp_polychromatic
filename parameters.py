# -*- coding: utf-8 -*-
"""Simulation parameters used throughout the project."""
import numpy as np

fac = int(2)
R = 126e-6 / 2  # 2.2499999999999998e-05/2
nz = 900
wl = 18
width = np.linspace(nz * 0.05, nz * 1.1, wl)
N = nz * 25  # Number of pixels set so heights of different orders are ok
ext = R / 0.30  # exact number not so important
te = np.linspace(0, ext, N)
lamda = 2.73e-9  # matched to experimental mean
dr = ext / N
fr = np.linspace(0, 1 / (2 * dr), N)
kr = 2 * np.pi * fr
k = 2 * np.pi / lamda
