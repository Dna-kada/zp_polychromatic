# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:11:10 2025

@author: Dunka
"""
import numpy as np
from scipy.interpolate import CubicSpline

def find_cutoff(freq, MTFs, thresholds):
    cut_offs = []
    for MTF in MTFs:
            thresholds_MTF = []
            for q in thresholds:
                for a in range(len(freq)):
                    if MTF[a] < q:
                        thresholds_MTF.append((freq[a] + freq[a - 1]) / 2)
                        break
            cut_offs.append(thresholds_MTF)
    
    return np.array(cut_offs)  # shape (n_q, nY)

def DOF_poly_PSFs(z,peaks):
    zl = len(z)
    Ys = len(peaks)
    foc_rec = np.linspace(z[zl // 2], z[-1], 1000, endpoint=True)
    foc_rec2 = (np.linspace(z[0], z[zl // 2], 1000, endpoint=True))
    f_p = CubicSpline(z, peaks)
    f_rec = f_p(foc_rec, nu=0)
    f_rec_2 = f_p(foc_rec2, nu=0)
    f_rec_2 = f_rec_2 / np.max(f_rec_2)
    f_rec = f_rec / np.max(f_rec)
    DOF1 = 0
    DOF2 = 0
    for k in range(len(f_rec)):
        if f_rec[k] < 0.8:
            DOF1 = (foc_rec[k - 1] + foc_rec[k]) / 2 - foc_rec[0]
            break
    for k in range(len(f_rec_2)):
        if np.flip(f_rec_2)[k] < 0.8:
            DOF2 = np.flip(foc_rec2)[0] - (np.flip(foc_rec2)[k - 1] + np.flip(foc_rec2)[k]) / 2
            break

    return DOF1 + DOF2
    