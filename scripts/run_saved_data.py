# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:25:17 2025

@author: Dunka
"""

import numpy as np
import os, sys
# add src directory to path so we can import zpxrm package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optics.transforms import HT
from optics.params import ZonePlate, Spectrum
from simulation.simulation import Simulation
import matplotlib.pyplot as plt
import matplotlib as mpl
from scripts.plotting import configure_plotting, plot_mtf_curves, plot_fractions
from scripts.compute_curves import find_cutoff, DOF_poly_PSFs
from params.par import *

nz = 900
all_results_freq = [np.load(f"results_MTF_{y:.2f}_nz_{nz}.npz")['a'] for y in Y_values][0]
all_results_MTF = [np.load(f"results_MTF_{y:.2f}_nz_{nz}.npz")['b'] for y in Y_values]
all_results_PSF = [np.load(f"results_PSF_{y:.2f}_nz_{nz}.npz")['b'] for y in Y_values]

flat_results_MTF = np.array(all_results_MTF)#np.array([item for sublist in all_results_MTF for item in sublist])
flat_results_PSF = np.array(all_results_PSF)#np.array([item for sublist in all_results_PSF for item in sublist])

configure_plotting()

# load combined results from MPI gather...
freq = all_results_freq / cut_off_freq
DOF_mid = DOF_points // 2
Ys = len(Y_values)

# select a few Y values
sel = np.array([0, 2, 5, 8, 12], dtype = np.int16)
thresholds = [0.05, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# out-of-focus curves
oof_curves = np.array([flat_results_MTF[i][0] for i in range(Ys)])
# in-focus curves
inf_curves = np.array([flat_results_MTF[i][DOF_mid] for i in range(Ys)])

inf_PSF =  np.array([flat_results_PSF[i][DOF_mid] for i in range(Ys)])

# save data
# np.savez("mtf_data.npz", freq=freq, Y=Y_values,
#          oof=np.stack(oof_curves), inf=np.stack(inf_curves))

# plotting
linestyles = ["k-", "k:", "k--", "k-.", (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (3, 5, 1, 5, 1, 5))]


plot_mtf_curves(freq, oof_curves[sel], Y_values[sel],
                "oof_MTF.pdf", linestyles)
plot_mtf_curves(freq, inf_curves[sel], Y_values[sel],
                "MTF_in_focus.pdf", linestyles)



cut_offs = find_cutoff(all_results_freq, inf_curves,thresholds).T

plt.figure()
for v in [1]: 
    plt.plot(Y_values, cut_offs[v] / cut_offs[v][0], linestyles[v], markersize = 5)
plt.xlabel("Y"); plt.ylabel(r"$\nu/\nu_{m}$")
plt.grid()
plt.legend(["10%"])
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig("const_r_demonstration.pdf")
plt.show()

val_mono = cut_offs[1][0]

plt.figure()
# plt.plot(Y_values[1:],1.22*zp.dr_min/Res,'k')
plt.plot(Y_values, cut_offs[1] / val_mono, linestyles[1])
plt.xlabel("$Y$"); plt.ylabel(r"$\frac{\nu_p}{\nu_{m}}$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout(); plt.grid()
plt.savefig("cut_off_reduction.pdf")
plt.show()

m = np.sum(Y_values ** 2 * ((val_mono / cut_offs[1]) ** 2 - 1)) / np.sum((Y_values ** 2) ** 2)
print("Frequency slope value:")
print(m)

plt.figure()
# plt.plot(Y_values[1:],1.22*zp.dr_min/Res,'k')
plt.plot((Y_values) ** 2, (val_mono / cut_offs[1]) ** 2, linestyles[2], markersize = 5)
plt.plot((Y_values) ** 2, m * (Y_values) ** 2 + 1)
plt.xlabel("$Y^2$"); plt.ylabel(r"$\left(\frac{\nu_{m}}{\nu_p}\right)^2$")
plt.grid()
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.legend(["simulated","fitted"])
plt.tight_layout()
plt.savefig("cut_off_reduction_one_curves.pdf")
plt.show()

val_mono_half = cut_offs[3][0]
m1 = np.sum(Y_values**2 * ((val_mono_half/cut_offs[3])**2 - 1)) / np.sum((Y_values**2)**2)
print("Frequency slope value half:")
print(m1)

plt.figure()
# plt.plot(Y_values[1:],1.22*zp.dr_min/Res,'k')
plt.plot((Y_values)**2,(val_mono_half/cut_offs[3])**2,linestyles[2],markersize=5)
plt.plot((Y_values)**2,m1*(Y_values)**2 + 1)
plt.xlabel("$Y^2$"); plt.ylabel(r"$\left(\frac{\nu_{m}}{\nu_p}\right)^2$")
plt.grid()
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.legend(["simulated","fitted"])
plt.tight_layout()
plt.savefig("cut_off_reduction_half.pdf")
plt.show()


mss = []
for i in cut_offs:
    mss.append(np.sum(Y_values**2 * (i[0]/i)**2 - 1)/ np.sum((Y_values**2)**2))
plt.figure()

# plt.plot(Y_values[1:],1.22*zp.dr_min/Res,'k')
plt.plot(thresholds,mss,linestyles[2],markersize=5)
plt.ylabel("$Slopes$"); plt.xlabel(r"$Thresholds$")
plt.grid()
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.legend(["simulated","fitted"])
plt.tight_layout()
plt.savefig("cut_off_reduction_vals.pdf")
plt.show()


#%%

# Compute values for DOF
DOF_poly = []
peaks_cen = []
peaks_tot = []
for i in range(Ys):
    peaks = flat_results_PSF[i][:,0]
    peaks_tot.append(peaks)
    peaks_cen.append(np.max(abs(peaks)))
    DOF_poly.append(DOF_poly_PSFs(z_vals, peaks))
DOF_poly = np.array(DOF_poly)
pc = np.array(peaks_cen)

#Compute values for Peak Intenity
plt.figure()
for i in range(Ys):
    plt.plot(peaks_tot[i]/np.max(peaks_tot[i]))
plt.xlabel("z"); plt.ylabel("intensity")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout();plt.grid()
plt.savefig("z_plots.pdf")
plt.show()



# Plot DOF ratio vs Y
plt.figure()
plt.xlabel(r"$Y$"); plt.ylabel(r"$\frac{\Delta z_{p}}{\Delta z_{m}}$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot(Y_values,DOF_poly / DOF_mono,'kd-')
plt.tight_layout();plt.grid()
plt.savefig("DOF_poly.pdf")
plt.show()

#Plot DOF ratio ^2 vs Y^2
p = np.polyfit((Y_values) ** 2, ((DOF_poly / DOF_mono)) ** 2, 1)
rrr = np.corrcoef((Y_values) ** 2, ((DOF_poly / DOF_mono)) ** 2)
print("DOF fit values:")
print(p)
print(rrr)
plt.figure()
plt.xlabel(r"$Y^2$"); plt.ylabel(r"$\frac{\Delta z_{p}}{\Delta z_{m}}$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot((Y_values) ** 2,((DOF_poly / DOF_mono)) ** 2,'kd-')
plt.plot((Y_values) ** 2, p[0] * (Y_values) ** 2 + p[1])
plt.legend(["Simulated","Fitted"])
plt.tight_layout();plt.grid()
plt.savefig("DOF_poly_square.pdf")
plt.show()



# Plot inverse peak intensity vs Y^2
p= np.polyfit((Y_values) ** 2, pc[0] ** 2 / pc ** 2, 1)
rrr = np.corrcoef((Y_values) ** 2, pc[0] ** 2 / pc ** 2)
print("peak int fit values:")
print(p)
print(rrr)
plt.figure()
plt.xlabel(r"$Y^2$")
plt.ylabel(r"$\left(\frac{I_m}{I_p}\right)^2$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot((Y_values) ** 2, pc[0]**2 / pc ** 2,'kd-')
plt.plot((Y_values) ** 2, p[0] * (Y_values) ** 2 + p[1])
plt.legend(["Simulated","Fitted"])
plt.tight_layout();plt.grid()
plt.savefig("peak_int.pdf")
plt.show()