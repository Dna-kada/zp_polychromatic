# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:03:16 2026

@author: Dunka
"""
from mpi4py import MPI
import numpy as np
import os, sys
# add src directory to path so we can import zpxrm package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optics.transforms import HT
from optics.params import ZonePlate, Spectrum_flc
from simulation.simulation_focal_length_const import Simulation
import matplotlib.pyplot as plt
import matplotlib as mpl
from scripts.plotting import configure_plotting, plot_mtf_curves_flc, plot_fractions
from scripts.compute_curves import find_cutoff, DOF_poly_PSFs
from params.par_const_foc import *

flat_results_MTF = np.array([item for sublist in all_results_MTF for item in sublist])
flat_results_PSF = np.array([item for sublist in all_results_PSF for item in sublist])
z_tot_flat = np.array([item for sublist in z_total for item in sublist])


configure_plotting()

# load combined results from MPI gather...
freq = res_local_fr[0]
DOF_mid = len(z_vals) // 2
Ys = len(Y_values)

# select a few Y values
sel = np.array([0, 1, 2, 3, 4, 7, 8], dtype = np.int16)
thresholds = [0.05, 0.10, 0.15]
# out-of-focus curves
oof_curves = np.array([flat_results_MTF[i][0] for i in range(Ys)])
# in-focus curves
inf_curves = np.array([flat_results_MTF[i][DOF_mid] for i in range(Ys)])

inf_PSF =  np.array([flat_results_PSF[i][DOF_mid] for i in range(Ys)])

# save data
# np.savez("mtf_data.npz", freq=freq, Y=Y_values,
         # oof=np.stack(oof_curves), inf=np.stack(inf_curves))

# plotting
linestyles = ["k-", "k:", "k--", "k-.",(0, (3, 5, 1, 5)), (0, (5, 5)), (0, (3, 5, 1, 5, 1, 5))]
plot_mtf_curves_flc(freq, oof_curves[sel], Y_values[sel],
                r"J:\PhD resources\Thesis\images\oof_MTF_const_foc.pdf", linestyles)

plt.figure()
plt.plot(freq)
plot_mtf_curves_flc(freq, inf_curves[sel], Y_values[sel],
                r"J:\PhD resources\Thesis\images\MTF_in_focus_const_foc.pdf", linestyles)

# fig, ax = plt.subplots()
# ax.set_xlabel(r"$f/f_{\mathrm{c}}$")
# ax.set_ylabel("modulation")
# # ax.set_xlim(0, 2.2)

# i = 0
# for curve, ls in zip(inf_curves[sel], linestyles):
#     ax.plot(freq*Y_values[sel[i]], curve, ls,markersize= 4)
#     i += 1
# labels = [f"{y:.2f}" for y in Y_values]
# ax.legend(labels, title="Y values", loc="upper right")
# ax.grid(True)
# fig.tight_layout()
# # plt.show()
# fig.savefig(r"J:\PhD resources\Thesis\images\MTF_in_focus_y_xscale_const_foc.pdf")



cut_offs = find_cutoff(sim.fr_reduced, inf_curves, thresholds).T


plt.figure()
plt.plot()
#%%
# now for the constant focus cut off frequency No. of zones/ zp radius
fco =  (Y_values * zeta) / np.sqrt(focal_length * Y_values * zeta * lamda)
fco_m =  zeta / np.sqrt(focal_length * zeta * lamda)

plt.figure()
plt.plot(Y_values, fco / fco_m)
for v in [1]:
    plt.plot(Y_values, cut_offs[v] / cut_offs[v][0], linestyles[v], markersize = 5)
plt.xlabel("Y"); plt.ylabel(r"$f/f_{m}$")
# plt.ylim(0.4,1.05); plt.xlim(1,6)
plt.grid()
plt.legend(['Monochromatic',"10%"])
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig(r"J:\PhD resources\Thesis\images\const_r_demonstration_const_foc.pdf")
# plt.show()


#%%

bw = n_zones
NZ = bw*Y_values
val_mono = cut_offs[1][0]
mono_curve = Y_values# * val_mono

labels = ['monochromatic', 'polychromatic']
plot_fractions(Y_values, 
               [cut_offs[1] / cut_offs[1][1]],
               r"$f/f_{m}$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_const_foc.pdf",
               linestyles[:2])
plot_fractions(Y_values**2, 
               [(cut_offs[0] / val_mono) ** 2],
               r"$(f/f_{m})^2$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_line_const_foc.pdf",
               linestyles[:2])
plot_fractions((Y_values) ** 2, 
               [(val_mono / cut_offs[0]) ** 2, (val_mono / cut_offs[1]) ** 2,(val_mono / cut_offs[2]) ** 2],
               r"$f_{m}/f$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_few_curves_const_foc.pdf",
               linestyles[:2])
plot_fractions((Y_values)**2, 
               [(val_mono / cut_offs[1]) ** 2],
               r"$f_{m}/f$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_few_curves_const_foc.pdf",
               linestyles[:2])

#%%

n_zones_arr = (Y_values * zeta).astype('i') # number of zones for the zone plate
zp_radius = np.sqrt(focal_length * n_zones_arr * lamda)

# the energy for each is scaled down by reducing the energy by the ratio
# of the areas of the zone plate according to the first simulated zone plate
# (2 * pi * r_0 ** 2) / (2 * pi * r_i **2)
energy_scale = []
for i in range(len(zp_radius)):
    energy_scale.append(zp_radius[0]**2 / zp_radius[i]**2)


DOF_poly = []
peaks_cen = []
peaks_tot = []
for i in range(Ys):
    # preserves total energy on zone plate by applying energy_scale
    peaks = flat_results_PSF[i][:,0] * energy_scale [i]
    peaks_tot.append(peaks)
    peaks_cen.append(np.max(peaks))
    DOF_poly.append(DOF_poly_PSFs(z_tot_flat[i], peaks))
DOF_poly = np.array(DOF_poly)
pc = np.array(peaks_cen)

plt.figure()
for i in range(Ys):
    plt.plot(peaks_tot[i]/np.max(peaks_tot[i]))
plt.plot(np.arange(0, 17), np.exp(-(np.arange(0, 17) - 8) ** 2 / 2*(1 / 2.75)**2))
plt.plot(np.arange(0, 17), np.exp(-(np.arange(0, 17) - 8) ** 2 / 2 * (2.75) ** 2))
plt.xlabel("z"); plt.ylabel("intensity")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig("z_plots_const_foc.png")

plt.figure()
plt.xlabel(r"$Y^2$")
plt.ylabel(r"$(\frac{DOF_{p}}{DOF_{m}})^2$")
plt.plot(Y_values ** 2, (DOF_poly / DOF_mono) ** 2,'k') 
plt.grid()
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig(r"J:\PhD resources\Thesis\images\DOF_poly_square_const_foc.pdf")

plot_fractions(Y_values, 
               [DOF_poly / (focal_length / ( 3.52 * zeta))],
               r"$\frac{DOF_p}{f/(3.52 \, \zeta)}$",
               [],
               r"J:\PhD resources\Thesis\images\DOF_poly_const_foc.pdf",
               linestyles[:2])


plt.figure()
plt.xlabel("Y"); plt.ylabel(r"$\frac{I_m}{I_p}$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot(Y_values ** 2, pc[0] / pc)
plt.tight_layout()
plt.savefig(r"J:\PhD resources\Thesis\images\peak_int_const_foc_norm.pdf")

plt.figure()
plt.xlabel(r"$Y^2$"); plt.ylabel(r"$\left(\frac{I_m}{I_p}\right)^2$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot(Y_values ** 2, pc[0] ** 2 / pc ** 2)
plt.tight_layout()
plt.savefig(r"J:\PhD resources\Thesis\images\peak_int_const_square_foc_norm.pdf")



plt.figure()
plt.xlabel("Y"); plt.ylabel(r"$\frac{I_p}{(\zeta / f)}$")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot(Y_values, pc * focal_length / zeta, 'kd-')
plt.tight_layout(); plt.grid()
plt.savefig(r"J:\PhD resources\Thesis\images\peak_int_const_foc.pdf")
plt.show()