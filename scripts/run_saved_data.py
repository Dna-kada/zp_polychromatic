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

flat_results_MTF = [item for sublist in all_results_MTF for item in sublist]
flat_results_PSF = [item for sublist in all_results_PSF for item in sublist]



configure_plotting()

# load combined results from MPI gather...
freq = all_results_freq
DOF_mid = len(z_vals)//2
Ys = len(Y_values)

# select a few Y values
sel = np.array([0,2,5,7,12],dtype = np.int16)
thresholds = [0.05, 0.10, 0.15]
# out-of-focus curves
oof_curves = np.array([flat_results_MTF[i][0] for i in range(Ys)])
# in-focus curves
inf_curves = np.array([flat_results_MTF[i][DOF_mid] for i in range(Ys)])

inf_PSF =  np.array([flat_results_PSF[i][DOF_mid] for i in range(Ys)])

# save data
np.savez("mtf_data.npz", freq=freq, Y=Y_values,
         oof=np.stack(oof_curves), inf=np.stack(inf_curves))

# plotting
linestyles = ["k-", "k:", "k--", "k-.","k-d","k-v","k-s","k-s"]
plot_mtf_curves(freq[sel], oof_curves[sel], Y_values[sel],
                r"J:\PhD resources\Thesis\images\oof_MTF_const_foc.png", linestyles)
plot_mtf_curves(freq[sel], inf_curves[sel], Y_values[sel],
                r"J:\PhD resources\Thesis\images\MTF_in_focus_const_foc.png", linestyles)
plot_mtf_curves(freq[sel], Y_values[sel][:, None]*inf_curves[sel], Y_values[sel],
                r"J:\PhD resources\Thesis\images\MTF_in_focus_y_yscale_const_foc.png", linestyles)

fig, ax = plt.subplots()
ax.set_xlabel(r"$f/f_{\mathrm{cutoff}}$")
ax.set_ylabel("modulation")
# ax.set_xlim(0, 2.2)

i = 0
for curve, ls in zip(inf_curves[sel], linestyles):
    ax.plot(freq[sel]*Y_values[sel[i]], curve, ls,markersize= 1)
    i += 1
labels = [f"{y:.2f}" for y in Y_values]
ax.legend(labels, title="Y values", loc="upper right")
ax.grid(True)
fig.tight_layout()
plt.show()
fig.savefig(r"J:\PhD resources\Thesis\images\MTF_in_focus_y_xscale_const_foc.png")



cut_offs = find_cutoff(sim.fr_reduced, inf_curves,thresholds).T


plt.figure()
plt.plot()
#%%
Res = 0.61*np.sqrt(zp.radius * 2 *zp.dr_min*Y_values[1:]/n_zones)
plt.figure()
plt.plot(Y_values[1:],1.22*zp.dr_min/Res,'k')
for v in [1]: 
    plt.plot(Y_values,cut_offs[v]/cut_offs[v][0],linestyles[v],markersize=5)
plt.xlabel("Y"); plt.ylabel(r"$f/f_{mono}$")
plt.ylim(0.4,1.05); plt.xlim(1,6)
plt.grid()
plt.legend(["Analytic","10%","15%"])
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig(r"J:\PhD resources\Thesis\images\const_r_demonstration_const_foc.png")
plt.show()


#%%

bw = n_zones
NZ = bw*Y_values
val_mono = cut_offs[1][0]
mono_curve = Y_values# * val_mono

labels = ['monochromatic','polychromatic']
plot_fractions(Y_values, 
               [cut_offs[1]/val_mono],
               r"$f/f_{Y \rightarrow 0}$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_const_foc.png",
               linestyles[:2])
plot_fractions(Y_values**2, 
               [(cut_offs[0]/val_mono)**2],
               r"$(f/f_{Y \rightarrow 0})^2$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_line_const_foc.png",
               linestyles[:2])
plot_fractions((Y_values)**2, 
               [(val_mono/cut_offs[0])**2,(val_mono/cut_offs[1])**2,(val_mono/cut_offs[2])**2],
               r"$f/f_{Y \rightarrow 0}$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_few_curves_const_foc.png",
               linestyles[:2])
plot_fractions((Y_values)**2, 
               [(val_mono/cut_offs[1])**2],
               r"$f/f_{Y \rightarrow 0}$",
               [],
               r"J:\PhD resources\Thesis\images\cut_off_reduction_few_curves_const_foc.png",
               linestyles[:2])
plot_fractions(Y_values,
               [mono_curve,Y_values*cut_offs[1]/val_mono],
               r"$f/f_{Y = 1}$",
               labels,
               r"J:\PhD resources\Thesis\images\increasing_N_res_const_foc.png",
               linestyles[:2])

#%%


DOF_poly = []
peaks_cen = []
peaks_tot = []
for i in range(Ys):
    peaks = flat_results_PSF[i][:,0]
    peaks_tot.append(peaks)
    peaks_cen.append(np.max(peaks))
    DOF_poly.append(DOF_poly_PSFs(z_vals, peaks))
DOF_poly = np.array(DOF_poly)
pc = np.array(peaks_cen)


plt.figure()
for i in range(Ys):
    plt.plot(peaks_tot[i]/np.max(peaks_tot[i]))
plt.plot(np.arange(0,17),np.exp(-(np.arange(0,17) - 8)**2 / 2*(1/2.75)**2))
plt.plot(np.arange(0,17),np.exp(-(np.arange(0,17) - 8)**2 / 2*(2.75)**2))
plt.xlabel("z"); plt.ylabel("intensity")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig("z_plots.png")

plt.figure()
plt.xlabel("Y"); plt.ylabel("peak intensity")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.plot(Y_values**2,pc[1]**2/pc**2)
plt.tight_layout()


plot_fractions(Y_values, 
               [DOF_poly/DOF_mono],
               r"$\frac{DOF_{poly}}{DOF_{mono}}$",
               [],
               r"J:\PhD resources\Thesis\images\DOF_poly.png",
               linestyles[:2])
plot_fractions(Y_values**2, 
               [(DOF_poly/DOF_mono)**2],
               r"$\frac{DOF_{poly}}{DOF_{mono}}$",
               [],
               r"J:\PhD resources\Thesis\images\DOF_poly_square.png",
               linestyles[:2])
# plot_fractions(Y_values, 
#                [Y_values,Y_values*DOF_mono/DOF_poly],
#                r"$\frac{DOF(Y=1)}{DOF(Y)}$",
#                labels,
#                "increasing_N_DOF.png",
#                linestyles[:2])

# plt.figure()
# plt.xlabel("Y")
# plt.ylabel(r"$\frac{DOF(Y=1)}{DOF(Y)}$")
# inv_DOF = NZ/(2 * zp.radius*np.sqrt(8*lamda))
# inv_DOF_norm = NZ[2]/(2 * zp.radius*np.sqrt(8*lamda))
# plt.plot(Y_values,inv_DOF/inv_DOF_norm,'k') 
# plt.plot(Y_values,(DOF_mono/DOF_poly)*inv_DOF/inv_DOF_norm,'k-.')
# plt.legend(["monochromatic","polychromatic"])
# plt.grid()
# plt.rc("xtick", direction="in", top=True)
# plt.rc("ytick", direction="in", right=True)
# plt.tight_layout()
# plt.savefig(r"J:\PhD resources\Thesis\images\increasing_DOF_inv.png")
# plt.show()
plt.figure()
plt.xlabel("Y")
plt.ylabel(r"$\frac{DOF(Y=1)}{DOF(Y)}$")
inv_DOF = (2*zp.radius)**2 / ( 4 * NZ**2 * lamda)
inv_DOF_norm = (2*zp.radius)**2 / ( 4 * NZ[2]**2 * lamda)
plt.plot(Y_values,inv_DOF/inv_DOF_norm,'k') 
plt.plot(Y_values,(DOF_mono/DOF_poly)*inv_DOF/inv_DOF_norm,'k-.')
plt.legend(["monochromatic","polychromatic"])
plt.grid()
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.tight_layout()
plt.savefig(r"J:\PhD resources\Thesis\images\increasing_DOF_inv.png")
plt.show()