#!/usr/bin/env python
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

def main():
    """
    description:
        The computation first defines the field, the zone plate, and the 
        simulation that produces the data. The sampling is quite high at first
        and then is reduced to reduce the run time. The largest part of the run
        time is using a very basic method of computing the Hankel transform
        which can't be vectorised for large number of zones (memory issues)
        and any fast Hankel transforms tend to produce some errors when trying
        to compute for zone plates.
        The reduction factors are present to reduce the size of the field
        where possible. In that respect the code is quite flexible.

    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
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
    #%%
    # Only rank 0 constructs the ZonePlate and spectra
    if rank == 0:
        spectra = Spectrum_flc(lamda, FWHM)
    else:
        spectra = None
    
    
    
    # Run simulation on each CPU core
    chunks = np.array_split(Y_values, size)
    Y_local = chunks[rank]
    
    
    #arrays to collect results that may come from mulitple Y values per core
    res_local_MTF = []
    res_local_PSF = []
    res_local_fr = []
    z_vals_local = []
    
    for y in Y_local:
        n_zones = int(y * zeta) # number of zones for the zone plate
        zp_radius = np.sqrt(focal_length * n_zones * lamda)
        L = zp_radius / field_fraction
        #the res factor could be higher but it's more than adequate at this number.
        r = np.linspace(0, L, N) # real space array
        
        fr = np.linspace(0, 1 / (2 * dr), N) # frequency space array
        kr = 2 * np.pi * fr # angular frequency space
        
        zp = ZonePlate(r, n_zones, lamda, zp_radius, kr) # pass FZP to other cores
        
        
        delta_z = (0.5 * lamda / (zp.NA ** 2)) # factor for zones 
        DOF_mono = delta_z # monochromatic DOF
        mult = 4.5 # multiple of 'DOF_mono's to measure over
        DOF_points = 17 # samples for DOF
        focal_lower = zp.focal_length - mult * delta_z
        focal_upper = zp.focal_length + mult * delta_z
        # define the range on the z-axis 
        z_vals = np.linspace(focal_lower, focal_upper, DOF_points)
        z_vals_local.append(z_vals)
        
        spec = comm.bcast(spectra, root=0)
        sim = Simulation(zp,
                         spec,
                         N,
                         field_fraction,
                         reduction_factor,
                         scale,
                         cut_off_freq_max)
        
        
        # Define the full set of axial positions and partition among ranks
        
        # Each rank computes its subset
        results_local_MTF, results_local_PSF = sim.compute_MTF(z_vals)#sim.compute_MTF(z_vals)
        res_local_fr.append(sim.fr_reduced)
        res_local_MTF.append(results_local_MTF)
        res_local_PSF.append(results_local_PSF)
        np.savez_compressed(f"results_MTF_const_foc_{y:.2f}_nz_{n_zones}",a = sim.kr_reduced, b = results_local_MTF, c = y)
        np.savez_compressed(f"results_PSF_const_foc_{y:.2f}_nz_{n_zones}",a = sim.kr_reduced, b = results_local_PSF, c = y)
    
    
    
    
    
    
    
    z_total = comm.gather(z_vals_local, root = 0)
    all_results_freq = comm.gather(res_local_fr, root = 0)
    all_results_MTF = comm.gather(res_local_MTF, root = 0)
    all_results_PSF = comm.gather(res_local_PSF, root = 0)
    
    if rank == 0:
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
        
        
        # plotting
        linestyles = ["k-", "k:", "k--", "k-.",(0, (3, 5, 1, 5)), (0, (5, 5)), (0, (3, 5, 1, 5, 1, 5))]
        plot_mtf_curves_flc(freq, oof_curves[sel], Y_values[sel],
                        "oof_MTF_const_foc.pdf", linestyles)
        
        plt.figure()
        plt.plot(freq)
        plot_mtf_curves_flc(freq, inf_curves[sel], Y_values[sel],
                        "MTF_in_focus_const_foc.pdf", linestyles)
        
        
        
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
        plt.xlabel("Y"); plt.ylabel(r"$\nu/\nu_{m}$")
        # plt.ylim(0.4,1.05); plt.xlim(1,6)
        plt.grid()
        plt.legend(['Monochromatic',"10%"])
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.tight_layout()
        plt.savefig("const_r_demonstration_const_foc.pdf")
        # plt.show()
        
        
        #%%
        
        bw = n_zones
        NZ = bw*Y_values
        val_mono = cut_offs[1][0]
        mono_curve = Y_values# * val_mono
        
        labels = ['monochromatic', 'polychromatic']
        plot_fractions(Y_values, 
                       [cut_offs[1] / cut_offs[1][7]],
                       r"$\nu/\nu_{Y = 3.3}$",
                       [],
                       "cut_off_reduction_const_foc.pdf",
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
        plt.xlabel("z"); plt.ylabel("intensity")
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.tight_layout()
        plt.savefig("z_plots_const_foc.pdf")
        
        plt.figure()
        plt.xlabel(r"$Y^2$")
        plt.ylabel(r"$(\frac{\Delta z_{p}}{\Delta z_{m}})^2$")
        plt.plot(Y_values ** 2, (DOF_poly / DOF_mono) ** 2,'k') 
        plt.grid()
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.tight_layout()
        plt.savefig("DOF_poly_square_const_foc.pdf")
        
        plot_fractions(Y_values, 
                       [DOF_poly / (focal_length / ( 3.52 * zeta))],
                       r"$\frac{\Delta z_p}{f/(3.52 \, \zeta)}$",
                       [],
                       "DOF_poly_const_foc.pdf",
                       linestyles[:2])
        
        
        plt.figure()
        plt.xlabel("Y"); plt.ylabel(r"$\frac{I_m}{I_p}$")
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.plot(Y_values ** 2, pc[0] / pc)
        plt.tight_layout()
        plt.savefig("peak_int_const_foc_norm.pdf")
        
        plt.figure()
        plt.xlabel(r"$Y^2$"); plt.ylabel(r"$\left(\frac{I_m}{I_p}\right)^2$")
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.plot(Y_values ** 2, pc[0] ** 2 / pc ** 2)
        plt.tight_layout()
        plt.savefig("peak_int_const_square_foc_norm.pdf")
        
        
        
        plt.figure()
        plt.xlabel("Y"); plt.ylabel(r"$\frac{I_p}{(\zeta / f)}$")
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.plot(Y_values, pc * focal_length / zeta, 'kd-')
        plt.tight_layout(); plt.grid()
        plt.savefig("peak_int_const_foc.pdf")
        plt.show()
        


if __name__ == '__main__':
    main()