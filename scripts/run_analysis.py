#!/usr/bin/env python
from mpi4py import MPI
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

def main():
    """
    description:
        The computation first defines the field, the zone plate, and the 
        simulation that prdouces the data. The sampling is quite high at first
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
    n_zones = 900 # number of zones for the zone plate
    #the res factor could be higher but it's more than adequate at this number.
    resolution_factor = 25 # this is simply a rule of thumb
    N = n_zones*resolution_factor # number of samples
    #the quantity being invsetigated
    Y_values = np.array([0, 0.8, 1., 1.25, 1.5, 1.875, 2., 2.5, 3., 3.5, 4., 4.5, 5. ])
    
    mod_val = [0.05, 0.1, 0.15] # modulation cut-offs to measure
    
    #fraction of field taken up by zone plate
    # 1 - field_fraction gives fraction of array that is zero padding
    field_fraction = 0.3 
    scale = 20 # scale down the physical space for producing PSF
    
    zp_radius = 126e-6 / 2 # radius of zone plate
    L = zp_radius / field_fraction # total length of field
    r = np.linspace(0, L, N) # real space array
    dr = L/N # increment in real space
    fr = np.linspace(0, 1 / (2 * dr), N) # frequency space array
    kr = 2 * np.pi * fr # angular frequency space
    
    reduction_factor = int(3) # reduces the no. of samples in computed array
    # this reduces the number of computed points in the resulting field. 
    
    
    # Only rank 0 constructs the ZonePlate and spectra
    if rank == 0:
        zp = ZonePlate(r, n_zones, lamda, zp_radius, kr) # generate FZP
        print(f"Zone plate focal length:{zp.focal_length}")
        print(f"Zone plate dr min:{zp.dr_min}")
        print(f"Zone plate DOF:{zp.DOF}")
    else:
        zp = None
        spectra = None
    
    # Broadcast to all ranks so everyone has the same objects
    zp = comm.bcast(zp, root=0) # pass FZP to other cores
    
    delta_z = (0.5 * lamda / (zp.NA ** 2)) # factor for zones 
    DOF_mono = delta_z # monochromatic DOF
    mult = 4.5 # multiple of 'DOF_mono's to measure over
    DOF_points = 19 # samples for DOF
    focal_lower = zp.focal_length - mult * delta_z
    focal_upper = zp.focal_length + mult * delta_z
    # define the range on the z-axis 
    z_vals = np.linspace(focal_lower, focal_upper, DOF_points)
    
    # Run simulation on each CPU core
    chunks = np.array_split(Y_values, size)
    Y_local = chunks[rank]
    
    
    #arrays to collect results that may come from mulitple Y values per core
    res_local_MTF = []
    res_local_PSF = []
    for y in Y_local:
        spec = Spectrum(lamda, y, n_zones)
        sim = Simulation(zp,
                         spec,
                         N,
                         field_fraction,
                         reduction_factor,
                         scale,
                         zp.cut_off_freq)
        
        # Define the full set of axial positions and partition among ranks
        
        # Each rank computes its subset
        results_local_MTF, results_local_PSF = sim.compute_MTF(z_vals)#sim.compute_MTF(z_vals)
        
        res_local_MTF.append(results_local_MTF)
        res_local_PSF.append(results_local_PSF)
        np.savez_compressed(f"results_MTF_{y:.2f}_nz_{n_zones}",a = sim.kr_reduced, b = results_local_MTF, c = y)
        np.savez_compressed(f"results_PSF_{y:.2f}_nz_{n_zones}",a = sim.kr_reduced, b = results_local_PSF, c = y)
        
        
    all_results_MTF = comm.gather(res_local_MTF, root = 0)
    all_results_PSF = comm.gather(res_local_PSF, root = 0)
    if rank == 0:
        flat_results_MTF = np.array([item for sublist in all_results_MTF for item in sublist])
        flat_results_PSF = np.array([item for sublist in all_results_PSF for item in sublist])
        
        configure_plotting()
        
        # load combined results from MPI gather...
        freq = sim.fr_reduced / zp.cut_off_freq
        DOF_mid = len(z_vals) // 2
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
        
        
        
        cut_offs = find_cutoff(sim.fr_reduced, inf_curves,thresholds).T
        
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
        
        

if __name__ == '__main__':
    main()