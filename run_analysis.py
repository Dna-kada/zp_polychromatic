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

def sort(n,t):
    """
    n - number of processors
    t - number of tasks
    Sort the number of tasks.
    """
    td = t//n # the number of tasks per processor
    r = t%n # should be the leftover tasks
    l = td*np.ones(n) # make an array for the allegation of tasks
    for i in range(r): # for loop to add the extra tasks to some processors
        l[i] = l[i] + 1
    return l

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
    n_zones = 70 # number of zones for the zone plate
    #the res factor could be higher but it's more than adequate at this number.
    resolution_factor = 25 # this is simply a rule of thumb
    N = n_zones*resolution_factor # number of samples
    #the quantity being invsetigated
    Y_values = np.array([0,0.8,*np.arange(1,6.5,0.5)])
    # Y = 0 is treated as monochromatic, but in theory undefined
    
    mod_val = [0.05, 0.1, 0.15] # modulation cut-offs to measure
    
    #fraction of field taken up by zone plate
    # 1 - field_fraction gives fraction of array that is zero padding
    field_fraction = 0.3 
    scale = 20 # scale down the physical space for producing PSF
    
    zp_radius = 126e-6 #/ 2 # radius of zone plate
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
    else:
        zp = None
        spectra = None
    
    # Broadcast to all ranks so everyone has the same objects
    zp = comm.bcast(zp, root=0) # pass FZP to other cores
    
    delta_z = (0.5 * lamda / (zp.NA ** 2)) # factor for zones 
    DOF_mono = 2 * delta_z # monochromatic DOF
    mult = 4 # multiple of 'DOF_mono's to measure over
    DOF_points = 17 # samples for DOF
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
        spec = Spectrum(lamda,y,n_zones)
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
        print(np.shape(flat_results_PSF))
        
        
        configure_plotting()
        
        # load combined results from MPI gather...
        freq = sim.fr_reduced / zp.cut_off_freq
        DOF_mid = len(z_vals)//2
        Ys = len(Y_values)
        
        # select a few Y values
        sel = np.array([0,2,5,9,12],dtype = np.int16)
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
        linestyles = ["k-", "k:", "k--", "k-."]
        plot_mtf_curves(freq, oof_curves[sel], Y_values[sel],
                        "oof_MTF.png", linestyles)
        plot_mtf_curves(freq, inf_curves[sel], Y_values[sel],
                        "MTF_in_focus.png", linestyles)
        
        
        
        cut_offs = find_cutoff(sim.fr_reduced, inf_curves,thresholds).T
        
        #%%
        Res = 0.61*np.sqrt(zp.radius * 2 *zp.dr_min*Y_values[1:]/n_zones)
        l = ['kd-','k-.','k--']
        plt.figure()
        plt.plot(Y_values[1:],1.22*zp.dr_min/Res,'k')
        for v in [1]: 
            plt.plot(Y_values,cut_offs[v]/cut_offs[v][0],linestyles[v],markersize=5)
        plt.xlabel("Y"); plt.ylabel(r"$f/f_{mono}$")
        plt.ylim(0.4,1.05); plt.xlim(1,6)
        plt.grid()
        plt.legend(["Analytic","5%","10%","15%"])
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.tight_layout()
        plt.savefig("const_r_demonstration.png")
        plt.show()
        
        
        #%%
        
        bw = n_zones
        NZ = bw*Y_values
        val_mono = cut_offs[1][0]
        mono_curve = Y_values# * val_mono
        
        labels = ['monochromatic','polychromatic']
        # plot_fractions(Y_values, 
        #                [cut_offs[1]/val_mono],
        #                [],
        #                r"$f/f_{Y \rightarrow 0}$",
        #                "cut_off_reduction.png",
        #                linestyles[:2])
        plot_fractions(Y_values,
                       [mono_curve,Y_values*cut_offs[1]/val_mono],
                       r"$f/f_{Y = 1}$",
                       labels,
                       "increasing_N_res.png",
                       linestyles[:2])
        
        #%%
        
        
        DOF_poly = []
        for i in range(Ys):
            peaks = flat_results_PSF[i][:,0]
            DOF_poly.append(DOF_poly_PSFs(z_vals, peaks))
        DOF_poly = np.array(DOF_poly)
        
        plot_fractions(Y_values, 
                       [DOF_poly/DOF_mono],
                       r"$\frac{DOF_{poly}}{DOF_{mono}}$",
                       [],
                       "DOF_poly.png",
                       linestyles[:2])
        # plot_fractions(Y_values, 
        #                [Y_values,Y_values*DOF_mono/DOF_poly],
        #                r"$\frac{DOF(Y=1)}{DOF(Y)}$",
        #                labels,
        #                "increasing_N_DOF.png",
        #                linestyles[:2])
        
        plt.figure()
        plt.xlabel("Y")
        plt.ylabel(r"$\frac{DOF(Y=1)}{DOF(Y)}$")
        inv_DOF = NZ/(2 * zp.radius*np.sqrt(8*lamda))
        inv_DOF_norm = NZ[2]/(2 * zp.radius*np.sqrt(8*lamda))
        plt.plot(Y_values,inv_DOF/inv_DOF_norm,'k') 
        plt.plot(Y_values,(DOF_mono/DOF_poly)*inv_DOF/inv_DOF_norm,'k-.')
        plt.legend(["monochromatic","polychromatic"])
        plt.grid()
        plt.rc("xtick", direction="in", top=True)
        plt.rc("ytick", direction="in", right=True)
        plt.tight_layout()
        plt.savefig("increasing_DOF_inv.png")
        plt.show()
        
        


if __name__ == '__main__':
    main()