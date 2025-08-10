from optics.params import ZonePlate, Spectrum
from optics.transforms import HT,iHT
from optics.propagation import prop
import numpy as np

class Simulation:
    
    def __init__(self,
                 zp: ZonePlate,
                 spec: Spectrum,
                 N: int,
                 field_fraction: float,
                 fac: int,
                 scale_real: float,
                 cut_off: float):
        """
        

        Parameters
        ----------
        zp : ZonePlate
            The zone plate for the simulation, it doesn't change between
            simulations.
        spec : Spectrum
            The spectrum here modelled from the Gaussian profile.
        N : int
            number of sampling points used.
        
        fac: int
            To reduce the computation time, the number of point in the
            resultant field may be reduced.
        Returns
        -------
        None.

        """
        self.zp = zp
        self.spec = spec
        self.N = N
        self.fac = fac
        self.field_fraction = field_fraction
        self.scale_real = scale_real
        self.dr = self.zp.r[1] - self.zp.r[0]
        self.cut_off = cut_off
        self.prepare_reduced_grids() # 
    
    # these grids cut down computation and focus region
    def prepare_reduced_grids(self):
        self.N_reduced = self.N // self.fac
        self.L_reduced = self.zp.radius / self.field_fraction / self.scale_real
        self.r_reduced = np.linspace(0,self.L_reduced, self.N_reduced)
        self.dr_reduced = self.L_reduced / (self.N_reduced)
        # self.fr_reduced = np.linspace(0, 1 / (2 * self.dr_reduced), self.N_reduced) / self.scale_kr
        self.fr_reduced = np.linspace(0, 2.2 * self.cut_off, self.N_reduced)
        self.kr_reduced = 2 * np.pi * self.fr_reduced
    
    def compute_psf(self,
                    z_offset: float) -> np.ndarray:
        
        psf_summed = np.zeros(self.N_reduced,dtype=np.complex128)
        
        for i in range(len(self.spec.spectrum_range)):
            propagated_field = prop(self.zp.profile_transform,z_offset,self.spec.spectrum_range[i],self.dr,self.zp.kr)
            psf_single = iHT(self.zp.r / self.scale_real,propagated_field,self.zp.kr,self.fac)
            psf_summed +=  self.spec.profile[i] * abs(psf_single) ** 2
            
        return psf_summed
    
    
        
    
    def compute_MTF(self,
                    z_vals):
        results_MTF = []
        results_PSF = []
        for z in z_vals:
            psf = self.compute_psf(z)
            results_PSF.append(psf)
            MTF = abs(HT(self.r_reduced,psf,self.kr_reduced))
            results_MTF.append(MTF/np.max(MTF))
        
        return results_MTF, results_PSF