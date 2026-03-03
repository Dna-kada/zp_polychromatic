import numpy as np
from dataclasses import dataclass, field
from optics.transforms import HT

@dataclass
class ZonePlate:
    r: np.ndarray
    n_zones: int
    lamda: float
    radius: float
    kr: np.ndarray
    
    @property
    def focal_length(self) -> float:
        return (self.radius**2 - (self.n_zones * self.lamda / 2)**2) / (self.n_zones * self.lamda)
    
    @property
    def NA(self) -> float:
        return np.sin( np.arctan(self.radius / self.focal_length))
    
    @property 
    def first_null(self) -> float:
        return 0.61 * self.lamda / self.NA
    
    @property
    def cut_off_freq(self) -> float:
        return 2 * self.NA / self.lamda
    
    @property
    def dr_min(self) -> float:
        return self.lamda / (2 * self.NA)
    
    @property
    def DOF(self) -> float:
        return 0.5 * self.lamda / self.NA ** 2
    
    
    @property
    def profile(self) -> np.ndarray:
        
        h = np.ones(len(self.r), dtype = np.complex128)
        radii = [np.sqrt(j * self.lamda * self.focal_length + 0.25 *\
                         (j * self.lamda) ** 2) for j in range(self.n_zones + 1)]
        
        radii = np.array(radii)
        radii = self.radius * radii / np.max(radii)

        for i in range(1, len(radii), 2):
            in_ring = (np.abs(self.r) >= radii[i-1]) & (np.abs(self.r) <= radii[i])
            h[in_ring] *= -1.0  # amplitude ZP

        
        ap1 = (self.r < radii[-1]) # zero outside of the zone plate 
        # ap2 = (self.r > radii[1]) # central block
        
        return h * ap1 #* ap2
    
    @property
    def rads(self) -> np.ndarray:
        radii = [np.sqrt(j * self.lamda * self.focal_length + 0.25 *\
                         (j * self.lamda) ** 2) for j in range(self.n_zones + 1)]
        
        radii = np.array(radii)
        # radii = self.radius * radii / np.max(radii)
        
        return radii
    
    @property
    def profile_transform(self) -> np.ndarray:
        return HT(self.r,self.profile,self.kr)

@dataclass()
class Spectrum:
    lamda0: float
    Y: float
    n_zones: float
    no_points: float = 19
    dlamda_mult = 3
    
    @property
    def FWHM(self):
        return self.Y * self.lamda0 / self.n_zones
    
    @property
    def dlamda(self) -> float:
        return self.FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    @property
    def spec_range(self) -> float:
        return self.dlamda * self.dlamda_mult
    
    
    @property 
    def spectrum_range(self) -> np.ndarray:
        
        return np.linspace(self.lamda0 - self.spec_range, self.lamda0 + self.spec_range, self.no_points)
    @property 
    def profile(self) -> np.ndarray:
        if self.Y == 0:
            t = 3 / 50 # this value is just used to create a range, it just has to be non-zero
            spec = np.linspace(self.lamda0 - self.lamda0 * t, self.lamda0 + self.lamda0 * t, self.no_points)
            return np.exp( - ((spec - self.lamda0) ** 2) / (2 * (self.lamda0 / 50) ** 2))
        else:
            return np.exp( - ((self.spectrum_range - self.lamda0) ** 2) / (2 * self.dlamda ** 2))


@dataclass
class Spectrum_flc:
    lamda0: float
    FWHM: float
    no_points: float = 15
    dlamda_mult = 3
    
    @property
    def dlamda(self) -> float:
        return self.FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    @property
    def spec_range(self) -> float:
        return self.dlamda * self.dlamda_mult
    
    @property
    def zeta(self) -> float:
        return self.lamda0 / self.FWHM
    
    
    @property 
    def spectrum_range(self) -> np.ndarray:
        return np.linspace(self.lamda0 - self.spec_range, self.lamda0 + self.spec_range, self.no_points)
    @property 
    def profile(self) -> np.ndarray:
        return np.exp( - ((self.spectrum_range - self.lamda0) ** 2) / (2 * self.dlamda ** 2))
        