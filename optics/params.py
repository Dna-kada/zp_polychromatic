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
        return self.radius ** 2 / (self.n_zones * self.lamda)
    
    @property 
    def first_null(self) -> float:
        return self.radius / self.n_zones 
    
    @property
    def cut_off_freq(self) -> float:
        return 1/self.first_null
    
    @property
    def NA(self) -> float:
        return np.sin( np.arctan(self.radius / self.focal_length))
    
    @property
    def dr_min(self) -> float:
        return self.lamda / (2 * self.NA)
    
    
    @property
    def DOF(self) -> float:
        return self.lamda / self.NA ** 2
    
    
    @property
    def profile(self) -> np.ndarray:
        
        h = np.ones(len(self.r), dtype = np.complex128)
        radii = [np.sqrt(j * self.lamda * self.focal_length + 0.25 *\
                         (j * self.lamda) ** 2) for j in range(self.n_zones + 1)]
        
        radii = np.array(radii)
        radii = self.radius * radii / np.max(radii)

        for i in range(0, len(radii), 2):
            for q in range(len(self.r)):
                if radii[i - 1] <= abs(self.r[q]) <= radii[i]:
                    h[q] *= np.exp(-1j * np.pi)
        
        ap1 = (self.r < radii[-1]) # zero outside of the zone plate 
        ap2 = (self.r > radii[1]) # central block
        
        return h * ap1 * ap2
    
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

@dataclass(slots=True)
class Spectrum:
    lamda0: float
    Y: float
    n_zones: float
    no_points: float = 15
    
    @property
    def FWHM(self):
        return self.Y * self.lamda0 / self.n_zones
    
    @property
    def dlamda(self) -> float:
        return self.FWHM / 2.3548
    
    @property
    def spec_range(self) -> float:
        return self.dlamda * 3
    
    @property
    def zeta(self) -> float:
        if self.FWHM == 0:
            return "monochromatic"
        else:
            return self.lamda0 / self.FWHM
    
    
    @property 
    def spectrum_range(self) -> np.ndarray:
        return np.linspace(self.lamda0 - self.spec_range, self.lamda0 + self.spec_range, self.no_points)
    @property 
    def profile(self) -> np.ndarray:
        if self.zeta == "monochromatic":
            return np.ones(len(self.spectrum_range))
        else:
            return np.exp( - ((self.spectrum_range - self.lamda0) ** 2) / (2 * self.dlamda ** 2))
        