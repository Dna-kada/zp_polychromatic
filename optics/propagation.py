import numpy as np

def prop(HTr: np.ndarray,
        z: float,
        lamda: float,
        dr: float,
        kr: np.ndarray) -> np.ndarray:
    
    k = 2 * np.pi / lamda # wavenumber
    mask = kr <= k
    exp_fac = np.zeros_like(kr, dtype = np.complex128)
    exp_fac[mask] = np.exp(1j * z * np.sqrt(k ** 2 - kr[mask] ** 2))
    exp_fac[~mask] = 0
    return  HTr * exp_fac