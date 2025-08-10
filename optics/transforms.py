import numpy as np
from scipy.integrate import trapezoid
from scipy.special import j0

def HT(r: np.ndarray,
       f: np.ndarray,
       kr: np.ndarray) -> np.ndarray:
    """
    
    
    Parameters
    ----------
    r : np.ndarray
        Regular space array
    f : np.ndarray
        the function to be trasnformed, f(r)
    kr : np.ndarray
        Frequency space array
        
    Returns
    -------
    ht : np.ndarray
        The Hankel transfrom of f(r)
    
    """
    
    dr = r[1] - r[0]
    r_c = r * np.exp(0j)
    N = len(r)
    ht = np.zeros(N, dtype = np.complex128) # records result
    for i in range(N):
        jav = j0(kr[i] * r) * np.exp(0j) # Bessel function
        ht[i] = trapezoid(f * jav * r_c, dx = dr) # integral
    return ht

def iHT(r: np.ndarray,
       f: np.ndarray,
       kr: np.ndarray,
       fac: int) -> np.ndarray:
    """
    
    
    Parameters
    ----------
    r : np.ndarray
        Regular space array
    f : np.ndarray
        the function to be trasnformed, H{f(r)}
    kr : np.ndarray
        Frequency space array
    fac : int
        Factor reducing the number of points computed in the array
    
    Returns
    -------
    iht : np.ndarray
        The inverse Hankel trasform f(r)
        
    """
    kr_c = kr * np.exp(0j)
    N = len(r)
    dkr = kr[1] - kr[0]
    iht = np.zeros(N // fac, dtype=np.complex128)
    for i in range(N // fac):
        jav = j0(kr * r[i * fac]) * np.exp(0j)
        iht[i] = trapezoid(f * jav * kr_c, dx = dkr)
    return iht