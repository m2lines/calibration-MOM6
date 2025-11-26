import xarray as xr
import numpy as np
import sys
sys.path.append('../src-double-gyre')
from helpers.computational_tools import compute_isotropic_KE

def variability_metrics(_e, _u, _v, static, Time=slice(1825,3650), coarse_factor=4, compute_e=False, compute_sp=True):
    '''
    This function computes metrics of interest which we would like to optimize:
    * STD of interfaces
    * Isotropic EKE spectrum

    Note: we could compute EKE spectrum from interface itself,
    however, this does not translate to the subsurface layer
    
    Note: we use isotopic metrics to do not give directional information
    to the optimized ANN as it may overfit to predict zonal flows
    '''
    ds = xr.Dataset()
    if compute_sp:
        u = _u.sel(Time=Time)
        v = _v.sel(Time=Time)

        u_prime = u - u.mean('Time')
        v_prime = v - v.mean('Time')

        EKE_spectrum = compute_isotropic_KE(u_prime, v_prime, static.dxT, static.dyT).mean('Time').compute()

        ds['EKE_spectrum'] = EKE_spectrum

    if compute_e:
        e = _e.sel(Time=Time)
        if len(e.Time)==0:
            raise ValueError("Time slice results in empty dataset. Please check the Time slice provided.")
        e_std = e.std('Time').coarsen({'xh':coarse_factor, 'yh':coarse_factor}, boundary='trim').mean().compute()
        e_mean = e.mean('Time').coarsen({'xh':coarse_factor, 'yh':coarse_factor}, boundary='trim').mean().compute()
        ds['e_std'] = e_std.isel(zi=slice(0,2))
        ds['e_mean'] = e_mean.isel(zi=slice(0,2))

    return ds