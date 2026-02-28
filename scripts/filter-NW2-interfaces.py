import sys

import xarray as xr
import matplotlib.pyplot as plt
from xgcm import Grid
import numpy as np
from xgcm.grid_ufunc import as_grid_ufunc
import gcm_filters
import argparse
from dask.diagnostics import ProgressBar
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--time_idx", type=int, default=0)
parser.add_argument("--zl_idx", type=int, default=0)
args = parser.parse_args()

print(args)

FOLDER = 'R2_FGR3'
COARSENING_FACTOR = 16
FILTER_SCALE = COARSENING_FACTOR * 3
lores_static = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R2-long/bare/output/static.nc', decode_times=False)

file_zi = f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/{FOLDER}/interfaces/time_{args.time_idx}_zl_{args.zl_idx}.nc'

if os.path.exists(file_zi):
    print('Files already exist. Skip')
    sys.exit()

# Here we create arrays for V_{i+1,j} and V_{i,j+1}
@as_grid_ufunc(signature="(X:center)->(X:outer)", boundary_width={'X':(1,1)})
def index_plus_one(a):
    return a[...,1:]

# Central value also should be given on a grid of outer values.
@as_grid_ufunc(signature="(X:center)->(X:outer)", boundary_width={'X':(1,0)})
def index_center(a):
    return a[...,:]
    
class GMFilter():
  '''
  Filter which uses Gent-McWilliams parameterization
  to perform filtering of layer interfaces
  '''
  def __init__(self, static, lower_interface):
    '''
    Required fields in static:
    wet_u, wet_v
    wet
    '''
    self.static = static.fillna(0.)
    self.lower_interface = lower_interface.fillna(0.)

    self.grid = Grid(self.static, coords={
            'X': {'center': 'xh', 'outer': 'xq'},
            'Y': {'center': 'yh', 'outer': 'yq'}
            },
            boundary={'X': 'periodic', 'Y': 'fill'},
            fill_value = {'Y': 0})

  def limit_fluxes(self, eta, Fu, Fv):
    '''
    This function assumes that perform
    one time step with layer interfaces as follows:
    eta_new = eta + (diff(Fu,'X') + diff(Fv,'Y')) / area_t

    And we want to be sure that after time step interfaces do
    not drop below the bathymetry, that is:
    eta_new >= bathymetry for all layers
    as long as interfaces were above bathymetry before diffusion
    if  eta >= bathymetry for all layers

    We introduce the volume enclosed into the water column
    below the interface:
    V = (eta-bathymetry) * area_t
    Then the limited fluxes should satisfy:
    diff(Fu,'X') + diff(Fv,'Y') + V >= 0.

    We assume that 4 fluxes on edges can work IN CONCERT (see Zalesak 1979 for definition),
    but we want to limit them independently
    Thus, we require (see divisor 4):
    Fu_{i+1/2}>0 -> -Fu_{i+1/2} + V_{i+1}/4 >=0
    Fu_{i+1/2}<0 -> +Fu_{i+1/2} + V_{i}/4   >=0

    Fv_{j+1/2}>0 -> -Fv_{j+1/2} + V_{j+1}/4 >=0
    Fu_{j+1/2}<0 -> +Fv_{j+1/2} + V_{j}/4   >=0
    '''

    static = self.static
    grid = self.grid

    # Compute volume below interface
    V = (eta - self.lower_interface) * static.wet

    # See explanation of dask="parallelized" in
    # https://xgcm.readthedocs.io/en/latest/grid_ufuncs.html
    V_right = index_plus_one(grid, V, axis='X', dask="parallelized")
    V_top   = index_plus_one(grid, V, axis='Y', dask="parallelized")

    Vu_center = index_center(grid, V, axis='X', dask="parallelized")
    Vv_center = index_center(grid, V, axis='Y', dask="parallelized")

    Fu = xr.where(Fu>0, np.minimum(Fu, V_right * 0.25), np.maximum(Fu, -Vu_center * 0.25))
    Fv = xr.where(Fv>0, np.minimum(Fv, V_top   * 0.25), np.maximum(Fv, -Vv_center * 0.25))

    return Fu, Fv

  def diffusion_fixed_factor(self, eta, filter_scale=4, limit_fluxes=True):
    '''
    This function computes diffusivity fluxes (Fu,Fv)
    in finite volume formulation and performs time stepping as
    follows:
    eta = eta + (diff(Fu * kappa_u,'X') + diff(Fv * kappa_v,'Y')) / area_t

    where kappa_u and kappa_v have physical meaning of:
    kappa = diffusivity coefficient x time spacing

    and chosen as much as possible (while stable)
    '''

    static = self.static
    grid = self.grid

    # Niters * kappa_max = filter_scale**2 / 24
    # Here we assume that kappa_max=0.25 on uniform grid
    kappa_max=0.25
    Niters = filter_scale**2 / 24. / kappa_max
    Niters = int(np.ceil(Niters + 0.01)) # we add 0.01 to make sure that resulting kappa is less than 0.25 in all cases

    kappa = filter_scale**2 / 24. / Niters

    etaf = eta.fillna(0.).copy(deep=True)

    print('Number of iterations of filter:', Niters)
    print('Kappa', kappa, 'must be strictly less than 0.25')

    for iter in range(Niters):
      # Fu_{i+1/2} = eta_i+1 - eta_i
      Fu = grid.diff(etaf,'X') * static.wet_u
      Fv = grid.diff(etaf,'Y') * static.wet_v

      # Multiply by diffusivity
      Fu = Fu * kappa
      Fv = Fv * kappa
      if limit_fluxes:
        Fu, Fv = self.limit_fluxes(etaf, Fu, Fv)
      # If kappa was 0.25, it will be
      # eta_{ij} = 0.25 * (eta_i+1 + eta_i-1 + eta_j+1 + eta_j-1)
      etaf = (etaf + grid.diff(Fu, 'X') + grid.diff(Fv, 'Y')) * static.wet

    return xr.where(np.isnan(eta),np.nan,etaf)

print('Reading data...')
with ProgressBar():
    ref = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R32/snapshots_*', decode_times=False, chunks={'time':1, 'zl':1, 'zi':1}).e.isel(time=args.time_idx)
    ref_static = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R32/static.nc', decode_times=False).squeeze().drop_vars('time')

lower_interface = ref.isel(zi=-1).squeeze().drop_vars(['time','zi'])
gmfilter = GMFilter(ref_static, lower_interface)

# Filter scale of 24 is 32/4 * 3
print('Filtering interfaces')
with ProgressBar():
    ef = gmfilter.diffusion_fixed_factor(ref.isel(zi=args.zl_idx),filter_scale=FILTER_SCALE,limit_fluxes=True).compute()

print('Coarsening...; Here we simply interpolate to prevent deviation from real topography')
ds_coarse = xr.Dataset()
with ProgressBar():
    ds_coarse['ef'] = ef.interp(yh=lores_static.yh, xh=lores_static.xh).compute()#.transpose('zi',...)
    #hf = -np.diff(ds_coarse['ef'].values,axis=0) # minus because vertical indexing is downward
    #ds_coarse['hf'] = xr.DataArray(hf, dims=['zl', 'yh', 'xh'])
    
if not(os.path.exists(file_zi)):
    print(f'Saving to {file_zi}')
    ds_coarse.astype('float32').to_netcdf(file_zi)

print(f'Script is done')