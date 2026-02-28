import sys

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import gcm_filters
import argparse
from dask.diagnostics import ProgressBar
import os
import sys
import xgcm

'''
Run me as follows:
sbatch --array=0-1215 --wrap="/home/pp2681/.local/share/jupyter/kernels/my_env/python filter-R32-interfaces.py --time_idx=\$SLURM_ARRAY_TASK_ID"
'''

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

FGR = 3
inverse_resolution = 2
RR = f'R{inverse_resolution}'
filter_scale = FGR * (32 // inverse_resolution)
PATH_IN = '/scratch/pp2681/mom6/Feb2022/bare/R32/high_frequency/*.nc'
PATH = f'/scratch/pp2681/mom6/Feb2022/filtered/R32_{RR}_FGR{FGR}/high_frequency'

parser = argparse.ArgumentParser()
parser.add_argument("--time_idx", type=int, default=0)
args = parser.parse_args()

print(args)

os.system(f'mkdir -p {PATH}')

file = f'{PATH}/time_{args.time_idx}.nc'

from xgcm import Grid
from xgcm.grid_ufunc import as_grid_ufunc

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
    self.static = static.fillna(0.).pad({'xh':1, 'yh':1, 'xq':1, 'yq':1}, constant_values=0).chunk({'xh':-1, 'yh':-1})
    self.lower_interface = lower_interface.fillna(0.).pad({'xh':1, 'yh':1}, constant_values=0).chunk({'xh':-1, 'yh':-1})

    self.grid = Grid(self.static, coords={
            'X': {'center': 'xh', 'outer': 'xq'},
            'Y': {'center': 'yh', 'outer': 'yq'}
            },
            boundary={'X': 'fill', 'Y': 'fill'},
            fill_value = {'Y': 0})

    wet_u = self.grid.interp(self.static.wet,'X')
    self.static['wet_u'] = xr.where(wet_u<1., 0., 1.)

    wet_v = self.grid.interp(self.static.wet,'Y')
    self.static['wet_v'] = xr.where(wet_v<1., 0., 1.)

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

    etaf = eta.fillna(0.).copy(deep=True).pad({'xh':1, 'yh':1}, constant_values=0)

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

    return xr.where(np.isnan(eta),np.nan,etaf.isel({'xh':slice(1,-1), 'yh':slice(1,-1)}))


if os.path.exists(file):
    print('Files already exist. Skip')
    sys.exit()

print('Reading data...')
prog = xr.open_mfdataset(PATH_IN).sortby('Time').isel(Time=args.time_idx)
e = prog.e.isel(zi=slice(0,3)).load()
h = prog.h.load()
u = prog.u.load()
v = prog.v.load()
try:
    RV = prog.RV.load()
except:
    pass

static = xr.open_mfdataset('/scratch/pp2681/mom6/Feb2022/bare/R32/output/ocean_geometry.nc').rename({'lonh': 'xh', 'lath': 'yh', 'lonq': 'xq', 'latq': 'yq'})
lores_static = xr.open_mfdataset(f'/scratch/pp2681/mom6/Feb2022/bare/{RR}/output/ocean_geometry.nc').rename({'lonh': 'xh', 'lath': 'yh', 'lonq': 'xq', 'latq': 'yq'})
lores_prog = xr.open_mfdataset(f'/scratch/pp2681/mom6/Feb2022/bare/{RR}/output/prog_*').isel(Time=-1).squeeze()

lower_interface = e.isel(zi=-1)
gmfilter = GMFilter(static, lower_interface)

print('Filtering interfaces')
with ProgressBar():
    ef = xr.concat([gmfilter.diffusion_fixed_factor(e.isel(zi=slice(0,2)),filter_scale=filter_scale,limit_fluxes=True), lower_interface], dim='zi')

print('Coarsening...; Here we simply interpolate to prevent deviation from real topography')
ds_coarse = xr.Dataset()
with ProgressBar():
    ef = ef.interp(yh=lores_static.yh, xh=lores_static.xh).compute().transpose('zi',...)
    hf = -np.diff(ef.values,axis=0) # minus because vertical indexing is downward
    hf = xr.DataArray(hf, dims=['zl', 'yh', 'xh'])

    # Correct topography as it was modified by interpolation
    #import pdb
    #pdb.set_trace()
    topography_correction = (ef.isel(zi=-1)-lores_prog.e.isel(zi=-1)).drop_vars('zi')
    # Increase thickness as we move interface downward
    hf = hf * (hf.sum('zl')+topography_correction) / (hf.sum('zl'))
    ef[{'zi':-1}] = lores_prog.e.isel(zi=-1)
    ef[{'zi':-2}] = ef[{'zi':-1}] + hf.isel(zl=-1)
    ef[{'zi':0}] = ef[{'zi':-1}] + hf.sum('zl')
    ds_coarse['e'] = ef
    ds_coarse['h'] = hf

grid = xgcm.Grid(static, coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'}
        },
        boundary={'X': 'fill', 'Y': 'fill'},
        fill_value = {'Y': 0, 'X': 0})

grid_lores = xgcm.Grid(lores_static, coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'}
        },
        boundary={'X': 'fill', 'Y': 'fill'},
        fill_value = {'Y': 0, 'X': 0})

sys.path.append('../src-double-gyre')

from helpers.computational_tools import gaussian_filter

with ProgressBar():
    hf = gaussian_filter(h, static.wet, filter_scale=filter_scale).compute()
    ds_coarse['u_h'] = grid_lores.interp((gaussian_filter(h * grid.interp(u,'X'), static.wet, filter_scale=filter_scale) / (hf + 1e-10)).interp(yh=lores_static.yh, xh=lores_static.xh), 'X')
    ds_coarse['v_h'] = grid_lores.interp((gaussian_filter(h * grid.interp(v,'Y'), static.wet, filter_scale=filter_scale) / (hf + 1e-10)).interp(yh=lores_static.yh, xh=lores_static.xh), 'Y')

with ProgressBar():
    ds_coarse['u'] = grid_lores.interp(gaussian_filter(grid.interp(u,'X'), static.wet, filter_scale=filter_scale).interp(yh=lores_static.yh, xh=lores_static.xh), 'X')

with ProgressBar():
    ds_coarse['v'] = grid_lores.interp(gaussian_filter(grid.interp(v,'Y'), static.wet, filter_scale=filter_scale).interp(yh=lores_static.yh, xh=lores_static.xh), 'Y')

try:
    with ProgressBar():
        ds_coarse['RV'] = grid_lores.interp(gaussian_filter(grid.interp(RV, ['X', 'Y']), static.wet, filter_scale=filter_scale).interp(yh=lores_static.yh, xh=lores_static.xh), ['X', 'Y'])
except:
    pass

if not(os.path.exists(file)):
    print(f'Saving to {file}')
    ds_coarse.astype('float32').to_netcdf(file)

print(f'Script is done')