import sys

import xarray as xr
import matplotlib.pyplot as plt
import xgcm
import gcm_filters
import argparse
from dask.diagnostics import ProgressBar
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

FOLDER = 'R2_FGR3_RESTART'
COARSENING_FACTOR = 16
FILTER_SCALE = COARSENING_FACTOR * 3
lores_static = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R2-long/bare/output/static.nc', decode_times=False)

parser = argparse.ArgumentParser()
parser.add_argument("--time_idx", type=int, default=0)
parser.add_argument("--zl_idx", type=int, default=0)
args = parser.parse_args()

file_zl = f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/{FOLDER}/centers/time_{args.time_idx}_zl_{args.zl_idx}.nc'
#file_zi = f'/scratch/pp2681/mom6/Neverworld2/simulations/R32/{FOLDER}/interfaces/time_{args.time_idx}_zl_{args.zl_idx}.nc'

if os.path.exists(file_zl):
    print('Files already exist. Skip')
    sys.exit()

print(f"The parsed arguments are: {args}")
# The first vertical layer filters two interfaces because interfaces have one plus vertical grid point
if (args.zl_idx ==0):
    zi_idx = slice(0,2)
else:
    zi_idx = args.zl_idx+1

print(f"zi is: {zi_idx}")

grid_lores = xgcm.Grid(lores_static.isel(time=0).drop_vars('time').squeeze(), coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'}
    },
    boundary={"X": 'periodic', 'Y': 'fill'},
    fill_value = {'Y':0})

lores_static['dxBu'] = grid_lores.interp(lores_static.dxCu, 'Y')
lores_static['dyBu'] = grid_lores.interp(lores_static.dyCu, 'Y')

lores_static['dxT'] = grid_lores.interp(lores_static.dxCu, 'X')
lores_static['dyT'] = grid_lores.interp(lores_static.dyCu, 'X')

print('Reading data...')
with ProgressBar():
    #ref = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R32/snapshots_*', decode_times=False, chunks={'time':1, 'zl':1})[['u','v']].isel(time=args.time_idx, zl=args.zl_idx)
    ref = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R32/RESTART/MOM.res.nc', decode_times=False, chunks={'Time':1, 'Layer':1})[['u', 'v', 'h']].rename({'Time': 'time', 'Layer': 'zl', 'lonh': 'xh', 'lath': 'yh', 'lonq': 'xq', 'latq': 'yq'}).isel(time=args.time_idx, zl=args.zl_idx)
    ref_static = xr.open_mfdataset('/scratch/pp2681/mom6/Neverworld2/simulations/R32/static.nc', decode_times=False).squeeze().drop_vars('time')

# Generate missing grid information
grid_ref = xgcm.Grid(ref_static, coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'}
    },
    boundary={"X": 'periodic', 'Y': 'fill'},
    fill_value = {'Y':0})

ref_static['dxBu'] = grid_ref.interp(ref_static.dxCu, 'Y')
ref_static['dyBu'] = grid_ref.interp(ref_static.dyCu, 'Y')

ref_static['dxT'] = grid_ref.interp(ref_static.dxCu, 'X')
ref_static['dyT'] = grid_ref.interp(ref_static.dyCu, 'X')

def KE_Arakawa(u, v):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L1000-L1003
        '''
        param = ref_static
        grid = grid_ref
        
        areaCu = param.dxCu * param.dyCu
        areaCv = param.dxCv * param.dyCv
        areaT = param.dxT * param.dyT
        
        KEu = grid.interp(param.wet_u * areaCu * u**2, 'X')
        KEv = grid.interp(param.wet_v * areaCv * v**2, 'Y')

        return 0.5 * (KEu + KEv) / areaT * param.wet

def gradKE(u,v):
    '''
    https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L1029-L1034
    '''
    
    param = ref_static
    grid = grid_ref

    KE = KE_Arakawa(u,v)
    IdxCu = 1. / param.dxCu
    IdyCv = 1. / param.dyCv

    KEx = grid.diff(KE, 'X') * IdxCu * param.wet_u
    KEy = grid.diff(KE, 'Y') * IdyCv * param.wet_v
    return (KEx, KEy)

def relative_vorticity(u,v):
    '''
    https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L472
    '''
    param = ref_static
    grid = grid_ref
    
    dyCv = param.dyCv
    dxCu = param.dxCu
    IareaBu = 1. / (param.dxBu * param.dyBu)
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L309-L310
    dvdx = grid.diff(param.wet_v * v * dyCv,'X')
    dudy = grid.diff(param.wet_u * u * dxCu,'Y')
    return ((dvdx - dudy) * IareaBu * param.wet_c).squeeze()

def PV_cross_uv(u,v):
    '''
    https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L669-L671
    https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L788-L790
    fx = + q * vh
    fy = - q * uh
    '''
    param = ref_static
    grid = grid_ref
    
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L131
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_continuity_PPM.F90#L569-L570
    uh = u * param.dyCu * param.wet_u
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L133
    vh = v * param.dxCv * param.wet_v
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L484
    rel_vort = relative_vorticity(u,v)

    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L247
    Area_h = param.dxT * param.dyT
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L272-L273
    Area_q = grid.interp(Area_h, ['X', 'Y']) * 4
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L323
    hArea_u = grid.interp(Area_h,'X')
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L320
    hArea_v = grid.interp(Area_h,'Y')
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L488
    hArea_q = 2 * grid.interp(hArea_u,'Y') + 2 * grid.interp(hArea_v,'X')
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L489
    Ih_q = Area_q / hArea_q

    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L490
    q = rel_vort * Ih_q

    IdxCu = 1. / param.dxCu
    IdyCv = 1. / param.dyCv
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L669-L671
    CAu = + grid.interp(q * grid.interp(vh,'X'),'Y') * IdxCu * param.wet_u
    # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L788-L790
    CAv = - grid.interp(q * grid.interp(uh,'Y'),'X') * IdyCv * param.wet_v

    return (CAu, CAv)

def advection(u,v):
    '''
    https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L751
    https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L875

    $- (u nabla) u$ operator, i.e. it is advection acceleration in RHS
    '''
    CAu, CAv = PV_cross_uv(u,v)
    KEx, KEy = gradKE(u,v)
    return (CAu - KEx, CAv - KEy)

filter_u = gcm_filters.Filter(
                filter_scale=FILTER_SCALE, # filter scale accounts for change in resolution (32/4) and FGR (3)
                dx_min=1,
                filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                grid_vars={'wet_mask': ref_static.wet_u}
                )

filter_v = gcm_filters.Filter(
                filter_scale=FILTER_SCALE, # filter scale accounts for change in resolution (32/4) and FGR (3)
                dx_min=1,
                filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                grid_vars={'wet_mask': ref_static.wet_v}
                )

filter_h = gcm_filters.Filter(
                filter_scale=FILTER_SCALE, # filter scale accounts for change in resolution (32/4) and FGR (3)
                dx_min=1,
                filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                grid_vars={'wet_mask': ref_static.wet}
                )

# Creating filtered dataset
ds = xr.Dataset()

#print('Filtering u/v/h/e...')
#with ProgressBar():
    #ds['uf'] = filter_u.apply(ref.u, dims=['yh', 'xq'])
    #ds['vf'] = filter_v.apply(ref.v, dims=['yq', 'xh'])
    #ds['hf'] = filter_h.apply(ref.h, dims=['yh', 'xh'])
    #ds['ef'] = filter_h.apply(ref.e, dims=['yh', 'xh'])

print('Thicnkess-weighted u and v')
with ProgressBar():
    h = filter_h.apply(ref.h, dims=['yh', 'xh']).compute()
    ds['uf'] = grid_ref.interp(filter_h.apply(grid_ref.interp(ref.u, 'X'), dims=['yh', 'xh']) / (h + 1e-10), 'X')
    ds['vf'] = grid_ref.interp(filter_h.apply(grid_ref.interp(ref.v, 'Y'), dims=['yh', 'xh']) / (h + 1e-10), 'Y')

#print('Computing SGS force...')
#with ProgressBar():
#    ds['advx_hires'], ds['advy_hires'] = advection(ref.u, ref.v)
#    ds['advx_filtered_tendency'] = filter_u.apply(ds['advx_hires'], dims=['yh', 'xq'])
#    ds['advy_filtered_tendency'] = filter_v.apply(ds['advy_hires'], dims=['yq', 'xh'])
#    del ds['advx_hires'], ds['advy_hires']
#    ds['advx_filtered_state'], ds['advy_filtered_state'] = advection(ds['uf'], ds['vf'])

#    ds['SGSx'] = ds['advx_filtered_tendency'] - ds['advx_filtered_state']
#    ds['SGSy'] = ds['advy_filtered_tendency'] - ds['advy_filtered_state']
#    del ds['advx_filtered_tendency'], ds['advx_filtered_state'], ds['advy_filtered_tendency'], ds['advy_filtered_state']

#print('Computing subfilter flux...')
#with ProgressBar():
#    ds['Txx_hires'] = grid_ref.interp(ref.u * ref.u, 'X') * ref_static.wet
#    ds['Tyy_hires'] = grid_ref.interp(ref.v * ref.v, 'Y') * ref_static.wet
#    ds['Txy_hires'] = grid_ref.interp(ref.u, 'X') * grid_ref.interp(ref.v, 'Y') * ref_static.wet

#    ds['Txx_filtered_tendency'] = filter_h.apply(ds['Txx_hires'], dims=['yh', 'xh'])
#    ds['Tyy_filtered_tendency'] = filter_h.apply(ds['Tyy_hires'], dims=['yh', 'xh'])
#    ds['Txy_filtered_tendency'] = filter_h.apply(ds['Txy_hires'], dims=['yh', 'xh'])

#    del ds['Txx_hires'], ds['Tyy_hires'], ds['Txy_hires']

#    ds['Txx_filtered_state'] = grid_ref.interp(ds['uf'] * ds['uf'], 'X') * ref_static.wet
#    ds['Tyy_filtered_state'] = grid_ref.interp(ds['vf'] * ds['vf'], 'Y') * ref_static.wet
#    ds['Txy_filtered_state'] = grid_ref.interp(ds['uf'], 'X') * grid_ref.interp(ds['vf'], 'Y') * ref_static.wet

#    ds['Txx'] = ds['Txx_filtered_state'] - ds['Txx_filtered_tendency']
#    ds['Tyy'] = ds['Tyy_filtered_state'] - ds['Tyy_filtered_tendency']
#    ds['Txy'] = ds['Txy_filtered_state'] - ds['Txy_filtered_tendency']

#    del ds['Txx_filtered_state'], ds['Tyy_filtered_state'], ds['Txy_filtered_state'], ds['Txx_filtered_tendency'], ds['Tyy_filtered_tendency'], ds['Txy_filtered_tendency']

print('Coarsening...')
with ProgressBar():
    tmp = ds.astype('float32').coarsen({'xq':COARSENING_FACTOR, 'xh':COARSENING_FACTOR, 'yq': COARSENING_FACTOR, 'yh':COARSENING_FACTOR}, boundary='pad').mean()
    ds_coarse = tmp.interp(yq = lores_static.yq, xq = lores_static.xq)[['uf','vf']].compute()
    #ds_coarse_interface = tmp.interp(yq = lores_static.yq, xq = lores_static.xq)[['ef']].compute()

del ds
################ Computing features on a coarse grid #################

def velocity_gradients(u=None, v=None):
    param = lores_static
    grid = grid_lores

    dudx = grid.diff(u * param.wet_u / param.dyCu, 'X') * param.dyT / param.dxT * param.wet
    dvdy = grid.diff(v * param.wet_v / param.dxCv, 'Y') * param.dxT / param.dyT * param.wet

    dudy = grid.diff(u * param.wet_u / param.dxCu, 'Y') * param.dxBu / param.dyBu * param.wet_c
    dvdx = grid.diff(v * param.wet_v / param.dyCv, 'X') * param.dyBu / param.dxBu * param.wet_c
    
    sh_xx = dudx-dvdy
    sh_xy_h = grid.interp(dvdx+dudy, ['X', 'Y']) * param.wet
    vort_xy_h=grid.interp(dvdx-dudy, ['X', 'Y']) * param.wet
    div = dudx+dvdy
    
    return sh_xx, sh_xy_h, vort_xy_h, div

#print('Velocity gradients...')
#with ProgressBar():
#    ds_coarse['sh_xx'], ds_coarse['sh_xy_h'], ds_coarse['vort_xy_h'], ds_coarse['div'] = velocity_gradients(ds_coarse.uf, ds_coarse.vf)

#print('SGS dissipation...')
#with ProgressBar():
#    Tdd = 0.5 * (ds_coarse['Txx'] - ds_coarse['Tyy'])
#    Ttr = 0.5 * (ds_coarse['Txx'] + ds_coarse['Tyy'])
#    # Positive number means backscatter
#    ds_coarse['SGS_back'] = - (Tdd * ds_coarse['sh_xx'] + Ttr * ds_coarse['div'] + ds_coarse['Txy'] * ds_coarse['sh_xy_h'])
#print('dEdt...')
#with ProgressBar():
    # Subgrid forcing times the velocity
    # Positive number means backscatter
#    ds_coarse['dEdt'] = (grid_lores.interp(ds_coarse.SGSx * ds_coarse.uf, 'X') + grid_lores.interp(ds_coarse.SGSy * ds_coarse.vf, 'Y')) * lores_static.wet

if not(os.path.exists(file_zl)):
    print(f'Saving to {file_zl}')
    try:
        ds_coarse = ds_coarse.drop_vars('zi')
    except:
        pass
    ds_coarse.astype('float32').to_netcdf(file_zl)

# if not(os.path.exists(file_zi)):
#     print(f'Saving to {file_zi}')
#     ds_coarse_interface.astype('float32').drop_vars('zl').to_netcdf(file_zi)

print(f'Script is done')
