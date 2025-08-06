import xarray as xr
import numpy as np
import os
import xesmf as xe

def ave3d(array, param):
    return (array * param.dV).sum(['lon','lat','z_l']) / param.dV.sum(['lon','lat','z_l'])

def ave2d(array, param):
    return (array * param.dS).sum(['lon','lat']) / param.dS.sum(['lon','lat'])

# def ave2d(array, param):
#     return (array * param.dxt * param.dyt * param.mask_nan).sum(['xh','yh']) / (param.dxt * param.dyt * param.mask_nan).sum(['xh','yh'])

def return_regridder():
    param_coarse = xr.open_dataset('../data-coarsegrain/param_coarse.nc')
    param = xr.open_dataset('../data-coarsegrain/param_fine.nc').isel(time=0).drop_vars('time')
    
    # Coordinate information for OM4 output
    coords_in = xr.Dataset()
    coords_in['lon'] = param.geolon
    coords_in['lat'] = param.geolat
    coords_in['lon_b'] = param.geolon_c.pad({'yq':(1,0)}, mode='symmetric').pad({'xq':(1,0)}, mode='wrap').drop_vars({'xq','yq'})
    coords_in['lat_b'] = param.geolat_c.pad({'yq':(1,0)}, mode='symmetric').pad({'xq':(1,0)}, mode='wrap').drop_vars({'xq','yq'})

    # Coordinate information for WOA grid
    coords_out = xr.Dataset()
    coords_out['lon'] = param_coarse.lon
    coords_out['lat'] = param_coarse.lat
    # This is one-degree grid
    coords_out['lon_b'] = xr.DataArray((coords_out.lon + 0.5).values, dims='lon_b').pad({'lon_b':(1,0)}, mode='wrap')
    coords_out['lat_b'] = xr.DataArray((coords_out.lat + 0.5).values, dims='lat_b').pad({'lat_b':(1,0)}, mode='symmetric')

    regridder = xe.Regridder(coords_in, coords_out, "conservative", ignore_degenerate=True, periodic=True, unmapped_to_nan=True, 
                             weights='../data-coarsegrain/regridder_om4_to_woa.nc')
    return regridder

def read_nyf_data(exp='unparameterized'):
    return xr.open_dataset(os.path.join('/scratch/pp2681/calibration/OM4-NYF/1-degree-35-levels', f'{exp}.nc'))

def read_woa():
    ds = xr.open_dataset('../data-coarsegrain/woa.nc')
    param = xr.open_dataset('../data-coarsegrain/param_coarse.nc')
    return ds, param
    
def TnS2vec(dataset, woa, param, max_depth=None, center=False):
    """
    Convert TnS fields to vector field.
    
    Parameters:
    dataset (xarray.Dataset): the dataset with TnS fields.
    woa (xarray.Dataset): the World Ocean Atlas dataset for reference.
    param (xarray.Dataset): the grid parameters dataset.
    
    Returns:
    numpy array: Normalizes the TnS fields by norm of WOA, multplies 
    by a square root of the volume, removes nans and concatenate TnS into a vector.
    """
    thetao = dataset['thetao'].sel(z_l=slice(None,max_depth))
    so = dataset['so'].sel(z_l=slice(None,max_depth))

    thetao_woa = woa['thetao'].sel(z_l=slice(None,max_depth))
    so_woa = woa['so'].sel(z_l=slice(None,max_depth))

    # Center data at each depth independently
    # While this will not affect analysis of responses and model errors,
    # This still might help for computation of scalar products if needed later
    # Also, this will give a better estimate of the normalization factor
    thetao_2d = ave2d(thetao_woa, param)
    so_2d = ave2d(so_woa, param)

    thetao_woa = thetao_woa - thetao_2d
    so_woa = so_woa - so_2d

    if center:
        thetao = thetao - thetao_2d
        so = so - so_2d

    # Normalize by WOA. We use a single normalization coefficient for the full
    # field to be closer to the actual 3D RMSE loss
    normalization_thetao = ave3d(thetao_woa**2, param)**0.5
    normalization_so = ave3d(so_woa**2, param)**0.5

    thetao = thetao / normalization_thetao
    so = so / normalization_so
    thetao_woa = thetao_woa / normalization_thetao
    so_woa = so_woa / normalization_so

    # Incorporate square root of volume to use a conventional scalar product
    thetao = thetao * np.sqrt(param.dV)
    so = so * np.sqrt(param.dV)
    thetao_woa = thetao_woa * np.sqrt(param.dV)
    so_woa = so_woa * np.sqrt(param.dV)

    # stack to a vector and remove nans
    stack = lambda x: x.stack(i=['z_l','lat','lon'])

    thetao = stack(thetao)
    so = stack(so)
    thetao_woa = stack(thetao_woa)
    so_woa = stack(so_woa)
    
    # Find mask of non-nan values 
    mask = ~np.isnan(thetao)

    vector = np.concatenate([
        thetao[mask],
        so[mask]]
    ).astype('float64')

    vector_woa = np.concatenate([
        thetao_woa[mask],
        so_woa[mask]]
    ).astype('float64')

    return dict(vector=vector,
                vector_woa=vector_woa,
                mask = mask,
                thetao=thetao,
                so=so,
                thetao_woa=thetao_woa,
                so_woa=so_woa)
    

def vec2TnS(vector, datastructure):
    '''
    Here vector is 1D numpy array
    and datastructure is a dictionary with keys
    from the output of the previous function
    '''
    thetao = xr.zeros_like(datastructure['thetao']) * np.nan
    so = xr.zeros_like(datastructure['so']) * np.nan

    thetao[datastructure['mask']] = vector[:len(vector)//2]
    so[datastructure['mask']] = vector[len(vector)//2:]

    return dict(thetao=thetao, so=so)