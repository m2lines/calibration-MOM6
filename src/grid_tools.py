import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq' or 'xh' or 'lon'
    as xarray
    '''
    for key in ['xq', 'xh', 'lon']:
        if key in array.coords:
            return array[key]

def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh' or 'lat'
    as xarray
    '''
    for key in ['yq', 'yh', 'lat']:
        if key in array.coords:
            return array[key]

def select_LatLon(array, Lat=(35,45), Lon=(5,15), time=None):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    x = array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})
    
    if 'time' in x.dims:
        if time is None:
            pass
        else:
            x = x.isel(time=time)
    return x

def select_NA(array, time=None):
    return select_LatLon(array, Lat=(15, 65), Lon=(-90,-10), time=time)

def select_Pacific(array, time=None):
    return select_LatLon(array, Lat=(10, 65), Lon=(-250,-130), time=time)

def select_Cem(array, time=None):
    return select_LatLon(array, Lat=(-10,15), Lon=(-260,-230), time=time)

def select_globe(array, time=None):
    return select_LatLon(array, Lat=(None,None), Lon=(None,None), time=time)

def select_Equator(array, time=None):
    return select_LatLon(array, Lat=(-30,30), Lon=(-190,-130), time=time)

def select_ACC(array, time=None):
    return select_LatLon(array, Lat=(-70,-30), Lon=(-40,0), time=time)

def select_rings(array, time=None):
    return select_LatLon(array, Lat=(-60,-10), Lon=(-80,50), time=time)

# Juricke regions below:

def select_Gulf(array):
    return select_LatLon(array, Lat=(30, 60), Lon=(-80,-20))

def select_Kuroshio(array):
    return select_LatLon(array, Lat=(20, 50), Lon=(120,180))

def select_SO(array):
    return select_LatLon(array, Lat=(-70,-30), Lon=(0,360))

def select_Aghulas(array):
    return select_LatLon(array, Lat=(-60,-30), Lon=(0,60))

def select_Malvinas(array):
    return select_LatLon(array, Lat=(-60,-30), Lon=(-60,0))

# Select time-series

def select_NA_series(array):
    return select_LatLon(array, Lat=(25, 45), Lon=(-60,-40))

def select_Pacific_series(array):
    return select_LatLon(array, Lat=(25, 45), Lon=(-200,-180))

def select_center(array):
    x = x_coord(array)
    y = y_coord(array)
    central_latitude = float(y.mean())
    central_longitude = float(x.mean())

    return array.sel({x.name:central_longitude, y.name:central_latitude}, method='nearest').drop_vars([x.name, y.name])

def plot(control, mask=None, vmax=None, vmin=None, selector=select_NA, cartopy=True, cmap=cmocean.cm.balance):
    if mask is not None:
        mask_nan = selector(mask).data.copy()
        mask_nan[mask_nan==0.] = np.nan
    else:
        mask_nan = 1

    control = (mask_nan * selector(control)).compute()
    
    if vmax is None:
        control_mean = control.mean()
        control_std = control.std()
        vmax = control_mean + control_std * 4
        vmin = control_mean - control_std * 4
    else:
        control_mean = 0.
        if vmin is None:
            vmin = - vmax
    
    central_latitude = float(y_coord(control).mean())
    central_longitude = float(x_coord(control).mean())
    if cartopy:
        fig, ax = plt.subplots(1,1, figsize=(20, 15), subplot_kw={'projection': ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)})
        ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True
        kw = {'transform': ccrs.PlateCarree()}
    else:
        ax = plt.gca()
        kw = {}
    cmap.set_bad('gray')
    im = selector(control).plot(ax=ax, vmax=vmax, vmin=vmin, cmap=cmap, add_colorbar=True, **kw)
    plt.title('')
    return im

# We compare masked fields because outside there may be 1e+20 values
def compare(tested, control, mask=None, vmax=None, vmin = None, selector=select_NA, cmap=cmocean.cm.balance, 
            label_test = 'Tested field', label_control = 'Control field'):
    if mask is not None:
        mask_nan = mask.data.copy()
        mask_nan[mask_nan==0.] = np.nan
        mask_nan = mask_nan + mask*0
        tested = tested * mask_nan
        control = control * mask_nan
    tested = selector(tested).compute()
    control = selector(control).compute()
    
    if vmax is None:
        control_mean = control.mean()
        control_std = control.std()
        vmax = control_mean + control_std * 4
        vmin = control_mean - control_std * 4
    else:
        control_mean = 0.
        if vmin is None:
            vmin = - vmax
    
    central_latitude = float(y_coord(control).mean())
    central_longitude = float(x_coord(control).mean())
    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    cmap.set_bad('gray')
    
    ax = axes[0][0];# ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    im = tested.plot(ax=ax, vmax=vmax, vmin=vmin, cmap=cmap, add_colorbar=False)
    ax.set_title(label_test)
    ax = axes[0][1];# ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    control.plot(ax=ax, vmax=vmax, vmin=vmin, cmap=cmap, add_colorbar=False)
    ax.set_title(label_control)
    ax = axes[1][0];# ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    (tested-control).plot(ax=ax, vmax=vmax-control_mean, vmin=vmin-control_mean, cmap=cmap, add_colorbar=False)
    ax.set_title(f'{label_test} $-$ {label_control}')
    plt.tight_layout()
    plt.colorbar(im, ax=axes, shrink=0.9, aspect=30, extend='both')
    axes[1][1].remove()
    
    ########## Metrics ##############
    error = tested-control
    relative_error = np.abs(error).mean() / np.abs(control).mean()
    R2 = 1 - (error**2).mean() / (control**2).mean()
    optimal_scaling = (tested*control).mean() / (tested**2).mean()
    error = tested * optimal_scaling - control
    R2_max = 1 - (error**2).mean() / (control**2).mean()
    corr = xr.corr(tested, control)
    scal_prod = (tested * control).mean() / np.sqrt((tested**2).mean() / (control**2).mean())
    print('Correlation:', float(corr))
    print('Relative Error:', float(relative_error))
    print('R2 = ', float(R2))
    print('R2 max = ', float(R2_max))
    print('Optinal scaling:', float(optimal_scaling))
    print('Scalar product:', float(optimal_scaling))
    print(f'Nans [test/control]: [{int(np.sum(np.isnan(tested)))}, {int(np.sum(np.isnan(control)))}]')

    return axes