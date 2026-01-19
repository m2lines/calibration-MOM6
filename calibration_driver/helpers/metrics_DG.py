import xarray as xr
import numpy as np

def return_climate_metrics(exp_path, daymax, *metrics):
    try:
        prog = xr.open_mfdataset(f'{exp_path}/prog_*.nc', decode_times=False).astype('float64').sortby('Time').sel(Time=slice(365*10,None)).isel(zi=slice(0,2))
        series = 1e-15 * xr.open_mfdataset(f'{exp_path}/ocean.stats.nc', decode_times=False).astype('float64').sel(Time=slice(365*10,None)).isel(Interface=slice(0,2)).rename({'Layer': 'zl', 'Interface': 'zi'})[['KE', 'APE']]
    except:
        return False

    if daymax not in series.Time:
        return False

    metrics_data = {}
    for metric in metrics:
        match metric:
            case "KE_mean":
                metrics_data[metric] = series.KE.mean('Time').values
            case "KE_std":
                metrics_data[metric] = series.KE.std('Time').values
            case "APE_mean":
                metrics_data[metric] = series.APE.mean('Time').values
            case "APE_std":
                metrics_data[metric] = series.APE.std('Time').values
            case "sqrt_KE_mean":
                metrics_data[metric] = np.sqrt(series.KE.mean('Time').values)
            case "sqrt_KE_std":
                metrics_data[metric] = np.sqrt(series.KE.std('Time').values)
            case "sqrt_APE_mean":
                metrics_data[metric] = np.sqrt(series.APE.mean('Time').values)
            case "sqrt_APE_std":
                metrics_data[metric] = np.sqrt(series.APE.std('Time').values)
            case "e_mean":
                metrics_data[metric] = prog.e.mean('Time').values
            case "e_std":
                metrics_data[metric] = prog.e.std('Time').values
            case "u_mean":
                metrics_data[metric] = prog.u.mean('Time').values
            case "v_mean":
                metrics_data[metric] = prog.v.mean('Time').values
            case "u_std":
                metrics_data[metric] = prog.u.std('Time').values
            case "v_std":
                metrics_data[metric] = prog.v.std('Time').values

    prog.close()
    series.close()
    return metrics_data