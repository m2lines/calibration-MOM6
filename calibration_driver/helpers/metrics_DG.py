import xarray as xr
import numpy as np
import os

def return_climate_metrics(exp_path, daymax, *metrics):
    try:
        prog = xr.open_mfdataset(f'{exp_path}/prog_*.nc', decode_times=False).astype('float64').sortby('Time').sel(Time=slice(365*10,None)).isel(zi=slice(0,2)).fillna(0.)
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

def assemble_G_matrix_and_store_metrics(iteration_path, optimization_folder_pwd, iteration,
                                        observation_netcdf, params,
                                        daymax,
                                        len_obs, ens_size, 
                                        outlier_scale, metrics_function_name,
                                        observation_vector_names, gamma_vector_names,
                                        observation_validation_names, gamma_validation_names):
    '''
    This algorithm includes multiple steps:
    1) Compute raw metrics for an ensemble of experiments
    2) Assemble forward model evaluation matrix (G)
    3) Compute distance to observations
    4) Remove outliers
    5) Evaluate ensemble-mean prediction

    observation_vector_names, gamma_vector_names are metrics used to compute loss function
    observation_validation_names, gamma_validation_names are all metrics computed and stored for analysis
    '''
    # Create forward model evaluation matrix
    g_ens = np.full(
        (len_obs, ens_size),
        np.nan,
        dtype="float64"
    )

    # Each list element is metrics for an ensemble member
    metrics_netcdf_list = []
    for ens_member in range(ens_size):
        exp_path = f"{iteration_path}/ens-member-{ens_member:02d}/output"
        
        # Compute metrics used for the optimization as a dictionaty
        metrics_function = eval(metrics_function_name) 
        metrics_data = metrics_function(exp_path, daymax, *observation_validation_names)
        
        # Concatenate metrics to a vector
        if isinstance(metrics_data, dict):
            g_ens[:,ens_member] = np.concatenate([metrics_data[metric].ravel() for metric in observation_vector_names])
            print(f'Ensemble member {ens_member} succesfully ingested')
        else:
            print(f'Ensemble member {ens_member} failed. Filled with NaNs')

        # Store metrics for a given experiment as netcdf file
        metrics_netcdf = xr.Dataset()
        for metric in observation_validation_names:
            if isinstance(metrics_data, dict):
                metrics_netcdf[metric] = observation_netcdf[metric]*0 + metrics_data[metric]
            else:
                metrics_netcdf[metric] = observation_netcdf[metric]*np.nan

        # Compute distance to the observation
        for metric, metric_var in zip(observation_validation_names, gamma_validation_names):
            # Compute error and obs covariance
            error = metrics_netcdf[metric] - observation_netcdf[metric]
            variance = observation_netcdf[metric_var]
            # Determine spatial dimensions for averaging/summation
            spatial_ave_dims = []
            for dim in ['xh', 'yh', 'xq', 'yq']:
                if dim in error.dims:
                    spatial_ave_dims.append(dim)
            
            # Compute weigted squared error, as it enters the loss function
            metrics_netcdf[metric+'_WSE'] = (error * error / variance).sum(spatial_ave_dims, skipna=False)
            # Compute RMSE for simler assesment
            metrics_netcdf[metric+'_RMSE'] = np.sqrt((error * error).mean(spatial_ave_dims, skipna=False))
        
        # Compute total weighted mean squared error as it is computed by Ensemble Kalman Processes.jl
        # Here we consider only those metrics which are in the loss function
        metrics_netcdf['WMSE'] = np.sum([metrics_netcdf[metric+'_WSE'] for metric in observation_vector_names]) / len_obs
        
        # Append the experiment to the list
        metrics_netcdf_list.append(metrics_netcdf)

    # Concatenate over the ensemble members
    metrics_netcdf = xr.concat(metrics_netcdf_list, dim='ens')
    metrics_netcdf['param'] = xr.DataArray(params, dims=['pdim', 'ens']).transpose('ens',...)

    # Find outliers to be excluded from the optimization
    min_WMSE = float(metrics_netcdf['WMSE'].min())
    mask_outlier = metrics_netcdf['WMSE'] > min_WMSE * outlier_scale
    g_ens[:,mask_outlier] = np.nan
    for metric in observation_validation_names:
        metrics_netcdf[metric][mask_outlier] = np.nan
        metrics_netcdf[metric+'_WSE'][mask_outlier] = np.nan
        metrics_netcdf[metric+'_RMSE'][mask_outlier] = np.nan
    metrics_netcdf['WMSE'][mask_outlier] = np.nan
    print('Filtered out outliers: ', np.where(mask_outlier)[0])

    # Evaluate ensemble-mean prediction
    for metric, metric_var in zip(observation_validation_names, gamma_validation_names):
        error = metrics_netcdf[metric].mean('ens') - observation_netcdf[metric]
        variance = observation_netcdf[metric_var]
        spatial_ave_dims = []
        for dim in ['xh', 'yh', 'xq', 'yq']:
            if dim in error.dims:
                spatial_ave_dims.append(dim)
        metrics_netcdf[metric+'_WSE_MAP'] = (error * error / variance).sum(spatial_ave_dims)
        metrics_netcdf[metric+'_RMSE_MAP'] = np.sqrt((error * error).mean(spatial_ave_dims, skipna=False))
    metrics_netcdf['WMSE_MAP'] = np.sum([metrics_netcdf[metric+'_WSE_MAP'] for metric in observation_vector_names]) / len_obs
    
    # Expand dimenion for iteration
    metrics_netcdf = metrics_netcdf.expand_dims(iter=[iteration]).transpose('iter', 'ens',...)

    metrics_netcdf.astype('float32').to_netcdf(f'{optimization_folder_pwd}/metrics_{iteration:02d}.nc')

    del metrics_netcdf

    return g_ens