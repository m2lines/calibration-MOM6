import os
import sys
sys.path.append('/home/pp2681/calibration/calibration_driver')
import numpy as np
import xarray as xr
from helpers.slurm_DG import *
from helpers.parameters import *
from helpers.julia_functions import *
from helpers.metrics_DG import *
import argparse
import yaml

########################## USAGE ################################
# python-jl /home/pp2681/calibration/calibration_driver/driver.py

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument('--latest_iteration', type=int, default=0) 
args = parser.parse_args()

print(args)
print(config)

optimization_folder_pwd = os.path.join(config["paths"]["base"], config["paths"]["optimization_folder"])
os.makedirs(f'{optimization_folder_pwd}', exist_ok=True)

############### Create initial ensemble ################
ANN_netcdf_default = xr.open_dataset(f'{config["paths"]["ann"]}/eANN.nc').load()
np.random.seed(config["eki"]["seed"])
initial_ensemble, num_of_parameters = generate_ensemble(ANN_netcdf_default, 
                                        config["eki"]["trainable_parameters"],
                                        config["eki"]["ens_spread"],
                                        config["eki"]["ens_size"])

############ Prepare observational vector ##############
observation_netcdf = xr.open_dataset(config["paths"]["observation"]).astype('float64')
observation_vector = []
for key in config["eki"]["observation_vector"]:
    observation_vector.append(observation_netcdf[key].values.ravel())
observation_vector = np.concatenate(observation_vector)

############ Prepare gamma vector ##############
gamma_vector = []
for key in config["eki"]["gamma_vector"]:
    gamma_vector.append(observation_netcdf[key].values.ravel())
gamma_vector = np.concatenate(gamma_vector)

############ Initialize EKI process #############
initialize_eki(observation_vector, gamma_vector, initial_ensemble, config["eki"]["scheduler"], config["eki"]["inversion"], config["eki"]["seed"], optimization_folder_pwd)

for iteration in range(args.latest_iteration, config["eki"]["n_iterations"]):
    print(f'################ iteration {iteration} ####################')
    params = eki_get_params()
    iteration_path = f'{optimization_folder_pwd}/iteration-{iteration:02d}'

    params_file = f'{iteration_path}-params.txt'

    if not(os.path.exists(params_file)):
        print('Saving parameters to file', params_file)
        np.savetxt(params_file, params)
    else:
        params_old = np.loadtxt(params_file)
        if not(np.allclose(params, params_old)):
            print('Parameters changed! Check the optimization algorithm.')
            sys.exit(1)   # terminate immediately with error code
        else:
            print('Parameters are the same. Keep going...')
    
    if os.path.exists(iteration_path):
        print('Folder with experiments exists. Preparing to update eki with new data')

        # Create forward model evaluation matrix
        g_ens = np.full(
            (len(observation_vector), config["eki"]["ens_size"]),
            np.nan,
            dtype="float64"
        )
        metrics_netcdf_list = []
        for ens_member in range(config["eki"]["ens_size"]):
            exp_path = f"{iteration_path}/ens-member-{ens_member:02d}/output"
            metrics_function = eval(config["eki"]["metrics_function"]) 
            metrics_data = metrics_function(exp_path, config["mom6_namelist"]["DAYMAX"], *config["eki"]["observation_vector"])

            if isinstance(metrics_data, dict):
                g_ens[:,ens_member] = np.concatenate([metrics_data[metric].ravel() for metric in config["eki"]["observation_vector"]])
                print(f'Ensemble member {ens_member} succesfully ingested')
            else:
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')

            # Store metrics for a given experiment
            metrics_netcdf = xr.Dataset()
            for metric in config["eki"]["observation_vector"]:
                if isinstance(metrics_data, dict):
                    metrics_netcdf[metric] = observation_netcdf[metric]*0 + metrics_data[metric]
                else:
                    metrics_netcdf[metric] = observation_netcdf[metric]*np.nan

            # Compute Weighted Squared Errors
            for metric, metric_var in zip(config["eki"]["observation_vector"], config["eki"]["gamma_vector"]):
                error = metrics_netcdf[metric] - observation_netcdf[metric]
                variance = observation_netcdf[metric_var]
                spatial_ave_dims = []
                for dim in ['xh', 'yh', 'xq', 'yq']:
                    if dim in error.dims:
                        spatial_ave_dims.append(dim)
                metrics_netcdf[metric+'_WSE'] = (error * error / variance).sum(spatial_ave_dims, skipna=False)
                metrics_netcdf[metric+'_RMSE'] = np.sqrt((error * error).mean(spatial_ave_dims, skipna=False))
            metrics_netcdf['WMSE'] = np.sum([metrics_netcdf[metric+'_WSE'] for metric in config["eki"]["observation_vector"]]) / len(observation_vector)
            metrics_netcdf_list.append(metrics_netcdf)

        print('Saving metrics to disk')
        metrics_netcdf = xr.concat(metrics_netcdf_list, dim='ens')
        metrics_netcdf['param'] = xr.DataArray(params, dims=['pdim', 'ens']).transpose('ens',...)

        # Find outliers
        min_WMSE = float(metrics_netcdf['WMSE'].min())
        mask_outlier = metrics_netcdf['WMSE'] > min_WMSE * config["eki"]["outlier_scale"]
        g_ens[:,mask_outlier] = np.nan
        for metric in config["eki"]["observation_vector"]:
            metrics_netcdf[metric][mask_outlier] = np.nan
            metrics_netcdf[metric+'_WSE'][mask_outlier] = np.nan
            metrics_netcdf[metric+'_RMSE'][mask_outlier] = np.nan
        metrics_netcdf['WMSE'][mask_outlier] = np.nan
        print('Filtered out outliers: ', np.where(mask_outlier)[0])

        # Compute Weighted Squared Errors
        for metric, metric_var in zip(config["eki"]["observation_vector"], config["eki"]["gamma_vector"]):
            error = metrics_netcdf[metric].mean('ens') - observation_netcdf[metric]
            variance = observation_netcdf[metric_var]
            spatial_ave_dims = []
            for dim in ['xh', 'yh', 'xq', 'yq']:
                if dim in error.dims:
                    spatial_ave_dims.append(dim)
            metrics_netcdf[metric+'_WSE_MAP'] = (error * error / variance).sum(spatial_ave_dims)
            metrics_netcdf[metric+'_RMSE_MAP'] = np.sqrt((error * error).mean(spatial_ave_dims, skipna=False))
        metrics_netcdf['WMSE_MAP'] = np.sum([metrics_netcdf[metric+'_WSE_MAP'] for metric in config["eki"]["observation_vector"]]) / len(observation_vector)
        metrics_netcdf.to_netcdf(f'{optimization_folder_pwd}/metrics_{iteration:02d}.nc')

        print('Passing forward model evaluations to the EKI')
        eki_update_ensemble(g_ens)
        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')

        print('Saving EKI to disk')
        save_eki_on_disk(optimization_folder_pwd)

    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member in range(config["eki"]["ens_size"]):
            exp_path = f"{iteration_path}/ens-member-{ens_member:02d}"
            ANN_modified = parameter_vector_to_ANN(ANN_netcdf_default, config["eki"]["trainable_parameters"], num_of_parameters, params[:,ens_member])

            call_function = config["singularity_command"] + \
                            f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN={config["paths"]["ann"]}/Tall.nc --netcdf_eANN={exp_path}/INPUT/eANN.nc --netcdf_output={exp_path}/INPUT/Tall.nc"'

            hpc = HPC.add(name=config["tag"], time=config["slurm_mom6"]["time"], begin='1minute', executable=config["paths"]["executable"])

            # Model configuration
            exp_params = PARAMETERS.add(**configuration('R2')).add(**config["mom6_namelist"])
            
            run_experiment(exp_path, hpc, exp_params,
                config["paths"]["configuration"],
                call_function)
            
            # We save data after initializing experiment to do not interrupt workflow.
            os.makedirs(f'{exp_path}/INPUT', exist_ok=True)
            ANN_modified.astype('float32').to_netcdf(f'{exp_path}/INPUT/eANN.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        this_file = os.path.abspath(__file__)  # full path of current script
        script_name = os.path.basename(this_file)  # just the filename
        commandline = f'cd {os.getcwd()}; {config["slurm_eki"]} --dependency=singleton --export=NONE --job-name={config["tag"]} -o {optimization_folder_pwd}/slurm-%j.out -e {optimization_folder_pwd}/slurm-%j.err --wrap="python-jl {script_name} --latest_iteration={iteration}"'
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code
