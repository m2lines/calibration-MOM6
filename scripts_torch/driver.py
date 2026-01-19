import os
import sys
sys.path.append('helpers')
import numpy as np
import xarray as xr
from helpers.slurm_DG import *
from helpers.parameters import *
from helpers.julia_functions import *
from helpers.metrics_DG import *
import argparse
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

print(config)

this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
optimization_folder_pwd = os.path.join(config["paths"]["base"], config["paths"]["optimization_folder"])
commandline = f'cd /home/pp2681/calibration/scripts_torch; {config["slurm_eki"]} --dependency=singleton --export=NONE --job-name={TAG} -o {optimization_folder_pwd}/slurm-%j.out -e {optimization_folder_pwd}/slurm-%j.err --wrap="python-jl {script_name}"'
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
initialize_eki(observation_vector, gamma_vector, initial_ensemble, config["eki"]["scheduler"], config["eki"]["inversion"], config["eki"]["seed"])

for iteration in range(config["eki"]["n_iterations"]):
    print(f'################ iteration {iteration} ####################')
    params = eki_get_params()

    iteration_path = f'{optimization_folder_pwd}/iteration-{iteration:02d}'
    
    if os.path.exists(iteration_path):
        print('Folder with experiments exists. Preparing to update eki with new data')

        # Create forward model evaluation matrix
        g_ens = np.full(
            (len(observation_vector), config["eki"]["ens_size"]),
            np.nan,
            dtype="float64"
        )
        for ens_member in range(config["eki"]["ens_size"]):
            exp_path = f"{iteration_path}/ens-member-{ens_member:02d}/output"
            metrics_function = eval(config["eki"]["metrics_function"]) 
            metrics_data = metrics_function(exp_path, *config["eki"]["observation_vector"])

            if isinstance(metrics_data, dict):
                g_ens[:,ens_member] = np.concatenate([metrics_data[metric] for metric in config["eki"]["observation_vector"]])
                print(f'Ensemble member {ens_member} succesfully ingested')
            else:
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')
        
        print('Passing forward model evaluations to the EKI')
        eki_update_ensemble(g_ens)
        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member in range(config["eki"]["ens_size"]):
            exp_path = f"{iteration_path}/ens-member-{ens_member:02d}"
            ANN_modified = parameter_vector_to_ANN(ANN_netcdf_default, config["eki"]["observation_vector"], num_of_parameters, params[:,ens_member])

            call_function = ('singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro '
                            '--bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ '
                            ' /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif '
                            f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN={config["paths"]["ann"]}/Tall.nc --netcdf_eANN={exp_path}/INPUT/eANN.nc --netcdf_output={exp_path}/INPUT/Tall.nc"')

            hpc = HPC.add(name=config["tag"], time=config["slurm_mom6"]["time"], begin='1minute', executable=config["paths"]["executable"])

            # Model configuration
            exp_params = PARAMETERS.add(*config["mom6_namelist"]).add(**configuration('R2'))
            
            run_experiment(exp_path, hpc, exp_params,
                config["paths"]["configuration"],
                call_function)
            
            # We save data after initializing experiment to do not interrupt workflow.
            os.makedirs(f'{exp_path}/INPUT', exist_ok=True)
            ANN_modified.astype('float32').to_netcdf(f'{exp_path}/INPUT/eANN.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code