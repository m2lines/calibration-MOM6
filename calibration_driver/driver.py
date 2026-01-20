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

####################### USAGE ##########################
# To start calibration: python-jl /home/pp2681/calibration/calibration_driver/driver.py
# To continue calibration: python-jl /home/pp2681/calibration/calibration_driver/driver.py --latest_iteration=1
# Where latest_iteration is the last folder of experiments which was computed but not yet processed

######################## YAML config ###################
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

######################## ARGUMENT ######################
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
gamma_vector = config["eki"]["gamma_scaling_factor"] * np.concatenate(gamma_vector)

############ Initialize EKI process #############
initialize_eki(observation_vector, gamma_vector, initial_ensemble, config["eki"]["scheduler"], config["eki"]["inversion"], config["eki"]["seed_julia"], optimization_folder_pwd)

for iteration in range(args.latest_iteration, config["eki"]["n_iterations"]):
    print(f'################ iteration {iteration} ####################')
    params = eki_get_params()
    iteration_path = f'{optimization_folder_pwd}/iteration-{iteration:02d}'

    save_params_txt(params, iteration_path)
    
    if os.path.exists(iteration_path):
        print('Folder with experiments exists. Preparing to update eki with new data')

        print('Processing Forward model outputs...')
        g_ens = assemble_G_matrix_and_store_metrics(iteration_path, optimization_folder_pwd, iteration,
            observation_netcdf, params,
            config["mom6_namelist"]["DAYMAX"], 
            len(observation_vector), config["eki"]["ens_size"],
            config["eki"]["outlier_scale"], config["eki"]["metrics_function"],
            config["eki"]["observation_vector"], config["eki"]["gamma_vector"],
            config["eki"]["observation_validation"], config["eki"]["gamma_validation"],
            )
        print('Passing forward model evaluations to the EKI')
        eki_update_ensemble(g_ens)
        print('Forward model evaluations are passed to the EKI; Parameters are updated')

        print('Saving EKI to disk')
        save_eki_on_disk(optimization_folder_pwd)
        print('Going to next iterations')

    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member in range(config["eki"]["ens_size"]):
            exp_path = f"{iteration_path}/ens-member-{ens_member:02d}"
            
            ########## Create a new ANN object with perturbed parameters #############
            ANN_modified = parameter_vector_to_ANN(ANN_netcdf_default, config["eki"]["trainable_parameters"], num_of_parameters, params[:,ens_member])

            ############ Create a callback function to assemble a regular ANN from equivariant ANN #############
            call_function = config["singularity_command"] + \
                            f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN={config["paths"]["ann"]}/Tall.nc --netcdf_eANN={exp_path}/INPUT/eANN.nc --netcdf_output={exp_path}/INPUT/Tall.nc"'

            ############ Create HPC profile ######################
            hpc = HPC.add(name=config["tag"], time=config["slurm_mom6"]["time"], begin='1minute', executable=config["paths"]["executable"])

            ########### Create MOM6 namelist #####################
            exp_params = PARAMETERS.add(**configuration('R2')).add(**config["mom6_namelist"])
            
            ########### Submit sbatch job ########################
            run_experiment(exp_path, hpc, exp_params,
                config["paths"]["configuration"],
                call_function)
            
            ########### Same euqivariant ANN on disk #############
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

print('Optimization is complete')