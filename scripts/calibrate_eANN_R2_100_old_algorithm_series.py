import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python-jl calibrate_eANN_R2_100_old_algorithm_series.py --echo

## Global paths
TAG = 'ser'
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/hundred-years'
optimization_folder = 'R2-old-algorithm-series'
ANN_default_path = '/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/learning_rate/N8-forcing-fluxes/0.05/model/'
this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
commandline = f'cd /home/pp2681/calibration/scripts; sbatch --time=02:00:00 --mem=16GB --dependency=singleton --export=NONE --job-name={TAG} -o {base_path}/{optimization_folder}/slurm-%j.out -e {base_path}/{optimization_folder}/slurm-%j.err --wrap="python-jl {script_name}"'
# Here, we attenuate the spread of the initial ensemble as if not doing so, experiments explode
ENS_SPREAD = 0.25

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--echo", action="store_true",
    help="Print the command to start EKI optimization"
)
args = parser.parse_args()
if args.echo:
    print('To start the EKI optimization, run the following command')
    print(commandline)
    sys.exit(0)

# Model configuration
exp_params = PARAMETERS.add(DAYMAX=36500.0).add(USE_ZB2020='True',ZB2020_USE_ANN='True',ZB2020_ANN_FILE_TALL='INPUT/Tall.nc',USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset(f'{ANN_default_path}/eANN.nc').load()
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R64_R2/series.nc')

# EKI configuration
N_iterations = 10
N_ensemble = 100

np.random.seed(0)
# Initial ensemble for EKI
# 13 x 100 matrix; (For 64 neurons and N8 symmetries)
def generate_ensemble_for_parameter_vector(parameter_key):
    parameter_vector = ANN_netcdf_default[parameter_key].values
    n_param = len(parameter_vector)
    if n_param > 5:
        parameter_scale = float(parameter_vector.std())
    else:
        parameter_scale = float(np.abs(parameter_vector).mean())

    random_perturbation = ENS_SPREAD * parameter_scale * np.random.randn(n_param, N_ensemble)

    return parameter_vector.reshape(-1,1) + random_perturbation

initial_ensemble = []
for parameter_key in ['weights2', 'biases2']:
    initial_ensemble.append(generate_ensemble_for_parameter_vector(parameter_key))

initial_ensemble = np.concatenate(initial_ensemble)
print('default eANN path', ANN_default_path)
print('Initial enemble shape: ', initial_ensemble.shape)
print('SPREAD: ', ENS_SPREAD)

# Observation vector for EKI
# Each y1,...,y4 vector is 2 values
y1 = (observation.KE_mean).values.ravel().astype('float64')
y2 = (observation.KE_std).values.ravel().astype('float64')
y3 = (observation.APE_mean).values.ravel().astype('float64')
y4 = (observation.APE_std).values.ravel().astype('float64')
# 8 numbers in total
y = np.concatenate([y1,y2,y3,y4])

# Observation (+forward model) covariance matrix
# This factor multiplies the noise variance matrix by two, that way assuming that observation error
# and forward model error are independent and identically distributed
OBS_AND_FORWARD_FACTOR = 2.
var1 = (observation.KE_mean_var).values.ravel().astype('float64')
var2 = (observation.KE_std_var).values.ravel().astype('float64')
var3 = (observation.APE_mean_var).values.ravel().astype('float64')
var4 = (observation.APE_std_var).values.ravel().astype('float64')

# Observational noise for one resolution
Gamma = OBS_AND_FORWARD_FACTOR * np.concatenate([var1, var2, var3, var4])

from julia import Main

Main.eval("""
    using EnsembleKalmanProcesses, Random     
    using LinearAlgebra   
    Random.seed!(2)   # Fix random numbers globally
    """)

Main.y = y
Main.Γ = Gamma
Main.initial_ensemble = initial_ensemble

Main.eval("""
    eki = EnsembleKalmanProcess(
    initial_ensemble, y, Diagonal(Γ), Inversion(),
    #scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true)
    """)

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

metrics = xr.Dataset()
Nres = 1
nzl = 2
metrics['KE_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['KE_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['APE_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['APE_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, initial_ensemble.shape[0]]), dims=['iter', 'ens', 'pdim'])

for iteration in range(N_iterations):
    print(f'################ iteration {iteration} ####################')
    # Return the parameters in unconstrained space
    params = Main.eval("get_u_final(eki)")

    iteration_path = f'{base_path}/{optimization_folder}/iteration-{iteration:02d}'
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
        print('Folder with experiments exists')
        g_ens = np.zeros([len(y), N_ensemble]).astype('float64')
        for ens_member, param in enumerate(params.T):
            try:
                y_prediction = []
                for res_idx, coarse_factor, RR in zip([0], [1], ['R2']):
                    ds = xr.open_mfdataset(f'{iteration_path}/{RR}/ens-member-{ens_member:02d}/output/ocean.stats.nc', decode_times=False)
                    if len(ds.Time) == 36501:
                        # Select two upper layers and averaging interval
                        ds = ds.sel(Time=slice(365*10,None)).isel(Interface=slice(0,2)).astype('float64')
                        KE_mean = ds.KE.mean('Time').values
                        KE_std = ds.KE.std('Time').values
                        APE_mean = ds.APE.mean('Time').values
                        APE_std = ds.APE.std('Time').values
                        y_prediction.append(KE_mean)
                        y_prediction.append(KE_std)
                        y_prediction.append(APE_mean)
                        y_prediction.append(APE_std)
                    else:
                        raise Exception("Too short time series. Simulation exploded")

                    print(f'Ensemble member {ens_member} at {RR} succesfully ingested')

                    metrics['KE_mean'][iteration][ens_member][res_idx] = KE_mean
                    metrics['KE_std'][iteration][ens_member][res_idx] = KE_std
                    metrics['APE_mean'][iteration][ens_member][res_idx] = APE_mean
                    metrics['APE_std'][iteration][ens_member][res_idx] = APE_std

                g_ens[:,ens_member] = np.concatenate(y_prediction)
            except:
                # Experiment is not ready or exploded or runtime error
                g_ens[:,ens_member] = np.nan
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')
            
            #Save parameter even if simulation exploded
            metrics['param'][iteration][ens_member] = param
        
        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')

        Main.g_ens = g_ens
        Main.eval("update_ensemble!(eki, g_ens, deterministic_forward_map=false)")
        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member, param in enumerate(params.T):
            for RR in ['R2']:
                experiment_folder = f'{iteration_path}/{RR}/ens-member-{ens_member:02d}'
                weights_netcdf = ANN_netcdf_default.copy()
                weights_netcdf['weights2'] = xr.DataArray(param[:12], dims='pdim2')
                weights_netcdf['biases2'] = xr.DataArray(param[12:13], dims='pdim4')

                call_function = ('singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro '
                                '--bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ '
                                ' /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif '
                                f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN={ANN_default_path}/Tall.nc --netcdf_eANN={experiment_folder}/INPUT/eANN.nc --netcdf_output={experiment_folder}/INPUT/Tall.nc"')

                if RR == 'R2':
                    hpc = HPC.add(name=TAG, time=6, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
                if RR == 'R4':
                    hpc = HPC.add(name=TAG, time=12, ntasks=4, mem=2, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
                if RR == 'R8':
                    hpc = HPC.add(name=TAG, time=48, ntasks=16, mem=4, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
                
                run_experiment(experiment_folder, hpc, exp_params.add(**configuration(RR)), call_function)
                # We save data after initializing experiment to do not interrupt workflow.
                os.makedirs(f'{experiment_folder}/INPUT', exist_ok=True)
                weights_netcdf.astype('float32').to_netcdf(f'{experiment_folder}/INPUT/eANN.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code