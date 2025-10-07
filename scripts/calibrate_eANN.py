import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python-jl calibrate_eANN.py --echo

## Global paths
TAG = 'eANN'
hpc = HPC.add(name=TAG, time=2, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/variability-R2-eANN'
optimization_folder = 'EKI-Vanilla-N8-forcing-spread-0-1'
this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
commandline = f'cd /home/pp2681/calibration/scripts; sbatch --mem=16GB --dependency=singleton --export=NONE --job-name={TAG} -o {base_path}/{optimization_folder}/slurm-%j.out -e {base_path}/{optimization_folder}/slurm-%j.err --wrap="python-jl {script_name}"'
# Here, we attenuate the spread of the initial ensemble as if not doing so, experiments explode
ENS_SPREAD = 0.1

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
exp_params = PARAMETERS.add(**configuration('R2')).add(DAYMAX=3650.0).add(USE_ZB2020='True',ZB2020_USE_ANN='True',ZB2020_ANN_FILE_TALL='INPUT/Tall.nc',USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset('/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/16-N8-forcing/model/weights_and_perturbations.nc').load()
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R64_R2/full.nc')

# EKI configuration
N_iterations = 5
N_ensemble = 100

np.random.seed(0)
# Initial ensemble for EKI
# 23 x 100 matrix
weights1 = ANN_netcdf_default['weights1'] + ENS_SPREAD * xr.DataArray(ANN_netcdf_default['weights1_std'].mean().values * np.random.randn(len(ANN_netcdf_default['weights1']), N_ensemble), dims=['pdim1', 'ens'])
weights2 = ANN_netcdf_default['weights2'] + ENS_SPREAD * xr.DataArray(ANN_netcdf_default['weights2_std'].mean().values * np.random.randn(len(ANN_netcdf_default['weights2']), N_ensemble), dims=['pdim2', 'ens'])
biases1 = ANN_netcdf_default['biases1'] + ENS_SPREAD * xr.DataArray(ANN_netcdf_default['biases1_std'].mean().values * np.random.randn(len(ANN_netcdf_default['biases1']), N_ensemble), dims=['pdim3', 'ens'])
biases2 = ANN_netcdf_default['biases2'] + ENS_SPREAD * xr.DataArray(ANN_netcdf_default['biases2_std'].mean().values * np.random.randn(len(ANN_netcdf_default['biases2']), N_ensemble), dims=['pdim4', 'ens'])

initial_ensemble = np.concatenate([weights1.values, weights2.values, biases1.values, biases2.values])
print('Initial enemble shape: ', initial_ensemble.shape)
print('SPREAD: ', ENS_SPREAD)

# Observation vector for EKI
# 220 values of ssh mean
y1 = (observation.e_mean).values.ravel().astype('float64')
# 220 values of ssh std
y2 = (observation.e_std).values.ravel().astype('float64')
y = np.concatenate([y1,y2])

def MSE_weighted(e_mean, e_std):
    obs_length = len(y)
    out =  ((e_mean - observation.e_mean)**2 / observation.e_mean_var_ave + \
            (e_std  - observation.e_std )**2 / observation.e_mean_var_ave).sum(['xh', 'yh', 'zi']) / obs_length
    return xr.where(out > 0, out, np.nan)

# Observation (+forward model) covariance matrix
# This factor multiplies the noise variance matrix by two, that way assuming that observation error
# and forward model error are independent and identically distributed
OBS_AND_FORWARD_FACTOR = 2.
# We multiply the noise variance of spatial field by this factor to
# reduce its contribution to the loss related to different number of elements (i.e., MSE instead of SSE)
var1 = (observation.e_mean_var_ave).values.ravel().astype('float64')
var2 = (observation.e_std_var_ave).values.ravel().astype('float64')
Gamma = OBS_AND_FORWARD_FACTOR * np.diag(np.concatenate([var1, var2]))

from julia import Main

Main.eval("""
    using EnsembleKalmanProcesses, Random        
    Random.seed!(2)   # Fix random numbers globally
    """)

Main.y = y
Main.Γ = Gamma
Main.initial_ensemble = initial_ensemble

Main.eval("""
    eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true)
    """)

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

metrics = xr.Dataset()
nzl = 2
ny = 10
nx = 11
nfreq_r = 5
metrics['e_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, ny, nx]), dims=['iter', 'ens', 'zi', 'yh', 'xh'])
metrics['e_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, ny, nx]), dims=['iter', 'ens', 'zi', 'yh', 'xh'])
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, 23]), dims=['iter', 'ens', 'pdim'])
metrics['EKE_spectrum'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, nfreq_r]), dims=['iter', 'ens', 'zl', 'freq_r'])
metrics['WMSE'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble]), dims=['iter', 'ens'])
metrics['WMSE_MAP'] = xr.DataArray(np.nan * np.zeros([N_iterations]), dims='iter')

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
        g_ens = np.zeros([440, N_ensemble]).astype('float64')
        for ens_member, param in enumerate(params.T):
            try:
                ds = xr.open_mfdataset(f'{iteration_path}/ens-member-{ens_member:02d}/output/prog_*.nc', decode_times=False)
                static = xr.open_mfdataset(f'{iteration_path}/ens-member-{ens_member:02d}/output/ocean_geometry.nc', decode_times=False).rename({'lonh': 'xh', 'lath': 'yh'})
                data = variability_metrics(ds.e, ds.u, ds.v, static, coarse_factor=4, compute_e=True)
                
                y1 = (data.e_mean).values.ravel().astype('float64')
                y2 = (data.e_std).values.ravel().astype('float64')
                g_ens[:,ens_member] = np.concatenate([y1,y2])
                print(f'Ensemble member {ens_member} succesfully ingested')

                metrics['EKE_spectrum'][iteration][ens_member] = data.EKE_spectrum
                metrics['e_mean'][iteration][ens_member] = data.e_mean
                metrics['e_std'][iteration][ens_member] = data.e_std
            except:
                # Experiment is not ready or exploded or runtime error
                g_ens[:,ens_member] = np.nan
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')
            # Save parameter even if simulation exploded
            metrics['param'][iteration][ens_member] = param
        
        print('Computing and saving metrics')
        metrics['WMSE'] = MSE_weighted(metrics['e_mean'], metrics['e_std'])
        metrics['WMSE_MAP'] = MSE_weighted(metrics['e_mean'].mean('ens'), metrics['e_std'].mean('ens'))

        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')

        Main.g_ens = g_ens
        Main.eval("update_ensemble!(eki, g_ens, deterministic_forward_map=false)")
        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member, param in enumerate(params.T):
            experiment_folder = f'{iteration_path}/ens-member-{ens_member:02d}'
            weights_netcdf = xr.Dataset()
            weights_netcdf['weights1'] = xr.DataArray(param[:18], dims='pdim1')
            weights_netcdf['weights2'] = xr.DataArray(param[18:21], dims='pdim2')
            weights_netcdf['biases1'] = xr.DataArray(param[21:22], dims='pdim3')
            weights_netcdf['biases2'] = xr.DataArray(param[22:23], dims='pdim4')

            call_function = ('singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro '
                             '--bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ '
                             ' /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif '
                             f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN=/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/16-N8-forcing/model/Tall.nc --netcdf_eANN={experiment_folder}/INPUT/eANN.nc --netcdf_output={experiment_folder}/INPUT/Tall.nc"')
            run_experiment(experiment_folder, hpc, exp_params, call_function)
            # We save data after initializing experiment to do not interrupt workflow.
            os.makedirs(f'{experiment_folder}/INPUT', exist_ok=True)
            weights_netcdf.astype('float32').to_netcdf(f'{experiment_folder}/INPUT/eANN.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code
