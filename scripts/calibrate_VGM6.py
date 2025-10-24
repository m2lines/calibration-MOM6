import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python-jl calibrate_variability_multiobjective_DG.py --echo

## Global paths
TAG = 'vgm6'
hpc = HPC.add(name=TAG, time=6, executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-VGM6-Sep22-ver3')
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/variability-R2-VGM6'
optimization_folder = 'EKI-Full'
this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
commandline = f'cd /home/pp2681/calibration/scripts; sbatch --mem=16GB --dependency=singleton --export=NONE --job-name={TAG} -o {base_path}/{optimization_folder}/slurm-%j.out -e {base_path}/{optimization_folder}/slurm-%j.err --wrap="python-jl {script_name}"'
# Here, we attenuate the spread of the initial ensemble as if not doing so, experiments explode

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
exp_params = PARAMETERS.add(**configuration('R2')).add(DAYMAX=3650.0).add(USE_ZB2020='True', ZB_SCALING=1.0, ZB2020_USE_VGM6='True', VGM6_c1 = 0.0, VGM6_c2 = 0.0, VGM6_c3 = 0.0, VGM6_c4 = 0.0)
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R64_R2/full.nc')

# EKI configuration
N_iterations = 3
N_ensemble = 40

# Observation vector for EKI
# 220 values of ssh mean
y1 = (observation.e_mean).values.ravel().astype('float64')
# 220 values of ssh std
y2 = (observation.e_std).values.ravel().astype('float64')
y = np.concatenate([y1,y2])

# Observation (+forward model) covariance matrix
# This factor multiplies the noise variance matrix by two, that way assuming that observation error
# and forward model error are independent and identically distributed
OBS_AND_FORWARD_FACTOR = 2.
# We multiply the noise variance of spatial field by this factor to
# reduce its contribution to the loss related to different number of elements (i.e., MSE instead of SSE)
var1 = (observation.e_mean_var_ave).values.ravel().astype('float64')
var2 = (observation.e_std_var_ave).values.ravel().astype('float64')
Gamma = OBS_AND_FORWARD_FACTOR * np.diag(np.concatenate([var1, var2]))

print('diagonal Gamma shape', Gamma.shape)

from julia import Main

Main.eval("""
    using EnsembleKalmanProcesses, Random        
    Random.seed!(2)   # Fix random numbers globally
    """)

Main.y = y
Main.Γ = Gamma
Main.N_ensemble = N_ensemble

Main.eval("""
    using EnsembleKalmanProcesses.ParameterDistributions
    prior_1 = constrained_gaussian("VGM2", 1, 0.5, 0.0, 2.0)
    prior_2 = constrained_gaussian("VGM4", 1, 0.5, 0.0, 2.0)
    prior_3 = constrained_gaussian("VGM6", 1, 0.5, 0.0, 2.0)
    prior = combine_distributions([prior_1, prior_2, prior_3])

    initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
    """)

Main.eval("""
    eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true)
    """)

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

metrics = xr.Dataset()
nzl = 2
nfreq_r = 5
ny = 10
nx = 11
metrics['e_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, ny, nx]), dims=['iter', 'ens', 'zi', 'yh', 'xh'])
metrics['e_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, ny, nx]), dims=['iter', 'ens', 'zi', 'yh', 'xh'])
metrics['EKE_spectrum'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, nfreq_r]), dims=['iter', 'ens', 'zl', 'freq_r'])
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, 3]), dims=['iter', 'ens', 'pdim'])

for iteration in range(N_iterations):
    print(f'################ iteration {iteration} ####################')
    # Return the parameters in unconstrained space
    params = Main.eval("get_ϕ_final(prior, eki)")

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

        Main.g_ens = g_ens
        Main.eval("update_ensemble!(eki, g_ens, deterministic_forward_map=false)")
        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')
        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member, param in enumerate(params.T):
            experiment_folder = f'{iteration_path}/ens-member-{ens_member:02d}'
            MOM6_parameters = exp_params.add(
                VGM6_c1 = param[0],
                VGM6_c2 = param[1],
                VGM6_c3 = param[2]
            )

            run_experiment(experiment_folder, hpc, MOM6_parameters)
            # We save data after initializing experiment to do not interrupt workflow.
            os.makedirs(f'{experiment_folder}/INPUT', exist_ok=True)

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code
