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
TAG = 'eki4'
hpc = HPC.add(name=TAG, time=6, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/variability-R4'
optimization_folder = 'EKI-Vanilla-Gamma-e-mean-std-spread-0.1'
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
ANN_params = PARAMETERS.add(**configuration('R4')).add(DAYMAX=3650.0).add(USE_ZB2020='True',ZB2020_USE_ANN='True',ZB2020_ANN_FILE_TALL='INPUT/Tall.nc',USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset('/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/EXP1/model/Tall.nc').drop_vars(['x_test', 'y_test'])
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R64_R4/full.nc')

# EKI configuration
N_iterations = 4
N_ensemble = 100

# Initial ensemble for EKI
A1_mean = ANN_netcdf_default['A1'].values.reshape(-1)
b1_mean = ANN_netcdf_default['b1'].values.reshape(-1)
A1_std = ENS_SPREAD * ANN_netcdf_default['A1'].values.std()
b1_std = ENS_SPREAD * ANN_netcdf_default['b1'].values.std()

np.random.seed(0)
# 63 x 100 matrix
# Where 63 is the number of free parameters and 100 is the ensemble size
# Note: we do computations here in float64, but later convert to float32 for online experiments
initial_ensemble = np.concatenate([A1_mean, b1_mean]).reshape(-1,1) + np.concatenate([A1_std * np.random.randn(len(A1_mean), N_ensemble), b1_std * np.random.randn(len(b1_mean), N_ensemble)]).astype('float64')

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

############ Add covariance matrix of noise #################
# we assumed that collapsed ensemble from vanilla EKI can serve as 
# a proxy for covariance matrix estimate for the noise in the forward model
metrics_R4 = xr.open_dataset('/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/variability-R4/EKI-Vanilla-e-mean-std-spread-0.1/metrics.nc')
g_ens =  np.vstack([np.array(metrics_R4['e_mean'].isel(iter=-1)).reshape([100, -1]).astype('float64').T,
                    np.array(metrics_R4['e_std'].isel(iter=-1)).reshape([100, -1]).astype('float64').T])
mean=lambda x: x.mean(axis=-1,keepdims=True)
cov_yy_R4 = (g_ens-mean(g_ens))   @ (g_ens-mean(g_ens)).T / 100
# Here we add only one copy of covariance matrix because we do not have
# a similar matrix for noise of high-resolution model
Gamma += cov_yy_R4

# Enforce symmetry
Gamma = 0.5 * (Gamma + Gamma.T)

print('Full Gamma shape', Gamma.shape)

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
nfreq_r = 12
ny = 10
nx = 11
metrics['e_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, ny, nx]), dims=['iter', 'ens', 'zi', 'yh', 'xh'])
metrics['e_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, ny, nx]), dims=['iter', 'ens', 'zi', 'yh', 'xh'])
metrics['EKE_spectrum'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, nfreq_r]), dims=['iter', 'ens', 'zl', 'freq_r'])
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, 63]), dims=['iter', 'ens', 'pdim'])

# Copy true metrics
metrics['e_std_true'] = observation.e_std
metrics['e_mean_true'] = observation.e_mean
metrics['EKE_spectrum_true'] = observation.EKE_spectrum
metrics['e_std_var_ave'] = observation.e_std_var_ave
metrics['e_mean_var_ave'] = observation.e_mean_var_ave
metrics['EKE_spectrum_var_ave'] = observation.EKE_spectrum_var_ave

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
                data = variability_metrics(ds.e, ds.u, ds.v, static, coarse_factor=8, compute_e=True)
                
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
            A1 = param[:-3]
            b1 = param[-3:]
            ANN_netcdf = ANN_netcdf_default.copy()
            ANN_netcdf['A1'] = ANN_netcdf['A1']*0 + A1.reshape([20,3]).astype('float32').copy()
            ANN_netcdf['b1'] = ANN_netcdf['b1']*0 + b1.astype('float32').copy()
            run_experiment(experiment_folder, hpc, ANN_params)
            # We save data after initializing experiment to do not interrupt workflow.
            os.makedirs(f'{experiment_folder}/INPUT', exist_ok=True)
            ANN_netcdf.to_netcdf(f'{experiment_folder}/INPUT/Tall.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code