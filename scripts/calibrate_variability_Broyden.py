import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python calibrate_variability_Broyden.py --echo #

## Global paths
TAG = 'brdlm'
hpc = HPC.add(name=TAG, time=1, ntasks=3, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/variability-R2'
optimization_folder = 'Broyden-LM-1'
this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
commandline = f'cd /home/pp2681/calibration/scripts; sbatch --mem=16GB --dependency=singleton --export=NONE --job-name={TAG} -o {base_path}/{optimization_folder}/slurm-%j.out -e {base_path}/{optimization_folder}/slurm-%j.err --wrap="python-jl {script_name}"'
# Here, we attenuate the spread of the initial ensemble as if not doing so, experiments explode

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--echo", action="store_true",
    help="Print the command to start Broyden optimization"
)
args = parser.parse_args()
if args.echo:
    print('To start the Broyden optimization, run the following command')
    print(commandline)
    sys.exit(0)

# Model configuration
ANN_params = PARAMETERS.add(**configuration('R2')).add(DAYMAX=3650.0).add(USE_ZB2020='True',ZB2020_USE_ANN='True',ZB2020_ANN_FILE_TALL='INPUT/Tall.nc',USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset('/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/EXP1/model/Tall.nc').drop_vars(['x_test', 'y_test'])
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R64_R2/variability-2-degrees.nc')

# Broyden configuration
# As Broyden method converges very fast, 5 iterations should be enough.
# Also, because Broyden method reduces step length if Jacobian is noisy,
# We will likely see a stagnation of iteration pretty soon. Nevertheless
# To investigate the convergence properties, we will do 10 iterations.
# This might be justified by the fact that the optimization problem
# is strongly nonlinear.
N_iterations = 10
# We understand the ensemble size of Broyden method as the number of initial conditions
# used to restart the iteration.
# For simplicity, we start with the simplest approach or random sampling.
N_ensemble = 92
# Note that starting from a few best points or a few points closest to the 
# untuned model might be preferable to explore in future

# Levenberg-Marquardt Optimization
# We assume that EKI is equivalent to 
# LM algorithm with regularization matrix which is given by inverse 
# covariance matrix of x
# See "Iterative Ensemble Kalman Methods: A Unified Perspective with Some New Variants"
# 3.7-3.8 equations
LM_lambda = 1.

# Observation vector for Broyden method
# 10 values of EKE spectrum
y = (observation.EKE_spectrum).values.ravel().astype('float64')

OBS_AND_FORWARD_FACTOR = 2.
y_var = OBS_AND_FORWARD_FACTOR * (observation.EKE_spectrum_var_ave).values.ravel().astype('float64')
# Inverse Matrix of obs+model error covariance
Gamma_inv = np.diag(1./y_var)

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

initial_ensemble = xr.open_dataset(f'{base_path}/{optimization_folder}/initial_ensemble.nc')
# Forming initial approximation of the Jacobian using all available data
# arrays of size N_params x ens_size
initial_ens_size = len(initial_ensemble.ens)
params = np.array(initial_ensemble['param']).astype('float64').T
g_ens =  np.array(initial_ensemble['EKE_spectrum']).reshape([initial_ens_size, -1]).astype('float64').T

# find ensemble members (columns) that contain NaNs
bad_mask = np.any(np.isnan(g_ens), axis=0)

# keep only good members
params = params[:, ~bad_mask]
g_ens = g_ens[:, ~bad_mask]
initial_ens_size = params.shape[-1]

mean=lambda x: x.mean(axis=-1,keepdims=True)
cov_xx = (params-mean(params)) @ (params-mean(params)).T / initial_ens_size
# We know that the initial ensemble is sampled from gaussian disttibution
# with independent random variables along each parameter.
# Thus, such Jacobian estimator is simply equivalent to fitting the linear
# slope to a scatter plot of parameter_i x observation_component_j
# Such slope estimation is very similar to estimation of slope with finite differences
# Estimation in such form is very simple to analyze analytically
# This feature probably was found only to destroy the computation
#cov_xx = np.diag(np.diag(cov_xx))
cov_yx = (g_ens-mean(g_ens))   @ (params-mean(params)).T / initial_ens_size

# Initial Jacobian computed from big ensemble
cov_xx_inv = np.linalg.pinv(cov_xx, rcond=1e-8)
J_initial = cov_yx @ cov_xx_inv

# Forming set of initial conditions
idx = np.sort(np.random.choice(initial_ens_size, N_ensemble, replace=False))
print('Initial ensemble indices:', idx)
# N_params x N_ensemble
# Here we choose among ensemble members which did not explode
params = params[:,idx]
g_ens = g_ens[:, idx]
J_ens = np.stack([J_initial.copy() for _ in range(N_ensemble)], axis=2)
print(params.shape, g_ens.shape, J_ens.shape)

metrics = xr.Dataset()
nzl = 2
nfreq_r = 5
metrics['EKE_spectrum'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, nzl, nfreq_r]), dims=['iter', 'ens', 'zl', 'freq_r'])
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, 63]), dims=['iter', 'ens', 'pdim'])
metrics['EKE_spectrum'][0,:,:,:] = initial_ensemble['EKE_spectrum'][~bad_mask][idx]
metrics['param'][0,:,:] = initial_ensemble['param'][~bad_mask][idx]

# We start from the 1st iteration because 0th iteration is already computed
for iteration in range(1,N_iterations):
    print(f'################ iteration {iteration} ####################')
    # Query parameter vector to compute
    delta_params = np.zeros_like(params)
    for ens_member in range(N_ensemble):
        param = params[:,ens_member]
        g = g_ens[:,ens_member]
        J = J_ens[:,:,ens_member]
        # Here, we can think of pinv as Levenberg-Marquardt update with
        # regularization parameter approaching zero. Because uninvertible
        # matrix can potentially lead to infinitely large parameter update,
        # we can see this as a natural damping regularization where Levenberg-Marquardt
        # suggests infinitely small parameter updates with infinite regularization parameter
        delta_params[:,ens_member] = np.linalg.pinv(J.T@Gamma_inv@J + LM_lambda * cov_xx_inv, rcond=1e-8)@(J.T@Gamma_inv@(y - g))
        # Note: a good property of the good Broyden method is that there is no
        # scalar product in the observational space. Thus, we do not need to
        # account for Gamma matrix in the Jacobian update. 
    params = params + delta_params

    iteration_path = f'{base_path}/{optimization_folder}/iteration-{iteration:02d}'
    params_file = f'{iteration_path}-params.txt'

    if np.isnan(params).sum() == params.size:
        print('All parameters are NaNs. Stopping the optimization')
        sys.exit(1)   # terminate immediately with error code

    if not(os.path.exists(params_file)):
        print('Saving parameters to file', params_file)
        np.savetxt(params_file, params)
    else:
        params_old = np.loadtxt(params_file)
        if not(np.allclose(np.nan_to_num(params,1e+30), np.nan_to_num(params_old,1e+30))):
            print('Parameters changed! Check the optimization algorithm.')
            sys.exit(1)   # terminate immediately with error code
        else:
            print('Parameters are the same. Keep going...')

    if os.path.exists(iteration_path):
        print('Folder with experiments exists')
        for ens_member, param in enumerate(params.T):
            try:
                ds = xr.open_mfdataset(f'{iteration_path}/ens-member-{ens_member:02d}/output/prog_*.nc', decode_times=False)
                static = xr.open_mfdataset(f'{iteration_path}/ens-member-{ens_member:02d}/output/ocean_geometry.nc', decode_times=False).rename({'lonh': 'xh', 'lath': 'yh'})
                data = variability_metrics(ds.e, ds.u, ds.v, static)
                
                g = (data.EKE_spectrum).values.ravel().astype('float64')
                delta_g = g - g_ens[:,ens_member]
                delta_p = delta_params[:,ens_member]
                J = J_ens[:,:,ens_member]
                # Update Jacobian with good Broyden formula
                J_ens[:,:, ens_member] += np.outer(delta_g-J@delta_p, delta_p) / np.dot(delta_p,delta_p)
                # Update g_ens
                g_ens[:,ens_member] = g
                # Note: parameter vector is already update above

                print(f'Ensemble member {ens_member} succesfully ingested')

                metrics['EKE_spectrum'][iteration][ens_member] = data.EKE_spectrum
            except:
                # Experiment is not ready or exploded or runtime error
                g_ens[:,ens_member] = np.nan
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')
            # Save parameter even if simulation exploded
            metrics['param'][iteration][ens_member] = param

        print('Forward model evaluations are passed through Broyden Jacobian update. Going to the next iterations...')
        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member, param in enumerate(params.T):
            experiment_folder = f'{iteration_path}/ens-member-{ens_member:02d}'
            if np.isnan(param).any():
                print(f'Parameter vector for {ens_member} contains NaNs. Skipping the experiment')
                continue
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