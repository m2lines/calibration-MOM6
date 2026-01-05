import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python-jl calibrate_eANN_R2_FGR3_series_scale_simple_dt.py --echo

## Global paths
TAG = 'sdt'
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/20-years'
optimization_folder = 'R2_FGR3_series_scale_simple_dt'
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
exp_params = PARAMETERS.add(DAYMAX=7300.0).add(USE_ZB2020='True',ZB2020_USE_ANN='True',ZB2020_ANN_FILE_TALL='INPUT/Tall.nc',USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset(f'{ANN_default_path}/eANN.nc').load()
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R32/R2_FGR3_series_scale.nc').astype('float64')

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
# 1760*2 values of interface mean
y1 = (observation.e_mean).values.ravel()
# 1760*2 values of interface std
y2 = (observation.e_std).values.ravel()
# Additional 8 numbers for energetics
y3 = (observation.KE_mean).values.ravel()
y4 = (observation.KE_std).values.ravel()
y5 = (observation.APE_mean).values.ravel()
y6 = (observation.APE_std).values.ravel()
y = np.concatenate([y1,y2,y3,y4,y5,y6])

# Inverse number of degrees of freedom.
# This scaling factor converts 7048 degrees
# of freedom to 12 degrees of freedom
# It is used mostly for convenience of interpretation of the
# result. The scaling factor appears that 
# some groups of variables (2d fields) we consider as one
# degree of freedom. In total we have 8 scalar variables
# and 4 2d fields, thus the actual number of degrees of freedom is 12
SCALING_FACTOR = 12. / (40*44*4+8)
var1 = (observation.e_mean_var_scale).values.ravel()
var2 = (observation.e_std_var_scale).values.ravel()
# Additional variances for energetics
var3 = (observation.KE_mean_var_scale).values.ravel()
var4 = (observation.KE_std_var_scale).values.ravel()
var5 = (observation.APE_mean_var_scale).values.ravel()
var6 = (observation.APE_std_var_scale).values.ravel()
Gamma = SCALING_FACTOR * np.concatenate([var1,var2,var3,var4,var5,var6])

from julia import Main

Main.eval("""
    using EnsembleKalmanProcesses, Random     
    using LinearAlgebra   
    Random.seed!(2)   # Fix random numbers globally
    """)

Main.y = y
Main.Γ = Gamma
Main.initial_ensemble = initial_ensemble

# Here, we use default scheduler because we wish to manually
# update the "time step" based on our own learning rate scheduler
# which ensures that we do not make too big jumps in parameter space
Main.eval("""
    eki = EnsembleKalmanProcess(
    initial_ensemble, y, Diagonal(Γ), TransformInversion(),
    scheduler =  DefaultScheduler(1.0),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true)
    """)

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

metrics = xr.Dataset()
Nres = 1
nzl = 2
ny = 40
nx = 44
metrics['e_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl, ny, nx]), dims=['iter', 'ens', 'res', 'zi', 'yh', 'xh'])
metrics['e_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl, ny, nx]), dims=['iter', 'ens', 'res', 'zi', 'yh', 'xh'])
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, initial_ensemble.shape[0]]), dims=['iter', 'ens', 'pdim'])
metrics['RMSE_e_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zi'])
metrics['RMSE_e_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zi'])

metrics['KE_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['KE_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['APE_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zi'])
metrics['APE_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zi'])

metrics['RMSE_KE_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['RMSE_KE_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zl'])
metrics['RMSE_APE_mean'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zi'])
metrics['RMSE_APE_std'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, nzl]), dims=['iter', 'ens', 'res', 'zi'])

metrics['WMSE'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble]), dims=['iter', 'ens'])
metrics['WMSE_MAP'] = xr.DataArray(np.nan * np.zeros([N_iterations]), dims=['iter'])

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
            y_prediction = []
            for res_idx, coarse_factor, RR in zip([0], [1], ['R2']):
                try:
                    e = xr.open_mfdataset(f'{iteration_path}/{RR}/ens-member-{ens_member:02d}/output/prog_*.nc', decode_times=False).e.sel(Time=slice(365*10,None)).isel(zi=slice(0,2)).astype('float64')
                    e_mean = e.mean('Time')
                    e_std = e.std('Time')
                except:
                    e_mean = np.nan * np.zeros((nzl,ny,nx))
                    e_std = np.nan * np.zeros((nzl,ny,nx))

                series = xr.open_mfdataset(f'{iteration_path}/{RR}/ens-member-{ens_member:02d}/output/ocean.stats.nc', decode_times=False).astype('float64')
                if len(series.Time) == 7301:
                    series = series.sel(Time=slice(365*10,None)).isel(Interface=slice(0,2)).rename({'Layer': 'zl', 'Interface': 'zi'})[['KE', 'APE']]
                    series = series * 1e-15 # Convert energetics to PJ
                    KE_mean = series.KE.mean('Time').values
                    KE_std = series.KE.std('Time').values
                    APE_mean = series.APE.mean('Time').values
                    APE_std = series.APE.std('Time').values
                else:
                    KE_mean = np.nan * np.zeros(2)
                    KE_std = np.nan * np.zeros(2)
                    APE_mean = np.nan * np.zeros(2)
                    APE_std = np.nan * np.zeros(2)

                y_prediction.append(e_mean.values.ravel())
                y_prediction.append(e_std.values.ravel())
                y_prediction.append(KE_mean)
                y_prediction.append(KE_std)
                y_prediction.append(APE_mean)
                y_prediction.append(APE_std)

                metrics['e_mean'][iteration][ens_member][res_idx] = e_mean
                metrics['e_std'][iteration][ens_member][res_idx] = e_std
                metrics['KE_mean'][iteration][ens_member][res_idx] = KE_mean
                metrics['KE_std'][iteration][ens_member][res_idx] = KE_std
                metrics['APE_mean'][iteration][ens_member][res_idx] = APE_mean
                metrics['APE_std'][iteration][ens_member][res_idx] = APE_std

            g_ens[:,ens_member] = np.concatenate(y_prediction)
            
            if np.isnan(g_ens[:,ens_member]).sum() > 0:
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')
            else:
                print(f'Ensemble member {ens_member} succesfully ingested')

            metrics['WMSE'][iteration][ens_member] = ((y - g_ens[:,ens_member])**2 / Gamma).mean()
            
            #Save parameter even if simulation exploded
            metrics['param'][iteration][ens_member] = param

        # Filter out outliers
        MAX_RELATIVE_LOSS = 10
        relative_loss = metrics['WMSE'][iteration] / metrics['WMSE'][iteration].min()
        mask_out = relative_loss > MAX_RELATIVE_LOSS
        g_ens[:,mask_out] = np.nan
        for variable in ['WMSE', 'e_mean', 'e_std', 'KE_mean', 'KE_std', 'APE_mean', 'APE_std']:
            metrics[variable][iteration][mask_out] = np.nan
        
        print(f'Filtered out {mask_out.sum().item()} outliers from the ensemble')
        print('Their indices: ', np.where(mask_out)[0])

        print('Computing and saving metrics')
        metrics['RMSE_e_mean'] = np.sqrt(((metrics['e_mean'] - observation['e_mean'])**2).mean(dim=['xh','yh']))
        metrics['RMSE_e_std'] = np.sqrt(((metrics['e_std'] - observation['e_std'])**2).mean(dim=['xh','yh']))
        
        metrics['RMSE_KE_mean'] = np.sqrt((metrics['KE_mean']- observation['KE_mean'])**2)
        metrics['RMSE_APE_mean'] = np.sqrt((metrics['APE_mean']- observation['APE_mean'])**2)
        metrics['RMSE_KE_std'] = np.sqrt((metrics['KE_std']- observation['KE_std'])**2)
        metrics['RMSE_APE_std'] = np.sqrt((metrics['APE_std']- observation['APE_std'])**2)
        
        metrics['WMSE_MAP'][iteration] = ((y - np.nanmean(g_ens, axis=1))**2 / Gamma).mean()
        
        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')

        # Compute trace of C_yy
        tr_C_yy = np.sum(np.nanvar(g_ens, axis=1))
        # Compute trace of Gamma
        tr_Gamma = np.sum(Gamma)
        # Make sure that traces are of the same order
        # tr_C_yy = 1/dt * tr_Gamma
        dt = tr_Gamma / tr_C_yy

        print('EKI time step', dt)
        Main.dt = dt

        Main.g_ens = g_ens
        Main.eval("update_ensemble!(eki, g_ens, Δt_new=dt)")
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
                    hpc = HPC.add(name=TAG, time=2, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
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
