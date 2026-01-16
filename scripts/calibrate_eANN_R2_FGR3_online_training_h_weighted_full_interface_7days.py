import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python-jl calibrate_eANN_R2_FGR3_online_training_h_weighted_full_interface_7days.py --echo

## Global paths
TAG = '7df'
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/online_training'
optimization_folder = 'R2_FGR3_h_weighted_full_interface_7days'
ANN_default_path = '/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/learning_rate/N8-forcing-fluxes/0.05/model/'
this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
commandline = f'cd /home/pp2681/calibration/scripts; sbatch --time=00:30:00 --cpus-per-task=4 --mem=16GB --dependency=singleton --export=NONE --job-name={TAG} -o {base_path}/{optimization_folder}/slurm-%j.out -e {base_path}/{optimization_folder}/slurm-%j.err --wrap="python-jl {script_name}"'
# Here, we attenuate the spread of the initial ensemble as if not doing so, experiments explode
ENS_SPREAD = 0.3

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
exp_params = PARAMETERS.add(DAYMAX=7.0).add(USE_ZB2020='True',ZB2020_USE_ANN='True',ZB2020_ANN_FILE_TALL='INPUT/Tall.nc',USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset(f'{ANN_default_path}/eANN.nc').load()
# Read observation vector
observation = xr.open_dataset('/home/pp2681/calibration/scripts/R32/R2_FGR3_online_training.nc').isel(Time=slice(0,7))

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
for parameter_key in ['weights1', 'biases1', 'weights2', 'biases2']:
    initial_ensemble.append(generate_ensemble_for_parameter_vector(parameter_key))

initial_ensemble = np.concatenate(initial_ensemble)
print('default eANN path', ANN_default_path)
print('Initial enemble shape: ', initial_ensemble.shape)
print('SPREAD: ', ENS_SPREAD)

y = observation.e.values.ravel().astype('float64')

# Here we completely ignore the noise model
# We do not want to upweight bottom velocities
Gamma = observation.e_var.values.ravel().astype('float64')

from julia import Main

Main.eval("""
    using EnsembleKalmanProcesses, Random     
    using LinearAlgebra   
    Random.seed!(2)   # Fix random numbers globally
    """)

Main.y = y
Main.Γ = Gamma
Main.initial_ensemble = initial_ensemble

eki_state_file = f'{base_path}/{optimization_folder}/eki_state.jls' 
Main.eki_state_file = eki_state_file
if os.path.exists(eki_state_file):
    Main.eval("""
        using Serialization
        eki = deserialize(eki_state_file)
        """)
else:
    Main.eval("""
        eki = EnsembleKalmanProcess(
        initial_ensemble, y, Diagonal(Γ), TransformInversion(),
        scheduler = DefaultScheduler(1.0),
        #scheduler = DataMisfitController(terminate_at=1e+100),
        accelerator = DefaultAccelerator(),
        localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
        verbose=true)
        """)

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

if os.path.exists(f'{base_path}/{optimization_folder}/metrics.nc'):
    print('Metrics file already exists. Loading previous metrics...')
    metrics = xr.open_dataset(f'{base_path}/{optimization_folder}/metrics.nc').load()
else:
    metrics = xr.Dataset()
    Ntime = 7
    Nres = 1
    nzl = 2
    ny = 40
    nx = 44
    metrics['e'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, Ntime, nzl, ny, nx]), dims=['iter', 'ens', 'res', 'Time', 'zi', 'yh', 'xh'])
    metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, initial_ensemble.shape[0]]), dims=['iter', 'ens', 'pdim'])
    metrics['RMSE_e'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble, Nres, Ntime, nzl]), dims=['iter', 'ens', 'res', 'Time', 'zi'])
    metrics['WMSE'] = xr.DataArray(np.nan * np.zeros([N_iterations, N_ensemble]), dims=['iter', 'ens'])
    metrics['WMSE_MAP'] = xr.DataArray(np.nan * np.zeros([N_iterations]), dims=['iter'])
    metrics['latest_iteration'] = 0    

for iteration in range(int(metrics['latest_iteration'].item()), N_iterations):
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
                    data = xr.open_mfdataset(f'{iteration_path}/{RR}/ens-member-{ens_member:02d}/output/prog_*.nc', decode_times=False).sortby('Time').fillna(0.).isel(zi=slice(0,2))
                    y_prediction.append(data['e'].values.ravel().astype('float64'))

                    metrics['e'][iteration][ens_member][res_idx] = data['e'].values

                g_ens[:,ens_member] = np.concatenate(y_prediction)
            except:
                # Experiment is not ready or exploded or runtime error
                g_ens[:,ens_member] = np.nan
                metrics['e'][iteration][ens_member] = np.nan

            if np.isnan(g_ens[:,ens_member]).sum() > 0:
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')
            else:
                print(f'Ensemble member {ens_member} succesfully ingested')

            metrics['WMSE'][iteration][ens_member] = ((y - g_ens[:,ens_member])**2 / Gamma).mean()
                
            #Save parameter even if simulation exploded
            metrics['param'][iteration][ens_member] = param

        Main.g_ens = g_ens
        '''
        We know the contraction operator in ETKI method
        T = (I + dt G^T Γ^{-1} G)^{-1/2}
        If we simply set
        2 * trace(I) = dt * trace(G^T Γ^{-1} G),
        we will have constant contraction rate. In this case, with constant 2, the limiting contraction rate will be 0.33. It corresponds to rapid convergence but without immediate collapse of the ensemble.
        Contraction is related to shrinking directions of observational space which are observed.
        So the only criteria - is that something observed. Let's play with this.
        '''
        mask = ~np.isnan(g_ens[0,:])
        G = g_ens[:,mask] - np.nanmean(g_ens, axis=1).reshape(-1,1)
        Nens_nonan = mask.sum()
        G = G / np.sqrt(Nens_nonan - 1.0)

        dt = 2 * Nens_nonan / (np.trace(G.T @ (1. / Gamma[:,np.newaxis] * G)) + 1e-30)
        print('Computed trace-ratio ETKI step size dt = ', dt)
        Main.dt = dt

        Main.eval("update_ensemble!(eki, g_ens, Δt_new=dt)")

        os.system(f'rm -f {eki_state_file}')
        Main.eval("""
                using Serialization
                serialize(eki_state_file, eki)
            """)

        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')

        print('Computing and saving metrics')
        metrics['RMSE_e'] = np.sqrt(((metrics['e'] - observation['e'])**2).mean(dim=['xh','yh']))
        metrics['WMSE_MAP'][iteration] = ((y - np.nanmean(g_ens, axis=1))**2 / Gamma).mean()
        metrics['latest_iteration'] = metrics['latest_iteration'] + 1
        
        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member, param in enumerate(params.T):
            for RR in ['R2']:
                experiment_folder = f'{iteration_path}/{RR}/ens-member-{ens_member:02d}'
                weights_netcdf = ANN_netcdf_default.copy()
                weights_netcdf['weights1'] = xr.DataArray(param[:72], dims='pdim1')
                weights_netcdf['biases1'] = xr.DataArray(param[72:76], dims='pdim3')
                weights_netcdf['weights2'] = xr.DataArray(param[76:88], dims='pdim2')
                weights_netcdf['biases2'] = xr.DataArray(param[88:89], dims='pdim4')

                call_function = ('singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro '
                                '--bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ '
                                ' /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif '
                                f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN={ANN_default_path}/Tall.nc --netcdf_eANN={experiment_folder}/INPUT/eANN.nc --netcdf_output={experiment_folder}/INPUT/Tall.nc"')

                further_command = f'cp /scratch/pp2681/mom6/Feb2022/filtered/R32_R2_FGR3/RESTART_non_TWA/MOM_0.res.nc {experiment_folder}/INPUT/MOM.res.nc'

                if RR == 'R2':
                    hpc = HPC.add(name=TAG, time=1, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
                if RR == 'R4':
                    hpc = HPC.add(name=TAG, time=12, ntasks=4, mem=2, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
                if RR == 'R8':
                    hpc = HPC.add(name=TAG, time=48, ntasks=16, mem=4, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
                
                run_experiment(experiment_folder, hpc, exp_params.add(**configuration(RR)),
                    '/home/pp2681/MOM6-examples/build/configurations/double_gyre',
                    call_function, further_command)
                # We save data after initializing experiment to do not interrupt workflow.
                os.makedirs(f'{experiment_folder}/INPUT', exist_ok=True)
                weights_netcdf.astype('float32').to_netcdf(f'{experiment_folder}/INPUT/eANN.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code

# Run final experiement
exp_params = exp_params.add(DAYMAX=7300.0, ZB2020_ANN_FILE_TALL='../iteration-09/R2/ens-member-00/INPUT/Tall.nc')

hpc = HPC.add(name=TAG, time=1, ntasks=4, mem=2, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
experiment_folder = f'{base_path}/{optimization_folder}/final_experiment'
further_command = f'cp {experiment_folder}/diag_table.long {experiment_folder}/diag_table'
run_experiment(experiment_folder, hpc, exp_params.add(**configuration('R2')),
    '/home/pp2681/MOM6-examples/build/configurations/double_gyre', '', further_command)